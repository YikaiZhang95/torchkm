#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchkm.estimators import (  # noqa: E402
    TorchKMDWD,
    TorchKMLogit,
    TorchKMSVC,
    _check_binary_y,
    _make_foldid,
    _make_ulam,
    _pick_device_str,
)
from torchkm.functions import data_gen, standardize  # noqa: E402


BACKEND_MAP = {
    "svc": TorchKMSVC,
    "dwd": TorchKMDWD,
    "logit": TorchKMLogit,
}

MODE_CHOICES = ("exact", "low_rank")
SIZE_PRESETS = {
    "smoke": {
        "n_train": 320,
        "n_test": 96,
        "nm": 5,
        "p": 10,
        "mu": 2.0,
        "ro": 3.0,
        "nC": 4,
        "nfolds": 3,
        "max_iter": 80,
        "num_landmarks": 96,
        "nys_k": 48,
    },
    "medium": {
        "n_train": 800,
        "n_test": 200,
        "nm": 6,
        "p": 16,
        "mu": 2.0,
        "ro": 3.0,
        "nC": 6,
        "nfolds": 5,
        "max_iter": 140,
        "num_landmarks": 160,
        "nys_k": 96,
    },
    "stress": {
        "n_train": 1400,
        "n_test": 280,
        "nm": 8,
        "p": 20,
        "mu": 2.0,
        "ro": 3.0,
        "nC": 8,
        "nfolds": 5,
        "max_iter": 220,
        "num_landmarks": 256,
        "nys_k": 128,
    },
}


@dataclass(frozen=True)
class RunCase:
    backend: str
    mode: str
    size: str
    is_exact: int = 0

    @property
    def label(self) -> str:
        suffix = f"-exact{self.is_exact}" if self.backend == "svc" and self.mode == "exact" else ""
        return f"{self.size}-{self.backend}-{self.mode}{suffix}"


class TimingAccumulator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seconds: Dict[str, float] = {}

    def synchronize(self) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def add(self, name: str, seconds: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds

    def wrap_function(self, name: str, fn):
        def wrapped(*args, **kwargs):
            self.synchronize()
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            self.synchronize()
            self.add(name, time.perf_counter() - start)
            return result

        return wrapped

    def wrap_method(self, name: str, method):
        def wrapped(instance, *args, **kwargs):
            self.synchronize()
            start = time.perf_counter()
            result = method(instance, *args, **kwargs)
            self.synchronize()
            self.add(name, time.perf_counter() - start)
            return result

        return wrapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark torchkm wrappers and low-level backends.")
    parser.add_argument("--device", default="cuda", help="Device string passed to torchkm estimators.")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["svc", "dwd", "logit"],
        choices=sorted(BACKEND_MAP),
        help="Backends to benchmark.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["exact", "low_rank"],
        choices=list(MODE_CHOICES),
        help="Benchmark exact and/or low-rank paths.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["smoke", "medium"],
        choices=list(SIZE_PRESETS),
        help="Dataset sizes to benchmark.",
    )
    parser.add_argument(
        "--include-svc-exact1",
        action="store_true",
        help="Include the slower SVC exact projection path (is_exact=1).",
    )
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of timed repeats per case.")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of untimed warmup fits to run before measuring each case.",
    )
    parser.add_argument(
        "--profile-backend",
        action="store_true",
        help="Run a second low-level backend fit to capture coarse phase timings.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write results JSON after each completed case.",
    )
    return parser.parse_args()


def git_sha(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return None


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compat_sigest(x: torch.Tensor, frac: float = 0.5) -> float:
    m = x.shape[0]
    n = int(frac * m)
    index1 = torch.randint(0, m, (n,), dtype=torch.long, device=x.device)
    index2 = torch.randint(0, m, (n,), dtype=torch.long, device=x.device)
    temp = x[index1] - x[index2]
    dist = torch.sum(temp ** 2, dim=1)
    non_zero_dist = dist[dist != 0]
    q = torch.tensor([0.9, 0.5, 0.1], dtype=non_zero_dist.dtype, device=non_zero_dist.device)
    srange = 1.0 / torch.quantile(non_zero_dist, q)
    return torch.mean(srange[[0, 2]]).item()


def patch_sigest_compat(stack: ExitStack) -> None:
    for module_name, attr_name in (
        ("torchkm.functions", "sigest"),
        ("torchkm.estimators", "sigest"),
        ("torchkm.cvknyssvm", "sigest"),
        ("torchkm.cvknysdwd", "sigest"),
        ("torchkm.cvknyslogit", "sigest"),
    ):
        module = importlib.import_module(module_name)
        if hasattr(module, attr_name):
            stack.enter_context(patch.object(module, attr_name, new=compat_sigest))


def sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_memory_mb(device: str) -> Optional[float]:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return None


def reset_peak_memory(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def result_hash(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def make_dataset(size_name: str, seed: int) -> Dict[str, Any]:
    spec = SIZE_PRESETS[size_name]
    set_global_seed(seed)

    p = int(spec["p"])
    nm = int(spec["nm"])
    mu = float(spec["mu"])
    ro = float(spec["ro"])

    X_train, y_train_pm1, means = data_gen(
        int(spec["n_train"]),
        nm,
        p,
        p // 2,
        p // 2,
        mu,
        ro,
        sdn=seed,
    )
    X_test, y_test_pm1, _ = data_gen(
        int(spec["n_test"]),
        nm,
        p,
        p // 2,
        p // 2,
        mu,
        ro,
        means=means,
    )

    X_train = standardize(X_train).double()
    X_test = standardize(X_test).double()
    y_train = ((y_train_pm1 + 1.0) / 2.0).to(torch.int64)
    y_test = ((y_test_pm1 + 1.0) / 2.0).to(torch.int64)

    return {
        "X_train_t": X_train,
        "X_test_t": X_test,
        "X_train_np": X_train.cpu().numpy(),
        "X_test_np": X_test.cpu().numpy(),
        "y_train_np": y_train.cpu().numpy(),
        "y_test_np": y_test.cpu().numpy(),
        "spec": spec,
    }


def make_foldid_array(n_samples: int, nfolds: int, seed: int) -> np.ndarray:
    foldid = _make_foldid(
        n=n_samples,
        nfolds=nfolds,
        foldid=None,
        random_state=seed,
    )
    return foldid.detach().cpu().numpy()


def estimator_kwargs(case: RunCase, dataset: Dict[str, Any], device: str, seed: int) -> Dict[str, Any]:
    spec = dataset["spec"]
    kwargs: Dict[str, Any] = {
        "kernel": "rbf",
        "nC": int(spec["nC"]),
        "nfolds": int(spec["nfolds"]),
        "foldid": make_foldid_array(int(spec["n_train"]), int(spec["nfolds"]), seed),
        "max_iter": int(spec["max_iter"]),
        "tol": 1e-5,
        "solver_gamma": 1e-8,
        "device": device,
        "probability": False,
        "store_path": False,
        "low_rank": case.mode == "low_rank",
    }
    if case.backend == "svc":
        kwargs["is_exact"] = int(case.is_exact)
    if case.mode == "low_rank":
        kwargs["num_landmarks"] = int(spec["num_landmarks"])
        kwargs["nys_k"] = int(spec["nys_k"])
    return kwargs


def new_estimator(case: RunCase, dataset: Dict[str, Any], device: str, seed: int):
    estimator_cls = BACKEND_MAP[case.backend]
    return estimator_cls(**estimator_kwargs(case, dataset, device, seed))


def build_backend_from_estimator(estimator, dataset: Dict[str, Any], device: str):
    X_np = dataset["X_train_np"]
    y_np = dataset["y_train_np"]
    X_train_t = torch.as_tensor(X_np, dtype=torch.double)
    y_pm1, _, _ = _check_binary_y(y_np)
    y_train_t = torch.as_tensor(y_pm1, dtype=torch.double)

    dev = _pick_device_str(device)
    uC_t = _make_ulam(estimator.nC, estimator.Cs, estimator.C_max, estimator.C_min)
    ulam_t = 1.0 / (2.0 * X_np.shape[0] * uC_t)
    nlam = int(ulam_t.numel())

    foldid_param = estimator.foldid
    foldid_t = torch.as_tensor(foldid_param).reshape(-1).to(torch.int64)

    y_backend = y_train_t.to(dev)
    ulam_backend = ulam_t.to(dev)
    foldid_backend = foldid_t.to(dev)

    if estimator.low_rank:
        K_train = None
    elif estimator.kernel == "precomputed":
        K_train = torch.as_tensor(X_np, dtype=torch.double).to(dev)
    else:
        with ExitStack() as stack:
            patch_sigest_compat(stack)
            K_train, _ = estimator._compute_K_train(X_train_t)
        K_train = K_train.to(dev)

    backend = estimator._make_backend(
        low_rank=estimator.low_rank,
        dev=dev,
        X_train_t=X_train_t,
        K_train=K_train,
        y_backend=y_backend,
        nlam=nlam,
        ulam_backend=ulam_backend,
        foldid_backend=foldid_backend,
    )
    return backend, {
        "dev": dev,
        "X_train_t": X_train_t,
        "y_train_t": y_train_t,
        "ulam_backend": ulam_backend,
        "foldid_backend": foldid_backend,
        "K_train": K_train,
    }


def wrap_backend_phases(backend, device: str) -> Tuple[TimingAccumulator, ExitStack]:
    timings = TimingAccumulator(device)
    stack = ExitStack()

    module = importlib.import_module(backend.__class__.__module__)

    original_eigh = torch.linalg.eigh
    stack.enter_context(patch("torch.linalg.eigh", new=timings.wrap_function("linalg_eigh", original_eigh)))

    if hasattr(torch.linalg, "svd"):
        original_svd = torch.linalg.svd
        stack.enter_context(patch("torch.linalg.svd", new=timings.wrap_function("linalg_svd", original_svd)))

    for fn_name in ("sigest", "rbf_kernel", "kernelMult"):
        if hasattr(module, fn_name):
            original = getattr(module, fn_name)
            stack.enter_context(patch.object(module, fn_name, new=timings.wrap_function(fn_name, original)))

    if hasattr(backend.__class__, "golden_section_search"):
        original_gss = getattr(backend.__class__, "golden_section_search")
        stack.enter_context(
            patch.object(
                backend.__class__,
                "golden_section_search",
                new=timings.wrap_method("golden_section_search", original_gss),
            )
        )

    return timings, stack


def time_exact_kernel_build(estimator, dataset: Dict[str, Any], device: str, seed: int) -> Dict[str, float]:
    if estimator.low_rank or estimator.kernel == "precomputed":
        return {}
    set_global_seed(seed)
    X_train_t = torch.as_tensor(dataset["X_train_np"], dtype=torch.double)
    with ExitStack() as stack:
        patch_sigest_compat(stack)
        sync_device(device)
        start = time.perf_counter()
        estimator._compute_K_train(X_train_t)
        sync_device(device)
    return {"kernel_build": time.perf_counter() - start}


def run_wrapper_fit(estimator, dataset: Dict[str, Any], device: str, repeat_seed: int) -> Dict[str, Any]:
    X_train = dataset["X_train_np"]
    y_train = dataset["y_train_np"]
    X_test = dataset["X_test_np"]
    y_test = dataset["y_test_np"]

    set_global_seed(repeat_seed)
    reset_peak_memory(device)
    with ExitStack() as stack:
        patch_sigest_compat(stack)
        sync_device(device)
        fit_start = time.perf_counter()
        estimator.fit(X_train, y_train)
        sync_device(device)
    fit_seconds = time.perf_counter() - fit_start

    sync_device(device)
    predict_start = time.perf_counter()
    scores = estimator.decision_function(X_test)
    pred = estimator.predict(X_test)
    sync_device(device)
    predict_seconds = time.perf_counter() - predict_start

    accuracy = float(np.mean(pred == y_test))
    return {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "accuracy": accuracy,
        "best_C": float(estimator.best_C_),
        "cv_mis_min": float(np.min(estimator.cv_mis_)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "pred_hash": result_hash(pred.astype(np.int64)),
        "score_hash": result_hash(scores.astype(np.float64)),
        "peak_memory_mb": peak_memory_mb(device),
    }


def run_backend_profile(case: RunCase, dataset: Dict[str, Any], device: str, seed: int) -> Dict[str, Any]:
    estimator = new_estimator(case, dataset, device, seed)
    backend, context = build_backend_from_estimator(estimator, dataset, device)

    phase_seconds = time_exact_kernel_build(estimator, dataset, device, seed)
    timings, stack = wrap_backend_phases(backend, context["dev"])

    patch_sigest_compat(stack)
    with stack:
        reset_peak_memory(context["dev"])
        sync_device(context["dev"])
        start = time.perf_counter()
        backend.fit()
        sync_device(context["dev"])
        backend_fit_seconds = time.perf_counter() - start

    phase_seconds.update(timings.seconds)
    timed_subtotal = sum(phase_seconds.values())
    phase_seconds["backend_fit_total"] = backend_fit_seconds
    phase_seconds["iterative_solve_estimate"] = max(0.0, backend_fit_seconds - timed_subtotal)

    return {
        "phase_seconds": phase_seconds,
        "backend_stats": {
            "anlam": int(getattr(backend, "anlam", -1)),
            "jerr": int(getattr(backend, "jerr", 0)),
            "npass_total": int(torch.as_tensor(getattr(backend, "npass")).sum().item()),
            "cvnpass_total": int(torch.as_tensor(getattr(backend, "cvnpass")).sum().item()),
            "peak_memory_mb": peak_memory_mb(context["dev"]),
        },
    }


def mean_dict(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not entries:
        return {}

    result: Dict[str, Any] = {}
    numeric_keys = set()
    exemplar = entries[0]
    for key, value in exemplar.items():
        if isinstance(value, dict):
            result[key] = mean_dict([entry.get(key, {}) for entry in entries])
        elif isinstance(value, (int, float, np.floating)):
            numeric_keys.add(key)
        else:
            result[key] = value

    for key in numeric_keys:
        vals = [float(entry[key]) for entry in entries if entry.get(key) is not None]
        result[key] = float(np.mean(vals)) if vals else None
    return result


def run_case(case: RunCase, args: argparse.Namespace) -> Dict[str, Any]:
    dataset = make_dataset(case.size, args.seed)

    for warmup_idx in range(args.warmup_runs):
        estimator = new_estimator(case, dataset, args.device, args.seed)
        run_wrapper_fit(estimator, dataset, args.device, args.seed + 10000 + warmup_idx)

    wrapper_runs: List[Dict[str, Any]] = []
    for repeat_idx in range(args.repeats):
        estimator = new_estimator(case, dataset, args.device, args.seed)
        wrapper_runs.append(run_wrapper_fit(estimator, dataset, args.device, args.seed + repeat_idx))

    result: Dict[str, Any] = {
        "case": asdict(case),
        "dataset": {
            "n_train": int(dataset["spec"]["n_train"]),
            "n_test": int(dataset["spec"]["n_test"]),
            "p": int(dataset["spec"]["p"]),
            "nm": int(dataset["spec"]["nm"]),
            "nC": int(dataset["spec"]["nC"]),
            "nfolds": int(dataset["spec"]["nfolds"]),
            "max_iter": int(dataset["spec"]["max_iter"]),
            "num_landmarks": int(dataset["spec"]["num_landmarks"]),
            "nys_k": int(dataset["spec"]["nys_k"]),
        },
        "wrapper_metrics": mean_dict(wrapper_runs),
    }

    if args.profile_backend:
        result["backend_profile"] = run_backend_profile(case, dataset, args.device, args.seed)

    return result


def build_cases(args: argparse.Namespace) -> List[RunCase]:
    cases: List[RunCase] = []
    for size in args.sizes:
        for backend in args.backends:
            for mode in args.modes:
                if backend != "svc":
                    cases.append(RunCase(backend=backend, mode=mode, size=size))
                    continue

                cases.append(RunCase(backend=backend, mode=mode, size=size, is_exact=0))
                if mode == "exact" and args.include_svc_exact1:
                    cases.append(RunCase(backend=backend, mode=mode, size=size, is_exact=1))
    return cases


def serialize_results(results: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    dev = _pick_device_str(args.device)

    try:
        device_name = torch.cuda.get_device_name(0) if dev.startswith("cuda") and torch.cuda.is_available() else "cpu"
    except Exception:
        device_name = "unknown"

    results: Dict[str, Any] = {
        "metadata": {
            "git_sha": git_sha(REPO_ROOT),
            "torch_version": torch.__version__,
            "device": dev,
            "device_name": device_name,
            "cuda_available": bool(torch.cuda.is_available()),
            "seed": args.seed,
            "repeats": args.repeats,
            "warmup_runs": args.warmup_runs,
            "profile_backend": bool(args.profile_backend),
            "compat_sigest_patch": True,
            "hostname": os.uname().nodename,
        },
        "cases": [],
    }

    cases = build_cases(args)
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] running {case.label}", flush=True)
        try:
            case_result = run_case(case, args)
            results["cases"].append(case_result)
            fit_seconds = case_result["wrapper_metrics"].get("fit_seconds")
            acc = case_result["wrapper_metrics"].get("accuracy")
            print(
                f"  fit={fit_seconds:.3f}s acc={acc:.4f}",
                flush=True,
            )
        except Exception as exc:
            err = {
                "case": asdict(case),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            results["cases"].append(err)
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)

        if args.output_json is not None:
            serialize_results(results, args.output_json)

    if args.output_json is not None:
        print(f"wrote results to {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
