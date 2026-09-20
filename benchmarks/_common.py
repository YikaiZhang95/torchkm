"""Shared helpers for the TorchKM benchmark scripts.

Every ``bench_*.py`` script in this directory reports the same quantities in
the same machine-readable form so that the paper tables, the documentation
page and the archived results cannot drift apart:

* **Protocol.** One RBF bandwidth per repeat from :func:`torchkm.sigest`,
  shared by every library; a log-uniform grid of ``C`` values, converted to
  ``lambda = 1 / (2 n C)`` for TorchKM; identical stratified folds for every
  library; end-to-end wall-clock time that includes kernel or feature
  construction, the cross-validation sweep and the final refit.
* **Metrics.** Test accuracy, balanced accuracy and AUC (from the decision
  scores), so class imbalance never hides behind accuracy.
* **Memory.** Peak device memory from the PyTorch caching allocator (for
  TorchKM and Falkon) and from NVML sampling of this process (for every
  library, including cuML and ThunderSVM which do not use PyTorch's allocator).
* **Output.** A JSON document with an environment snapshot (GPU, driver, CUDA,
  PyTorch, library versions, TorchKM version and commit) and one record per
  dataset x library x repeat, written after every record so a long run that
  dies still leaves its results on disk.

The legacy ``table2_simulation.py`` / ``table3_benchmarks.py`` /
``table4_nystrom.py`` scripts reproduce the submitted paper's protocol and keep
using the small helpers at the top of this module (``timed``, ``warmup``,
``svm_objective``, ...).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Small helpers shared with the legacy table scripts
# ---------------------------------------------------------------------------


def get_device(pref: str | None = None) -> str:
    if pref:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def c_grid(n: int = 50, c_max: float = 1e3, c_min: float = 1e-3) -> np.ndarray:
    """``n`` log-uniform C values from ``c_max`` down to ``c_min``."""
    return np.logspace(np.log10(c_max), np.log10(c_min), num=n)


def lam_grid(n: int = 50) -> np.ndarray:
    """50 log-uniform lambda values over [1e-3, 1e3] (source-notebook grid).

    Convert to the LIBSVM/scikit-learn C parameterization with C = 1/(2*n_obs*lambda).
    """
    return np.logspace(3.0, -3.0, num=n)


def lam_from_c(C: np.ndarray, n_obs: int) -> np.ndarray:
    """TorchKM regularization weight for a LIBSVM-style ``C``: lambda = 1/(2 n C)."""
    return 1.0 / (2.0 * float(n_obs) * np.asarray(C, dtype=float))


def synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def free_cuda(device: str) -> None:
    """Release cached CUDA blocks between datasets/methods to limit peak memory."""
    import gc

    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class timed:
    """Context manager for CUDA-safe wall-clock timing (seconds)."""

    def __init__(self, device: str):
        self.device = device
        self.dt = float("nan")

    def __enter__(self) -> "timed":
        synchronize(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        synchronize(self.device)
        self.dt = time.perf_counter() - self._t0


def warmup(device: str) -> None:
    """One tiny fit to absorb CUDA initialization before timed runs."""
    if not str(device).startswith("cuda"):
        return
    from torchkm.estimators import TorchKMSVC

    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 4))
    y = np.where(rng.standard_normal(64) > 0, 1, -1)
    TorchKMSVC(kernel="rbf", Cs=c_grid(5), nC=5, cv=3, device=device).fit(X, y)
    synchronize(device)


def svm_objective(
    K: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor, intercept: float, lam: float
) -> float:
    """Kernel-SVM objective, equation (1) of the supplement:

        (1/n) * sum_i (1 - y_i f_i)_+  +  lam * alpha^T K alpha,   f = K alpha + b

    Every method is scored with this same objective functional at the same
    lambda, so the reported values are comparable. ``K`` is the training kernel
    and ``alpha``/``intercept`` are that method's fitted solution.
    """
    Ka = K @ alpha
    f = Ka + intercept
    hinge = torch.clamp(1.0 - y * f, min=0.0)
    return float(hinge.mean().item() + lam * torch.dot(alpha, Ka).item())


def mean_se(values) -> tuple[float, float]:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    se = a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0
    return float(a.mean()), float(se)


# ---------------------------------------------------------------------------
# Kernel bandwidth conversions
# ---------------------------------------------------------------------------
#
# TorchKM's RBF kernel is exp(-2 * sig * ||x - x'||^2) with ``sig`` from
# ``sigest``. Every other library is given the *same* kernel:
#   * scikit-learn / ThunderSVM / cuML use exp(-gamma * d^2)  -> gamma = 2 sig
#   * Falkon's GaussianKernel(s) uses exp(-d^2 / (2 s^2))     -> s = 1 / (2 sqrt(sig))


def gamma_from_sigest(sig: float) -> float:
    return 2.0 * float(sig)


def falkon_sigma_from_sigest(sig: float) -> float:
    return 1.0 / (2.0 * math.sqrt(float(sig)))


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------


def _cpu_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_bytes() -> Optional[int]:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _module_version(dist_name: str, module_name: Optional[str] = None) -> Optional[str]:
    try:
        return importlib.metadata.version(dist_name)
    except Exception:
        pass
    try:
        mod = importlib.import_module(module_name or dist_name)
        return str(getattr(mod, "__version__", "installed"))
    except Exception:
        return None


def _nvml_driver_version() -> Optional[str]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            v = pynvml.nvmlSystemGetDriverVersion()
            return v.decode() if isinstance(v, bytes) else str(v)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def env_snapshot() -> Dict[str, Any]:
    """Everything a reader needs to reproduce or discount a timing."""
    import torchkm

    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": int(props.multi_processor_count),
            "driver": _nvml_driver_version(),
            "device_count": torch.cuda.device_count(),
        }
    packages = {
        name: _module_version(dist, mod)
        for name, dist, mod in [
            ("numpy", "numpy", None),
            ("scipy", "scipy", None),
            ("scikit-learn", "scikit-learn", "sklearn"),
            ("thundersvm", "thundersvm", None),
            ("cuml", "cuml", None),
            ("falkon", "falkon", None),
            ("eigenpro", "eigenpro", None),
            ("pykeops", "pykeops", None),
            ("nvidia-ml-py", "nvidia-ml-py", "pynvml"),
            ("psutil", "psutil", None),
        ]
    }
    return {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": _ram_bytes(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "torchkm": torchkm.__version__,
        "torchkm_commit": _git_commit(),
        "packages": packages,
        "argv": sys.argv,
    }


# ---------------------------------------------------------------------------
# Peak memory
# ---------------------------------------------------------------------------


def host_rss_bytes() -> Optional[int]:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


class _NvmlSampler:
    """Background sampler of this process's (and the device's) NVML memory use."""

    def __init__(self, device: str, interval: float):
        self.interval = interval
        self.process_peak: Optional[int] = None
        self.device_peak: Optional[int] = None
        self.host_rss_peak: Optional[int] = host_rss_bytes()
        self.error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._pynvml = None
        self._cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        if self._cuda:
            try:
                import pynvml  # type: ignore

                pynvml.nvmlInit()
                self._pynvml = pynvml
                self._handle = self._find_handle(pynvml, device)
            except Exception as err:  # pragma: no cover - depends on the machine
                self.error = f"nvml unavailable: {err!r}"
                self._pynvml = None

    @staticmethod
    def _find_handle(pynvml, device: str):
        index = torch.device(device).index or 0
        # Match by UUID when PyTorch exposes it, so CUDA_VISIBLE_DEVICES
        # remaps cannot point NVML at the wrong card.
        props = torch.cuda.get_device_properties(index)
        uuid = getattr(props, "uuid", None)
        if uuid is not None:
            try:
                return pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode())
            except Exception:
                pass
        return pynvml.nvmlDeviceGetHandleByIndex(index)

    def _sample(self) -> None:
        rss = host_rss_bytes()
        if rss is not None:
            self.host_rss_peak = max(self.host_rss_peak or 0, rss)
        if self._pynvml is None or self._handle is None:
            return
        pynvml = self._pynvml
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.device_peak = max(self.device_peak or 0, int(info.used))
        except Exception as err:  # pragma: no cover
            self.error = f"nvml memory info failed: {err!r}"
        try:
            pid = os.getpid()
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
                if proc.pid == pid and proc.usedGpuMemory is not None:
                    self.process_peak = max(
                        self.process_peak or 0, int(proc.usedGpuMemory)
                    )
        except Exception as err:  # pragma: no cover
            self.error = f"nvml process query failed: {err!r}"

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)
        self._sample()

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:  # pragma: no cover
                pass


class PeakMemory:
    """Peak memory during a block, from the PyTorch allocator and from NVML.

    ``result`` holds, in bytes (``None`` when not measurable):

    * ``torch_max_allocated`` / ``torch_max_reserved``: PyTorch caching
      allocator peaks (TorchKM, Falkon and anything else that allocates
      through torch).
    * ``nvml_process_peak``: this process's device memory as reported by the
      driver, sampled every ``interval`` seconds. The number that is comparable
      across cuML, ThunderSVM and TorchKM. ``None`` when the driver does not
      expose per-process usage (some container runtimes).
    * ``nvml_device_peak``: whole-device used memory, the fallback when the
      per-process figure is unavailable (includes the CUDA context and any
      other process on the card).
    * ``host_rss_peak``: resident set size of this process (the CPU-side
      analogue, and what the CPU smoke runs report).
    """

    def __init__(self, device: str, interval: float = 0.05):
        self.device = str(device)
        self.interval = interval
        self.result: Dict[str, Optional[int]] = {}
        self._sampler: Optional[_NvmlSampler] = None

    def __enter__(self) -> "PeakMemory":
        cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sampler = _NvmlSampler(self.device, self.interval)
        self._sampler.start()
        return self

    def __exit__(self, *exc) -> None:
        cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
        assert self._sampler is not None
        self._sampler.stop()
        self.result = {
            "torch_max_allocated": (
                int(torch.cuda.max_memory_allocated(self.device)) if cuda else None
            ),
            "torch_max_reserved": (
                int(torch.cuda.max_memory_reserved(self.device)) if cuda else None
            ),
            "nvml_process_peak": self._sampler.process_peak,
            "nvml_device_peak": self._sampler.device_peak,
            "host_rss_peak": self._sampler.host_rss_peak,
            "nvml_error": self._sampler.error,
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def classification_metrics(
    y_true_pm1: np.ndarray, scores: np.ndarray
) -> Dict[str, Any]:
    """Accuracy, balanced accuracy and AUC of decision scores against +-1 labels."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    y = np.asarray(y_true_pm1).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    pred = np.where(s > 0, 1.0, -1.0)
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": None,
        "n_test": int(y.size),
        "test_pos_frac": float(np.mean(y > 0)),
    }
    if np.unique(y).size == 2 and np.all(np.isfinite(s)):
        out["auc"] = float(roc_auc_score(y, s))
    return out


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    u = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.where(u >= 0, tau * u, (tau - 1.0) * u)))


def quantile_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, tau: float
) -> Dict[str, Any]:
    """Check loss and empirical coverage P(y <= q_hat) of a quantile prediction."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    q = np.asarray(y_pred, dtype=float).reshape(-1)
    return {
        "pinball_loss": pinball_loss(y, q, tau),
        "coverage": float(np.mean(y <= q)),
        "target_coverage": float(tau),
        "n_test": int(y.size),
    }


# ---------------------------------------------------------------------------
# JSON results
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


class ResultWriter:
    """Accumulates records and rewrites the JSON file after each one."""

    def __init__(
        self, path: Optional[str], *, script: str, args: Any, protocol: Dict[str, Any]
    ):
        self.path = path
        self.doc: Dict[str, Any] = {
            "script": script,
            "args": vars(args) if isinstance(args, argparse.Namespace) else dict(args),
            "protocol": protocol,
            "environment": env_snapshot(),
            "records": [],
        }
        self.save()

    def add(self, record: Dict[str, Any]) -> None:
        self.doc["records"].append(_jsonable(record))
        self.save()

    def save(self) -> None:
        if not self.path:
            return
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(_jsonable(self.doc), fh, indent=2)
        os.replace(tmp, self.path)


def summarize_repeats(
    repeats: Sequence[Dict[str, Any]], keys: Iterable[str]
) -> Dict[str, Any]:
    """Mean and standard error over repeats for each key (missing/None skipped)."""
    summary: Dict[str, Any] = {"n_repeats": len(repeats)}
    for key in keys:
        vals = [r.get(key) for r in repeats]
        m, se = mean_se(vals)
        summary[f"{key}_mean"] = m
        summary[f"{key}_se"] = se
    return summary


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"

#: Named datasets used by the benchmark suite. ``train``/``test`` are LIBSVM
#: file names inside ``--data-dir`` (``.bz2`` / ``.xz`` accepted). Single-file
#: datasets get a stratified 80/20 split. ``classes`` picks two digits from a
#: multiclass file (first -> -1, second -> +1). ``subsample_train`` draws a
#: stratified subset so the problem fits the exact-mode envelope.
DATASETS: Dict[str, Dict[str, Any]] = {
    # Adult at increasing size: a scaling study, not six benchmarks.
    "a1a": dict(train="a1a", test="a1a.t", group="scaling"),
    "a3a": dict(train="a3a", test="a3a.t", group="scaling"),
    "a5a": dict(train="a5a", test="a5a.t", group="scaling"),
    "a7a": dict(train="a7a", test="a7a.t", group="scaling"),
    "a8a": dict(train="a8a", test="a8a.t", group="scaling"),
    "a9a": dict(train="a9a", test="a9a.t", group="scaling"),
    # Web page data: heavily imbalanced (about 3% positive).
    "w7a": dict(train="w7a", test="w7a.t", group="imbalanced"),
    "w8a": dict(train="w8a", test="w8a.t", group="imbalanced"),
    # Problems where the kernel matters, sized for exact mode.
    "ijcnn1_30k": dict(
        train="ijcnn1", test="ijcnn1.t", subsample_train=30_000, group="exact"
    ),
    "mnist_3v8": dict(
        train="mnist.scale", test="mnist.scale.t", classes=(3, 8), group="exact"
    ),
    "mnist_4v9": dict(
        train="mnist.scale", test="mnist.scale.t", classes=(4, 9), group="exact"
    ),
    "covtype_30k": dict(
        train="covtype.libsvm.binary.scale",
        test=None,
        subsample_train=30_000,
        subsample_test=20_000,
        group="exact",
    ),
    # Full-size problems for the Nyström path.
    "ijcnn1": dict(train="ijcnn1", test="ijcnn1.t", group="imbalanced"),
    "covtype": dict(train="covtype.libsvm.binary.scale", test=None, group="scale"),
    "mnist8m_4v6": dict(
        train="mnist8m.scale", test=None, classes=(4, 6), group="scale"
    ),
    # High-dimension, low-sample-size case for DWD.
    "gisette": dict(train="gisette_scale", test="gisette_scale.t", group="hdlss"),
}

SUITES: Dict[str, List[str]] = {
    "scaling": ["a1a", "a3a", "a5a", "a7a", "a8a", "a9a"],
    "exact": ["ijcnn1_30k", "mnist_3v8", "mnist_4v9", "covtype_30k", "w7a"],
    "imbalanced": ["w8a", "ijcnn1"],
    "scale": ["covtype", "mnist8m_4v6"],
}


def open_libsvm(path: str):
    """Open a LIBSVM file as a binary stream, handling .bz2 / .xz / plain."""
    if os.path.exists(path):
        return open(path, "rb")
    if os.path.exists(path + ".bz2"):
        import bz2

        return bz2.open(path + ".bz2", "rb")
    if os.path.exists(path + ".xz"):
        import lzma

        return lzma.open(path + ".xz", "rb")
    raise FileNotFoundError(
        f"none of {path}, {path}.bz2, {path}.xz exists; download it from {LIBSVM_URL}"
    )


def _to_pm1(
    y: np.ndarray, classes: Optional[Tuple[Any, Any]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mask, y_pm1). Binary files map their smaller label to -1."""
    y = np.asarray(y)
    if classes is None:
        neg = np.min(np.unique(y))
        return np.ones(y.shape[0], dtype=bool), np.where(y == neg, -1.0, 1.0)
    a, b = classes
    mask = (y == a) | (y == b)
    return mask, np.where(y[mask] == a, -1.0, 1.0)


def stratified_subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int):
    from sklearn.model_selection import train_test_split

    if n is None or n >= X.shape[0]:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=int(n), stratify=y, random_state=seed
    )
    return X_sub, y_sub


def load_dataset(
    name: str, data_dir: str, *, seed: int = 0, cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load a named dataset as dense float64 arrays with labels in {-1, +1}.

    Returns ``dict(Xtr, ytr, Xte, yte, name, n_train, n_test, p, pos_frac)``.
    """
    from sklearn.datasets import load_svmlight_file, load_svmlight_files
    from sklearn.model_selection import train_test_split

    cfg = dict(DATASETS[name] if cfg is None else cfg)
    classes = cfg.get("classes")
    train_path = os.path.join(data_dir, cfg["train"])

    if cfg.get("test"):
        test_path = os.path.join(data_dir, cfg["test"])
        with open_libsvm(train_path) as ftr, open_libsvm(test_path) as fte:
            Xtr, ytr, Xte, yte = load_svmlight_files((ftr, fte), dtype=np.float64)
        Xtr, Xte = Xtr.toarray(), Xte.toarray()
        mtr, ytr = _to_pm1(ytr, classes)
        mte, yte = _to_pm1(yte, classes)
        Xtr, Xte = Xtr[mtr], Xte[mte]
    else:
        with open_libsvm(train_path) as f:
            X, y = load_svmlight_file(f, dtype=np.float64)
        X = X.toarray()
        mask, y = _to_pm1(y, classes)
        X = X[mask]
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=cfg.get("test_size", 0.2), stratify=y, random_state=seed
        )

    Xtr, ytr = stratified_subsample(Xtr, ytr, cfg.get("subsample_train"), seed)
    Xte, yte = stratified_subsample(Xte, yte, cfg.get("subsample_test"), seed + 1)
    return dict(
        name=name,
        Xtr=np.ascontiguousarray(Xtr),
        ytr=np.asarray(ytr, dtype=np.float64),
        Xte=np.ascontiguousarray(Xte),
        yte=np.asarray(yte, dtype=np.float64),
        n_train=int(Xtr.shape[0]),
        n_test=int(Xte.shape[0]),
        p=int(Xtr.shape[1]),
        pos_frac=float(np.mean(ytr > 0)),
    )


def synthetic_dataset(
    n: int, p: int, seed: int, *, name: str = "synthetic"
) -> Dict[str, Any]:
    """Gaussian-mixture classification data from ``torchkm.data_gen`` (paper Table 2).

    The mixture has fast kernel-spectrum decay, which favours spectral and
    Nyström methods; it is a mechanism illustration, not a neutral benchmark.
    """
    from torchkm import data_gen, standardize

    nm, mu, ro = 5, 2.0, 3.0
    Xtr, ytr, _ = data_gen(n, nm, p, p // 2, p // 2, mu, ro, seed)
    Xte, yte, _ = data_gen(
        max(n // 5, 200), nm, p, p // 2, p // 2, mu, ro, seed + 10_000
    )
    Xtr, Xte = standardize(Xtr).numpy().astype(np.float64), standardize(
        Xte
    ).numpy().astype(np.float64)
    return dict(
        name=name,
        Xtr=Xtr,
        ytr=ytr.numpy().astype(np.float64),
        Xte=Xte,
        yte=yte.numpy().astype(np.float64),
        n_train=int(n),
        n_test=int(Xte.shape[0]),
        p=int(p),
        pos_frac=float(np.mean(ytr.numpy() > 0)),
    )


def synthetic_regression(n: int, p: int, seed: int, *, name: str = "synthetic_hetero"):
    """Heteroscedastic regression data with known conditional quantiles.

    y = sin(2 x1) + x2^2 / 2 + (0.5 + 0.5 |x1|) * eps,  eps ~ N(0, 1).
    Returns the arrays plus a callable ``true_quantile(X, tau)``.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)

    def draw(m):
        X = rng.uniform(-2.0, 2.0, size=(m, p))
        mean = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] ** 2
        scale = 0.5 + 0.5 * np.abs(X[:, 0])
        y = mean + scale * rng.standard_normal(m)
        return X, y

    Xtr, ytr = draw(n)
    Xte, yte = draw(max(n // 5, 200))

    def true_quantile(X, tau):
        X = np.asarray(X)
        mean = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] ** 2
        scale = 0.5 + 0.5 * np.abs(X[:, 0])
        return mean + scale * norm.ppf(tau)

    return dict(
        name=name,
        Xtr=Xtr,
        ytr=ytr,
        Xte=Xte,
        yte=yte,
        n_train=int(n),
        n_test=int(Xte.shape[0]),
        p=int(p),
        true_quantile=true_quantile,
    )


# ---------------------------------------------------------------------------
# Folds and cross-validation sweeps for external libraries
# ---------------------------------------------------------------------------


def make_folds(y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Stratified fold ids in ``1..n_folds`` (TorchKM's ``foldid`` convention)."""
    from sklearn.model_selection import KFold, StratifiedKFold

    y = np.asarray(y).reshape(-1)
    foldid = np.zeros(y.shape[0], dtype=np.int64)
    binary = np.unique(y).size == 2
    splitter = (
        StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        if binary
        else KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    )
    for k, (_, val_idx) in enumerate(splitter.split(y, y if binary else None), start=1):
        foldid[val_idx] = k
    return foldid


def fold_iter(foldid: np.ndarray) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    for k in np.unique(foldid):
        val = foldid == k
        yield int(k), np.flatnonzero(~val), np.flatnonzero(val)


class TimeCap(Exception):
    """Raised inside a sweep when the per-cell time cap is exceeded."""


def cv_sweep(
    fit_predict: Callable[[Any, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    foldid: np.ndarray,
    grid: Sequence[Any],
    *,
    score: str = "accuracy",
    time_cap_s: Optional[float] = None,
    t0: Optional[float] = None,
) -> Dict[str, Any]:
    """Grid x fold sweep for a library without integrated model selection.

    ``fit_predict(param, X_train, y_train, X_val)`` returns predictions for the
    validation rows (labels for ``score="accuracy"``, values for
    ``score="pinball:<tau>"``). Fold failures score NaN and are averaged out
    with ``nanmean``; a grid value whose every fold failed is skipped. The sweep
    stops early once ``time_cap_s`` seconds have elapsed since ``t0`` and marks
    the result ``capped``; selection then uses the grid values completed so far.
    """
    t0 = time.perf_counter() if t0 is None else t0
    tau = float(score.split(":")[1]) if score.startswith("pinball") else None
    cv_scores: List[float] = []
    for i, param in enumerate(grid):
        fold_scores: List[float] = []
        for _, tr, va in fold_iter(foldid):
            try:
                pred = np.asarray(fit_predict(param, X[tr], y[tr], X[va])).reshape(-1)
                if tau is None:
                    fold_scores.append(
                        float(np.mean(np.where(pred > 0, 1.0, -1.0) == y[va]))
                    )
                else:
                    fold_scores.append(-pinball_loss(y[va], pred, tau))
            except Exception:
                fold_scores.append(float("nan"))
        cv_scores.append(
            float(np.nanmean(fold_scores))
            if np.any(np.isfinite(fold_scores))
            else float("nan")
        )
        if time_cap_s is not None and (time.perf_counter() - t0) > time_cap_s:
            break
    completed = len(cv_scores)
    arr = np.asarray(cv_scores, dtype=float)
    if not np.any(np.isfinite(arr)):
        raise RuntimeError("every grid value failed in every fold")
    best = int(np.nanargmax(arr))
    return {
        "best_index": best,
        "best_param": grid[best],
        "cv_scores": cv_scores,
        "grid_completed": completed,
        "grid_size": len(grid),
        "capped": completed < len(grid),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_args(ap: argparse.ArgumentParser, *, repeats: int = 1) -> None:
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--repeats", type=int, default=repeats, help="runs per cell")
    ap.add_argument("--folds", type=int, default=10, help="cross-validation folds")
    ap.add_argument("--grid-size", type=int, default=50, help="number of C values")
    ap.add_argument("--c-max", type=float, default=1e3)
    ap.add_argument("--c-min", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--max-iter", type=int, default=100_000, help="TorchKM solver cap")
    ap.add_argument("--tol", type=float, default=1e-5, help="TorchKM solver tolerance")
    ap.add_argument(
        "--kkt-eps",
        type=float,
        default=None,
        help="TorchKM KKT stopping tolerance (default: the estimator default, 1e-3)",
    )
    ap.add_argument(
        "--kkt-scaled",
        action="store_true",
        help="TorchKM scale-aware KKT rule (n * sum(KKT^2) < KKTeps)",
    )
    ap.add_argument("--out", default=None, help="write JSON results here")
    ap.add_argument(
        "--time-cap",
        type=float,
        default=None,
        help="seconds allowed per library x dataset x repeat before the CV sweep stops",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny synthetic data, short grid, few folds: checks the script runs",
    )


def smoke_settings(args: argparse.Namespace) -> argparse.Namespace:
    """Shrink a run so it completes in seconds on a CPU."""
    if getattr(args, "smoke", False):
        args.folds = min(args.folds, 3)
        args.grid_size = min(args.grid_size, 4)
        args.max_iter = min(args.max_iter, 200)
        args.repeats = min(args.repeats, 2)
    return args


def protocol_dict(args: argparse.Namespace, **extra: Any) -> Dict[str, Any]:
    base = {
        "folds": getattr(args, "folds", None),
        "grid_size": getattr(args, "grid_size", None),
        "c_max": getattr(args, "c_max", None),
        "c_min": getattr(args, "c_min", None),
        "timing": "end-to-end: kernel/feature construction + CV sweep + final refit",
        "bandwidth": "sigest on the training features, shared by every library",
        "folds_shared": "identical stratified folds for every library",
    }
    base.update(extra)
    return base


def banner(title: str, **fields: Any) -> None:
    print(title)
    for k, v in fields.items():
        print(f"  {k}: {v}")
    print(flush=True)


def fmt_bytes(n: Optional[float]) -> str:
    if n is None or not np.isfinite(n):
        return "-"
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.0f} MB"
