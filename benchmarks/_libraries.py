"""Library runners shared by ``bench_gpu_libraries.py`` and ``bench_covtype_rank.py``.

Every runner has the signature ``run_x(data, sig, Cs, foldid, args, dev, seed,
**options)`` and returns one flat record: library, mode, parameters, status,
end-to-end time, peak memory, selected ``C``, and the test metrics from
:func:`_common.classification_metrics`. All libraries see the same kernel
(bandwidth ``sig`` from ``sigest``), the same ``C`` grid and the same folds.

External libraries are imported lazily so the scripts run wherever they are
started; a missing library is reported once and skipped.
"""

from __future__ import annotations

import importlib
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from _common import (
    PeakMemory,
    classification_metrics,
    cv_sweep,
    falkon_sigma_from_sigest,
    gamma_from_sigest,
    lam_from_c,
    timed,
)

#: Library key -> (import name, human name). ``torchkm*`` and ``linear`` need
#: nothing beyond the package's own dependencies.
EXTERNAL_IMPORTS = {
    "sklearn_svc": ("sklearn.svm", "scikit-learn SVC (CPU, libsvm SMO)"),
    "thundersvm": ("thundersvm", "ThunderSVM (GPU SMO)"),
    "cuml_svc": ("cuml.svm", "cuML SVC (RAPIDS, GPU SMO)"),
    "falkon": ("falkon", "Falkon (GPU Nyström KRR)"),
    "linear": ("sklearn.linear_model", "logistic regression / LinearSVC (CPU)"),
}

ALL_LIBRARIES = ["torchkm", "torchkm_nystrom"] + list(EXTERNAL_IMPORTS)


def library_availability(names) -> Dict[str, Optional[str]]:
    """Map library key -> None if importable, else the import error."""
    out: Dict[str, Optional[str]] = {}
    for name in names:
        if name not in EXTERNAL_IMPORTS:
            out[name] = None
            continue
        module = EXTERNAL_IMPORTS[name][0]
        try:
            importlib.import_module(module)
            out[name] = None
        except Exception as err:  # ImportError, OSError from missing CUDA libs, ...
            out[name] = f"{type(err).__name__}: {err}"
    return out


def decision_chunked(clf, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """``decision_function`` in row chunks so K_test never exceeds chunk x n_train."""
    parts = [
        clf.decision_function(X[i : i + chunk]) for i in range(0, X.shape[0], chunk)
    ]
    return np.concatenate([np.asarray(p).reshape(-1) for p in parts])


def _cast(X: np.ndarray, float32: bool) -> np.ndarray:
    return np.ascontiguousarray(X, dtype=np.float32 if float32 else np.float64)


def _sweep_record(
    *,
    library: str,
    mode: str,
    device: str,
    fit_predict: Callable[[Any, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    refit_decision: Callable[[Any, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    data: Dict[str, Any],
    grid: List[Any],
    foldid: np.ndarray,
    args,
    params: Dict[str, Any],
    grid_label: str = "C",
) -> Dict[str, Any]:
    """CV sweep + refit + scoring for a library without integrated tuning.

    The timed region covers the whole sweep (every fold of every grid value)
    and the final refit on the full training set, which is what a user pays
    to obtain a tuned model from such a library.
    """
    Xtr, ytr, Xte, yte = data["Xtr"], data["ytr"], data["Xte"], data["yte"]
    with PeakMemory(device) as pm, timed(device) as t:
        t0 = time.perf_counter()
        res = cv_sweep(
            fit_predict, Xtr, ytr, foldid, grid, time_cap_s=args.time_cap, t0=t0
        )
        scores = refit_decision(res["best_param"], Xtr, ytr, Xte)
    rec = dict(
        library=library,
        mode=mode,
        device=device,
        params=params,
        status="capped" if res["capped"] else "ok",
        time_s=t.dt,
        memory=pm.result,
        torch_peak_bytes=pm.result.get("torch_max_allocated"),
        cv=dict(
            grid_completed=res["grid_completed"],
            grid_size=res["grid_size"],
            best_index=res["best_index"],
            cv_scores=res["cv_scores"],
        ),
    )
    rec[f"best_{grid_label}"] = float(res["best_param"])
    rec.update(classification_metrics(yte, scores))
    return rec


# ---------------------------------------------------------------------------
# TorchKM
# ---------------------------------------------------------------------------


def run_torchkm(
    data,
    sig,
    Cs,
    foldid,
    args,
    dev,
    seed,
    *,
    low_rank: bool = False,
    landmarks: Optional[int] = None,
    rank: Optional[int] = None,
    estimator: str = "svm",
) -> Dict[str, Any]:
    """TorchKM with integrated model selection (exact or Nyström path)."""
    from torchkm.estimators import TorchKMDWD, TorchKMLogit, TorchKMSVC

    cls = {"svm": TorchKMSVC, "dwd": TorchKMDWD, "logit": TorchKMLogit}[estimator]
    kwargs: Dict[str, Any] = dict(
        kernel="rbf",
        rbf_sigma=float(sig),
        Cs=np.asarray(Cs, dtype=float),
        nC=len(Cs),
        cv=int(args.folds),
        foldid=foldid,
        device=dev,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        random_state=int(seed),
    )
    params: Dict[str, Any] = {
        "estimator": estimator,
        "max_iter": args.max_iter,
        "tol": args.tol,
    }
    if getattr(args, "kkt_eps", None) is not None:
        kwargs["KKTeps"] = float(args.kkt_eps)
        params["KKTeps"] = float(args.kkt_eps)
    if getattr(args, "kkt_scaled", False):
        kwargs["kkt_scaled"] = True
        params["kkt_scaled"] = True
    if low_rank:
        kwargs.update(low_rank=True, num_landmarks=int(landmarks), nys_k=int(rank))
        params.update(num_landmarks=int(landmarks), nys_k=int(rank))
    clf = cls(**kwargs)
    with PeakMemory(dev) as pm, timed(dev) as t:
        clf.fit(data["Xtr"], data["ytr"])
    scores = decision_chunked(clf, data["Xte"])
    conv = getattr(clf, "converged_", None)
    rec = dict(
        library="torchkm_nystrom" if low_rank else "torchkm",
        mode="nystrom" if low_rank else "exact",
        device=dev,
        params=params,
        status="ok",
        time_s=t.dt,
        memory=pm.result,
        torch_peak_bytes=clf.peak_gpu_memory_bytes_,
        best_C=float(clf.best_C_),
        converged_frac=None if conv is None else float(np.mean(conv)),
    )
    if low_rank:
        rec["params"]["nys_k_effective"] = getattr(clf, "nys_k_", None)
    rec.update(classification_metrics(data["yte"], scores))
    return rec


# ---------------------------------------------------------------------------
# SMO-based SVMs: scikit-learn (CPU), ThunderSVM (GPU), cuML (GPU)
# ---------------------------------------------------------------------------


def run_sklearn_svc(data, sig, Cs, foldid, args, dev, seed) -> Dict[str, Any]:
    from sklearn.svm import SVC

    gamma = gamma_from_sigest(sig)
    cache = float(getattr(args, "svc_cache_mb", 2000))

    def make(C):
        return SVC(kernel="rbf", C=float(C), gamma=gamma, cache_size=cache)

    def fit_predict(C, Xa, ya, Xb):
        return make(C).fit(Xa, ya).decision_function(Xb)

    return _sweep_record(
        library="sklearn_svc",
        mode="exact",
        device="cpu",
        fit_predict=fit_predict,
        refit_decision=fit_predict,
        data=data,
        grid=list(map(float, Cs)),
        foldid=foldid,
        args=args,
        params={"gamma": gamma, "cache_size_mb": cache},
    )


def run_thundersvm(data, sig, Cs, foldid, args, dev, seed) -> Dict[str, Any]:
    """ThunderSVM 0.3.4 tuned by the same fold loop.

    ThunderSVM can return zero support vectors at the strongly regularised end
    of the grid and crash inside its own csr conversion; those folds score NaN
    in the sweep (``cv_sweep`` averages with ``nanmean``).
    """
    from thundersvm import SVC

    gamma = gamma_from_sigest(sig)
    f32 = bool(getattr(args, "float32_baselines", False))
    d = dict(data)
    d["Xtr"], d["Xte"] = _cast(data["Xtr"], f32), _cast(data["Xte"], f32)

    def fit_predict(C, Xa, ya, Xb):
        model = SVC(kernel="rbf", C=float(C), gamma=gamma, tol=1e-3)
        model.fit(Xa, ya)
        return np.asarray(model.decision_function(Xb)).reshape(-1)

    return _sweep_record(
        library="thundersvm",
        mode="exact",
        device=dev,
        fit_predict=fit_predict,
        refit_decision=fit_predict,
        data=d,
        grid=list(map(float, Cs)),
        foldid=foldid,
        args=args,
        params={"gamma": gamma, "float32_input": f32},
    )


def run_cuml_svc(data, sig, Cs, foldid, args, dev, seed) -> Dict[str, Any]:
    """cuML ``SVC`` (RAPIDS) tuned by the same fold loop on the same grid."""
    from cuml.svm import SVC

    gamma = gamma_from_sigest(sig)
    f32 = bool(getattr(args, "float32_baselines", False))
    d = dict(data)
    d["Xtr"], d["Xte"] = _cast(data["Xtr"], f32), _cast(data["Xte"], f32)
    cache = float(getattr(args, "svc_cache_mb", 2000))

    def fit_predict(C, Xa, ya, Xb):
        model = SVC(
            kernel="rbf", C=float(C), gamma=gamma, cache_size=cache, output_type="numpy"
        )
        model.fit(Xa, ya)
        return np.asarray(model.decision_function(Xb)).reshape(-1)

    return _sweep_record(
        library="cuml_svc",
        mode="exact",
        device=dev,
        fit_predict=fit_predict,
        refit_decision=fit_predict,
        data=d,
        grid=list(map(float, Cs)),
        foldid=foldid,
        args=args,
        params={"gamma": gamma, "cache_size_mb": cache, "float32_input": f32},
    )


# ---------------------------------------------------------------------------
# Falkon: Nyström kernel ridge regression on GPU
# ---------------------------------------------------------------------------


def run_falkon(
    data, sig, Cs, foldid, args, dev, seed, *, centers: int
) -> Dict[str, Any]:
    """Falkon (squared loss, ``M`` Nyström centres) tuned over the same lambda grid.

    Falkon minimises (1/n) ||y - f||^2 + lambda ||f||^2, the same
    (loss + lambda * penalty) form as TorchKM, so the grid is lambda = 1/(2 n C).
    Predictions are thresholded at zero. Note the loss is squared, not hinge.
    """
    import falkon
    from falkon.kernels import GaussianKernel

    n = data["Xtr"].shape[0]
    lams = lam_from_c(np.asarray(Cs, dtype=float), n)
    kernel = GaussianKernel(sigma=falkon_sigma_from_sigest(sig))
    use_cpu = not str(dev).startswith("cuda")
    options = falkon.FalkonOptions(use_cpu=use_cpu, keops_active="no", debug=False)
    maxiter = int(getattr(args, "falkon_maxiter", 20))

    def as_t(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

    def fit_predict(lam, Xa, ya, Xb):
        model = falkon.Falkon(
            kernel=kernel,
            penalty=float(lam),
            M=int(min(centers, Xa.shape[0])),
            maxiter=maxiter,
            seed=int(seed),
            options=options,
        )
        model.fit(as_t(Xa), as_t(ya).reshape(-1, 1))
        return model.predict(as_t(Xb)).reshape(-1).cpu().numpy()

    rec = _sweep_record(
        library="falkon",
        mode="nystrom",
        device=dev,
        fit_predict=fit_predict,
        refit_decision=fit_predict,
        data=data,
        grid=list(map(float, lams)),
        foldid=foldid,
        args=args,
        params={"centers": int(centers), "maxiter": maxiter, "loss": "squared"},
        grid_label="lambda",
    )
    rec["best_C"] = float(1.0 / (2.0 * n * rec["best_lambda"]))
    return rec


# ---------------------------------------------------------------------------
# Linear baselines: what the kernel buys
# ---------------------------------------------------------------------------


def run_linear(
    data, sig, Cs, foldid, args, dev, seed, *, model: str = "logreg"
) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    def make(C):
        if model == "logreg":
            return LogisticRegression(C=float(C), max_iter=2000)
        return LinearSVC(C=float(C), dual=False, max_iter=5000)

    def fit_predict(C, Xa, ya, Xb):
        return make(C).fit(Xa, ya).decision_function(Xb)

    return _sweep_record(
        library=model,
        mode="linear",
        device="cpu",
        fit_predict=fit_predict,
        refit_decision=fit_predict,
        data=data,
        grid=list(map(float, Cs)),
        foldid=foldid,
        args=args,
        params={"model": model},
    )


def print_record(dataset: str, rec: Dict[str, Any]) -> None:
    """One console line per record, in the same order as the JSON fields."""
    from _common import fmt_bytes

    mem = rec.get("memory") or {}
    peak = (
        rec.get("torch_peak_bytes")
        or mem.get("nvml_process_peak")
        or mem.get("host_rss_peak")
    )
    acc = rec.get("accuracy")
    auc = rec.get("auc")
    print(
        f"{dataset:>12} {rec['library']:>16} {rec.get('mode', ''):>8} "
        f"{rec.get('status', ''):>8} "
        f"acc={acc if acc is None else f'{acc:.4f}'} "
        f"auc={auc if auc is None else f'{auc:.4f}'} "
        f"bal={rec.get('balanced_accuracy', float('nan')):.4f} "
        f"t={rec.get('time_s', float('nan')):8.1f}s mem={fmt_bytes(peak)}",
        flush=True,
    )
