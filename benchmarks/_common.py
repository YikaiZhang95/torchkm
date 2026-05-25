"""Shared helpers for the paper benchmark scripts (Tables 2-4).

These utilities implement the common protocol described in the paper supplement
(Appendix B.1): a 50-point regularization grid spaced log-uniformly over
C in [1e-3, 1e3] (LIBSVM/scikit-learn parameterization), 10-fold cross-validation
for model selection, and end-to-end wall-clock timing of the full
training-and-tuning pipeline with a CUDA warmup and synchronization.
"""

from __future__ import annotations

import time

import numpy as np
import torch


def get_device(pref: str | None = None) -> str:
    if pref:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def c_grid(n: int = 50) -> np.ndarray:
    """50 log-uniform C values over [1e-3, 1e3] (paper Appendix B.1)."""
    return np.logspace(3.0, -3.0, num=n)


def lam_grid(n: int = 50) -> np.ndarray:
    """50 log-uniform lambda values over [1e-3, 1e3] (source-notebook grid).

    Convert to the LIBSVM/scikit-learn C parameterization with C = 1/(2*n_obs*lambda).
    """
    return np.logspace(3.0, -3.0, num=n)


def synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


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

    Every method is scored with this same objective functional at its selected
    lambda, so the reported values are comparable (paper Section 3). ``K`` is the
    training kernel and ``alpha``/``intercept`` are that method's fitted solution.
    """
    Ka = K @ alpha
    f = Ka + intercept
    hinge = torch.clamp(1.0 - y * f, min=0.0)
    return float(hinge.mean().item() + lam * torch.dot(alpha, Ka).item())


def mean_se(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    se = a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0
    return float(a.mean()), float(se)


def _filter_classes(X, y, classes):
    if classes is None:
        # Map an arbitrary binary label set to {-1, +1}.
        neg = np.min(np.unique(y))
        return X, np.where(y == neg, -1, 1)
    a, b = classes
    mask = (y == a) | (y == b)
    return X[mask], np.where(y[mask] == a, -1, 1)


def load_train_test(train_path, test_path=None, classes=None, test_size=0.2, seed=0):
    """Load a LIBSVM train/test pair as dense float64 arrays mapped to +-1 labels.

    When ``test_path`` is None (e.g. covtype, MNIST8m, which ship as a single
    file) a deterministic train/test split is made instead. ``classes`` selects
    a two-class subset, e.g. (4, 6) for MNIST8m 4-vs-6.

    Memory note: features are densified for simplicity; the larger Table 4
    datasets require substantial RAM/GPU memory.
    """
    from sklearn.datasets import load_svmlight_file, load_svmlight_files

    if test_path is not None:
        Xtr, ytr, Xte, yte = load_svmlight_files([train_path, test_path])
        Xtr = np.asarray(Xtr.todense(), dtype=np.float64)
        Xte = np.asarray(Xte.todense(), dtype=np.float64)
        Xtr, ytr = _filter_classes(Xtr, ytr, classes)
        Xte, yte = _filter_classes(Xte, yte, classes)
        return Xtr, ytr, Xte, yte

    from sklearn.model_selection import train_test_split

    X, y = load_svmlight_file(train_path)
    X = np.asarray(X.todense(), dtype=np.float64)
    X, y = _filter_classes(X, y, classes)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return Xtr, ytr, Xte, yte
