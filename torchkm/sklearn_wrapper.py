from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import torch

from .functions import sigest, rbf_kernel as rbf_kernel_train, kernelMult

from .cvksvm import cvksvm
from .cvkdwd import cvkdwd
from .cvklogit import cvklogit
from .platt import PlattScalerTorch


# ---- sklearn is OPTIONAL: raise a clean error only when wrapper is imported ----
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
except Exception as e:
    raise ImportError(
        "torchkm.sklearn_wrapper requires scikit-learn.\n"
        "Install it via: pip install scikit-learn\n"
        "or (recommended) add an extra and do: pip install torchkm[sklearn]"
    ) from e


KernelName = Literal["rbf", "linear", "poly", "precomputed"]
BackendName = Literal["svm", "dwd", "logit"]


def _as_numpy(X: Any) -> np.ndarray:
    """Convert input to a dense numpy array (float64)."""
    if isinstance(X, np.ndarray):
        return X
    if torch.is_tensor(X):
        return X.detach().cpu().numpy()
    return np.asarray(X)


def _pick_device_str(device: Optional[Union[str, torch.device]]) -> str:
    """
    Return exactly 'cuda' or 'cpu' to stay compatible with your current
    internal checks like `if self.device == "cuda": ...` in cvksvm.py.
    """
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, torch.device):
        return "cuda" if device.type == "cuda" else "cpu"
    dev = str(device).lower()
    return "cuda" if dev.startswith("cuda") and torch.cuda.is_available() else "cpu"


def _make_ulam(nlam: int, ulam: Optional[Any], ulam_max: float, ulam_min: float) -> torch.Tensor:
    """
    torch.logspace uses base-10 exponents.
    Default matches your docs: torch.logspace(3, -3, steps=nlam). :contentReference[oaicite:2]{index=2}
    """
    if ulam is not None:
        u = torch.as_tensor(_as_numpy(ulam), dtype=torch.double)
        if u.ndim != 1:
            raise ValueError("ulam must be 1D (sequence of lambdas).")
        return u
    start = float(np.log10(ulam_max))
    end = float(np.log10(ulam_min))
    return torch.logspace(start, end, steps=int(nlam), dtype=torch.double)


def _make_foldid(n: int, nfolds: int, foldid: Optional[Any], random_state: Optional[int]) -> torch.Tensor:
    """
    Your internal generators use fold IDs in {1,...,nfolds}. We follow that.
    """
    if foldid is not None:
        f = torch.as_tensor(_as_numpy(foldid)).reshape(-1)
        if f.numel() != n:
            raise ValueError("foldid must have length n_samples.")
        return f.to(torch.int64)

    # deterministic folds if random_state is set
    g = torch.Generator()
    if random_state is not None:
        g.manual_seed(int(random_state))
    perm = torch.randperm(n, generator=g)
    return (perm % int(nfolds) + 1).to(torch.int64)


def _check_binary_y(y: np.ndarray) -> Tuple[np.ndarray, Any, Any]:
    """
    Map arbitrary binary labels to internal {-1, +1} used by torchkm solvers.
    Returns (y_internal_pm1, neg_label, pos_label).
    """
    y = np.asarray(y).reshape(-1)
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(f"Only binary classification is supported. Got classes={classes}.")
    neg_label, pos_label = classes[0], classes[1]
    y_pm1 = np.where(y == pos_label, 1.0, -1.0).astype(np.float64)
    return y_pm1, neg_label, pos_label


class _TorchKMBaseBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    Common sklearn wrapper for your torchkm large-margin *binary* classifiers.

    Notes:
    - Supports kernels: rbf / linear / poly / precomputed.
    - Uses your pathwise solver (nlam lambdas) and selects best lambda by CV
      using your built-in `model.cv(model.pred, y)` routine. :contentReference[oaicite:3]{index=3}
    """
    _BACKEND: BackendName = "svm"

    def __init__(
        self,
        kernel: KernelName = "rbf",
        nlam: int = 50,
        ulam: Optional[Any] = None,
        ulam_max: float = 1e3,
        ulam_min: float = 1e-3,
        nfolds: int = 5,
        foldid: Optional[Any] = None,
        eps: float = 1e-5,
        maxit: int = 1000,
        solver_gamma: float = 1e-8,
        is_exact: int = 0,  # only used by cvksvm/cvkdwd
        device: Optional[Union[str, torch.device]] = None,
        # RBF
        rbf_sigma: Optional[float] = None,
        sigest_frac: float = 0.5,
        # Poly
        poly_degree: int = 3,
        poly_coef0: float = 1.0,
        poly_gamma: float = 1.0,
        # Probability
        probability: bool = False,
        platt_device: Optional[Union[str, torch.device]] = None,
        random_state: Optional[int] = None,
        store_path: bool = False,  # store full path (big) or keep only best
    ):
        self.kernel = kernel
        self.nlam = nlam
        self.ulam = ulam
        self.ulam_max = ulam_max
        self.ulam_min = ulam_min
        self.nfolds = nfolds
        self.foldid = foldid
        self.eps = eps
        self.maxit = maxit
        self.solver_gamma = solver_gamma
        self.is_exact = is_exact
        self.device = device

        self.rbf_sigma = rbf_sigma
        self.sigest_frac = sigest_frac

        self.poly_degree = poly_degree
        self.poly_coef0 = poly_coef0
        self.poly_gamma = poly_gamma

        self.probability = probability
        self.platt_device = platt_device
        self.random_state = random_state
        self.store_path = store_path

    def _compute_K_train(self, X_t: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute training kernel matrix K(X,X).
        Returns (K_train, kernel_state) where kernel_state holds params needed for test kernel.
        """
        if self.kernel == "rbf":
            sigma = self.rbf_sigma
            if sigma is None:
                # your sigest returns a float gamma-like parameter :contentReference[oaicite:4]{index=4}
                sigma = float(sigest(X_t, frac=float(self.sigest_frac)))
            K = rbf_kernel_train(X_t, sigma)
            return K, {"sigma": sigma}

        if self.kernel == "linear":
            K = X_t @ X_t.T
            return K, {}

        if self.kernel == "poly":
            K = (self.poly_gamma * (X_t @ X_t.T) + self.poly_coef0) ** self.poly_degree
            return K, {}

        raise ValueError(f"Unsupported kernel={self.kernel} for non-precomputed mode.")

    def _compute_K_test(self, X_test_t: torch.Tensor, X_train_t: torch.Tensor, kernel_state: dict) -> torch.Tensor:
        """
        Compute test kernel K(X_test, X_train).
        """
        if self.kernel == "rbf":
            sigma = float(kernel_state["sigma"])
            return kernelMult(X_test_t, X_train_t, sigma)

        if self.kernel == "linear":
            return X_test_t @ X_train_t.T

        if self.kernel == "poly":
            return (self.poly_gamma * (X_test_t @ X_train_t.T) + self.poly_coef0) ** self.poly_degree

        raise ValueError(f"Unsupported kernel={self.kernel} for non-precomputed mode.")

    def fit(self, X: Any, y: Any):
        X_np, y_np = check_X_y(_as_numpy(X), _as_numpy(y), accept_sparse=False, ensure_2d=True)
        y_pm1, neg_label, pos_label = _check_binary_y(y_np)

        self.classes_ = np.array([neg_label, pos_label], dtype=object)
        self.n_features_in_ = X_np.shape[1]

        dev = _pick_device_str(self.device)
        self._device_str_ = dev

        # lambdas
        ulam_t = _make_ulam(self.nlam, self.ulam, self.ulam_max, self.ulam_min)
        nlam = int(ulam_t.numel())

        # folds (int64 on CPU, backend will move to device)
        foldid_t = _make_foldid(n=X_np.shape[0], nfolds=self.nfolds, foldid=self.foldid, random_state=self.random_state)
        # Store the actual fold assignment used (sklearn-style learned attribute)
        self.foldid_ = foldid_t.detach().cpu().to(torch.int64).numpy()

        # tensors
        X_train_t = torch.as_tensor(X_np, dtype=torch.float)  # keep on CPU for sklearn-ish behavior
        y_train_t = torch.as_tensor(y_pm1, dtype=torch.float)

        if self.kernel == "precomputed":
            # X is K_train: (n,n)
            K_train = torch.as_tensor(X_np, dtype=torch.double)
            if K_train.ndim != 2 or K_train.shape[0] != K_train.shape[1]:
                raise ValueError("For kernel='precomputed', X must be a square (n,n) kernel matrix.")
            kernel_state = {}
            self.X_fit_ = None
        else:
            K_train, kernel_state = self._compute_K_train(X_train_t)
            self.X_fit_ = X_np  # store original training features (CPU)

        # backend expects torch.Tensor inputs (it validates this in cvksvm/cvkdwd) :contentReference[oaicite:5]{index=5}
        K_train = K_train.to(dev)
        y_backend = y_train_t.to(dev)
        ulam_backend = ulam_t.to(dev)
        foldid_backend = foldid_t.to(dev)

        if self._BACKEND == "svm":
            backend = cvksvm(
                Kmat=K_train,
                y=y_backend,
                nlam=nlam,
                ulam=ulam_backend,
                foldid=foldid_backend,
                nfolds=int(self.nfolds),
                eps=float(self.eps),
                maxit=int(self.maxit),
                gamma=float(self.solver_gamma),
                is_exact=int(self.is_exact),
                device=dev,
            )
        elif self._BACKEND == "dwd":
            backend = cvkdwd(
                Kmat=K_train,
                y=y_backend,
                nlam=nlam,
                ulam=ulam_backend,
                foldid=foldid_backend,
                nfolds=int(self.nfolds),
                eps=float(self.eps),
                maxit=int(self.maxit),
                gamma=float(self.solver_gamma),
                device=dev,
            )
        elif self._BACKEND == "logit":
            # cvklogit requires foldid in its signature
            backend = cvklogit(
                Kmat=K_train,
                y=y_backend,
                nlam=nlam,
                ulam=ulam_backend,
                foldid=foldid_backend,
                nfolds=int(self.nfolds),
                eps=float(self.eps),
                maxit=int(self.maxit),
                gamma=float(self.solver_gamma),
                device=dev,
            )
        else:
            raise ValueError(f"Unknown backend {self._BACKEND}")

        backend.fit()

        # CV selection: backend.cv expects y on CPU shape (n,) :contentReference[oaicite:6]{index=6}
        cv_mis_t = backend.cv(backend.pred, y_train_t)  # returns tensor length nlam
        cv_mis = cv_mis_t.detach().cpu().numpy()
        best_ind = int(np.argmin(cv_mis))

        # extract best solution
        alpvec = backend.alpmat[:, best_ind].detach().cpu().to(torch.double)
        self.intercept_ = float(alpvec[0].item())
        self.alpha_ = alpvec[1:].numpy()  # length n_train
        self.best_ind_ = best_ind
        self.best_lambda_ = float(backend.ulam[best_ind].detach().cpu().item())
        self.cv_mis_ = cv_mis

        self.kernel_state_ = kernel_state
        self.n_samples_fit_ = int(X_np.shape[0])

        if self.store_path:
            self.alpmat_path_ = backend.alpmat.detach().cpu()
            self.pred_path_ = backend.pred.detach().cpu()
        else:
            self.alpmat_path_ = None
            self.pred_path_ = None

        # optional Platt scaling (uses raw decision values; your class expects that)
        self.platt_ = None
        if self.probability:
            platt_dev = _pick_device_str(self.platt_device)
            oof_scores = backend.pred[:, best_ind].detach().cpu().to(torch.double)
            self.platt_ = PlattScalerTorch(device=platt_dev).fit(oof_scores, y_train_t)

        # free big GPU kernel tensor ASAP
        del backend
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["alpha_", "intercept_", "classes_", "best_lambda_"])

        X_np = check_array(_as_numpy(X), accept_sparse=False, ensure_2d=True)
        dev = getattr(self, "_device_str_", "cpu")

        alpha_t = torch.as_tensor(self.alpha_, dtype=torch.double, device=dev)
        b = float(self.intercept_)

        if self.kernel == "precomputed":
            # X is K_test: (n_test, n_train)
            K_test = torch.as_tensor(X_np, dtype=torch.double, device=dev)
            if K_test.ndim != 2 or K_test.shape[1] != self.n_samples_fit_:
                raise ValueError(
                    f"For kernel='precomputed', X must have shape (n_test, {self.n_samples_fit_})."
                )
        else:
            X_train_t = torch.as_tensor(self.X_fit_, dtype=torch.double)  # CPU
            X_test_t = torch.as_tensor(X_np, dtype=torch.double)          # CPU
            K_test = self._compute_K_test(X_test_t, X_train_t, self.kernel_state_).to(dev)

        with torch.no_grad():
            scores = torch.mv(K_test, alpha_t) + b
        return scores.detach().cpu().numpy()

    def predict(self, X: Any) -> np.ndarray:
        scores = self.decision_function(X)
        neg_label, pos_label = self.classes_[0], self.classes_[1]
        return np.where(scores > 0, pos_label, neg_label)

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["alpha_", "intercept_", "classes_"])
        if self.platt_ is None:
            raise AttributeError(
                "probability=False (or Platt not fitted). Initialize with probability=True to enable predict_proba."
            )
        scores = self.decision_function(X)
        with torch.no_grad():
            proba_t = self.platt_.predict_proba(torch.as_tensor(scores, dtype=torch.double))
        return proba_t.detach().cpu().numpy()


class TorchKMSVC(_TorchKMBaseBinaryClassifier):
    """sklearn-style wrapper for torchkm.cvksvm. :contentReference[oaicite:7]{index=7}"""
    _BACKEND: BackendName = "svm"


class TorchKMDWD(_TorchKMBaseBinaryClassifier):
    """sklearn-style wrapper for torchkm.cvkdwd. :contentReference[oaicite:8]{index=8}"""
    _BACKEND: BackendName = "dwd"


class TorchKMLogit(_TorchKMBaseBinaryClassifier):
    """sklearn-style wrapper for torchkm.cvklogit. :contentReference[oaicite:9]{index=9}"""
    _BACKEND: BackendName = "logit"
