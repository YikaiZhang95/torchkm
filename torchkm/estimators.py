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
from .cvknyssvm import cvknyssvm
from .cvknysdwd import cvknysdwd
from .cvknyslogit import cvknyslogit


# ---- sklearn is OPTIONAL: raise a clean error only when wrapper is imported ----
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
except Exception as e:
    raise ImportError(
        "torchkm.estimators requires scikit-learn.\n"
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


def _make_ulam(nC: int, Cs: Optional[Any], C_max: float, C_min: float) -> torch.Tensor:
    """
    torch.logspace uses base-10 exponents.
    Default matches your docs: torch.logspace(3, -3, steps=nlam). :contentReference[oaicite:2]{index=2}
    """
    if Cs is not None:
        u = torch.as_tensor(_as_numpy(Cs), dtype=torch.double)
        if u.ndim != 1:
            raise ValueError("Cs must be 1D (sequence of C).")
        return u
    start = float(np.log10(C_max))
    end = float(np.log10(C_min))
    return torch.logspace(start, end, steps=int(nC), dtype=torch.double)


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
        raise ValueError(f"Only binary classification is supported. Got classes={classes}. For multiclass case, the problem can be addressed using either a one-vs-one or a one-vs-rest strategy.")
    neg_label, pos_label = classes[0], classes[1]
    y_pm1 = np.where(y == pos_label, 1.0, -1.0).astype(np.float64)
    return y_pm1, neg_label, pos_label


class _TorchKMBaseBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    Common sklearn wrapper for your torchkm large-margin *binary* classifiers.

    Notes:
    - Supports kernels: rbf / linear / poly / precomputed.
    - Uses your pathwise solver (nC Cs) and selects best C by CV
      using your built-in `model.cv(model.pred, y)` routine. :contentReference[oaicite:3]{index=3}
    """
    _BACKEND: BackendName = "svm"

    def __init__(
        self,
        kernel: KernelName = "rbf",
        nC: int = 50,
        Cs: Optional[Any] = None,
        C_max: float = 1e3,
        C_min: float = 1e-3,
        nfolds: int = 5,
        foldid: Optional[Any] = None,
        tol: float = 1e-5,
        max_iter: int = 1000,
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
        #Nystrom
        low_rank: bool = False,
        num_landmarks: int = 2000,
        nys_k: int = 1000,
    ):
        self.kernel = kernel
        self.nC = nC
        self.Cs = Cs
        self.C_max = C_max
        self.C_min = C_min
        self.nfolds = nfolds
        self.foldid = foldid
        self.tol = tol
        self.max_iter = max_iter
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

        self.low_rank = low_rank
        self.num_landmarks = num_landmarks
        self.nys_k = nys_k

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
        self.y_fit_original_ = np.asarray(y_np).copy()
        self.n_features_in_ = X_np.shape[1]
        self._validate_low_rank()

        dev = _pick_device_str(self.device)
        self._device_str_ = dev

        # lambdas
        uC_t = _make_ulam(self.nC, self.Cs, self.C_max, self.C_min)
        ulam_t = 1.0 / (2 * X_np.shape[0] * uC_t)
        nlam = int(ulam_t.numel())

        # folds (int64 on CPU, backend will move to device)
        foldid_t = _make_foldid(n=X_np.shape[0], nfolds=self.nfolds, foldid=self.foldid, random_state=self.random_state)
        # Store the actual fold assignment used (sklearn-style learned attribute)
        self.foldid_ = foldid_t.detach().cpu().to(torch.int64).numpy()

        # tensors
        X_train_t = torch.as_tensor(X_np, dtype=torch.double)  # keep on CPU for sklearn-ish behavior
        y_train_t = torch.as_tensor(y_pm1, dtype=torch.double)

        ulam_backend = ulam_t.to(dev)
        foldid_backend = foldid_t.to(dev)
        y_backend = y_train_t.to(dev)

        self.foldid_ = foldid_t.detach().cpu().to(torch.int64).numpy()
        self.y_fit_original_ = np.asarray(y_np).copy()

        if self.low_rank:
            self.X_fit_ = X_np
            self.kernel_state_ = {"low_rank": True}
            K_train = None
        else:
            if self.kernel == "precomputed":
                K_train = torch.as_tensor(X_np, dtype=torch.double)
                if K_train.ndim != 2 or K_train.shape[0] != K_train.shape[1]:
                    raise ValueError("For kernel='precomputed', X must be a square (n,n) kernel matrix.")
                self.X_fit_ = None
                self.kernel_state_ = {}
            else:
                K_train, kernel_state = self._compute_K_train(X_train_t)
                self.X_fit_ = X_np
                self.kernel_state_ = kernel_state

            K_train = K_train.to(dev)

            # self.X_fit_ = X_np
            # self.kernel_state_ = {"low_rank": True}

            # backend = cvknysdwd(
            #     Xmat=X_train_t,          # raw training features
            #     X_test=X_train_t,        # use train set here; wrapper handles future X via transform()
            #     y=y_backend,
            #     nlam=nlam,
            #     ulam=ulam_backend,
            #     foldid=foldid_backend,
            #     nfolds=int(self.nfolds),
            #     eps=float(self.tol),
            #     maxit=int(self.max_iter),
            #     gamma=float(self.solver_gamma),
            #     num_landmarks=int(self.num_landmarks),
            #     k=int(self.nys_k),
            #     device=dev,
            # )
            # backend.fit()

            # # backend.pred is (n_train, nlam), just like the exact wrapper expects
            # cv_mis_t = backend.cv(backend.pred, y_train_t)
            # cv_mis = cv_mis_t.detach().cpu().numpy()
            # best_ind = int(np.argmin(cv_mis))

            # alpvec = backend.alpmat[:, best_ind].detach().cpu().to(torch.double)

            # self.intercept_ = float(alpvec[0].item())
            # self.alpha_ = alpvec[1:].numpy()   # coefficients in Nyström feature space
            # self.low_rank_basis_dim_ = int(self.alpha_.shape[0])

            # self.best_ind_ = best_ind
            # self.best_C_ = float(1.0 / (2.0 * backend.ulam[best_ind].detach().cpu().item() * X_np.shape[0])) #transfer back
            # self.cv_mis_ = cv_mis
            # self.n_samples_fit_ = int(X_np.shape[0])

            # # keep backend; decision_function() needs transform()
            # self._nys_backend_ = backend

            # # useful fitted attrs for debugging / reproducibility
            # self.low_rank_landmark_indices_ = backend.indices.detach().cpu().numpy()
            # self.num_landmarks_ = int(getattr(backend, "landmarks_", X_train_t).shape[0])
            # self.nys_k_ = int(getattr(backend, "k_eff_", self.alpha_.shape[0]))

        # else: 
        #     if self.kernel == "precomputed":
        #         # X is K_train: (n,n)
        #         K_train = torch.as_tensor(X_np, dtype=torch.double)
        #         if K_train.ndim != 2 or K_train.shape[0] != K_train.shape[1]:
        #             raise ValueError("For kernel='precomputed', X must be a square (n,n) kernel matrix.")
        #         kernel_state = {}
        #         self.X_fit_ = None
        #     else:
        #         K_train, kernel_state = self._compute_K_train(X_train_t)
        #         self.X_fit_ = X_np  # store original training features (CPU)

        #     # backend expects torch.Tensor inputs (it validates this in cvksvm/cvkdwd) :contentReference[oaicite:5]{index=5}
        #     K_train = K_train.to(dev)
        #     y_backend = y_train_t.to(dev)
        #     ulam_backend = ulam_t.to(dev)
        #     foldid_backend = foldid_t.to(dev)

        #     if self._BACKEND == "svm":
        #         backend = cvksvm(
        #             Kmat=K_train,
        #             y=y_backend,
        #             nlam=nlam,
        #             ulam=ulam_backend,
        #             foldid=foldid_backend,
        #             nfolds=int(self.nfolds),
        #             eps=float(self.tol),
        #             maxit=int(self.max_iter),
        #             gamma=float(self.solver_gamma),
        #             is_exact=int(self.is_exact),
        #             device=dev,
        #         )
        #     elif self._BACKEND == "dwd":
        #         backend = cvkdwd(
        #             Kmat=K_train,
        #             y=y_backend,
        #             nlam=nlam,
        #             ulam=ulam_backend,
        #             foldid=foldid_backend,
        #             nfolds=int(self.nfolds),
        #             eps=float(self.tol),
        #             maxit=int(self.max_iter),
        #             gamma=float(self.solver_gamma),
        #             device=dev,
        #         )
        #     elif self._BACKEND == "logit":
        #         # cvklogit requires foldid in its signature
        #         backend = cvklogit(
        #             Kmat=K_train,
        #             y=y_backend,
        #             nlam=nlam,
        #             ulam=ulam_backend,
        #             foldid=foldid_backend,
        #             nfolds=int(self.nfolds),
        #             eps=float(self.tol),
        #             maxit=int(self.max_iter),
        #             gamma=float(self.solver_gamma),
        #             device=dev,
        #         )
        #     else:
        #         raise ValueError(f"Unknown backend {self._BACKEND}")
        backend = self._make_backend(
            low_rank=self.low_rank,
            dev=dev,
            X_train_t=X_train_t,
            K_train=K_train,
            y_backend=y_backend,
            nlam=nlam,
            ulam_backend=ulam_backend,
            foldid_backend=foldid_backend,
)
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
        self.best_C_ = float(1.0 / (2.0 * backend.ulam[best_ind].detach().cpu().item() * X_np.shape[0])) #transfer back
        self.cv_mis_ = cv_mis

        self.n_samples_fit_ = int(X_np.shape[0])
        
        if self.low_rank:
            self._low_rank_backend_ = backend
            self.low_rank_basis_dim_ = int(self.alpha_.shape[0])

            if hasattr(backend, "indices"):
                self.low_rank_landmark_indices_ = backend.indices.detach().cpu().numpy()
            if hasattr(backend, "landmarks_"):
                self.num_landmarks_ = int(backend.landmarks_.shape[0])
            if hasattr(backend, "k_eff_"):
                self.nys_k_ = int(backend.k_eff_)
        else:
            if self.store_path:
                self.alpmat_path_ = backend.alpmat.detach().cpu()
                self.pred_path_ = backend.pred.detach().cpu()
            else:
                self.alpmat_path_ = None
                self.pred_path_ = None
        # if self.store_path:
        #     self.alpmat_path_ = backend.alpmat.detach().cpu()
        #     self.pred_path_ = backend.pred.detach().cpu()
        # else:
        #     self.alpmat_path_ = None
        #     self.pred_path_ = None
        
        self.platt_ = None
        self.platt_scores_ = None
        self.platt_y_ = None
        self._platt_device_ = None

        if self.probability:
            platt_dev = _pick_device_str(self.platt_device if self.platt_device is not None else dev)
            self._platt_device_ = platt_dev

            # Scores used to fit the calibrator
            oof_scores = backend.pred[:, best_ind].detach().to(torch.double).to(platt_dev)
            y_platt = y_train_t.detach().to(torch.double).to(platt_dev)

            self.platt_ = PlattScalerTorch(device=platt_dev).fit(oof_scores, y_platt)

            # Store CPU copies for plotting/debugging
            self.platt_scores_ = oof_scores.detach().cpu().numpy()
            self.platt_y_ = np.asarray(y_np).copy()
        # optional Platt scaling (uses raw decision values; your class expects that)
        # self.platt_ = None
        # if self.probability:
        #     platt_dev = _pick_device_str(self.platt_device)
        #     oof_scores = backend.pred[:, best_ind].detach().cpu().to(torch.double)
        #     self.platt_ = PlattScalerTorch(device=platt_dev).fit(oof_scores, y_train_t)

        # free big GPU kernel tensor ASAP
        del backend
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["alpha_", "intercept_", "classes_", "best_C_"])

        X_np = check_array(_as_numpy(X), accept_sparse=False, ensure_2d=True)
        dev = getattr(self, "_device_str_", "cpu")

        alpha_t = torch.as_tensor(self.alpha_, dtype=torch.double, device=dev)
        b = float(self.intercept_)

        if self.low_rank:
            check_is_fitted(self, ["_low_rank_backend_"])

            X_test_t = torch.as_tensor(X_np, dtype=torch.double)

            with torch.no_grad():
                Z_test = self._low_rank_backend_.transform(X_test_t)
                scores = torch.mv(Z_test, alpha_t) + b

            return scores.detach().cpu().numpy()

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

        # Use the same device as the fitted Platt scaler
        platt_device = getattr(self, "_platt_device_", "cpu")
        scores_t = torch.as_tensor(scores, dtype=torch.double, device=platt_device)

        with torch.no_grad():
            proba_t = self.platt_.predict_proba(scores_t)

        return proba_t.detach().cpu().numpy()

    # def predict_proba(self, X: Any) -> np.ndarray:
    #     check_is_fitted(self, ["alpha_", "intercept_", "classes_"])
    #     if self.platt_ is None:
    #         raise AttributeError(
    #             "probability=False (or Platt not fitted). Initialize with probability=True to enable predict_proba."
    #         )
    #     scores = self.decision_function(X)
    #     with torch.no_grad():
    #         proba_t = self.platt_.predict_proba(torch.as_tensor(scores, dtype=torch.double))
    #     return proba_t.detach().cpu().numpy()

    def platt_plot(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        *,
        n_bins: int = 15,
        strategy: str = "uniform",
        annotate_counts: bool = True,
        figsize: Tuple[float, float] = (5.2, 5.2),
        title: str = "Calibration (Reliability) Curve",
        savepath: Optional[str] = None,
        dpi: int = 150,
        ax=None,
    ):
        """
        Plot a calibration / reliability curve for the fitted Platt scaler.

        Parameters
        ----------
        X : array-like or None
            If provided, compute predict_proba(X) and plot reliability against y.
            If omitted, use the stored training calibration scores from fit().

        y : array-like or None
            True labels corresponding to X.
            If X is None and y is None, stored training labels from fit() are used.

        n_bins : int
            Number of bins used in the reliability curve.

        strategy : {"uniform", "quantile"}
            How to bin probabilities.

        annotate_counts : bool
            If True, annotate each point with the number of samples in that bin.

        figsize : tuple
            Figure size when ax is None.

        title : str
            Plot title.

        savepath : str or None
            If provided, save the plot.

        dpi : int
            Save DPI.

        ax : matplotlib axis or None
            Existing axis to draw on.

        Returns
        -------
        ax : matplotlib axis
        stats : dict
            Contains ECE, Brier score, bin counts, and plotted points.
        """
        check_is_fitted(self, ["classes_"])

        if self.platt_ is None:
            raise AttributeError(
                "Platt scaler is not fitted. Fit with probability=True before calling platt_plot()."
            )

        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise ImportError(
                "platt_plot requires matplotlib. Install it with `pip install matplotlib` "
                "or add it to a visualization extra such as `torchkm[viz]`."
            ) from e

        # ------------------------------------------------------------
        # Get probabilities + labels
        # ------------------------------------------------------------
        if X is None:
            if self.platt_scores_ is None or self.platt_y_ is None:
                raise AttributeError(
                    "Stored calibration data not found. Fit with probability=True first, "
                    "or call platt_plot(X=..., y=...)."
                )

            scores_t = torch.as_tensor(
                self.platt_scores_,
                dtype=torch.double,
                device=getattr(self, "_platt_device_", "cpu"),
            )

            with torch.no_grad():
                proba_t = self.platt_.predict_proba(scores_t)

            proba = proba_t.detach().cpu().numpy()
            y_raw = np.asarray(self.platt_y_).reshape(-1)

        else:
            if y is None:
                raise ValueError("When X is provided, y must also be provided.")

            proba = self.predict_proba(X)
            y_raw = np.asarray(_as_numpy(y)).reshape(-1)

        if proba.ndim == 2:
            p_pos = proba[:, -1].astype(np.float64)
        else:
            p_pos = proba.reshape(-1).astype(np.float64)

        pos_label = self.classes_[1]
        y01 = (y_raw == pos_label).astype(np.float64)

        if p_pos.shape[0] != y01.shape[0]:
            raise ValueError("Predicted probabilities and labels must have the same length.")

        # ------------------------------------------------------------
        # Metrics: ECE and Brier
        # ------------------------------------------------------------
        brier = float(np.mean((p_pos - y01) ** 2))

        # ------------------------------------------------------------
        # Binning
        # ------------------------------------------------------------
        if strategy not in {"uniform", "quantile"}:
            raise ValueError("strategy must be 'uniform' or 'quantile'.")

        if strategy == "uniform":
            edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
        else:
            edges = np.quantile(p_pos, np.linspace(0.0, 1.0, int(n_bins) + 1))
            edges = np.unique(edges)
            if edges.size < 2:
                edges = np.array([0.0, 1.0], dtype=np.float64)

        bin_x = []
        bin_y = []
        bin_n = []

        n = p_pos.shape[0]
        ece = 0.0

        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]

            if i == len(edges) - 2:
                mask = (p_pos >= lo) & (p_pos <= hi)
            else:
                mask = (p_pos >= lo) & (p_pos < hi)

            count = int(mask.sum())
            if count == 0:
                continue

            conf = float(p_pos[mask].mean())   # average predicted probability
            acc = float(y01[mask].mean())      # empirical positive frequency

            bin_x.append(conf)
            bin_y.append(acc)
            bin_n.append(count)

            ece += (count / n) * abs(acc - conf)

        bin_x = np.asarray(bin_x, dtype=np.float64)
        bin_y = np.asarray(bin_y, dtype=np.float64)
        bin_n = np.asarray(bin_n, dtype=np.int64)

        # ------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # light grey background like your example
        fig.patch.set_facecolor("#EAEAF2")
        ax.set_facecolor("#EAEAF2")

        # perfect line
        ax.plot([0, 1], [0, 1], "--", linewidth=1.5, label="Perfect")

        # calibration curve
        label = f"Platt (ECE={ece:.3f}, Brier={brier:.3f})"
        ax.plot(bin_x, bin_y, marker="o", linewidth=1.8, label=label)

        # annotate counts
        if annotate_counts:
            for x_i, y_i, n_i in zip(bin_x, bin_y, bin_n):
                ax.text(x_i, y_i + 0.015, str(int(n_i)), ha="center", va="bottom", fontsize=9)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Predicted probability (bin average)")
        ax.set_ylabel("Observed frequency (empirical)")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper left")

        if savepath is not None:
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

        stats = {
            "ece": float(ece),
            "brier": float(brier),
            "bin_avg_proba": bin_x,
            "bin_empirical_freq": bin_y,
            "bin_count": bin_n,
        }

        return ax, stats

    def _validate_low_rank(self):
        if not self.low_rank:
            return

        if self.kernel == "precomputed":
            raise ValueError(
                "low_rank=True requires raw feature input; kernel='precomputed' is not supported."
            )

        if self.kernel != "rbf":
            raise ValueError(
                "low_rank=True currently supports only kernel='rbf', because cvknyssvm "
                "internally uses an RBF Nyström map."
            )

    def _make_backend(self, *,
    low_rank: bool,
    dev: str,
    X_train_t: torch.Tensor,
    K_train: Optional[torch.Tensor],
    y_backend: torch.Tensor,
    nlam: int,
    ulam_backend: torch.Tensor,
    foldid_backend: torch.Tensor,
    ):
        if low_rank:
            backend_cls = {
                "svm": cvknyssvm,
                "dwd": cvknysdwd,
                "logit": cvknyslogit,
            }[self._BACKEND]

            kwargs = dict(
                Xmat=X_train_t,
                X_test=X_train_t,   # placeholder unless your nys classes make X_test optional
                y=y_backend,
                nlam=nlam,
                ulam=ulam_backend,
                foldid=foldid_backend,
                nfolds=int(self.nfolds),
                eps=float(self.tol),
                maxit=int(self.max_iter),
                gamma=float(self.solver_gamma),
                # delta_len=int(self.delta_len),
                # KKTeps=float(self.KKTeps),
                # KKTeps2=float(self.KKTeps2),
                num_landmarks=int(self.num_landmarks),
                k=int(self.nys_k),
                device=dev,
            )

            # Optional: only keep these if your nys classes support them
            if self.rbf_sigma is not None:
                kwargs["sigma"] = float(self.rbf_sigma)
            if self.random_state is not None:
                kwargs["random_state"] = int(self.random_state)

            return backend_cls(**kwargs)

        # exact backends
        if self._BACKEND == "svm":
            return cvksvm(
                Kmat=K_train,
                y=y_backend,
                nlam=nlam,
                ulam=ulam_backend,
                foldid=foldid_backend,
                nfolds=int(self.nfolds),
                eps=float(self.tol),
                maxit=int(self.max_iter),
                gamma=float(self.solver_gamma),
                is_exact=int(self.is_exact),
                device=dev,
            )

        if self._BACKEND == "dwd":
            return cvkdwd(
                Kmat=K_train,
                y=y_backend,
                nlam=nlam,
                ulam=ulam_backend,
                foldid=foldid_backend,
                nfolds=int(self.nfolds),
                eps=float(self.tol),
                maxit=int(self.max_iter),
                gamma=float(self.solver_gamma),
                device=dev,
            )

        if self._BACKEND == "logit":
            return cvklogit(
                Kmat=K_train,
                y=y_backend,
                nlam=nlam,
                ulam=ulam_backend,
                foldid=foldid_backend,
                nfolds=int(self.nfolds),
                eps=float(self.tol),
                maxit=int(self.max_iter),
                gamma=float(self.solver_gamma),
                device=dev,
            )

        raise ValueError(f"Unknown backend {self._BACKEND}")


class TorchKMSVC(_TorchKMBaseBinaryClassifier):
    """sklearn-style wrapper for torchkm.cvksvm. :contentReference[oaicite:7]{index=7}"""
    _BACKEND: BackendName = "svm"


class TorchKMDWD(_TorchKMBaseBinaryClassifier):
    """sklearn-style wrapper for torchkm.cvkdwd. :contentReference[oaicite:8]{index=8}"""
    _BACKEND: BackendName = "dwd"


class TorchKMLogit(_TorchKMBaseBinaryClassifier):
    """sklearn-style wrapper for torchkm.cvklogit. :contentReference[oaicite:9]{index=9}"""
    _BACKEND: BackendName = "logit"
