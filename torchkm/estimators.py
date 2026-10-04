# SPDX-License-Identifier: MIT
from __future__ import annotations

import time
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import torch

from .functions import sigest, rbf_kernel as rbf_kernel_train, kernelMult
from .memory import exact_mode_oom_message

from .cvksvm import cvksvm
from .cvkdwd import cvkdwd
from .cvklogit import cvklogit
from .cvkqr import cvkqr
from .platt import PlattScalerTorch

# ---- sklearn is OPTIONAL: raise a clean error only when wrapper is imported ----
try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
except Exception as e:
    raise ImportError(
        "torchkm.estimators requires scikit-learn.\n"
        "Install TorchKM with its standard dependencies, or install scikit-learn "
        "directly with: pip install scikit-learn"
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
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, torch.device):
        return "cuda" if device.type == "cuda" else "cpu"
    dev = str(device).lower()
    return "cuda" if dev.startswith("cuda") and torch.cuda.is_available() else "cpu"


def _now(dev: str) -> float:
    """Wall clock once the device's queued work is done."""
    if dev == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def _kernel_times(compute_K_test, X_test, X_train, state, alpha, entries=2**25):
    """K(X_test, X_train) @ alpha, the test kernel built on the inputs' device in
    blocks of test rows, so at most ``entries`` of its entries exist at once."""
    rows = max(1, entries // max(1, X_train.shape[0]))
    return torch.cat(
        [
            torch.mv(compute_K_test(X_test[i : i + rows], X_train, state), alpha)
            for i in range(0, X_test.shape[0], rows)
        ]
    )


def _make_ulam(nC: int, Cs: Optional[Any], C_max: float, C_min: float) -> torch.Tensor:
    if Cs is not None:
        u = torch.as_tensor(_as_numpy(Cs), dtype=torch.double)
        if u.ndim != 1:
            raise ValueError("Cs must be 1D (sequence of C).")
        return u
    start = float(np.log10(C_max))
    end = float(np.log10(C_min))
    return torch.logspace(start, end, steps=int(nC), dtype=torch.double)


def _make_foldid(
    n: int, nfolds: int, foldid: Optional[Any], random_state: Optional[int]
) -> torch.Tensor:
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
        raise ValueError(
            "Only binary classification is supported. "
            f"Got classes={classes}. For multiclass case, the problem can be "
            "addressed using either a one-vs-one or a one-vs-rest strategy."
        )
    neg_label, pos_label = classes[0], classes[1]
    y_pm1 = np.where(y == pos_label, 1.0, -1.0).astype(np.float64)
    return y_pm1, neg_label, pos_label


class _TruncatedSVMBackend:
    """The exact SVM backend's interface over
    :class:`torchkm.experimental.SpectralSVMPath`, for ``spectrum="truncated"``.

    The kernel is the same; the solver's curvature keeps only the top ``rank``
    eigenpairs, so there is no full eigendecomposition (peak about 1.2 ``n x n``
    matrices instead of 5), and every lambda and fold stops at the certified
    relative duality gap ``gap_tol``.
    """

    def __init__(
        self, K, y, ulam, foldid, *, rank, gap_tol, seed, block=1, fit_cap=None
    ):
        self.ulam = ulam
        self._problem = (K, y.to(K.dtype), ulam.detach().cpu().tolist(), foldid)
        self._options = dict(
            spectrum="truncated", rank=rank, gap_tol=gap_tol, seed=seed, block=block
        )
        if fit_cap is not None:
            self._options["fit_cap"] = int(fit_cap)

    def fit(self):
        from .experimental import SpectralSVMPath

        K, y, lambdas, foldid = self._problem
        self._problem = None  # hold no reference to the kernel after the fit
        m = SpectralSVMPath(K, y, lambdas, foldid, **self._options).fit()
        self.alpmat, self.pred = m.alphas, m.cv_scores
        self.converged = m.converged & m.fold_converged.all(0)
        self.gaps, self.fold_gaps = m.gaps, m.fold_gaps
        self.timing = dict(m.timing)
        self.npass = torch.tensor(m.path_iterations)
        self.cvnpass = torch.tensor(m.cv_iterations)
        return self

    @staticmethod
    def cv(pred, y):
        """Misclassification rate of the held-out scores per lambda, as cvksvm."""
        pred_label = torch.where(pred > 0, 1, -1).to(device="cpu")
        return (pred_label != y[:, None]).float().mean(dim=0)


class _TorchKMBaseBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    Common sklearn wrapper for your torchkm large-margin *binary* classifiers.
    """

    _BACKEND: BackendName = "svm"

    def __init__(
        self,
        kernel: KernelName = "rbf",
        nC: int = 50,
        Cs: Optional[Any] = None,
        C_max: float = 1e3,
        C_min: float = 1e-3,
        cv: int = 5,
        foldid: Optional[Any] = None,
        tol: float = 1e-5,
        max_iter: int = 1000,
        solver_gamma: float = 1e-8,
        is_exact: int = 0,  # only used by cvksvm/cvkdwd
        KKTeps: float = 1e-3,
        delta_len: int = 8,  # only used by cvksvm
        kkt_scaled: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: str = "float64",  # only used by cvksvm (exact mode)
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
        # truncated spectrum (SVM only)
        spectrum: str = "full",
        spectrum_rank: int = 400,
        gap_tol: float = 1e-3,
        spectrum_block: int = 10,
    ):
        self.kernel = kernel
        self.nC = nC
        self.Cs = Cs
        self.C_max = C_max
        self.C_min = C_min
        self.cv = cv
        self.foldid = foldid
        self.tol = tol
        self.max_iter = max_iter
        self.solver_gamma = solver_gamma
        self.is_exact = is_exact
        self.KKTeps = KKTeps
        self.delta_len = delta_len
        self.kkt_scaled = kkt_scaled
        self.device = device
        self.dtype = dtype

        self.rbf_sigma = rbf_sigma
        self.sigest_frac = sigest_frac

        self.poly_degree = poly_degree
        self.poly_coef0 = poly_coef0
        self.poly_gamma = poly_gamma

        self.probability = probability
        self.platt_device = platt_device
        self.random_state = random_state
        self.store_path = store_path

        self.spectrum = spectrum
        self.spectrum_rank = spectrum_rank
        self.gap_tol = gap_tol
        self.spectrum_block = spectrum_block

    def _low_rank(self) -> bool:
        """TorchKMSVC(low_rank=True): the matrix-free truncated-spectrum SVM."""
        return bool(getattr(self, "low_rank", False))

    def _check_spectrum(self) -> None:
        if self.spectrum not in ("full", "truncated"):
            raise ValueError(
                f"spectrum must be 'full' or 'truncated', got {self.spectrum!r}."
            )
        if self.spectrum == "truncated" or self._low_rank():
            if self._BACKEND != "svm":
                raise ValueError(
                    "spectrum='truncated' is supported by TorchKMSVC only."
                )
            if self.is_exact != 0:
                raise ValueError(
                    "spectrum='truncated' and low_rank=True do not take is_exact=1."
                )

    def _clear_fit_state(self) -> None:
        fitted_attrs = (
            "classes_",
            "y_fit_original_",
            "n_features_in_",
            "foldid_",
            "X_fit_",
            "kernel_state_",
            "_device_str_",
            "intercept_",
            "alpha_",
            "best_ind_",
            "best_C_",
            "cv_mis_",
            "converged_",
            "duality_gaps_",
            "fold_duality_gaps_",
            "n_samples_fit_",
            "alpmat_path_",
            "pred_path_",
            "platt_",
            "platt_scores_",
            "platt_y_",
            "_platt_device_",
            "peak_gpu_memory_bytes_",
            "_work_dtype_",
            "fit_timing_",
            "n_passes_",
            "fit_profile_",
        )
        for attr in fitted_attrs:
            if hasattr(self, attr):
                delattr(self, attr)

    def _work_dtype(self) -> torch.dtype:
        """Precision of the exact solver's kernel matrix and solution path."""
        if self.dtype not in ("float64", "float32"):
            raise ValueError(
                f"dtype must be 'float64' or 'float32', got {self.dtype!r}."
            )
        if self.dtype == "float32" and self._BACKEND != "svm":
            raise ValueError("dtype='float32' is supported by TorchKMSVC only.")
        return torch.float32 if self.dtype == "float32" else torch.float64

    def _compute_K_train(
        self, X_t: torch.Tensor, sigma: Optional[float] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute training kernel matrix K(X,X).
        Returns (K_train, kernel_state) where kernel_state holds params needed for test kernel.
        ``sigma`` rebuilds an RBF kernel with a fitted bandwidth (no new sigest draw).
        """
        if self.kernel == "rbf":
            if sigma is None:
                sigma = self.rbf_sigma
            if sigma is None:
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

    def _compute_K_test(
        self, X_test_t: torch.Tensor, X_train_t: torch.Tensor, kernel_state: dict
    ) -> torch.Tensor:
        """
        Compute test kernel K(X_test, X_train).
        """
        if self.kernel == "rbf":
            sigma = float(kernel_state["sigma"])
            return kernelMult(X_test_t, X_train_t, sigma)

        if self.kernel == "linear":
            return X_test_t @ X_train_t.T

        if self.kernel == "poly":
            return (
                self.poly_gamma * (X_test_t @ X_train_t.T) + self.poly_coef0
            ) ** self.poly_degree

        raise ValueError(f"Unsupported kernel={self.kernel} for non-precomputed mode.")

    def fit(self, X: Any, y: Any):
        try:
            return self._fit_impl(X, y)
        except torch.cuda.OutOfMemoryError as err:
            if self._low_rank():
                raise
            n = int(_as_numpy(X).shape[0])
            raise torch.cuda.OutOfMemoryError(
                exact_mode_oom_message(
                    n,
                    getattr(self, "_device_str_", "cuda"),
                    getattr(self, "_work_dtype_", torch.float64),
                )
            ) from err

    def _fit_impl(self, X: Any, y: Any):
        self._clear_fit_state()

        X_np, y_np = check_X_y(
            _as_numpy(X), _as_numpy(y), accept_sparse=False, ensure_2d=True
        )
        y_pm1, neg_label, pos_label = _check_binary_y(y_np)

        self.classes_ = np.array([neg_label, pos_label], dtype=object)
        self.y_fit_original_ = np.asarray(y_np).copy()
        self.n_features_in_ = X_np.shape[1]
        self._validate_low_rank()
        self._check_spectrum()
        self._work_dtype_ = self._work_dtype()

        dev = _pick_device_str(self.device)
        self._device_str_ = dev
        if dev == "cuda":
            # Peak-memory accounting for the whole fit (kernel build, solver,
            # calibration); read back into ``peak_gpu_memory_bytes_``.
            torch.cuda.reset_peak_memory_stats(dev)
        t_start = _now(dev)

        # lambdas
        uC_t = _make_ulam(self.nC, self.Cs, self.C_max, self.C_min)
        ulam_t = 1.0 / (2 * X_np.shape[0] * uC_t)
        nlam = int(ulam_t.numel())

        # folds (int64 on CPU, backend will move to device)
        foldid_t = _make_foldid(
            n=X_np.shape[0],
            nfolds=self.cv,
            foldid=self.foldid,
            random_state=self.random_state,
        )
        # Store the actual fold assignment used (sklearn-style learned attribute)
        self.foldid_ = foldid_t.detach().cpu().to(torch.int64).numpy()

        # tensors
        X_train_t = torch.as_tensor(
            X_np, dtype=torch.double
        )  # keep on CPU for sklearn-ish behavior
        y_train_t = torch.as_tensor(y_pm1, dtype=torch.double)

        ulam_backend = ulam_t.to(dev)
        foldid_backend = foldid_t.to(dev)
        y_backend = y_train_t.to(dev)

        self.foldid_ = foldid_t.detach().cpu().to(torch.int64).numpy()
        self.y_fit_original_ = np.asarray(y_np).copy()

        t_kernel = _now(dev)
        # Given a way to build the kernel again, the exact solvers factorize it in
        # place (one n x n copy less at the peak) and rebuild it; a precomputed
        # kernel is the caller's array, so it is never overwritten.
        rebuild_kmat = None
        if self._low_rank():
            # the kernel is never stored: products recompute it (fused on CUDA
            # in float32), so memory grows like n x columns, not n^2
            X_dev = X_train_t.to(device=dev, dtype=self._work_dtype_)
            sigma = self.rbf_sigma
            if sigma is None:
                sigma = float(sigest(X_dev, frac=float(self.sigest_frac)))
            self.X_fit_ = X_np
            self.kernel_state_ = {"sigma": float(sigma), "low_rank": True}
            K_train = self._kernel_operator(X_dev, float(sigma))
        else:
            if self.kernel == "precomputed":
                K_train = torch.as_tensor(X_np, dtype=self._work_dtype_)
                if K_train.ndim != 2 or K_train.shape[0] != K_train.shape[1]:
                    raise ValueError(
                        "For kernel='precomputed', X must be a square (n,n) kernel matrix."
                    )
                self.X_fit_ = None
                self.kernel_state_ = {}
            else:
                # Build the kernel on the target device: no host-side n x n
                # copy and no host-to-device transfer of the full matrix.
                X_dev = X_train_t.to(device=dev, dtype=self._work_dtype_)
                K_train, kernel_state = self._compute_K_train(X_dev)
                self.X_fit_ = X_np
                self.kernel_state_ = kernel_state

                def rebuild():  # the same kernel again, with the fitted bandwidth
                    return self._compute_K_train(X_dev, kernel_state.get("sigma"))[0]

                rebuild_kmat = rebuild

            K_train = K_train.to(dev)
        kernel_s = _now(dev) - t_kernel

        backend = self._make_backend(
            low_rank=self._low_rank(),
            dev=dev,
            X_train_t=X_train_t,
            K_train=K_train,
            y_backend=y_backend,
            nlam=nlam,
            ulam_backend=ulam_backend,
            foldid_backend=foldid_backend,
            rebuild_kmat=rebuild_kmat,
        )
        backend.fit()

        # Per-lambda convergence status (None for backends that do not track it)
        conv = getattr(backend, "converged", None)
        self.converged_ = None if conv is None else conv.detach().cpu().numpy().copy()
        # certified relative duality gaps (truncated spectrum and low_rank=True)
        gaps = getattr(backend, "gaps", None)
        fold_gaps = getattr(backend, "fold_gaps", None)
        self.duality_gaps_ = None if gaps is None else np.asarray(gaps, dtype=float)
        self.fold_duality_gaps_ = (
            None if fold_gaps is None else np.asarray(fold_gaps, dtype=float)
        )

        # CV selection: backend.cv expects y on CPU shape (n,)
        cv_mis_t = backend.cv(backend.pred, y_train_t)  # returns tensor length nlam
        cv_mis = cv_mis_t.detach().cpu().numpy()
        best_ind = int(np.argmin(cv_mis))

        # extract best solution
        alpvec = backend.alpmat[:, best_ind].detach().cpu().to(torch.double)
        self.intercept_ = float(alpvec[0].item())
        self.alpha_ = alpvec[1:].numpy()  # length n_train
        self.best_ind_ = best_ind
        self.best_C_ = float(
            1.0 / (2.0 * backend.ulam[best_ind].detach().cpu().item() * X_np.shape[0])
        )  # transfer back
        self.cv_mis_ = cv_mis

        self.n_samples_fit_ = int(X_np.shape[0])

        if self.store_path:
            self.alpmat_path_ = backend.alpmat.detach().cpu()
            self.pred_path_ = backend.pred.detach().cpu()
        else:
            self.alpmat_path_ = None
            self.pred_path_ = None

        self.platt_ = None
        self.platt_scores_ = None
        self.platt_y_ = None
        self._platt_device_ = None

        if self.probability:
            platt_dev = _pick_device_str(
                self.platt_device if self.platt_device is not None else dev
            )
            self._platt_device_ = platt_dev

            oof_scores = (
                backend.pred[:, best_ind].detach().to(torch.double).to(platt_dev)
            )
            y_platt = y_train_t.detach().to(torch.double).to(platt_dev)

            self.platt_ = PlattScalerTorch(device=platt_dev).fit(oof_scores, y_platt)

            self.platt_scores_ = oof_scores.detach().cpu().numpy()
            self.platt_y_ = np.asarray(y_np).copy()

        self.peak_gpu_memory_bytes_ = (
            int(torch.cuda.max_memory_allocated(dev)) if dev == "cuda" else None
        )
        # Where the fit's time went, in seconds: building the kernel, the
        # backend's own phases where it records them (the exact SVM solver:
        # eigendecomposition, its error check, the lambda path, the
        # cross-validation fits), and the whole fit. Passes are solver
        # iterations, one matrix-vector product with the kernel each.
        self.fit_timing_ = dict(
            kernel=kernel_s,
            **(getattr(backend, "timing", None) or {}),
            total=_now(dev) - t_start,
        )
        npass = getattr(backend, "npass", None)
        cvnpass = getattr(backend, "cvnpass", None)
        self.n_passes_ = (
            None
            if npass is None or cvnpass is None
            else dict(path=int(npass.sum()), cross_validation=int(cvnpass.sum()))
        )
        # The same per lambda, and the fold fits' iterations per fold and lambda
        # (the folds of a lambda are fitted together, so they share its time).
        lam_t = getattr(backend, "lambda_timing", None)
        self.fit_profile_ = (
            None
            if lam_t is None
            else dict(
                lambdas=backend.ulam.detach().cpu().tolist(),
                path_seconds=list(lam_t["path"]),
                cv_seconds=list(lam_t["cross_validation"]),
                path_passes=npass.detach().cpu().tolist(),
                cv_passes=backend.fold_passes.tolist(),
            )
        )

        # free big GPU kernel tensor ASAP
        del backend
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["alpha_", "intercept_", "classes_", "best_C_"])

        X_np = check_array(_as_numpy(X), accept_sparse=False, ensure_2d=True)
        dev = getattr(self, "_device_str_", "cpu")

        alpha_t = torch.as_tensor(self.alpha_, dtype=torch.double, device=dev)
        b = float(self.intercept_)

        wdt = getattr(self, "_work_dtype_", torch.float64)
        if self._low_rank():
            # K(X, X_fit) alpha without forming the test kernel
            op = self._kernel_operator(
                torch.as_tensor(self.X_fit_, dtype=wdt, device=dev),
                float(self.kernel_state_["sigma"]),
            )
            X_test_t = torch.as_tensor(X_np, dtype=wdt, device=dev)
            with torch.no_grad():
                scores = op.cross(X_test_t, alpha_t.to(wdt)) + b
            return scores.detach().cpu().double().numpy()

        if self.kernel == "precomputed":
            # X is K_test: (n_test, n_train)
            K_test = torch.as_tensor(X_np, dtype=wdt, device=dev)
            if K_test.ndim != 2 or K_test.shape[1] != self.n_samples_fit_:
                raise ValueError(
                    f"For kernel='precomputed', X must have shape (n_test, {self.n_samples_fit_})."
                )
            with torch.no_grad():
                scores = torch.mv(K_test, alpha_t.to(wdt)) + b
            return scores.detach().cpu().numpy()

        # the test kernel is built on the fit's device (the GPU when there is one)
        X_train_t = torch.as_tensor(self.X_fit_, dtype=wdt, device=dev)
        X_test_t = torch.as_tensor(X_np, dtype=wdt, device=dev)
        with torch.no_grad():
            scores = (
                _kernel_times(
                    self._compute_K_test,
                    X_test_t,
                    X_train_t,
                    self.kernel_state_,
                    alpha_t.to(wdt),
                )
                + b
            )
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

        platt_device = getattr(self, "_platt_device_", "cpu")
        scores_t = torch.as_tensor(scores, dtype=torch.double, device=platt_device)

        with torch.no_grad():
            proba_t = self.platt_.predict_proba(scores_t)

        return proba_t.detach().cpu().numpy()

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
            raise ValueError(
                "Predicted probabilities and labels must have the same length."
            )

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

            conf = float(p_pos[mask].mean())  # average predicted probability
            acc = float(y01[mask].mean())  # empirical positive frequency

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
                ax.text(
                    x_i,
                    y_i + 0.015,
                    str(int(n_i)),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

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
        if not self._low_rank():
            return
        if self.kernel != "rbf":
            raise ValueError(
                "low_rank=True needs kernel='rbf' and raw features: the kernel is "
                "recomputed from the training rows on every product, never stored "
                f"(got kernel={self.kernel!r})."
            )

    def _kernel_operator(self, X_dev: torch.Tensor, sigma: float):
        """The RBF kernel of ``X_dev`` as a never-stored operator: fused
        products on CUDA in float32, blocks of rows otherwise."""
        from .experimental import RBFKernelOperator

        fused = X_dev.device.type == "cuda" and X_dev.dtype == torch.float32
        return RBFKernelOperator(X_dev, sigma, fused=fused)

    def _make_backend(
        self,
        *,
        low_rank: bool,
        dev: str,
        X_train_t: torch.Tensor,
        K_train: Optional[torch.Tensor],
        y_backend: torch.Tensor,
        nlam: int,
        ulam_backend: torch.Tensor,
        foldid_backend: torch.Tensor,
        rebuild_kmat=None,
    ):
        if low_rank:  # K_train is the never-stored kernel operator
            return _TruncatedSVMBackend(
                K_train,
                y_backend,
                ulam_backend,
                foldid_backend,
                rank=int(self.spectrum_rank),
                gap_tol=float(self.gap_tol),
                seed=0 if self.random_state is None else int(self.random_state),
                block=int(self.spectrum_block),
                fit_cap=int(self.max_iter),
            )

        # exact backends
        if self._BACKEND == "svm" and self.spectrum == "truncated":
            return _TruncatedSVMBackend(
                K_train,
                y_backend,
                ulam_backend,
                foldid_backend,
                rank=int(self.spectrum_rank),
                gap_tol=float(self.gap_tol),
                seed=0 if self.random_state is None else int(self.random_state),
                block=int(self.spectrum_block),
            )

        if self._BACKEND == "svm":
            return cvksvm(
                Kmat=K_train,
                y=y_backend,
                nlam=nlam,
                ulam=ulam_backend,
                foldid=foldid_backend,
                nfolds=int(self.cv),
                eps=float(self.tol),
                maxit=int(self.max_iter),
                gamma=float(self.solver_gamma),
                is_exact=int(self.is_exact),
                delta_len=int(self.delta_len),
                KKTeps=float(self.KKTeps),
                kkt_scaled=bool(self.kkt_scaled),
                device=dev,
                dtype=self._work_dtype_,
                rebuild_kmat=rebuild_kmat,
            )

        if self._BACKEND == "dwd":
            return cvkdwd(
                Kmat=K_train,
                y=y_backend,
                nlam=nlam,
                ulam=ulam_backend,
                foldid=foldid_backend,
                nfolds=int(self.cv),
                eps=float(self.tol),
                maxit=int(self.max_iter),
                gamma=float(self.solver_gamma),
                KKTeps=float(self.KKTeps),
                device=dev,
                rebuild_kmat=rebuild_kmat,
            )

        if self._BACKEND == "logit":
            return cvklogit(
                Kmat=K_train,
                y=y_backend,
                nlam=nlam,
                ulam=ulam_backend,
                foldid=foldid_backend,
                nfolds=int(self.cv),
                eps=float(self.tol),
                maxit=int(self.max_iter),
                gamma=float(self.solver_gamma),
                KKTeps=float(self.KKTeps),
                device=dev,
                rebuild_kmat=rebuild_kmat,
            )

        raise ValueError(f"Unknown backend {self._BACKEND}")


class TorchKMSVC(_TorchKMBaseBinaryClassifier):
    """Kernel support vector classifier with integrated model selection.

    ``TorchKMSVC`` is the scikit-learn-style wrapper around
    :class:`torchkm.cvksvm.cvksvm`. It builds a kernel matrix from feature
    input, fits a path of candidate regularization values, selects ``best_C_``
    by cross-validation, and exposes familiar prediction methods.

    Parameters
    ----------
    kernel : {"rbf", "linear", "poly", "precomputed"}, default="rbf"
        Kernel used by the estimator. ``"precomputed"`` expects a square
        training kernel matrix in ``fit`` and a test-by-train kernel matrix in
        ``decision_function`` or ``predict``.
    nC : int, default=50
        Number of candidate ``C`` values when ``Cs`` is not provided.
    Cs : array-like, optional
        Candidate regularization values under the scikit-learn/LIBSVM
        ``C`` convention. Internally these are converted to solver
        regularization values.
    C_max, C_min : float, default=1e3, 1e-3
        Endpoints for the log-spaced ``C`` grid used when ``Cs`` is omitted.
    cv : int, default=5
        Number of cross-validation folds used to choose ``best_C_``.
    foldid : array-like, optional
        Optional fold assignment of length ``n_samples``. Fold labels follow
        the low-level solver convention and are typically in ``1, ..., cv``.
    tol : float, default=1e-5
        Solver convergence tolerance.
    max_iter : int, default=1000
        Maximum number of iterations used by the low-level solver.
    solver_gamma : float, default=1e-8
        Small numerical regularizer passed to the solver.
    is_exact : int, default=0
        Solver option used by the exact SVM backend.
    KKTeps : float, default=1e-3
        Tolerance of the KKT stopping rule applied after each smoothing
        stage (``sum(KKT**2) / max(lambda, 1)**2 < KKTeps``). The squared
        KKT residual scales like ``1/n``, so the default is loose for large
        ``n`` and weak regularization; ``1e-6`` or smaller recovers the
        exact optimum at a modest cost in solver passes. See the model
        selection page of the user guide.
    delta_len : int, default=8
        Number of smoothing stages of the finite-smoothing SVM solver.
    kkt_scaled : bool, default=False
        Use the scale-aware KKT rule (``n * sum(KKT**2) < KKTeps``), under which
        ``KKTeps`` means the same relative accuracy at every ``n``.
    device : {"cpu", "cuda"} or torch.device, optional
        Device used for computation. If ``None``, CUDA is used when available;
        otherwise CPU is used. Requests for CUDA fall back to CPU when CUDA is
        unavailable.
    dtype : {"float64", "float32"}, default="float64"
        Precision of the kernel matrix, its eigendecomposition and the solution
        path. ``"float32"`` halves the memory of every ``n x n`` matrix (see
        :class:`torchkm.cvksvm.cvksvm`); with ``low_rank=True`` on CUDA it runs
        the kernel products fused.
    rbf_sigma : float, optional
        RBF kernel scale. If omitted, ``sigest`` estimates a scale from the
        training data.
    sigest_frac : float, default=0.5
        Fraction passed to ``sigest`` when estimating the RBF scale.
    poly_degree, poly_coef0, poly_gamma : int or float
        Polynomial-kernel parameters.
    probability : bool, default=False
        If ``True``, fit a Platt scaler on the selected out-of-fold scores and
        enable ``predict_proba`` and ``platt_plot``.
    platt_device : {"cpu", "cuda"} or torch.device, optional
        Device used for Platt calibration. Defaults to the estimator device.
    random_state : int, optional
        Seed used for deterministic fold construction and for the truncated
        spectrum's random start. ``None`` draws folds from the global torch RNG.
    store_path : bool, default=False
        If ``True``, keep the full coefficient and out-of-fold prediction path.
    low_rank : bool, default=False
        Large-n mode for problems whose kernel matrix does not fit in memory
        (e.g. the whole covtype.binary, 464,809 rows). The exact RBF kernel
        model, fitted by the truncated-spectrum solver
        (:class:`torchkm.experimental.SpectralSVMPath`, as ``spectrum=
        "truncated"``), but the kernel is never stored: every product with it
        is recomputed from the training rows
        (:class:`torchkm.experimental.RBFKernelOperator`), fused into one GPU
        kernel on CUDA with ``dtype="float32"``. Memory grows like ``n`` times
        the columns of a block (``spectrum_block x (cv + 1)``) instead of
        ``n^2``; each product costs the kernel's arithmetic again, so a stored
        kernel is faster when it fits. ``spectrum_rank``, ``gap_tol`` and
        ``spectrum_block`` apply, ``spectrum`` is ignored, and ``max_iter`` is
        the iteration budget of each lambda (``fit_cap``): fits that reach it
        are kept and reported as not converged (``converged_``). Needs
        ``kernel="rbf"``.
    spectrum : {"full", "truncated"}, default="full"
        The exact-mode solver. ``"full"`` eigendecomposes the kernel matrix
        (:class:`torchkm.cvksvm.cvksvm`). ``"truncated"`` (experimental) keeps
        the exact kernel but only its top ``spectrum_rank`` eigenpairs, and
        stops every lambda and fold at the certified relative duality gap
        ``gap_tol`` (:class:`torchkm.experimental.SpectralSVMPath`). Its peak
        memory is about 1.2 ``n x n`` matrices instead of 5, and it is not
        bound by the GPU eigensolver's size limit. ``tol``, ``max_iter``,
        ``KKTeps``, ``delta_len`` and ``kkt_scaled`` do not apply to it.
    spectrum_rank : int, default=400
        Eigenpairs kept by ``spectrum="truncated"``.
    gap_tol : float, default=1e-3
        Certified relative duality gap at which ``spectrum="truncated"`` stops
        each lambda and each fold.
    spectrum_block : int, default=10
        Lambdas that ``spectrum="truncated"`` fits together with all their
        folds, sharing each product with the kernel matrix (the ``block``
        option of :class:`torchkm.experimental.SpectralSVMPath`); 1 fits the
        path serially. Without cross-validation folds the path is serial.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Original binary class labels, ordered as negative then positive.
    best_C_ : float
        Regularization value selected by cross-validation.
    best_ind_ : int
        Index of the selected value in the candidate path.
    cv_mis_ : ndarray of shape (nC,)
        Cross-validation misclassification scores for the candidate path.
    alpha_ : ndarray
        Coefficients for the selected model.
    intercept_ : float
        Intercept for the selected model.
    foldid_ : ndarray
        Fold assignment used during fitting.
    n_features_in_ : int
        Number of input features seen during fitting.
    n_samples_fit_ : int
        Number of training samples.
    kernel_state_ : dict
        Kernel parameters needed for prediction, such as the fitted RBF scale.
    converged_ : ndarray of bool or None
        Per candidate value, whether the whole-data fit and every fold fit
        converged (for ``spectrum="truncated"`` and ``low_rank=True``: reached
        the certified gap ``gap_tol``).
    duality_gaps_, fold_duality_gaps_ : ndarray or None
        ``spectrum="truncated"`` and ``low_rank=True``: the certified relative
        duality gap of each whole-data fit (``nC``) and fold fit (``cv x nC``);
        ``None`` otherwise.
    peak_gpu_memory_bytes_ : int or None
        Peak CUDA memory (bytes, as tracked by the PyTorch allocator) used by
        the whole ``fit`` call: kernel construction, the solver, and Platt
        calibration. ``None`` when the fit ran on CPU. Exact mode scales as
        ``n^2``; see :mod:`torchkm.memory` and the "Operating envelope" page.
    fit_timing_ : dict
        Wall-clock seconds of the last ``fit`` by phase: ``kernel`` (building
        the kernel matrix), in exact mode ``eigendecomposition``,
        ``factorization_error`` (the check of its rounding error), ``path``
        (the whole-data fits along the lambda grid) and ``cross_validation``
        (the fold fits), and ``total`` (the whole call).
    n_passes_ : dict or None
        Solver iterations of the last ``fit``, ``path`` and
        ``cross_validation`` (one per fold and lambda iteration); each costs a
        few matrix-vector products with the ``n x n`` kernel.
    fit_profile_ : dict or None
        Exact mode, per lambda of the path (in path order): ``lambdas``,
        ``path_seconds`` and ``cv_seconds`` (the whole-data fit and the fold
        fits, which run together), ``path_passes``, and ``cv_passes``, the
        iterations of each fold (a ``cv x nC`` nested list).

    Notes
    -----
    The high-level wrapper accepts any two distinct class labels and maps them
    internally to the ``{-1, +1}`` convention used by the low-level solvers.
    Predictions are mapped back to the original labels.

    The methods ``decision_function`` and ``predict`` are available after
    fitting. ``predict_proba`` and ``platt_plot`` require
    ``probability=True`` at construction time.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from sklearn.datasets import make_circles
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler
    >>> from torchkm.estimators import TorchKMSVC
    >>> X, y = make_circles(n_samples=120, factor=0.4, noise=0.08,
    ...                     random_state=0)
    >>> X = StandardScaler().fit_transform(X)
    >>> Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
    ...                                       random_state=0)
    >>> Cs = np.logspace(2, -2, num=4)
    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> clf = TorchKMSVC(kernel="rbf", Cs=Cs, nC=len(Cs), cv=5,
    ...                  device=device, max_iter=40)
    >>> clf.fit(Xtr, ytr)
    TorchKMSVC(...)
    >>> clf.best_C_ > 0
    True
    >>> clf.predict(Xte[:3]).shape
    (3,)
    """

    _BACKEND: BackendName = "svm"

    def __init__(
        self,
        kernel: KernelName = "rbf",
        nC: int = 50,
        Cs: Optional[Any] = None,
        C_max: float = 1e3,
        C_min: float = 1e-3,
        cv: int = 5,
        foldid: Optional[Any] = None,
        tol: float = 1e-5,
        max_iter: int = 1000,
        solver_gamma: float = 1e-8,
        is_exact: int = 0,  # only used by cvksvm/cvkdwd
        KKTeps: float = 1e-3,
        delta_len: int = 8,  # only used by cvksvm
        kkt_scaled: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: str = "float64",  # only used by cvksvm (exact mode)
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
        # truncated spectrum (SVM only)
        spectrum: str = "full",
        spectrum_rank: int = 400,
        gap_tol: float = 1e-3,
        spectrum_block: int = 10,
        low_rank: bool = False,
    ):
        super().__init__(
            kernel=kernel,
            nC=nC,
            Cs=Cs,
            C_max=C_max,
            C_min=C_min,
            cv=cv,
            foldid=foldid,
            tol=tol,
            max_iter=max_iter,
            solver_gamma=solver_gamma,
            is_exact=is_exact,
            KKTeps=KKTeps,
            delta_len=delta_len,
            kkt_scaled=kkt_scaled,
            device=device,
            dtype=dtype,
            rbf_sigma=rbf_sigma,
            sigest_frac=sigest_frac,
            poly_degree=poly_degree,
            poly_coef0=poly_coef0,
            poly_gamma=poly_gamma,
            probability=probability,
            platt_device=platt_device,
            random_state=random_state,
            store_path=store_path,
            spectrum=spectrum,
            spectrum_rank=spectrum_rank,
            gap_tol=gap_tol,
            spectrum_block=spectrum_block,
        )
        self.low_rank = low_rank


class TorchKMDWD(_TorchKMBaseBinaryClassifier):
    """Kernel distance-weighted discrimination classifier.

    ``TorchKMDWD`` uses the same scikit-learn-style interface and model
    selection machinery as ``TorchKMSVC``, but delegates fitting to
    :class:`torchkm.cvkdwd.cvkdwd`. It accepts binary labels, maps them to the
    solver's ``{-1, +1}`` convention internally, and returns predictions in the
    original label space.

    Parameters are inherited from the shared binary-classifier wrapper. The
    most common options are ``kernel``, ``Cs``/``nC``, ``cv``, ``device``,
    and ``probability``.

    Attributes include ``best_C_``, ``cv_mis_``, ``alpha_``, ``intercept_``,
    ``classes_``, and ``foldid_`` after fitting. ``predict_proba`` and
    ``platt_plot`` are available only when ``probability=True``.
    """

    _BACKEND: BackendName = "dwd"


class TorchKMLogit(_TorchKMBaseBinaryClassifier):
    """Kernel logistic-regression classifier.

    ``TorchKMLogit`` wraps :class:`torchkm.cvklogit.cvklogit` with the same
    estimator interface used by the other TorchKM binary classifiers. It fits
    a path over candidate ``C`` values, chooses ``best_C_`` by cross-validation,
    and supports CPU or CUDA execution through the ``device`` parameter.

    The estimator accepts any two distinct class labels and maps them
    internally to the low-level solver convention. Use ``decision_function`` for
    fitted scores and ``predict`` for class labels. Set ``probability=True`` to
    fit Platt calibration and enable ``predict_proba``.
    """

    _BACKEND: BackendName = "logit"


class _TorchKMBaseKernelQuantileRegressor(BaseEstimator, RegressorMixin):
    """Shared implementation for the public kernel quantile regressor."""

    def __init__(
        self,
        kernel: KernelName = "rbf",
        nC: int = 50,
        Cs: Optional[Any] = None,
        C_max: float = 1e3,
        C_min: float = 1e-3,
        cv: int = 5,
        foldid: Optional[Any] = None,
        tau: float = 0.5,
        tol: float = 1e-5,
        max_iter: int = 1000,
        solver_gamma: float = 1e-8,
        is_exact: int = 0,
        delta_len: int = 4,
        mproj: int = 2,
        KKTeps: float = 1e-3,
        KKTeps2: float = 1e-3,
        kkt_scaled: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        rbf_sigma: Optional[float] = None,
        sigest_frac: float = 0.5,
        poly_degree: int = 3,
        poly_coef0: float = 1.0,
        poly_gamma: float = 1.0,
        random_state: Optional[int] = None,
        store_path: bool = False,
        gap_tol: float = 1e-3,
        max_tighten: int = 0,
    ):
        self.kernel = kernel
        self.nC = nC
        self.Cs = Cs
        self.C_max = C_max
        self.C_min = C_min
        self.cv = cv
        self.foldid = foldid
        self.tau = tau
        self.gap_tol = gap_tol
        self.max_tighten = max_tighten
        self.tol = tol
        self.max_iter = max_iter
        self.solver_gamma = solver_gamma
        self.is_exact = is_exact
        self.delta_len = delta_len
        self.mproj = mproj
        self.KKTeps = KKTeps
        self.KKTeps2 = KKTeps2
        self.kkt_scaled = kkt_scaled
        self.device = device
        self.rbf_sigma = rbf_sigma
        self.sigest_frac = sigest_frac
        self.poly_degree = poly_degree
        self.poly_coef0 = poly_coef0
        self.poly_gamma = poly_gamma
        self.random_state = random_state
        self.store_path = store_path

    def _clear_fit_state(self) -> None:
        fitted_attrs = (
            "n_features_in_",
            "n_samples_fit_",
            "tau_",
            "_device_str_",
            "foldid_",
            "X_fit_",
            "kernel_state_",
            "intercept_",
            "alpha_",
            "best_ind_",
            "best_C_",
            "cv_loss_",
            "alpmat_path_",
            "pred_path_",
            "peak_gpu_memory_bytes_",
        )
        for attr in fitted_attrs:
            if hasattr(self, attr):
                delattr(self, attr)

    def _compute_K_train(
        self, X_t: torch.Tensor, sigma: Optional[float] = None
    ) -> Tuple[torch.Tensor, dict]:
        if self.kernel == "rbf":
            if sigma is None:
                sigma = self.rbf_sigma
            if sigma is None:
                sigma = float(sigest(X_t, frac=float(self.sigest_frac)))
            return rbf_kernel_train(X_t, sigma), {"sigma": sigma}
        if self.kernel == "linear":
            return X_t @ X_t.T, {}
        if self.kernel == "poly":
            K = (self.poly_gamma * (X_t @ X_t.T) + self.poly_coef0) ** self.poly_degree
            return K, {}
        raise ValueError(f"Unsupported kernel={self.kernel} for non-precomputed mode.")

    def _compute_K_test(
        self, X_test_t: torch.Tensor, X_train_t: torch.Tensor, kernel_state: dict
    ) -> torch.Tensor:
        if self.kernel == "rbf":
            return kernelMult(X_test_t, X_train_t, float(kernel_state["sigma"]))
        if self.kernel == "linear":
            return X_test_t @ X_train_t.T
        if self.kernel == "poly":
            return (
                self.poly_gamma * (X_test_t @ X_train_t.T) + self.poly_coef0
            ) ** self.poly_degree
        raise ValueError(f"Unsupported kernel={self.kernel} for non-precomputed mode.")

    def _make_backend(
        self,
        *,
        K_train: Optional[torch.Tensor],
        X_train_t: torch.Tensor,
        y_backend: torch.Tensor,
        nlam: int,
        ulam_backend: torch.Tensor,
        foldid_backend: torch.Tensor,
        device: str,
        rebuild_kmat=None,
    ):
        return cvkqr(
            Kmat=K_train,
            y=y_backend,
            nlam=nlam,
            ulam=ulam_backend,
            tau=float(self.tau),
            foldid=foldid_backend,
            nfolds=int(self.cv),
            eps=float(self.tol),
            maxit=int(self.max_iter),
            gamma=float(self.solver_gamma),
            is_exact=int(self.is_exact),
            delta_len=int(self.delta_len),
            mproj=int(self.mproj),
            KKTeps=float(self.KKTeps),
            KKTeps2=float(self.KKTeps2),
            kkt_scaled=bool(self.kkt_scaled),
            device=device,
            rebuild_kmat=rebuild_kmat,
            gap_tol=float(self.gap_tol),
            max_tighten=int(self.max_tighten),
        )

    def fit(self, X: Any, y: Any):
        try:
            return self._fit_impl(X, y)
        except torch.cuda.OutOfMemoryError as err:
            n = int(_as_numpy(X).shape[0])
            raise torch.cuda.OutOfMemoryError(
                exact_mode_oom_message(n, getattr(self, "_device_str_", "cuda"))
            ) from err

    def _fit_impl(self, X: Any, y: Any):
        self._clear_fit_state()

        tau = float(self.tau)
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must be in (0, 1).")

        X_np, y_np = check_X_y(
            _as_numpy(X),
            _as_numpy(y),
            accept_sparse=False,
            ensure_2d=True,
            y_numeric=True,
        )
        y_np = np.asarray(y_np, dtype=np.float64).reshape(-1)
        self.n_features_in_ = X_np.shape[1]
        self.n_samples_fit_ = int(X_np.shape[0])
        self.tau_ = tau

        dev = _pick_device_str(self.device)
        self._device_str_ = dev
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)

        uC_t = _make_ulam(self.nC, self.Cs, self.C_max, self.C_min)
        ulam_t = 1.0 / (2 * X_np.shape[0] * uC_t)
        nlam = int(ulam_t.numel())

        foldid_t = _make_foldid(
            n=X_np.shape[0],
            nfolds=self.cv,
            foldid=self.foldid,
            random_state=self.random_state,
        )
        self.foldid_ = foldid_t.detach().cpu().to(torch.int64).numpy()

        X_train_t = torch.as_tensor(X_np, dtype=torch.double)
        y_train_t = torch.as_tensor(y_np, dtype=torch.double)

        ulam_backend = ulam_t.to(dev)
        foldid_backend = foldid_t.to(dev)
        y_backend = y_train_t.to(dev)

        rebuild_kmat = None  # see the classifier path
        if self.kernel == "precomputed":
            K_train = torch.as_tensor(X_np, dtype=torch.double)
            if K_train.ndim != 2 or K_train.shape[0] != K_train.shape[1]:
                raise ValueError(
                    "For kernel='precomputed', X must be a square (n,n) kernel matrix."
                )
            self.X_fit_ = None
            self.kernel_state_ = {}
        else:
            # Build the kernel on the target device (see the classifier path).
            X_dev = X_train_t.to(dev)
            K_train, kernel_state = self._compute_K_train(X_dev)
            self.X_fit_ = X_np
            self.kernel_state_ = kernel_state

            def rebuild():  # the same kernel again, with the fitted bandwidth
                return self._compute_K_train(X_dev, kernel_state.get("sigma"))[0]

            rebuild_kmat = rebuild

        if K_train is not None:
            K_train = K_train.to(dev)

        backend = self._make_backend(
            K_train=K_train,
            X_train_t=X_train_t,
            y_backend=y_backend,
            nlam=nlam,
            ulam_backend=ulam_backend,
            foldid_backend=foldid_backend,
            device=dev,
            rebuild_kmat=rebuild_kmat,
        )
        backend.fit()

        cv_loss_t = backend.cv(backend.pred, y_train_t.to(backend.pred.device))
        cv_loss = cv_loss_t.detach().cpu().numpy()
        best_ind = int(np.nanargmin(cv_loss))

        alpvec = backend.alpmat[:, best_ind].detach().cpu().to(torch.double)
        self.intercept_ = float(alpvec[0].item())
        self.alpha_ = alpvec[1:].numpy()
        self.best_ind_ = best_ind
        self.best_C_ = float(
            1.0 / (2.0 * backend.ulam[best_ind].detach().cpu().item() * X_np.shape[0])
        )
        self.cv_loss_ = cv_loss

        if self.store_path:
            self.alpmat_path_ = backend.alpmat.detach().cpu()
            self.pred_path_ = backend.pred.detach().cpu()
        else:
            self.alpmat_path_ = None
            self.pred_path_ = None

        self.peak_gpu_memory_bytes_ = (
            int(torch.cuda.max_memory_allocated(dev)) if dev == "cuda" else None
        )

        del backend
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["alpha_", "intercept_"])
        X_np = check_array(_as_numpy(X), accept_sparse=False, ensure_2d=True)
        dev = getattr(self, "_device_str_", "cpu")

        alpha_t = torch.as_tensor(self.alpha_, dtype=torch.double, device=dev)
        b = float(self.intercept_)

        if self.kernel == "precomputed":
            K_test = torch.as_tensor(X_np, dtype=torch.double, device=dev)
            if K_test.ndim != 2 or K_test.shape[1] != self.n_samples_fit_:
                raise ValueError(
                    f"For kernel='precomputed', X must have shape (n_test, {self.n_samples_fit_})."
                )
            with torch.no_grad():
                scores = torch.mv(K_test, alpha_t) + b
            return scores.detach().cpu().numpy()

        # the test kernel is built on the fit's device (the GPU when there is one)
        X_train_t = torch.as_tensor(self.X_fit_, dtype=torch.double, device=dev)
        X_test_t = torch.as_tensor(X_np, dtype=torch.double, device=dev)
        with torch.no_grad():
            scores = (
                _kernel_times(
                    self._compute_K_test,
                    X_test_t,
                    X_train_t,
                    self.kernel_state_,
                    alpha_t,
                )
                + b
            )
        return scores.detach().cpu().numpy()

    def score(self, X: Any, y: Any) -> float:
        y_true = np.asarray(_as_numpy(y), dtype=np.float64).reshape(-1)
        y_pred = self.predict(X).reshape(-1)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y have incompatible lengths.")

        residual = y_true - y_pred
        loss = np.where(
            residual >= 0,
            float(self.tau_) * residual,
            (float(self.tau_) - 1.0) * residual,
        )
        return -float(np.mean(loss))


class TorchKMKQR(_TorchKMBaseKernelQuantileRegressor):
    """Kernel quantile regressor with integrated model selection.

    ``TorchKMKQR`` uses :class:`torchkm.cvkqr.cvkqr` (exact mode).

    With ``is_exact=0`` (the default), each lambda and
    each fold stops early once its certified relative duality gap is at most
    ``gap_tol``; fits that end above it are reported in one
    ``ConvergenceWarning`` (``cvkqr``'s ``gaps`` and ``fold_gaps``).
    ``max_tighten`` lets the solver re-solve with a smaller ``eps`` to reach
    ``gap_tol`` (slower; see :class:`torchkm.cvkqr.cvkqr`).
    """
