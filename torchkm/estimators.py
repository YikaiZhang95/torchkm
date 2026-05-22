from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import torch

from .functions import sigest, rbf_kernel as rbf_kernel_train, kernelMult

from .cvksvm import cvksvm
from .cvkdwd import cvkdwd
from .cvklogit import cvklogit
from .cvkqr import cvkqr
from .platt import PlattScalerTorch
from .cvknyssvm import cvknyssvm
from .cvknysdwd import cvknysdwd
from .cvknyslogit import cvknyslogit
from .cvknysqr import cvknysqr

# ---- sklearn is OPTIONAL: raise a clean error only when wrapper is imported ----
try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
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
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, torch.device):
        return "cuda" if device.type == "cuda" else "cpu"
    dev = str(device).lower()
    return "cuda" if dev.startswith("cuda") and torch.cuda.is_available() else "cpu"


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
            f"Only binary classification is supported. Got classes={classes}. For multiclass case, the problem can be addressed using either a one-vs-one or a one-vs-rest strategy."
        )
    neg_label, pos_label = classes[0], classes[1]
    y_pm1 = np.where(y == pos_label, 1.0, -1.0).astype(np.float64)
    return y_pm1, neg_label, pos_label


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
        # Nystrom
        low_rank: bool = False,
        num_landmarks: int = 2000,
        nys_k: int = 1000,
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
        X_np, y_np = check_X_y(
            _as_numpy(X), _as_numpy(y), accept_sparse=False, ensure_2d=True
        )
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

        if self.low_rank:
            self.X_fit_ = X_np
            self.kernel_state_ = {"low_rank": True}
            K_train = None
        else:
            if self.kernel == "precomputed":
                K_train = torch.as_tensor(X_np, dtype=torch.double)
                if K_train.ndim != 2 or K_train.shape[0] != K_train.shape[1]:
                    raise ValueError(
                        "For kernel='precomputed', X must be a square (n,n) kernel matrix."
                    )
                self.X_fit_ = None
                self.kernel_state_ = {}
            else:
                K_train, kernel_state = self._compute_K_train(X_train_t)
                self.X_fit_ = X_np
                self.kernel_state_ = kernel_state

            K_train = K_train.to(dev)

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
            X_test_t = torch.as_tensor(X_np, dtype=torch.double)  # CPU
            K_test = self._compute_K_test(X_test_t, X_train_t, self.kernel_state_).to(
                dev
            )

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
    ):
        if low_rank:
            backend_cls = {
                "svm": cvknyssvm,
                "dwd": cvknysdwd,
                "logit": cvknyslogit,
            }[self._BACKEND]

            kwargs = dict(
                Xmat=X_train_t,
                X_test=X_train_t,  # placeholder unless your nys classes make X_test optional
                y=y_backend,
                nlam=nlam,
                ulam=ulam_backend,
                foldid=foldid_backend,
                nfolds=int(self.cv),
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
                nfolds=int(self.cv),
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
                nfolds=int(self.cv),
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
                nfolds=int(self.cv),
                eps=float(self.tol),
                maxit=int(self.max_iter),
                gamma=float(self.solver_gamma),
                device=dev,
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
    device : {"cpu", "cuda"} or torch.device, optional
        Device used for computation. If ``None``, CUDA is used when available;
        otherwise CPU is used. Requests for CUDA fall back to CPU when CUDA is
        unavailable.
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
        Seed used for deterministic fold construction.
    store_path : bool, default=False
        If ``True``, keep the full coefficient and out-of-fold prediction path.
    low_rank : bool, default=False
        If ``True``, use the Nyström SVM backend. The low-rank path currently
        supports raw-feature RBF-kernel workflows, not ``kernel="precomputed"``.
    num_landmarks : int, default=2000
        Number of Nyström landmarks when ``low_rank=True``.
    nys_k : int, default=1000
        Rank used by the Nyström feature map when ``low_rank=True``.

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
    low_rank_basis_dim_ : int
        Effective low-rank feature dimension when ``low_rank=True``.
    low_rank_landmark_indices_ : ndarray
        Landmark indices when exposed by the Nyström backend.
    num_landmarks_ : int
        Number of landmarks used by the fitted Nyström backend, when available.
    nys_k_ : int
        Effective Nyström rank, when available.

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


class TorchKMDWD(_TorchKMBaseBinaryClassifier):
    """Kernel distance-weighted discrimination classifier.

    ``TorchKMDWD`` uses the same scikit-learn-style interface and model
    selection machinery as ``TorchKMSVC``, but delegates fitting to
    :class:`torchkm.cvkdwd.cvkdwd`. It accepts binary labels, maps them to the
    solver's ``{-1, +1}`` convention internally, and returns predictions in the
    original label space.

    Parameters are inherited from the shared binary-classifier wrapper. The
    most common options are ``kernel``, ``Cs``/``nC``, ``cv``, ``device``,
    ``probability``, ``low_rank``, ``num_landmarks``, and ``nys_k``.

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


class TorchKMKQR(BaseEstimator, RegressorMixin):
    """Kernel quantile regressor with integrated model selection.

    ``TorchKMKQR`` delegates fitting to :class:`torchkm.cvkqr.cvkqr`. It fits
    kernel quantile regression for a continuous response, selects ``best_C_``
    by cross-validation check loss, and exposes predictions through
    ``predict``.

    Parameters
    ----------
    kernel : {"rbf", "linear", "poly", "precomputed"}, default="rbf"
        Kernel used by the estimator. ``"precomputed"`` expects a square
        training kernel matrix in ``fit`` and a test-by-train kernel matrix in
        ``predict``.
    nC : int, default=50
        Number of candidate ``C`` values when ``Cs`` is not provided.
    Cs : array-like, optional
        Candidate regularization values under the scikit-learn/LIBSVM
        ``C`` convention.
    cv : int, default=5
        Number of cross-validation folds used to choose ``best_C_``.
    tau : float, default=0.5
        Quantile level in ``(0, 1)``.
    device : {"cpu", "cuda"} or torch.device, optional
        Device used for computation. If ``None``, CUDA is used when available;
        otherwise CPU is used.

    Attributes
    ----------
    best_C_ : float
        Regularization value selected by cross-validation.
    best_ind_ : int
        Index of the selected value in the candidate path.
    cv_loss_ : ndarray of shape (nC,)
        Cross-validation check-loss scores for the candidate path.
    alpha_ : ndarray
        Coefficients for the selected model.
    intercept_ : float
        Intercept for the selected model.
    foldid_ : ndarray
        Fold assignment used during fitting.
    n_features_in_ : int
        Number of input features seen during fitting.

    Notes
    -----
    ``TorchKMKQR`` uses continuous regression targets; there is no class-label
    remapping. The lower-level solver uses check loss for cross-validation.
    """

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
        device: Optional[Union[str, torch.device]] = None,
        rbf_sigma: Optional[float] = None,
        sigest_frac: float = 0.5,
        poly_degree: int = 3,
        poly_coef0: float = 1.0,
        poly_gamma: float = 1.0,
        random_state: Optional[int] = None,
        store_path: bool = False,
    ):
        self.kernel = kernel
        self.nC = nC
        self.Cs = Cs
        self.C_max = C_max
        self.C_min = C_min
        self.cv = cv
        self.foldid = foldid
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.solver_gamma = solver_gamma
        self.is_exact = is_exact
        self.delta_len = delta_len
        self.mproj = mproj
        self.KKTeps = KKTeps
        self.KKTeps2 = KKTeps2
        self.device = device
        self.rbf_sigma = rbf_sigma
        self.sigest_frac = sigest_frac
        self.poly_degree = poly_degree
        self.poly_coef0 = poly_coef0
        self.poly_gamma = poly_gamma
        self.random_state = random_state
        self.store_path = store_path

    def _compute_K_train(self, X_t: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if self.kernel == "rbf":
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

    def fit(self, X: Any, y: Any):
        X_np, y_np = check_X_y(
            _as_numpy(X), _as_numpy(y), accept_sparse=False, ensure_2d=True, y_numeric=True
        )
        self.n_features_in_ = X_np.shape[1]

        dev = _pick_device_str(self.device)
        self._device_str_ = dev

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

        if self.kernel == "precomputed":
            K_train = torch.as_tensor(X_np, dtype=torch.double)
            if K_train.ndim != 2 or K_train.shape[0] != K_train.shape[1]:
                raise ValueError(
                    "For kernel='precomputed', X must be a square (n,n) kernel matrix."
                )
            self.X_fit_ = None
            self.kernel_state_ = {}
        else:
            K_train, kernel_state = self._compute_K_train(X_train_t)
            self.X_fit_ = X_np
            self.kernel_state_ = kernel_state
        K_train = K_train.to(dev)

        backend = cvkqr(
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
            device=dev,
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
        self.n_samples_fit_ = int(X_np.shape[0])

        if self.store_path:
            self.alpmat_path_ = backend.alpmat.detach().cpu()
            self.pred_path_ = backend.pred.detach().cpu()
        else:
            self.alpmat_path_ = None
            self.pred_path_ = None

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
        else:
            X_train_t = torch.as_tensor(self.X_fit_, dtype=torch.double)
            X_test_t = torch.as_tensor(X_np, dtype=torch.double)
            K_test = self._compute_K_test(X_test_t, X_train_t, self.kernel_state_).to(dev)

        with torch.no_grad():
            scores = torch.mv(K_test, alpha_t) + b
        return scores.detach().cpu().numpy()


class TorchKMNysKQR(BaseEstimator, RegressorMixin):
    """Nyström kernel quantile regressor.

    ``TorchKMNysKQR`` delegates fitting to :class:`torchkm.cvknysqr.cvknysqr`
    and uses a low-rank Nyström feature map instead of the full training kernel
    matrix.

    Notes
    -----
    This estimator expects raw feature input and does not support
    ``kernel="precomputed"``. It uses continuous regression targets and selects
    ``best_C_`` by cross-validation check loss.
    """

    def __init__(
        self,
        nC: int = 50,
        Cs: Optional[Any] = None,
        C_max: float = 1e3,
        C_min: float = 1e-3,
        cv: int = 5,
        foldid: Optional[Any] = None,
        tau: float = 0.5,
        tol: float = 1e-5,
        max_iter: int = 1000,
        solver_gamma: float = 1.0,
        is_exact: int = 0,
        delta_len: int = 4,
        mproj: int = 2,
        KKTeps: float = 1e-3,
        KKTeps2: float = 1e-3,
        num_landmarks: int = 2000,
        k: int = 1000,
        device: Optional[Union[str, torch.device]] = None,
        random_state: Optional[int] = None,
        store_path: bool = False,
    ):
        self.nC = nC
        self.Cs = Cs
        self.C_max = C_max
        self.C_min = C_min
        self.cv = cv
        self.foldid = foldid
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.solver_gamma = solver_gamma
        self.is_exact = is_exact
        self.delta_len = delta_len
        self.mproj = mproj
        self.KKTeps = KKTeps
        self.KKTeps2 = KKTeps2
        self.num_landmarks = num_landmarks
        self.k = k
        self.device = device
        self.random_state = random_state
        self.store_path = store_path

    def fit(self, X: Any, y: Any):
        X_np, y_np = check_X_y(
            _as_numpy(X), _as_numpy(y), accept_sparse=False, ensure_2d=True, y_numeric=True
        )
        self.n_features_in_ = X_np.shape[1]

        dev = _pick_device_str(self.device)
        self._device_str_ = dev

        uC_t = _make_ulam(self.nC, self.Cs, self.C_max, self.C_min)
        ulam_t = 1.0 / (2 * X_np.shape[0] * uC_t)
        nlam = int(ulam_t.numel())

        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))

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

        backend = cvknysqr(
            Xmat=X_train_t,
            X_test=X_train_t,
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
            num_landmarks=int(self.num_landmarks),
            k=int(self.k),
            device=dev,
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
        self.n_samples_fit_ = int(X_np.shape[0])

        self._backend_ = backend
        self.X_fit_ = X_np

        if self.store_path:
            self.alpmat_path_ = backend.alpmat.detach().cpu()
            self.pred_path_ = backend.pred.detach().cpu()
        else:
            self.alpmat_path_ = None
            self.pred_path_ = None

        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["alpha_", "intercept_", "_backend_"])
        X_np = check_array(_as_numpy(X), accept_sparse=False, ensure_2d=True)
        dev = getattr(self, "_device_str_", "cpu")

        alpha_t = torch.as_tensor(self.alpha_, dtype=torch.double, device=dev)
        b = float(self.intercept_)

        X_test_t = torch.as_tensor(X_np, dtype=torch.double)
        with torch.no_grad():
            Z_test = self._backend_.transform(X_test_t)
            scores = torch.mv(Z_test, alpha_t) + b
        return scores.detach().cpu().numpy()
