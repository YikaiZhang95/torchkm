from __future__ import annotations

import torch

from .cvkqr import cvkqr
from .functions import kernelMult, rbf_kernel, sigest


class cvknyqr:
    """Nyström backend for kernel quantile regression.

    This backend constructs a Nyström approximation to the RBF kernel and then
    delegates the quantile-regression optimization to ``cvkqr`` using the
    approximate training kernel.

    The high-level estimator calls this backend when
    ``TorchKMKQR(low_rank=True)``. There is intentionally no separate
    high-level ``TorchKMNysKQR`` estimator.
    """

    def __init__(
        self,
        Xmat,
        X_test=None,
        y=None,
        nlam=50,
        ulam=None,
        tau=0.5,
        foldid=None,
        nfolds=5,
        eps=1e-5,
        maxit=1000,
        gamma=1.0,
        is_exact=0,
        delta_len=4,
        mproj=2,
        KKTeps=1e-3,
        KKTeps2=1e-3,
        num_landmarks=2000,
        k=1000,
        sigma=None,
        random_state=None,
        device=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        if not isinstance(Xmat, torch.Tensor):
            raise TypeError("Xmat must be a torch.Tensor.")
        if y is None:
            raise ValueError("y is required.")
        if not isinstance(y, torch.Tensor):
            raise TypeError("y must be a torch.Tensor.")
        if ulam is None:
            raise ValueError("ulam is required.")
        if not isinstance(ulam, torch.Tensor):
            raise TypeError("ulam must be a torch.Tensor.")

        tau = float(tau)
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must be in (0, 1).")

        self.Xmat = Xmat.double().to(self.device)
        self.X_test = X_test
        self.y = y.double().to(self.device)
        self.nobs = int(self.Xmat.shape[0])

        if self.y.ndim != 1 or self.y.shape[0] != self.nobs:
            raise ValueError("y must have shape (n_samples,).")

        self.nlam = int(nlam)
        self.ulam = ulam.double().to(self.device)
        self.tau = tau
        self.foldid = foldid
        self.nfolds = int(nfolds)
        self.eps = float(eps)
        self.maxit = int(maxit)
        self.gamma = float(gamma)
        self.is_exact = int(is_exact)
        self.delta_len = int(delta_len)
        self.mproj = int(mproj)
        self.KKTeps = float(KKTeps)
        self.KKTeps2 = float(KKTeps2)
        self.num_landmarks = int(num_landmarks)
        self.k = int(k)
        self.sigma = sigma
        self.random_state = random_state

        self.indices = None
        self.landmarks_ = None
        self.sig_w_ = None
        self.M_ = None
        self.k_eff_ = None
        self.Z_train_ = None
        self.K_approx_ = None
        self._exact_backend = None

        self.alpmat = torch.zeros(
            (self.nobs + 1, self.nlam), dtype=torch.double, device=self.device
        )
        self.pred = torch.zeros(
            (self.nobs, self.nlam), dtype=torch.double, device=self.device
        )
        self.npass = torch.zeros(self.nlam, dtype=torch.int32, device=self.device)
        self.cvnpass = torch.zeros(self.nlam, dtype=torch.int32, device=self.device)
        self.anlam = 0
        self.jerr = 0

    def _make_foldid(self):
        if self.foldid is not None:
            if not isinstance(self.foldid, torch.Tensor):
                raise TypeError("foldid must be a torch.Tensor.")
            foldid = self.foldid.to(self.device).to(torch.int64)
            if foldid.numel() != self.nobs:
                raise ValueError("foldid must have length n_samples.")
            return foldid

        if self.nfolds == self.nobs:
            return torch.arange(1, self.nobs + 1, device=self.device, dtype=torch.int64)

        generator = torch.Generator(device="cpu")
        if self.random_state is not None:
            generator.manual_seed(int(self.random_state))

        perm = torch.randperm(self.nobs, generator=generator).to(self.device)
        return (perm % self.nfolds + 1).to(torch.int64)

    def _fit_nystrom_state(self):
        n = self.nobs
        m = min(max(1, int(self.num_landmarks)), n)
        k_eff = min(max(1, int(self.k)), m)

        generator = torch.Generator(device="cpu")
        if self.random_state is not None:
            generator.manual_seed(int(self.random_state))

        indices = torch.randperm(n, generator=generator)[:m].to(self.device)
        X_work = self.Xmat.float()
        landmarks = X_work[indices]

        sigma = self.sigma
        if sigma is None:
            sigma = float(sigest(landmarks))

        W = rbf_kernel(landmarks, sigma)
        evals, evecs = torch.linalg.eigh(W)

        evals = evals[-k_eff:].flip(0)
        evecs = evecs[:, -k_eff:].flip(1)

        eps = torch.finfo(evals.dtype).eps
        evals = evals.clamp_min(eps)

        M = evecs * torch.rsqrt(evals)
        C = kernelMult(X_work, landmarks, sigma)
        Z_train = torch.mm(C, M).double()

        K_approx = torch.mm(Z_train, Z_train.T)
        K_approx = 0.5 * (K_approx + K_approx.T)

        self.indices = indices.detach().cpu().to(torch.int64)
        self.landmarks_ = landmarks.detach()
        self.sig_w_ = float(sigma)
        self.M_ = M.detach()
        self.k_eff_ = int(k_eff)
        self.Z_train_ = Z_train.detach()
        self.K_approx_ = K_approx.detach()

        return K_approx

    def fit(self):
        foldid = self._make_foldid()
        self.foldid = foldid

        K_approx = self._fit_nystrom_state()

        backend = cvkqr(
            Kmat=K_approx,
            y=self.y,
            nlam=self.nlam,
            ulam=self.ulam,
            tau=self.tau,
            foldid=foldid,
            nfolds=self.nfolds,
            eps=self.eps,
            maxit=self.maxit,
            gamma=self.gamma,
            is_exact=self.is_exact,
            delta_len=self.delta_len,
            mproj=self.mproj,
            KKTeps=self.KKTeps,
            KKTeps2=self.KKTeps2,
            device=self.device,
        )
        backend.fit()

        self._exact_backend = backend
        self.alpmat = backend.alpmat
        self.pred = backend.pred
        self.npass = backend.npass
        self.cvnpass = backend.cvnpass
        self.anlam = getattr(backend, "anlam", 0)
        self.jerr = getattr(backend, "jerr", 0)
        self.ulam = backend.ulam

        return self

    def transform(self, X_new):
        """Transform raw features into the fitted Nyström feature space."""
        if self.landmarks_ is None or self.M_ is None or self.sig_w_ is None:
            raise RuntimeError("Call fit() before transform().")

        if not isinstance(X_new, torch.Tensor):
            raise TypeError("X_new must be a torch.Tensor.")

        X_new = X_new.float().to(self.device)
        C_new = kernelMult(X_new, self.landmarks_, self.sig_w_)
        return torch.mm(C_new, self.M_).double()

    def approx_kernel_to_train(self, X_new):
        """Approximate K(X_new, X_train) using the fitted Nyström map."""
        if self.Z_train_ is None:
            raise RuntimeError("Call fit() before approx_kernel_to_train().")
        Z_new = self.transform(X_new)
        return torch.mm(Z_new, self.Z_train_.T)

    def cv(self, pred, y):
        if self._exact_backend is not None:
            return self._exact_backend.cv(pred, y.to(self.device))

        y_expanded = y.to(self.device)[:, None]
        residuals = y_expanded - pred
        return self.check_loss(residuals, self.tau).mean(dim=0)

    @staticmethod
    def check_loss(u, tau):
        return torch.where(u >= 0, tau * u, (tau - 1.0) * u)

    def predict(self, X_new, alp_b):
        """Predict from raw features using fitted state and coefficients."""
        if alp_b.ndim != 1:
            raise ValueError("alp_b must be a one-dimensional tensor.")

        K_new = self.approx_kernel_to_train(X_new)
        return torch.mv(K_new, alp_b[1:].to(self.device)) + alp_b[0].to(self.device)
