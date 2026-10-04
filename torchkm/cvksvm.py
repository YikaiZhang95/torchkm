# SPDX-License-Identifier: MIT
import math
import time
import warnings

import torch

from .exceptions import ConvergenceWarning
from .functions import *
from .functions import brent_minimize, brent_minimize_batch
from .memory import kernel_eigh


def _factorization_error(Kmat, Umat, eigens, iters=20):
    """Power-iteration estimate of the spectral norm of U diag(eigens) U^T - K."""
    gen = torch.Generator().manual_seed(0)
    v = torch.randn(Kmat.shape[0], generator=gen, dtype=Kmat.dtype).to(Kmat.device)
    v /= v.norm()
    norm = 0.0
    for _ in range(iters):
        w = torch.mv(Umat, eigens * torch.mv(Umat.T, v)) - torch.mv(Kmat, v)
        norm = float(w.norm())
        if norm == 0.0:
            break
        v = w / norm
    return norm


class cvksvm:
    """
    Kernel SVM with Regularization and Acceleration.

    This function initializes the optimization process for a kernel SVM model,
    supporting advanced features like GPU acceleration and iterative projection methods
    for large-scale data.

    Parameters
    ----------
    Kmat : ndarray or tensor
        The kernel matrix of shape (n_samples, n_samples).

    y : ndarray or tensor
        Target labels for each sample, of shape (n_samples,). Typically, -1 or 1.

    nlam : int
        The number of regularization parameters to consider in the optimization.

    ulam : ndarray or tensor
        User-specified regularization parameters, of shape (nlam,).

    foldid : ndarray, default=None
        Array indicating the fold assignment for cross-validation. Each element is an
        integer corresponding to a fold.

    nfolds : int, default=5
        The number of cross-validation folds to use.

    eps : float, default=1e-5
        Tolerance for convergence in the optimization.

    maxit : int, default=1000
        Maximum number of iterations allowed for the optimization process.

    gamma : float, default=1.0
        Regularization parameter for kernel methods, controlling the trade-off between
        margin width and misclassification.

    is_exact : int, default=0
        Indicates whether projection step is used (1 for exact, 0 for approximate).

    delta_len : int, default=8
        Length of delta vector used in projection steps.

    mproj : int, default=10
        Number of projection steps to perform for iterative optimization.

    KKTeps : float, default=1e-3
        Tolerance for KKT conditions in the primary optimization problem.

    KKTeps2 : float, default=1e-3
        Tolerance for KKT conditions in secondary checks.

    kkt_scaled : bool, default=False
        Scale-aware KKT stopping rule: compare ``n * sum(KKT**2)`` (the squared
        residual in units of its natural scale ``1/n``) with ``KKTeps`` instead of
        the absolute ``sum(KKT**2)``. With the default rule the threshold gets
        easier to meet as ``n`` grows; with ``kkt_scaled=True`` a given ``KKTeps``
        means the same relative accuracy at every ``n``.

    device : {'cuda', 'cpu'}, default='cuda'
        Device to perform computations on. Default is GPU ('cuda') for improved performance.

    dtype : {torch.float64, torch.float32}, default=torch.float64
        Working precision of the kernel matrix, its eigendecomposition and the
        solution path. ``torch.float32`` halves the memory of every ``n x n``
        matrix (the kernel, the eigenvectors and the eigendecomposition
        workspace), so exact mode fits about 1.4 times as many rows on the same
        device, and runs at the device's single-precision rate. In our checks
        cross-validation chose the same lambda as in float64 and the objectives
        agreed to about 1e-5 (median over the path). At weak regularization a
        smoothing round in float32 can end at the precision floor rather than
        at ``eps`` (see ``__init__``). ``is_exact=1`` needs ``torch.float64``.

    rebuild_kmat : callable, optional
        Returns ``Kmat`` again, with the same values. When given, the
        eigendecomposition overwrites ``Kmat``'s storage with the eigenvectors
        instead of factorizing a copy, which lowers the peak by one ``n x n``
        matrix (from 6 to 5 on the GPU), and ``Kmat`` is rebuilt with it
        afterwards. The fit is the same. The estimators pass their kernel
        construction here.

    Attributes
    ----------
    self.alpmat : ndarray or tensor
        Matrix of optimized alpha values after fitting the data, of shape (n_samples, nlam).

    self.npass : int
        Number of passes made over the data during the optimization.

    self.cvnpass : int
        Number of passes made during cross-validation.

    self.jerr : int
        Error flag to indicate any issues during computation (0 for success, non-zero for errors).

    self.pred : ndarray or tensor
        Predicted values based on the optimization, of shape (n_samples,).

    Notes
    -----
    This implementation is designed for large-scale data problems and leverages GPU
    acceleration for improved computational efficiency. Regularization is controlled
    through multiple hyperparameters, allowing fine-tuned trade-offs between accuracy
    and computational cost.

    Examples
    --------
    >>> from torchkm.cvksvm import cvksvm
    >>> from torchkm.functions import *
    >>> import torch
    >>> import numpy
    >>> nn = 1000 # Number of samples
    >>> nm = 5   # Number of clusters per class
    >>> pp = 10  # Number of features
    >>> p1 = p2 = pp // 2    # Number of positive/negative centers
    >>> mu = 2.0  # Mean shift
    >>> ro = 3  # Standard deviation for normal distribution
    >>> sdn = 42  # Seed for reproducibility

    >>> nlam = 50
    >>> torch.manual_seed(sdn)
    >>> ulam = torch.logspace(3, -3, steps=nlam)

    >>> X_train, y_train, means_train = data_gen(nn, nm, pp, p1, p2, mu, ro, sdn)
    >>> X_test, y_test, means_test = data_gen(nn // 10, nm, pp, p1, p2, mu, ro, sdn)
    >>> X_train = standardize(X_train)
    >>> X_test = standardize(X_test)

    >>> sig = sigest(X_train)
    >>> Kmat = rbf_kernel(X_train, sig)

    >>> torch.manual_seed(sdn)
    >>> nfolds = 10
    >>> if nfolds == nn:
    >>>     foldid = torch.arange(nn) # Each row gets its own fold ID
    >>> else:
    >>>     # Randomly assign fold IDs across the rows
    >>>     # foldid = torch.tensor(np.random.permutation(np.repeat(np.arange(1, nfolds + 1), nn // nfolds + 1)[:nn]))
    >>>     foldid = torch.randperm(nn) % nfolds + 1
    >>> model = cvksvm(Kmat=Kmat, y=y_train, nlam=nlam, ulam=ulam, nfolds=nfolds, eps=1e-5, maxit=100000, gamma=1e-8, is_exact=0, device='cuda')
    >>> model.fit()
    """

    def __init__(
        self,
        Kmat,
        y,
        nlam,
        ulam,
        foldid=None,
        nfolds=5,
        eps=1e-5,
        maxit=1000,
        gamma=1.0,
        is_exact=0,
        delta_len=8,
        mproj=10,
        KKTeps=1e-3,
        KKTeps2=1e-3,
        device=None,
        kkt_scaled=False,
        dtype=torch.float64,
        rebuild_kmat=None,
    ):
        self.rebuild_kmat = rebuild_kmat
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype not in (torch.float64, torch.float32):
            raise ValueError(
                f"dtype must be torch.float64 or torch.float32, got {dtype}"
            )
        if dtype == torch.float32 and is_exact != 0:
            raise ValueError("is_exact=1 needs dtype=torch.float64")
        self.dtype = dtype
        # In float32 the steps stop shrinking at a level set by the rounding of
        # K alpha, which at weak regularization can lie above eps; a smoothing
        # round then ends once its largest step has not reached a new low for
        # this many iterations. float64's floor is far below eps.
        self._stall_patience = 100 if dtype == torch.float32 else None

        # --- Check Kmat ---
        if not isinstance(Kmat, torch.Tensor):
            raise TypeError("Kmat must be a torch.Tensor")
        Kmat = Kmat.to(device=self.device, dtype=self.dtype)
        self.Kmat = Kmat
        self.nobs = Kmat.shape[0]

        if not isinstance(y, torch.Tensor):
            raise TypeError("y must be a torch.Tensor")
        y = y.to(device=self.device, dtype=self.dtype)

        # --- Label check ---
        unique_labels = torch.unique(y)
        if unique_labels.numel() > 2:
            raise ValueError(
                f"Multi-class detected: labels = {unique_labels.tolist()}. Only -1 and 1 allowed."
            )
        if not torch.all((unique_labels == -1) | (unique_labels == 1)):
            raise ValueError(
                f"Invalid labels: {unique_labels.tolist()}. Must be only -1 and 1."
            )
        self.y = y

        # --- Check ulam ---
        if not isinstance(ulam, torch.Tensor):
            raise TypeError("ulam must be a torch.Tensor")
        ulam = ulam.double().to(self.device)

        # --- Check foldid ---
        if foldid is not None:
            if not isinstance(foldid, torch.Tensor):
                raise TypeError("foldid must be a torch.Tensor")
            foldid = foldid.to(self.device)
        else:
            if nfolds == self.nobs:
                foldid = torch.arange(self.nobs)  # Each row gets its own fold ID
            else:
                # Randomly assign fold IDs across the rows
                # foldid = torch.tensor(np.random.permutation(np.repeat(np.arange(1, nfolds + 1), nn // nfolds + 1)[:nn]))
                foldid = torch.randperm(self.nobs) % nfolds + 1
            foldid = foldid.to(self.device)

        # --- Shape check ---
        if Kmat.shape[0] != Kmat.shape[1]:
            raise ValueError("Kmat must be a square matrix")
        if Kmat.shape[0] != y.shape[0]:
            raise ValueError("Kmat and y size mismatch")
        # self.Kmat = None
        # self.y = None

        self.nlam = nlam
        self.ulam = ulam.double()
        self.eps = eps
        self.maxit = maxit
        self.gamma = gamma
        self.is_exact = is_exact
        self.delta_len = delta_len
        self.mproj = mproj
        self.KKTeps = KKTeps
        self.KKTeps2 = KKTeps2
        self.kkt_scaled = bool(kkt_scaled)
        # Each KKT entry has natural scale 1/n; the scale-aware rule compares
        # the squared norm in units of (1/n)^2 so KKTeps means the same relative
        # accuracy at every n (see docs/user_guide/model_selection.md).
        self._kkt_scale = float(self.nobs) if self.kkt_scaled else 1.0
        self.nfolds = nfolds
        self.nmaxit = self.nlam * self.maxit
        self.foldid = foldid

        # Initialize outputs
        self.alpmat = torch.zeros((self.nobs + 1, self.nlam), dtype=self.dtype).to(
            self.device
        )
        self.anlam = 0
        self.npass = torch.zeros(self.nlam, dtype=torch.int32).to(self.device)
        self.converged = torch.zeros(self.nlam, dtype=torch.bool).to(self.device)
        self.cvnpass = torch.zeros(self.nlam, dtype=torch.int32).to(self.device)
        self.pred = torch.zeros((self.nobs, self.nlam), dtype=self.dtype).to(
            self.device
        )
        self.jerr = 0
        # seconds spent in each phase of the last fit, in total and per lambda,
        # and each fold's solver iterations per lambda (see ``fit``)
        self.timing = None
        self.lambda_timing = None
        self.fold_passes = None

    def fit(self):
        nobs = self.nobs
        nlam = self.nlam
        y = self.y
        Kmat = self.Kmat
        nfolds = self.nfolds

        r = torch.zeros(nobs, dtype=self.dtype).to(self.device)
        alpmat = torch.zeros((nobs + 1, nlam), dtype=self.dtype).to(self.device)
        # Iterations per lambda, counted on the host: reading a device counter
        # on every iteration would make the CPU wait for the GPU each time.
        npass = [0] * nlam
        cvnpass = [0] * nlam
        alpvec = torch.zeros(nobs + 1, dtype=self.dtype).to(self.device)
        pred = torch.zeros((self.nobs, self.nlam), dtype=self.dtype).to(self.device)
        converged = torch.zeros(nlam, dtype=torch.bool).to(self.device)
        jerr = 0
        eps2 = 1.0e-5
        one = torch.ones((), dtype=self.dtype, device=self.device)
        step_buf = torch.empty(nobs + 1, dtype=self.dtype, device=self.device)

        # Kinv = torch.linalg.inv(Kmat)

        # Wall-clock seconds per phase, device work included: the kernel's
        # eigendecomposition, the check of its rounding error, the whole-data
        # lambda path and the cross-validation fits.
        timing = dict.fromkeys(
            ("eigendecomposition", "factorization_error", "path", "cross_validation"),
            0.0,
        )
        # the path and the (batched) fold fits of each lambda, and each fold's
        # iterations per lambda
        lambda_timing = dict(path=[0.0] * nlam, cross_validation=[0.0] * nlam)
        fold_passes = torch.zeros((nfolds, nlam), dtype=torch.int64, device=self.device)
        t = self._now()
        # Given rebuild_kmat, the eigenvectors overwrite Kmat's storage (one
        # n x n copy less at the peak) and Kmat is rebuilt for the rest of the fit.
        eigens, Umat = kernel_eigh(Kmat, overwrite=self.rebuild_kmat is not None)
        if self.rebuild_kmat is not None:
            Kmat = self.rebuild_kmat().to(device=self.device, dtype=self.dtype)
            self.Kmat = Kmat
        timing["eigendecomposition"] = self._now() - t
        # K is positive semi-definite, so a negative eigenvalue is rounding error.
        eigens.clamp_min_(0.0)
        # The steps bound the loss curvature with U diag(eigens) U^T, which equals
        # K only up to the eigendecomposition's rounding error: about 1e-13 |K| in
        # float64 but 1e-6 |K| in float32. Once 4 n delta lambda is small, that
        # error lets the float32 bound fall a few percent below K and the
        # accelerated steps oscillate instead of converging; adding twice the
        # error to the eigenvalues keeps the bound above K.
        t = self._now()
        error = _factorization_error(Kmat, Umat, eigens)
        timing["factorization_error"] = self._now() - t
        eigens += self.gamma + 2.0 * error
        Usum = torch.sum(Umat, dim=0)
        einv = 1 / eigens
        # The regularised inverse K^{-1} = U diag(einv) U^T is applied on the
        # fly in the projection step below instead of materialising an
        # extra n x n matrix (see docs/user_guide/operating_envelope.md).

        vareps = 1.0e-8

        lpUsum = torch.zeros(
            (nobs, self.delta_len), dtype=self.dtype, device=self.device
        )
        lpinv = torch.zeros(
            (nobs, self.delta_len), dtype=self.dtype, device=self.device
        )
        svec = torch.zeros((nobs, self.delta_len), dtype=self.dtype, device=self.device)
        vvec = torch.zeros((nobs, self.delta_len), dtype=self.dtype, device=self.device)
        gval = torch.zeros((self.delta_len), dtype=self.dtype, device=self.device)

        for l in range(nlam):
            t = self._now()
            al = self.ulam[l].item()
            delta = 1.0
            delta_id = 0
            delta_save = 0
            oldalpvec = torch.zeros(nobs + 1, dtype=self.dtype).to(self.device)

            while delta_id < self.delta_len:
                delta_id += 1
                opdelta = 1.0 + delta
                omdelta = 1.0 - delta
                oddelta = 1.0 / delta

                if delta_id > delta_save:
                    lpinv[:, delta_id - 1] = 1.0 / (
                        eigens + 4.0 * float(nobs) * delta * al
                    )
                    lpUsum[:, delta_id - 1] = lpinv[:, delta_id - 1] * Usum
                    vvec[:, delta_id - 1] = torch.mv(
                        Umat, eigens * lpUsum[:, delta_id - 1]
                    )
                    svec[:, delta_id - 1] = torch.mv(Umat, lpUsum[:, delta_id - 1])
                    gval[delta_id - 1] = self._gval(
                        Usum, lpUsum[:, delta_id - 1], delta, al, nobs, vareps
                    )
                    delta_save = delta_id

                # Compute residual r
                told = one
                stall = [float("inf"), 0]
                ka = torch.mv(Kmat, alpvec[1:])
                r = y * (alpvec[0] + ka)
                # Update alpha
                # alpha loop
                for iteration in range(self.maxit):
                    zvec = torch.where(
                        r < omdelta,
                        -y,
                        torch.where(
                            r > opdelta,
                            torch.zeros(1, device=self.device),
                            0.5 * y * oddelta * (r - opdelta),
                        ),
                    )
                    gamvec = zvec + 2.0 * float(nobs) * al * alpvec[1:]  ##
                    hval = self._hval(
                        zvec,
                        alpvec,
                        svec[:, delta_id - 1],
                        vvec[:, delta_id - 1],
                        delta,
                        al,
                        nobs,
                        vareps,
                    )

                    tnew = 0.5 + 0.5 * torch.sqrt(one + 4.0 * told * told)
                    mul = 1.0 + (told - 1.0) / tnew
                    told = tnew

                    # Update step using Pinv
                    if delta_id > self.delta_len:
                        print("Exceeded maximum delta_id")
                        break

                    # Compute dif vector

                    step_buf[0] = -2.0 * mul * delta * gval[delta_id - 1] * hval
                    step_buf[1:] = -step_buf[0] * svec[
                        :, delta_id - 1
                    ] - 2.0 * mul * delta * torch.mv(
                        Umat, gamvec @ Umat * lpinv[:, delta_id - 1]
                    )
                    alpvec += step_buf

                    # Update residual
                    ka = torch.mv(Kmat, alpvec[1:])
                    r = y * (alpvec[0] + ka)
                    npass[l] += 1

                    # Check convergence
                    step2 = torch.max(step_buf**2)
                    if step2 < (self.eps * mul * mul) or self._stalled(step2, stall):
                        break

                    if sum(npass) > self.maxit:
                        jerr = -l - 1
                        break

                # Check KKT conditions
                dif_step = oldalpvec - alpvec
                ka = torch.mv(Kmat, alpvec[1:])
                aka = torch.dot(ka, alpvec[1:])
                obj_value = self.objfun(alpvec[0], aka, ka, y, al, nobs)
                # eps_float64 = np.finfo(np.float64).eps
                # optimal_intercept = minimize_scalar(self.objfun, args=(aka, ka, y, al, nobs), bracket=(-100.0, 100.0), method="brent")
                # obj_value_new = self.objfun(optimal_intercept.x, aka, ka, y, al, nobs)
                golden_s = self.golden_section_search(
                    -100.0, 100.0, nobs, ka, aka, y, al
                )
                int_new = golden_s[0]
                obj_value_new = golden_s[1]
                if obj_value_new < obj_value:
                    dif_step[0] = dif_step[0] + int_new - alpvec[0]
                    r = r + y * (int_new - alpvec[0])
                    alpvec[0] = int_new

                oldalpvec = alpvec.clone()

                zvec = torch.where(
                    r < 1.0,
                    -y,
                    torch.where(r > 1.0, torch.zeros(1).to(self.device), -0.5 * y),
                )
                KKT = zvec / float(nobs) + 2.0 * al * alpvec[1:]
                uo = max(al, 1.0)
                KKT_norm = self._kkt_scale * torch.sum(KKT**2) / (uo**2)
                if KKT_norm < self.KKTeps:
                    # Check convergence
                    dif_norm = torch.max(dif_step**2)
                    if dif_norm < float(nobs) * (self.eps * mul * mul):
                        if self.is_exact == 0:
                            converged[l] = True
                            break
                        else:
                            is_exit = False
                            alptmp = alpvec.clone()
                            for nn in range(self.mproj):
                                elbowid = torch.zeros(nobs, dtype=torch.bool)
                                elbchk = True
                                # Compute rmg and check elbow condition
                                rmg = torch.abs(1.0 - r)
                                elbowid = rmg < delta
                                elbchk = torch.all(rmg[elbowid] <= 1e-3).item()

                                if elbchk:
                                    break

                                # Projection update
                                told = one
                                for _ in range(self.maxit):
                                    ka = torch.mv(Kmat, alptmp[1:])
                                    aKa = torch.dot(ka, alptmp[1:])
                                    obj_value = self.objfun(
                                        alptmp[0], aka, ka, y, al, nobs
                                    )

                                    # Optimize intercept
                                    # optimal_intercept = minimize_scalar(self.objfun, args=(aka, ka, y, al, nobs), bracket=(-100.0, 100.0), method = 'brent')
                                    # obj_value_new = self.objfun(optimal_intercept.x, aka, ka, y, al, nobs)
                                    golden_s = self.golden_section_search(
                                        -100.0, 100.0, nobs, ka, aka, y, al
                                    )
                                    int_new = golden_s[0]
                                    obj_value_new = golden_s[1]
                                    if obj_value_new < obj_value:
                                        dif_step[0] = dif_step[0] + int_new - alptmp[0]
                                        alptmp[0] = int_new

                                    r = y * (alptmp[0] + ka)
                                    zvec = torch.where(
                                        r < omdelta,
                                        -y,
                                        torch.where(
                                            r > opdelta,
                                            torch.zeros(1, device=self.device),
                                            0.5 * y * oddelta * (r - opdelta),
                                        ),
                                    )
                                    gamvec = (
                                        zvec + 2.0 * float(nobs) * al * alptmp[1:]
                                    )  ##
                                    hval = self._hval(
                                        zvec,
                                        alptmp,
                                        svec[:, delta_id - 1],
                                        vvec[:, delta_id - 1],
                                        delta,
                                        al,
                                        nobs,
                                        vareps,
                                    )

                                    tnew = 0.5 + 0.5 * torch.sqrt(
                                        one + 4.0 * told * told
                                    )
                                    mul = 1.0 + (told - 1.0) / tnew
                                    told = tnew

                                    # Compute dif vector

                                    # dif_step = torch.zeros((nobs + 1), dtype=torch.double, device=self.device)
                                    dif_step[0] = (
                                        -2.0 * mul * delta * gval[delta_id - 1] * hval
                                    )
                                    dif_step[1:] = -dif_step[0] * svec[
                                        :, delta_id - 1
                                    ] - 2.0 * mul * delta * torch.mv(
                                        Umat, gamvec @ Umat * lpinv[:, delta_id - 1]
                                    )
                                    alptmp += dif_step

                                    ka = torch.mv(Kmat, alptmp[1:])
                                    r = y * (alptmp[0] + ka)
                                    npass[l] += 1
                                    alp_old = alptmp.clone()

                                    if torch.sum(elbowid).item() > 1:
                                        theta = torch.mv(Kmat, alptmp[1:])
                                        theta[elbowid] += y[elbowid] * (
                                            1.0 - r[elbowid]
                                        )
                                        alptmp[1:] = torch.mv(
                                            Umat, einv * torch.mv(Umat.T, theta)
                                        )

                                    dif_step = dif_step + alptmp - alp_old
                                    r = y * (alptmp[0] + torch.mv(Kmat, alptmp[1:]))
                                    mdd = torch.max(dif_step**2)
                                    # Check convergence
                                    if mdd < self.eps * mul**2:
                                        break
                                    elif mdd > nobs and npass[l] > 2:
                                        is_exit = True
                                        break
                                    if sum(npass) > self.maxit:
                                        is_exit = True
                                        break

                            # Check KKT condition
                            if is_exit:
                                break
                            zvec = torch.where(
                                r < 1.0,
                                -y,
                                torch.where(
                                    r > 1.0, torch.zeros(1).to(self.device), -0.5 * y
                                ),
                            )
                            KKT = zvec / nobs + 2.0 * al * alptmp[1:]
                            uo = max(al, 1.0)

                            if (
                                self._kkt_scale * torch.sum(KKT**2) / (uo**2)
                                < self.KKTeps
                            ):
                                alpvec = alptmp.clone()
                                converged[l] = True
                                break
                # else:
                #     # Reduce delta
                #     delta *= 0.125
                if delta_id >= self.delta_len:
                    print(f"Exceeded maximum delta iterations for lambda {l}")
                    break
                delta *= 0.125
            # Save the alpha vector for current lambda
            alpmat[:, l] = alpvec
            # Update anlam
            self.anlam = l
            t_cv = self._now()
            timing["path"] += t_cv - t
            lambda_timing["path"][l] = t_cv - t

            # Check if maximum iterations exceeded
            if sum(npass) > self.maxit:
                self.jerr = -l - 1
                break
            # print(f'Single fitting:{time.time() - start}')

            ######### cross-validation
            if self.is_exact == 0:
                pred[:, l] = self._cv_batched_lambda(
                    Kmat=Kmat,
                    y=y,
                    alpvec=alpvec,
                    r=r,
                    al=al,
                    nobs=nobs,
                    nfolds=nfolds,
                    vareps=vareps,
                    eps2=eps2,
                    Umat=Umat,
                    eigens=eigens,
                    Usum=Usum,
                    lpinv=lpinv,
                    lpUsum=lpUsum,
                    svec=svec,
                    vvec=vvec,
                    gval=gval,
                    delta_save=delta_save,
                    cvnpass=cvnpass,
                    fold_passes=fold_passes,
                    l=l,
                    one=one,
                )
                self.anlam = l
                lambda_timing["cross_validation"][l] = self._now() - t_cv
                timing["cross_validation"] += lambda_timing["cross_validation"][l]
                continue
            for nf in range(nfolds):
                # start = time.time()
                yn = y.clone()

                # Set the current fold's labels to zero
                yn[self.foldid == (nf + 1)] = 0.0

                loor = r.clone()  # Initial residuals
                looalp = alpvec.clone()  # Initial alphas

                delta = 1.0
                delta_id = 0

                # while delta_id < self.delta_len:
                while True:
                    delta_id += 1
                    opdelta = 1.0 + delta
                    omdelta = 1.0 - delta
                    oddelta = 1.0 / delta

                    if delta_id > delta_save:
                        lpinv[:, delta_id - 1] = 1.0 / (
                            eigens + 4.0 * float(nobs) * delta * al
                        )
                        lpUsum[:, delta_id - 1] = lpinv[:, delta_id - 1] * Usum
                        vvec[:, delta_id - 1] = torch.mv(
                            Umat, eigens * lpUsum[:, delta_id - 1]
                        )
                        svec[:, delta_id - 1] = torch.mv(Umat, lpUsum[:, delta_id - 1])
                        gval[delta_id - 1] = self._gval(
                            Usum, lpUsum[:, delta_id - 1], delta, al, nobs, vareps
                        )
                        delta_save = delta_id

                    # Compute residual r
                    told = one
                    ka = torch.mv(Kmat, looalp[1:])
                    loor = yn * (looalp[0] + ka)

                    while sum(cvnpass) <= self.nmaxit:
                        zvec = torch.where(
                            loor < omdelta,
                            -yn,
                            torch.where(
                                loor > opdelta,
                                torch.zeros(1).to(self.device),
                                yn * torch.tensor(0.5) * oddelta * (loor - opdelta),
                            ),
                        )
                        gamvec = zvec + 2.0 * float(nobs) * al * looalp[1:]  ##
                        hval = self._hval(
                            zvec,
                            looalp,
                            svec[:, delta_id - 1],
                            vvec[:, delta_id - 1],
                            delta,
                            al,
                            nobs,
                            vareps,
                        )

                        tnew = 0.5 + 0.5 * torch.sqrt(one + 4.0 * told * told)
                        mul = 1.0 + (told - 1.0) / tnew
                        told = tnew

                        # Compute dif vector

                        step_buf[0] = -2.0 * mul * delta * gval[delta_id - 1] * hval
                        step_buf[1:] = -step_buf[0] * svec[
                            :, delta_id - 1
                        ] - 2.0 * mul * delta * torch.mv(
                            Umat, gamvec @ Umat * lpinv[:, delta_id - 1]
                        )
                        looalp += step_buf

                        # zvec = torch.where(loor < omdelta, -yn, torch.where(loor > opdelta, torch.zeros(1).to(self.device), yn * torch.tensor(0.5) * oddelta * (loor - opdelta)))

                        # rds = torch.zeros(nobs + 1, dtype=torch.double).to(self.device)
                        # rds[0] = torch.sum(zvec) + 2.0 * nobs * vareps * looalp[0]
                        # rds[1:] = torch.mv(Kmat, zvec + 2.0 * float(nobs) * al * looalp[1:])

                        # tnew = 0.5 + 0.5 * torch.sqrt(torch.tensor(1.0).to(self.device) + 4.0 * told ** 2)
                        # mul = 1.0 + (told - 1.0) / tnew
                        # told = tnew.item()

                        # dif_step = -2.0 * delta * mul * torch.mv(Pinv[:, :, delta_id - 1], rds)
                        # looalp += dif_step

                        loor = yn * (looalp[0] + torch.mv(Kmat, looalp[1:]))

                        cvnpass[l] += 1
                        fold_passes[nf, l] += 1

                        # Check convergence
                        if torch.max(step_buf**2) < eps2 * (mul**2):
                            break
                    if sum(cvnpass) > self.nmaxit:
                        break
                    dif_step = step_buf.clone()
                    # dif_step = oldalpvec - alpvec
                    # print(f'Fitting alp time:{time.time() - start}')

                    ka = torch.mv(Kmat, looalp[1:])
                    aka = torch.dot(ka, looalp[1:])

                    obj_value = self.objfun(looalp[0], aka, ka, yn, al, nobs)
                    # optimal_intercept = minimize_scalar(self.objfun, args=(aka, ka, yn, al, nobs), bracket=(-100.0, 100.0), method="brent")
                    # obj_value_new = self.objfun(optimal_intercept.x, aka, ka, yn, al, nobs)
                    golden_s = self.golden_section_search(
                        -100.0, 100.0, nobs, ka, aka, yn, al
                    )
                    int_new = golden_s[0]
                    obj_value_new = golden_s[1]
                    if obj_value_new < obj_value:
                        dif_step[0] = dif_step[0] + int_new - looalp[0]
                        loor = loor + y * (int_new - looalp[0])
                        looalp[0] = int_new

                    # print(f'Fitting intercpt time:{time.time() - start}')
                    oldalpvec = looalp.clone()

                    zvec = torch.where(
                        loor < 1.0,
                        -yn,
                        torch.where(
                            loor > 1.0,
                            torch.zeros(1).to(self.device),
                            -torch.tensor(0.5) * yn,
                        ),
                    )
                    KKT = zvec / float(nobs) + 2.0 * al * looalp[1:]
                    uo = max(al, 1.0)
                    KKT_norm = self._kkt_scale * torch.sum(KKT**2) / (uo**2)

                    if KKT_norm < self.KKTeps2:
                        # Check convergence
                        # print(f'dif_step{dif_step}')
                        # dif_norm = torch.max(dif_step ** 2)
                        # print(f'dif:{dif_norm}')
                        # print(f'mul:{mul}')
                        # print(f'dif_cont:{float(nobs) * self.eps * mul * mul}')
                        # if dif_norm < float(nobs) * (self.eps * mul * mul):
                        if self.is_exact == 0:
                            break
                        else:
                            is_exit = False
                            alptmp = looalp.clone()
                            for nn in range(self.mproj):
                                elbowid = torch.zeros(nobs, dtype=torch.bool)
                                elbchk = True
                                # Compute rmg and check elbow condition
                                rmg = torch.abs(1.0 - loor)
                                elbowid = rmg < delta
                                elbchk = torch.all(rmg[elbowid] <= 1e-2).item()

                                if elbchk:
                                    break

                                # Projection update
                                told = one
                                for _ in range(self.maxit):
                                    ka = torch.mv(Kmat, alptmp[1:])
                                    aKa = torch.dot(ka, alptmp[1:])

                                    obj_value = self.objfun(
                                        alptmp[0], aka, ka, yn, al, nobs
                                    )

                                    # Optimize intercept
                                    golden_s = self.golden_section_search(
                                        -100.0, 100.0, nobs, ka, aka, yn, al
                                    )
                                    int_new = golden_s[0]
                                    obj_value_new = golden_s[1]
                                    if obj_value_new < obj_value:
                                        dif_step[0] = dif_step[0] + int_new - alptmp[0]
                                        alptmp[0] = int_new

                                    loor = yn * (alptmp[0] + ka)
                                    zvec = torch.where(
                                        loor < omdelta,
                                        -yn,
                                        torch.where(
                                            loor > opdelta,
                                            torch.zeros(1).to(self.device),
                                            0.5 * yn * oddelta * (loor - opdelta),
                                        ),
                                    )

                                    # rds = torch.zeros(nobs + 1, dtype=torch.double).to(self.device)
                                    # rds[0] = torch.sum(zvec) + 2.0 * float(nobs) * vareps * alptmp[0]
                                    # rds[1:] = torch.mv(Kmat, zvec + 2.0 * float(nobs) * al * alptmp[1:])

                                    # tnew = 0.5 + 0.5 * torch.sqrt(torch.tensor(1.0).to(self.device) + 4.0 * told ** 2)
                                    # mul = 1.0 + (told - 1.0) / tnew
                                    # told = tnew.item()

                                    # dif_step = - 2.0 * delta * mul * torch.mv(Pinv[:, :, delta_id - 1], rds)
                                    # alptmp += dif_step

                                    gamvec = (
                                        zvec + 2.0 * float(nobs) * al * alptmp[1:]
                                    )  ##
                                    hval = self._hval(
                                        zvec,
                                        alptmp,
                                        svec[:, delta_id - 1],
                                        vvec[:, delta_id - 1],
                                        delta,
                                        al,
                                        nobs,
                                        vareps,
                                    )

                                    tnew = 0.5 + 0.5 * torch.sqrt(
                                        one + 4.0 * told * told
                                    )
                                    mul = 1.0 + (told - 1.0) / tnew
                                    told = tnew

                                    # Compute dif vector

                                    # dif_step = torch.zeros((nobs + 1), dtype=torch.double, device=self.device)
                                    dif_step[0] = (
                                        -2.0 * mul * delta * gval[delta_id - 1] * hval
                                    )
                                    dif_step[1:] = -dif_step[0] * svec[
                                        :, delta_id - 1
                                    ] - 2.0 * mul * delta * torch.mv(
                                        Umat, gamvec @ Umat * lpinv[:, delta_id - 1]
                                    )
                                    alptmp += dif_step

                                    ka = torch.mv(Kmat, alptmp[1:])
                                    loor = yn * (alptmp[0] + ka)
                                    alp_old = alptmp.clone()

                                    if torch.sum(elbowid).item() > 1:
                                        theta = torch.mv(Kmat, alptmp[1:])
                                        theta[elbowid] += yn[elbowid] * (
                                            1.0 - loor[elbowid]
                                        )
                                        alptmp[1:] = torch.mv(
                                            Umat, einv * torch.mv(Umat.T, theta)
                                        )

                                    dif_step = dif_step + alptmp - alp_old
                                    loor = yn * (alptmp[0] + torch.mv(Kmat, alptmp[1:]))
                                    cvnpass[l] += 1
                                    fold_passes[nf, l] += 1
                                    mdd = torch.max(dif_step**2)
                                    # Check convergence
                                    if mdd < nobs * eps2 * mul**2:
                                        break
                                    elif mdd > nobs and cvnpass[l] > 2:
                                        is_exit = True
                                        break
                                    if sum(cvnpass) > self.nmaxit:
                                        is_exit = True
                                        break
                                if is_exit:
                                    break
                            if is_exit:
                                break
                            looalp = alptmp.clone()
                            break
                    if delta_id >= self.delta_len:
                        print(f"Exceeded maximum delta iterations for lambda {l}")
                        break
                    delta *= 0.125

                # for j in range(nobs):
                #     if self.foldid[j] == (nf + 1):
                #         looalp[j + 1] = 0.0
                loo_ind = self.foldid == (nf + 1)
                looalp[1:][loo_ind] = 0.0
                pred[loo_ind, l] = looalp[1:] @ Kmat[:, loo_ind] + looalp[0]
                # print(pred[loo_ind, l][:10])
                # for j in range(nobs):
                #     if self.foldid[j] == (nf + 1):
                #         pred[j, l] = torch.sum(Kmat[:, j] * looalp[1:]) + looalp[0]
                # print(pred[loo_ind, l][:10])
                # print(f'{nf}-fold: {time.time() - start}')
            self.anlam = l
            lambda_timing["cross_validation"][l] = self._now() - t_cv
            timing["cross_validation"] += lambda_timing["cross_validation"][l]

        self.timing = timing
        self.lambda_timing = lambda_timing
        self.fold_passes = fold_passes.cpu()
        self.alpmat = alpmat
        self.npass = torch.tensor(npass, dtype=torch.int32, device=self.device)
        self.cvnpass = torch.tensor(cvnpass, dtype=torch.int32, device=self.device)
        self.converged = converged
        self.jerr = jerr
        self.pred = pred
        self._warn_not_converged()

    def _warn_not_converged(self):
        n_not_converged = int((~self.converged).sum().item())
        if n_not_converged > 0:
            warnings.warn(
                f"cvksvm did not converge for {n_not_converged} of {self.nlam} "
                f"lambda values within maxit={self.maxit} iterations. The "
                "solution may be inaccurate. Consider increasing maxit or "
                "loosening the tolerances (eps/KKTeps).",
                ConvergenceWarning,
            )
        if int(torch.sum(self.cvnpass).item()) > self.nmaxit:
            warnings.warn(
                f"cvksvm cross-validation hit the iteration cap "
                f"(nlam * maxit = {self.nmaxit}); CV predictions, and any "
                "regularization value selected from them, may be inaccurate. "
                "Consider increasing maxit.",
                ConvergenceWarning,
            )

    def _cv_batched_lambda(
        self,
        *,
        Kmat,
        y,
        alpvec,
        r,
        al,
        nobs,
        nfolds,
        vareps,
        eps2,
        Umat,
        eigens,
        Usum,
        lpinv,
        lpUsum,
        svec,
        vvec,
        gval,
        delta_save,
        cvnpass,
        fold_passes,
        l,
        one,
    ):
        fold_ids = torch.arange(1, nfolds + 1, device=self.device)
        fold_masks = self.foldid.unsqueeze(1) == fold_ids.unsqueeze(0)
        fold_col_index = self.foldid.to(dtype=torch.long) - 1
        row_index = torch.arange(nobs, device=self.device)

        yn_batch = y.unsqueeze(1).expand(-1, nfolds).clone()
        yn_batch[fold_masks] = 0.0

        looalp_batch = alpvec.unsqueeze(1).expand(-1, nfolds).clone()
        loor_batch = r.unsqueeze(1).expand(-1, nfolds).clone()
        cv_step_buf = torch.zeros(
            (nobs + 1, nfolds), dtype=self.dtype, device=self.device
        )

        active = torch.ones(nfolds, dtype=torch.bool, device=self.device)
        delta = 1.0
        delta_id = 0

        while torch.any(active):
            delta_id += 1
            opdelta = 1.0 + delta
            omdelta = 1.0 - delta
            oddelta = 1.0 / delta

            if delta_id > delta_save:
                lpinv[:, delta_id - 1] = 1.0 / (eigens + 4.0 * float(nobs) * delta * al)
                lpUsum[:, delta_id - 1] = lpinv[:, delta_id - 1] * Usum
                vvec[:, delta_id - 1] = torch.mv(Umat, eigens * lpUsum[:, delta_id - 1])
                svec[:, delta_id - 1] = torch.mv(Umat, lpUsum[:, delta_id - 1])
                gval[delta_id - 1] = self._gval(
                    Usum, lpUsum[:, delta_id - 1], delta, al, nobs, vareps
                )
                delta_save = delta_id

            active_cols = torch.nonzero(active, as_tuple=False).squeeze(1)
            told = torch.ones(nfolds, dtype=self.dtype, device=self.device)
            best_step2 = torch.full_like(told, float("inf"))
            since_best = torch.zeros(nfolds, dtype=torch.int64, device=self.device)
            ka_batch = torch.mm(Kmat, looalp_batch[1:, active_cols])
            loor_batch[:, active_cols] = yn_batch[:, active_cols] * (
                looalp_batch[0, active_cols].unsqueeze(0) + ka_batch
            )

            active_iter = active.clone()
            while torch.any(active_iter):
                iter_cols = torch.nonzero(active_iter, as_tuple=False).squeeze(1)
                yn_iter = yn_batch[:, iter_cols]
                loor_iter = loor_batch[:, iter_cols]
                alp_iter = looalp_batch[:, iter_cols]
                told_iter = told[iter_cols]

                zvec = torch.where(
                    loor_iter < omdelta,
                    -yn_iter,
                    torch.where(
                        loor_iter > opdelta,
                        0.0,
                        0.5 * yn_iter * oddelta * (loor_iter - opdelta),
                    ),
                )
                gamvec = zvec + 2.0 * float(nobs) * al * alp_iter[1:, :]
                hval = self._hval(
                    zvec,
                    alp_iter,
                    svec[:, delta_id - 1],
                    vvec[:, delta_id - 1],
                    delta,
                    al,
                    nobs,
                    vareps,
                )

                tnew = 0.5 + 0.5 * torch.sqrt(one + 4.0 * told_iter * told_iter)
                mul = 1.0 + (told_iter - 1.0) / tnew
                told[iter_cols] = tnew

                cv_step_buf[0, iter_cols] = (
                    -2.0 * mul * delta * gval[delta_id - 1] * hval
                )
                spectral = torch.mm(Umat.T, gamvec)
                spectral.mul_(lpinv[:, delta_id - 1].unsqueeze(1))
                proj_term = torch.mm(Umat, spectral)
                cv_step_buf[1:, iter_cols] = (
                    -cv_step_buf[0, iter_cols].unsqueeze(0)
                    * svec[:, delta_id - 1].unsqueeze(1)
                    - 2.0 * delta * mul.unsqueeze(0) * proj_term
                )
                looalp_batch[:, iter_cols] += cv_step_buf[:, iter_cols]

                ka_batch = torch.mm(Kmat, looalp_batch[1:, iter_cols])
                loor_batch[:, iter_cols] = yn_iter * (
                    looalp_batch[0, iter_cols].unsqueeze(0) + ka_batch
                )

                cvnpass[l] += iter_cols.numel()
                fold_passes[iter_cols, l] += 1
                if sum(cvnpass) > self.nmaxit:
                    break

                step2 = torch.max(cv_step_buf[:, iter_cols] ** 2, dim=0).values
                converged = step2 < eps2 * (mul**2)
                if self._stall_patience is not None:  # see _stalled
                    improved = step2 < best_step2[iter_cols]
                    best_step2[iter_cols] = torch.where(
                        improved, step2, best_step2[iter_cols]
                    )
                    since_best[iter_cols] = torch.where(
                        improved, 0, since_best[iter_cols] + 1
                    )
                    converged |= since_best[iter_cols] >= self._stall_patience
                active_iter[iter_cols[converged]] = False

            if sum(cvnpass) > self.nmaxit:
                break

            # The unfinished folds together: one product with K for all of them,
            # their intercepts by searches run in step (one objective evaluation
            # for all folds per step), then their KKT tests.
            cols = torch.nonzero(active, as_tuple=False).squeeze(1)
            alp = looalp_batch[:, cols]
            yn = yn_batch[:, cols]
            ka = torch.mm(Kmat, alp[1:])
            aka = (ka * alp[1:]).sum(dim=0)
            obj_value = self.objfun(alp[0], aka, ka, yn, al, nobs)
            int_new, obj_value_new = brent_minimize_batch(
                lambda b: self.objfun(
                    torch.as_tensor(b, dtype=ka.dtype, device=ka.device),
                    aka,
                    ka,
                    yn,
                    al,
                    nobs,
                ),
                -100.0,
                100.0,
                cols.numel(),
            )
            int_new = torch.as_tensor(int_new, dtype=alp.dtype, device=alp.device)
            obj_value_new = torch.as_tensor(
                obj_value_new, dtype=obj_value.dtype, device=obj_value.device
            )
            better = obj_value_new < obj_value
            shift = torch.where(better, int_new - alp[0], torch.zeros_like(int_new))
            loor = loor_batch[:, cols] + y.unsqueeze(1) * shift
            looalp_batch[0, cols] = torch.where(better, int_new, alp[0])
            loor_batch[:, cols] = loor
            zvec = torch.where(loor < 1.0, -yn, torch.where(loor > 1.0, 0.0, -0.5 * yn))
            KKT = zvec / float(nobs) + 2.0 * al * alp[1:]
            uo = max(al, 1.0)
            KKT_norm = self._kkt_scale * torch.sum(KKT**2, dim=0) / (uo**2)
            active[cols[KKT_norm < self.KKTeps2]] = False

            if delta_id >= self.delta_len:
                print(f"Exceeded maximum delta iterations for lambda {l}")
                break
            delta *= 0.125

        cv_alpha = looalp_batch[1:, :].clone()
        cv_alpha[fold_masks] = 0.0
        cv_scores = torch.mm(Kmat, cv_alpha) + looalp_batch[0, :].unsqueeze(0)
        return cv_scores[row_index, fold_col_index]

    def _now(self):
        """Wall clock once the device's queued work is done, for ``timing``."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _stalled(self, step2, stall):
        """True once the largest step has not reached a new low for
        ``_stall_patience`` iterations (float32 only; see ``__init__``).
        ``stall`` holds the lowest step so far and the iterations since."""
        if self._stall_patience is None:
            return False
        step2 = float(step2)
        if step2 < stall[0]:
            stall[0], stall[1] = step2, 0
            return False
        stall[1] += 1
        return stall[1] >= self._stall_patience

    @staticmethod
    def _gval(Usum, lpUsum_d, delta, al, nobs, vareps):
        """1 / (n + 4 n delta vareps - sum(vvec)), the intercept's step factor.

        With c = 4 n delta lambda and U orthogonal (so n = |U^T 1|^2),
        n - sum(vvec) = c * Usum . lpUsum. The direct form subtracts two numbers
        of size n that agree to within about c, which float32 cannot resolve
        once c is small; this form has no such difference.
        """
        return 1.0 / (4.0 * nobs * delta * (vareps + al * torch.dot(Usum, lpUsum_d)))

    @staticmethod
    def _hval(zvec, alp, svec_d, vvec_d, delta, al, nobs, vareps):
        """sum(zvec) + 2 n vareps alp[0] - vvec . gamvec: the intercept's step.

        Written with 1 - vvec = c * svec (c = 4 n delta lambda), for the reason
        given in ``_gval``. ``zvec`` and ``alp`` may hold one column per fold.
        """
        return (
            4.0 * nobs * delta * al * (svec_d @ zvec)
            - 2.0 * nobs * al * (vvec_d @ alp[1:])
            + 2.0 * nobs * vareps * alp[0]
        )

    def cv(self, pred, y):
        pred_label = torch.where(pred > 0, 1, -1).to(device="cpu")
        y_expanded = y[:, None]
        misclass_matrix = (pred_label != y_expanded).float()
        misclass_rate = misclass_matrix.mean(dim=0)
        return misclass_rate

    def predict(self, Kmat_new, y_new, alp_b):
        result = torch.mv(Kmat_new, alp_b[1:]) + alp_b[0]
        ypred = torch.where(result > 0, torch.tensor(1), torch.tensor(-1))
        acc = torch.mean((ypred == y_new).float())
        return ypred, acc

    def obj_value(self, alp_b, lam_b):
        intcpt = alp_b[0]
        alp = alp_b[1:]
        Kmat = self.Kmat.to(alp.device)
        alp = alp.to(Kmat.dtype)
        ka = torch.mv(Kmat, alp)
        aka = torch.dot(alp, ka)
        y_train = self.y.to(alp.device)
        obj = self.objfun(intcpt, aka, ka, y_train, lam_b, self.nobs)
        return obj

    def objfun(self, intcpt, aka, ka, y, lam, nobs):
        """
        Compute the objective function value for SVM.

        Parameters:
        - intcpt (float): Intercept term.
        - aka (torch.Tensor): Regularization term (alpha * K * alpha).
        - ka (torch.Tensor): Kernel matrix dot alpha vector (K * alpha).
        - y (torch.Tensor): Labels vector of shape (nobs,).
        - lam (float): Regularization parameter.
        - nobs (int): Number of observations.

        Returns:
        - objval (float): Objective function value.
        """
        # Compute f_hat (fh) and the hinge loss xi
        fh = ka + intcpt
        xi_tmp = 1.0 - y * fh
        xi = torch.where(xi_tmp > 0, xi_tmp, torch.zeros_like(xi_tmp))

        # Compute the objective value
        # per column when y and ka hold one column per fold
        objval = lam * aka + torch.sum(xi, dim=0) / nobs

        return objval

    def golden_section_search(self, lmin, lmax, nobs, ka, aka, y, lam):
        """Intercept minimising ``objfun`` on [lmin, lmax] by Brent's method
        (``functions.brent_minimize``); returns (intercept, objective) as
        floats."""
        return brent_minimize(
            lambda b: self.objfun(b, aka, ka, y, lam, nobs), lmin, lmax
        )


# ---------------------------------------------------------------------------
# Truncated-spectrum SVM path (TorchKMSVC(spectrum="truncated") and
# TorchKMSVC(low_rank=True)). The same code as
# torchkm.experimental.spectral_svm, which stays as the place to explore
# changes; this copy is the one the estimators use.
#
# For each lambda the solver minimises the smoothed-hinge objective
#
#     F(b, alpha) = sum_i phi_delta(y_i (b + K_i alpha)) + n lam alpha'K alpha + n eps b^2
#
# over a decreasing smoothing schedule delta = 1, 1/8, ..., and accepts the
# lambda (and each cross-validation fold) only when a duality gap for the
# unsmoothed hinge problem is below ``gap_tol``. The curvature matrix of the
# majorization steps is either K's full eigendecomposition or a truncated
# spectrum: the top-r Ritz pairs plus a flat tail that bounds the rest.
# ---------------------------------------------------------------------------


def smoothed_hinge(r, delta):
    """phi_delta(r): 1 - r below 1 - delta, 0 above 1 + delta, quadratic between."""
    return torch.where(
        r < 1.0 - delta,
        1.0 - r,
        torch.where(r > 1.0 + delta, 0.0, (1.0 + delta - r) ** 2 / (4.0 * delta)),
    )


def smoothed_hinge_grad(r, delta):
    """d phi_delta / d r, in [-1, 0]."""
    return torch.where(
        r < 1.0 - delta,
        -1.0,
        torch.where(r > 1.0 + delta, 0.0, (r - 1.0 - delta) / (2.0 * delta)),
    )


# Kernel DWD (q = 1): V(u) = 1 - u for u <= 1/2, 1 / (4u) above. V is smooth
# with V'' <= 4, the curvature of a smoothed hinge of width 1/8, so the same
# majorization steps apply at that fixed width: no smoothing schedule.
_DWD_DELTA = 0.125


def dwd_loss(r):
    """V(r) of kernel DWD with q = 1."""
    return torch.where(r <= 0.5, 1.0 - r, 0.25 / torch.clamp(r, min=0.5))


def dwd_loss_grad(r):
    """V'(r), in [-1, 0)."""
    return torch.where(r <= 0.5, -1.0, -0.25 / torch.clamp(r, min=0.5) ** 2)


def _loss(r, delta, loss):
    """The (smoothed, for the hinge) loss the steps minimise."""
    return smoothed_hinge(r, delta) if loss == "hinge" else dwd_loss(r)


def _loss_grad(r, delta, loss):
    return smoothed_hinge_grad(r, delta) if loss == "hinge" else dwd_loss_grad(r)


def _margin_loss(r, loss):
    """The loss of the problem itself: the hinge, or DWD's V."""
    return torch.clamp(1.0 - r, min=0.0) if loss == "hinge" else dwd_loss(r)


def _dual_linear(beta, n, loss):
    """The dual's separable part per column: sum(beta) for the hinge,
    sum(sqrt(n beta)) / n for DWD (V*(-t) = -sqrt(t), t = n beta in [0, 1])."""
    if loss == "hinge":
        return beta.sum(dim=0)
    return torch.sqrt(torch.clamp(n * beta, min=0.0)).sum(dim=0) / n


def project_dual(beta, y, upper, iters=64):
    """Euclidean projection of each column of ``beta`` onto
    {0 <= beta <= upper, sum(beta * y) = 0}: clip(beta - theta y, 0, upper),
    with theta found by bisection (the constraint sum is non-increasing in
    theta). float64 in and out."""
    reach = beta.abs().amax(dim=0) + upper.amax(dim=0) + 1.0
    lo, hi = -reach, reach.clone()
    for _ in range(iters):
        theta = 0.5 * (lo + hi)
        s = (y * torch.clamp(beta - theta * y, min=0.0).minimum(upper)).sum(dim=0)
        lo = torch.where(s > 0, theta, lo)
        hi = torch.where(s > 0, hi, theta)
    theta = 0.5 * (lo + hi)
    return torch.clamp(beta - theta * y, min=0.0).minimum(upper)


def hinge_duality_gap(
    K, y, alpha, b, lam, *, Ka=None, delta=None, refine=10, lmax=None, target=None,
    loss="hinge",
):
    """Certified relative gap of the unsmoothed kernel SVM, one column per problem.

    ``loss="dwd"``: kernel DWD instead, P with V(y f) in place of the hinge and
    D(beta) = sum_{i in T} sqrt(n beta_i) / n - (beta y)'K(beta y) / (4 lam) on
    the same set; its candidates are 2 lam y alpha and -V'(y f) / n, and it is
    not refined (the derivative of sqrt is unbounded at 0).

    Primal: P(alpha, b) = sum_{i in T} max(0, 1 - y_i (K_i alpha + b)) / n + lam alpha'K alpha,
    where T holds the rows with y_i != 0 (a cross-validation fold zeroes its
    held-out labels). Dual: D(beta) = sum(beta) - (beta y)'K(beta y) / (4 lam),
    0 <= beta_i <= 1/n on T, beta_i = 0 off T, sum(beta y) = 0. Weak duality
    D(beta) <= P* <= P(alpha, b) holds for every positive semi-definite K,
    so (P - D) / P bounds the relative suboptimality of (alpha, b).

    The dual point is the best of a few candidates built from the iterate
    (the smoothed-hinge derivative at ``delta``, margin-based points, and
    2 lam y alpha), improved by ``refine`` projected-gradient steps (step
    2 lam / lmax, where lmax bounds K's largest eigenvalue; any step keeps the
    bound valid because only feasible points are scored). One product with K
    per candidate batch and per refinement step. With ``target``, refinement
    stops once every gap is at most ``target``, and is skipped unless some
    column's best candidate is within 30 times ``target`` (farther, the
    primal itself is not there yet).

    ``lam`` is a number or one value per column.

    Returns (gap, P, D) as float64 tensors with one entry per column.
    """
    squeeze = alpha.dim() == 1
    if squeeze:
        y, alpha = y.unsqueeze(1), alpha.unsqueeze(1)
        Ka = None if Ka is None else Ka.unsqueeze(1)
    n, m = alpha.shape
    b = torch.as_tensor(b, dtype=torch.float64, device=alpha.device).reshape(-1)
    lam = torch.as_tensor(lam, dtype=torch.float64, device=alpha.device).reshape(-1)
    if Ka is None:
        Ka = K @ alpha
    # y is +-1 or 0: exact in any dtype, and promotes to float64 in products
    # with float64 terms; so it stays in its own dtype, as do alpha and K alpha
    # (each used once), and the mask is boolean
    y64 = y if y.dtype == torch.float64 else y.float()
    on = y != 0
    r = y64 * (Ka.double() + b)
    P = (_margin_loss(r, loss) * on).sum(dim=0) / n + lam * _row_sums(
        lambda a, ka: (a.double() * ka.double()).sum(dim=0), alpha, Ka
    )
    upper = on.double() / n

    def score(beta, yy, lam):
        by = beta * yy
        Kby = (K @ by.to(K.dtype)).double()
        return _dual_linear(beta, n, loss) - (by * Kby).sum(dim=0) / (4.0 * lam), Kby

    # candidates one at a time (one n x m block each, not all c stacked); per
    # column the first best is kept
    def candidates():
        yield 2.0 * lam * y64 * alpha.double()
        if loss == "dwd":
            yield -dwd_loss_grad(r) * upper
            return
        mid = torch.clamp(2.0 * lam * y64 * alpha.double(), min=0.0).minimum(upper)
        for t in (1e-2, 1e-3, 1e-4):
            yield torch.where(r < 1.0 - t, upper, torch.where(r > 1.0 + t, 0.0, mid))
        if delta is not None:
            yield -smoothed_hinge_grad(r, delta) * upper

    beta, D = None, None
    for cand in candidates():
        cand = project_dual(cand, y64, upper)
        Dc, _ = score(cand, y64, lam)
        if beta is None:
            beta, D = cand, Dc
        else:
            better = Dc > D
            beta = torch.where(better, cand, beta)
            D = torch.where(better, Dc, D)
        del cand

    if loss == "dwd":
        refine = 0
    if target is not None:
        g0 = (P - D) / P.abs().clamp_min(1e-300)
        if not bool(((g0 > target) & (g0 <= 30.0 * target)).any()):
            refine = 0
    if refine > 0:
        if lmax is None:
            lmax = float(torch.linalg.matrix_norm(K.double(), 2)) if n <= 2000 else None
        if lmax is None:
            raise ValueError("pass lmax (an upper bound on K's largest eigenvalue)")
        # accelerated projected gradient; every scored point is feasible
        step = 2.0 * lam / lmax
        prev, point, t = beta, beta, 1.0
        for i in range(refine + 1):
            Dp, Kby = score(point, y64, lam)
            D = torch.maximum(D, Dp)
            done = (
                (P - D) / P.abs().clamp_min(1e-300) <= target
                if target is not None
                else None
            )
            if i == refine or (done is not None and bool(done.all())):
                break
            grad = (1.0 - y64 * Kby / (2.0 * lam)) * on
            cur = project_dual(point + step * grad, y64, upper)
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            point = project_dual(cur + ((t - 1.0) / t_next) * (cur - prev), y64, upper)
            prev, t = cur, t_next
    gap = (P - D) / P.abs().clamp_min(1e-300)
    if squeeze:
        return gap[0], P[0], D[0]
    return gap, P, D


def _project_dual_tiled(src, Y, n, m, out, iters=64):
    """``project_dual`` of the n x m point whose rows i .. j are src(i, j),
    with labels Y (a _Labels), written into ``out`` (float64): the bisection
    sums are accumulated over row tiles and the point is rebuilt for each."""
    tiles = _tiles(n, m)
    up = 1.0 / n
    reach = torch.zeros(m, dtype=torch.float64, device=out.device)
    for i, j in tiles:
        reach = torch.maximum(reach, src(i, j).abs().amax(dim=0))
    lo = -(reach + up + 1.0)
    hi = reach + up + 1.0
    rows = [(i, j, Y.rows(i, j).double()) for i, j in tiles] if len(tiles) == 1 else None

    def labels(i, j):
        return rows[0][2] if rows is not None else Y.rows(i, j).double()

    for _ in range(iters):
        theta = 0.5 * (lo + hi)
        tot = torch.zeros(m, dtype=torch.float64, device=out.device)
        for i, j in tiles:
            yt = labels(i, j)
            upper = (yt != 0).double() * up
            tot += (yt * torch.clamp(src(i, j) - theta * yt, min=0.0).minimum(upper)).sum(0)
        lo = torch.where(tot > 0, theta, lo)
        hi = torch.where(tot > 0, hi, theta)
    theta = 0.5 * (lo + hi)
    for i, j in tiles:
        yt = labels(i, j)
        upper = (yt != 0).double() * up
        out[i:j] = torch.clamp(src(i, j) - theta * yt, min=0.0).minimum(upper)
    return out


def _hinge_duality_gap_tiled(
    K, Y, alpha, b, lam, *, Ka, delta=None, refine=10, lmax=None, target=None,
    loss="hinge",
):
    """``hinge_duality_gap`` for labels Y given as a _Labels: the same
    candidates, projection, scores and refinement, with the candidates and
    projections built row tile by row tile. Stored n x m: the projected
    candidate and the best point (float64), and the input and output of each
    product with K."""
    n, m = alpha.shape
    dev = alpha.device
    b = torch.as_tensor(b, dtype=torch.float64, device=dev).reshape(-1)
    lam = torch.as_tensor(lam, dtype=torch.float64, device=dev).reshape(-1)
    tiles = _tiles(n, m)
    up = 1.0 / n

    def P_rows(Yt, a, ka):
        r = Yt.double() * (ka.double() + b)
        return (_margin_loss(r, loss) * (Yt != 0)).sum(0) / n + lam * (
            a.double() * ka.double()
        ).sum(0)

    P = _row_sums(P_rows, Y, alpha, Ka)

    def score(beta):
        by = torch.empty(n, m, dtype=K.dtype, device=dev)
        bsum = torch.zeros(m, dtype=torch.float64, device=dev)
        for i, j in tiles:
            by[i:j] = (beta[i:j] * Y.rows(i, j).double()).to(K.dtype)
            bsum += _dual_linear(beta[i:j], n, loss)
        Kby = K @ by
        quad = torch.zeros(m, dtype=torch.float64, device=dev)
        for i, j in tiles:
            quad += (beta[i:j] * Y.rows(i, j).double() * Kby[i:j].double()).sum(0)
        del by
        return bsum - quad / (4.0 * lam), Kby

    def margins(i, j):
        Yt = Y.rows(i, j).double()
        return Yt, Yt * (Ka[i:j].double() + b), (Yt != 0).double() * up

    def cand(k):
        def src(i, j):
            Yt, r, upper = margins(i, j)
            base = 2.0 * lam * Yt * alpha[i:j].double()
            if k == 0:
                return base
            if loss == "dwd":
                return -dwd_loss_grad(r) * upper
            if k <= 3:
                tt = (1e-2, 1e-3, 1e-4)[k - 1]
                mid = torch.clamp(base, min=0.0).minimum(upper)
                return torch.where(r < 1.0 - tt, upper, torch.where(r > 1.0 + tt, 0.0, mid))
            return -smoothed_hinge_grad(r, delta) * upper

        return src

    work = torch.empty(n, m, dtype=torch.float64, device=dev)
    beta = torch.empty(n, m, dtype=torch.float64, device=dev)
    D = None
    ks = (0, 4) if loss == "dwd" else range(5 if delta is not None else 4)
    for k in ks:
        _project_dual_tiled(cand(k), Y, n, m, work)
        Dc, _ = score(work)
        if D is None:
            beta.copy_(work)
            D = Dc
        else:
            better = Dc > D
            for i, j in tiles:
                beta[i:j] = torch.where(better, work[i:j], beta[i:j])
            D = torch.where(better, Dc, D)

    if loss == "dwd":
        refine = 0
    if target is not None:
        g0 = (P - D) / P.abs().clamp_min(1e-300)
        if not bool(((g0 > target) & (g0 <= 30.0 * target)).any()):
            refine = 0
    if refine > 0:
        if lmax is None:
            raise ValueError("pass lmax (an upper bound on K's largest eigenvalue)")
        step = 2.0 * lam / lmax
        prev, point, t = beta, beta.clone(), 1.0  # point: the scored iterate
        cur = work
        for it in range(refine + 1):
            Dp, Kby = score(point)
            D = torch.maximum(D, Dp)
            done = (P - D) / P.abs().clamp_min(1e-300) <= target if target is not None else None
            if it == refine or (done is not None and bool(done.all())):
                break

            def ascent(i, j):
                Yt = Y.rows(i, j).double()
                grad = (1.0 - Yt * Kby[i:j].double() / (2.0 * lam)) * (Yt != 0)
                return point[i:j] + step * grad

            _project_dual_tiled(ascent, Y, n, m, cur)
            del Kby
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            mom = (t - 1.0) / t_next

            def extrapolate(i, j):
                return cur[i:j] + mom * (cur[i:j] - prev[i:j])

            _project_dual_tiled(extrapolate, Y, n, m, point)
            prev, cur = cur, prev  # cur's buffer is free for the next ascent
            t = t_next
    gap = (P - D) / P.abs().clamp_min(1e-300)
    return gap, P, D


_ROW_BLOCK = 2**24  # elements per row block of the float64 sums below
_TILE_BLOCK = 2**21  # elements per row tile of the tiled path (about 20 live)


class _Labels:
    """The n x m label matrix of a set of columns, never stored: column j holds
    y with the rows of fold ``cf[j]`` set to zero (cf[j] = -1: no fold, the
    whole data). ``rows(i, j)`` builds rows i .. j; ``cols(idx)`` selects
    columns."""

    def __init__(self, y, fold, cf):
        self.y, self.fold, self.cf = y, fold, cf
        self.shape = (y.shape[0], cf.shape[0])
        self.dtype, self.device = y.dtype, y.device

    def rows(self, i, j):
        held = self.fold[i:j, None] == self.cf[None, :]
        return torch.where(held, 0.0, self.y[i:j, None]).to(self.dtype)

    def cols(self, idx):
        return _Labels(self.y, self.fold, self.cf[idx])


def _tile(t, i, j):
    return t.rows(i, j) if isinstance(t, _Labels) else t[i:j]


def _tiles(n, m, block=None):
    """Row ranges of about ``block`` elements (default _TILE_BLOCK, the tiled
    path's) for m columns."""
    step = max(1, (_TILE_BLOCK if block is None else block) // max(m, 1))
    return [(i, min(n, i + step)) for i in range(0, n, step)]


def _row_sums(fn, *rows):
    """sum over row blocks of fn(block of each tensor in ``rows``): column sums
    of float64 expressions without n x m float64 temporaries. A _Labels
    argument is built block by block."""
    n, m = rows[0].shape
    tiled = any(isinstance(t, _Labels) for t in rows)
    total = None
    for i, j in _tiles(n, m, None if tiled else _ROW_BLOCK):
        part = fn(*(_tile(t, i, j) for t in rows))
        total = part if total is None else total + part
    return total


def _dF_rows(R, Y, KA, da, Kda, db, lam, delta, n, loss="hinge"):
    """Rows' share of the smoothed objective's change along a step (float64)."""
    R64 = R.double()
    dR = (Y * (db + Kda)).double()
    dloss = (
        (_loss(R64 + dR, delta, loss) - _loss(R64, delta, loss)) * (Y != 0)
    ).sum(0)
    da64, Kda64 = da.double(), Kda.double()
    return dloss + n * lam * (2.0 * (da64 * KA.double()).sum(0) + (da64 * Kda64).sum(0))


def _F_rows(R, Y, A, KA, lam, delta, n, loss="hinge"):
    """Rows' share of the smoothed objective, without the intercept's ridge."""
    value = (_loss(R.double(), delta, loss) * (Y != 0)).sum(0)
    return value + n * lam * (A.double() * KA.double()).sum(0)


class _CountedK:
    """K for the certificate: products counted under reads["certificate"]."""

    def __init__(self, K, reads):
        self.K, self.reads = K, reads
        self.dtype, self.shape = K.dtype, K.shape

    def __matmul__(self, X):
        self.reads["certificate"] += 1
        return self.K @ X

    def double(self):
        return self.K.double()


class _Counter:
    """Products with an n x n matrix (K, or U for the full spectrum)."""

    def __init__(self):
        self.reads = 0

    def mm(self, M, X):
        self.reads += 1
        return M @ X


class _FullSpectrum:
    """K = U diag(e) U' (torch.linalg.eigh). The curvature eigenvalues are e,
    clamped at zero, plus twice the factorization's rounding error (as in
    cvksvm), so the bound stays above K in float32 too. K P is applied as
    U diag(e / (et + c)) U'."""

    name = "full"

    def __init__(self, K, count):
        e, U = kernel_eigh(K)
        e.clamp_min_(0.0)
        err = _factorization_error(K, U, e)
        self.U, self.e, self.et, self.count = U, e, e + 2.0 * err, count
        self.u1 = (
            U.T @ torch.ones(K.shape[0], dtype=K.dtype, device=K.device)
        ).double()
        self.lmax = float(self.et.max())
        self.info = dict(rank=K.shape[0], factorization_error=err, lmax=self.lmax)

    def apply(self, G, KG, c):
        """P G and K P G; ``c`` (float64) holds one shift per column of G."""
        m = G.shape[1]
        inv = (1.0 / (self.et.double()[:, None] + c)).to(G.dtype)
        UG = self.count.mm(self.U.T, G)
        out = self.count.mm(
            self.U, torch.cat([UG * inv, UG * (self.e[:, None] * inv)], dim=1)
        )
        return out[:, :m], out[:, m:]

    def coupling(self, c, K1):
        """Per shift in ``c``: s = P 1, K s, v = P K 1 (n x len(c)) and n - 1'v
        (in a cancellation-free form)."""
        et, e = self.et.double()[:, None], self.e.double()[:, None]
        inv = 1.0 / (et + c)
        u1 = self.u1[:, None]
        s = self.count.mm(self.U, (u1 * inv).to(self.U.dtype))
        Ks = self.count.mm(self.U, (u1 * e * inv).to(self.U.dtype))
        den0 = (u1**2 * (et - e + c) * inv).sum(0)
        return s, Ks, Ks, den0


class _TruncatedSpectrum:
    """Kt = V T V' + tau (I - V V') >= K from r Ritz pairs of K.

    V, T: q passes of subspace iteration on a random block of r + oversample
    columns, then Rayleigh-Ritz. rho = ||K V - V T|| and tau0, a
    Kuczynski-Wozniakowski bound on the largest eigenvalue of K on the
    complement of V (Lanczos, ``lanczos_steps`` steps, factor 1 / (1 - eps)),
    give T + rho and tau = tau0 + rho, which dominate K. K V is kept, so K P
    costs O(n r)."""

    name = "truncated"

    def __init__(
        self, K, count, rank, passes, oversample, lanczos_steps, lanczos_eps, generator
    ):
        n = K.shape[0]
        k = min(rank + oversample, n)
        omega = torch.randn(n, k, generator=generator, dtype=torch.float64).to(
            K.device, K.dtype
        )
        Q = torch.linalg.qr(count.mm(K, omega)).Q
        for _ in range(passes):
            Q = torch.linalg.qr(count.mm(K, Q)).Q
        KQ = count.mm(K, Q)
        T = Q.T @ KQ
        th, S = torch.linalg.eigh(0.5 * (T + T.T))
        th, S = th[-rank:], S[:, -rank:]
        V = Q @ S
        del Q
        KV = KQ @ S
        del KQ
        # ||K V - V T||: its Gram matrix over row blocks (no n x r residual)
        step = max(1, _ROW_BLOCK // max(rank, 1))
        gram = torch.zeros(rank, rank, dtype=K.dtype, device=K.device)
        for i in range(0, n, step):
            Rk = KV[i : i + step] - V[i : i + step] * th
            gram += Rk.T @ Rk
        del Rk
        rho = math.sqrt(max(float(torch.linalg.eigvalsh(gram.double())[-1]), 0.0))
        rho += 10.0 * torch.finfo(K.dtype).eps * float(th[-1])  # rounding of K V
        tau0 = _lanczos_max(K, V, count, lanczos_steps, generator) / (1.0 - lanczos_eps)
        self.V, self.KV, self.count = V, KV, count
        self.th, self.tau = th + rho, tau0 + rho
        self.v1 = (V.T @ torch.ones(n, dtype=K.dtype, device=K.device)).double()
        self.lmax = max(float(self.th[-1]), self.tau)
        self.info = dict(
            rank=rank,
            passes=passes,
            rho=rho,
            tau0=tau0,
            tau=self.tau,
            theta1=float(th[-1]),
            lmax=self.lmax,
        )

    def apply(self, G, KG, c):
        """P G and K P G; ``c`` (float64) holds one shift per column of G."""
        dt = G.dtype
        tail = (1.0 / (self.tau + c)).to(dt)
        dc = (1.0 / (self.th.double()[:, None] + c) - 1.0 / (self.tau + c)).to(dt)
        W = dc * (self.V.T @ G)
        PG = G * tail + self.V @ W
        KPG = None if KG is None else KG * tail + self.KV @ W
        return PG, KPG

    def weights(self, VtG, c):
        """W of ``apply`` from V'G (the subspace part), for row-tiled steps."""
        dc = 1.0 / (self.th.double()[:, None] + c) - 1.0 / (self.tau + c)
        return dc.to(VtG.dtype) * VtG

    def apply_rows(self, G, KG, W, c, i, j):
        """Rows i .. j of ``apply``, given W = weights(V'G)."""
        tail = (1.0 / (self.tau + c)).to(G.dtype)
        return G * tail + self.V[i:j] @ W, KG * tail + self.KV[i:j] @ W

    def coupling(self, c, K1):
        """Per shift in ``c``: s = P 1, K s, v = P K 1 (n x len(c)) and n - 1'v."""
        m = c.numel()
        one = torch.ones_like(K1)[:, None].expand(-1, m)
        K1m = K1[:, None].expand(-1, m)
        s, Ks = self.apply(one, K1m, c)
        v, _ = self.apply(K1m, None, c)
        s64 = s.double()
        Kt1 = (
            self.tau
            + (
                self.V @ ((self.th.double() - self.tau) * self.v1).to(self.V.dtype)
            ).double()
        )
        den0 = c * s64.sum(0) + (Kt1 - K1.double()) @ s64
        return s, Ks, v, den0


def _lanczos_max(K, V, count, steps, generator):
    """Largest Ritz value of (I - V V') K (I - V V') after ``steps`` Lanczos
    steps with full reorthogonalisation, from a random start."""
    n = K.shape[0]
    q = torch.randn(n, generator=generator, dtype=torch.float64).to(K.device, K.dtype)
    q = q - V @ (V.T @ q)
    q = q / q.norm()
    basis, alphas, betas = [q], [], []
    for _ in range(min(steps, n - V.shape[1])):
        w = count.mm(K, q)
        w = w - V @ (V.T @ w)
        alphas.append(float(q @ w))
        B = torch.stack(basis, dim=1)
        w = w - B @ (B.T @ w)
        w = w - B @ (B.T @ w)
        w = w - V @ (V.T @ w)
        beta = float(w.norm())
        if beta <= 1e-12 * max(abs(alphas[-1]), 1.0):
            break
        betas.append(beta)
        q = w / beta
        basis.append(q)
    m = len(alphas)
    T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    off = torch.tensor(betas[: m - 1], dtype=torch.float64)
    T += torch.diag(off, 1) + torch.diag(off, -1)
    return float(torch.linalg.eigvalsh(T)[-1])


class SpectralSVMPath:
    """Kernel SVM regularization path and K-fold CV, stopped by a duality gap.

    Parameters
    ----------
    K : tensor (n, n) or operator
        Kernel matrix; its dtype and device are the working ones. With the
        truncated spectrum only products ``K @ B`` are used, so ``K`` may also
        be an operator that is never stored, such as
        :class:`torchkm.functions.RBFKernelOperator` (matrix-free).
    y : tensor (n,)
        Labels in {-1, +1}.
    lambdas : sequence of float
        Decreasing penalties; the objective is mean hinge + lam alpha'K alpha.
    foldid : tensor (n,), optional
        Fold of each row, 1..F. Each fold refits with its rows' labels set to
        zero, warm-started from the whole-data fit at the same lambda (as
        cvksvm).
    spectrum : {"truncated", "full"}
        Curvature matrix of the majorization steps.
    rank, passes, oversample, lanczos_steps, lanczos_eps
        Truncated spectrum: Ritz pairs kept, subspace-iteration passes, extra
        block columns, Lanczos steps and the 1 / (1 - eps) factor of the tail
        bound.
    gap_tol : float
        A lambda (or fold) is accepted when its certified relative gap
        (P - D) / P is at most this.
    eps : float
        Loose rounds end when a step's predicted decrease is at most eps times
        the objective (certifying rounds: eps / 1e4). With ``knee``, the
        truncated spectrum uses eps / 10 in loose rounds where
        c = 4 n delta lam is below tau.
    max_rounds, chunk, min_delta, round_cap
        Smoothing rounds delta = 1, 1/8, ... (at most ``round_cap`` iterations
        each) stop shrinking once the smoothing bias is below half the
        tolerance; then iterations run in chunks of ``chunk`` (doubling, up to
        16 x), each followed by the certificate. ``max_rounds`` caps rounds
        and chunks per lambda.
    safeguard : bool
        Check each step (see EIGENDECOMPOSITION_REVIEW_REPLY.md, item 2) and
        fall back to the scalar majorizer sigma I when it fails.
    fit_cap : int
        Iterations, loose and certifying rounds together, after which one
        lambda (or one batch of folds) stops: its certificate is computed
        once more and unfinished columns are reported as not converged.
    warm_levels : int or None
        A warm-started fit (the next lambda, or the folds) starts smoothing
        ``warm_levels`` levels above the delta at which its starting point was
        certified; None starts every fit at delta = 1 (as cvksvm).
    loss : {"hinge", "dwd"}
        The SVM's hinge (smoothed along the schedule above), or kernel DWD
        with q = 1, V(u) = 1 - u for u <= 1/2 and 1 / (4u) above: smooth, so
        its steps run at a fixed width of 1/8 (V'' <= 4) and every round
        certifies, against DWD's dual. Same objective convention: mean loss
        plus lam alpha'K alpha.
    tile : bool or None
        Wide blocks with the truncated spectrum: build every n x m quantity
        except the coefficients, their products with K and the buffers of
        each product row tile by row tile (about 2**24 elements), and never
        store the label matrix. The same arithmetic; peak memory of the
        iterations drops to about six n x m float32 matrices. None: on when
        n x block x (folds + 1) exceeds 2**24.
    bias : float
        Certification starts once the smoothing bias bound, delta / 4 times
        the share of rows inside the smoothing band, is at most ``bias`` times
        ``gap_tol`` times the objective; on a stall, delta shrinks while the
        bound exceeds half of that. The certificate itself is exact, so a
        larger value only starts certifying at a coarser, cheaper delta.
    block : int
        Lambdas fitted together. 1 (serial): the whole-data fit at each lambda
        warm-starts from the previous lambda, and its folds from it. Above 1
        (wide, with ``foldid``): the whole-data fit and every fold of
        ``block`` consecutive lambdas are one set of columns, so each product
        with K serves block x (folds + 1) fits; the whole-data columns start
        from the previous block's last whole-data fit and each fold from the
        same fold's last fit (zeros in the first block). The fitted problems
        and their certificates are the same; only the starting points differ.
        Each block may take ``fit_cap`` x ``block`` iterations.

    After ``fit``: ``alphas`` (n + 1, L; intercepts in row 0), ``gaps`` and
    ``converged`` (L), ``cv_scores`` (n, L), ``cv_error`` (L), ``fold_gaps``
    and ``fold_converged`` (F, L), ``counts`` (iterations, fallbacks, n x n
    products per phase), ``timing`` (seconds per phase) and
    ``spectrum_info``.
    """

    def __init__(
        self,
        K,
        y,
        lambdas,
        foldid=None,
        *,
        spectrum="truncated",
        rank=400,
        passes=4,
        oversample=20,
        lanczos_steps=40,
        lanczos_eps=0.05,
        gap_tol=1e-3,
        eps=1e-5,
        knee=True,
        max_rounds=60,
        chunk=50,
        min_delta=1e-7,
        refine=20,
        safeguard=True,
        eta=0.5,
        armijo=1e-4,
        refresh=50,
        round_cap=5_000,
        fit_cap=20_000,
        max_iter=2_000_000,
        warm_levels=1,
        block=1,
        bias=4.0,
        tile=None,
        loss="hinge",
        seed=0,
    ):
        if spectrum not in ("truncated", "full"):
            raise ValueError("spectrum must be 'truncated' or 'full'")
        if spectrum == "full" and not torch.is_tensor(K):
            raise ValueError(
                "spectrum='full' eigendecomposes K, so it needs the matrix itself; "
                "a kernel operator needs spectrum='truncated'"
            )
        self.K = K
        self.y = torch.as_tensor(y, dtype=K.dtype, device=K.device).reshape(-1)
        self.lambdas = [float(v) for v in lambdas]
        self.foldid = (
            None
            if foldid is None
            else torch.as_tensor(foldid, device=K.device).reshape(-1)
        )
        self.spectrum = spectrum
        self.rank = min(int(rank), K.shape[0] - 1)
        self.passes, self.oversample = int(passes), int(oversample)
        self.lanczos_steps, self.lanczos_eps = int(lanczos_steps), float(lanczos_eps)
        self.gap_tol, self.eps, self.knee = float(gap_tol), float(eps), bool(knee)
        self.max_rounds, self.chunk = int(max_rounds), int(chunk)
        self.min_delta, self.refine = float(min_delta), int(refine)
        self.safeguard, self.eta, self.armijo = (
            bool(safeguard),
            float(eta),
            float(armijo),
        )
        self.refresh, self.round_cap, self.max_iter = (
            int(refresh),
            int(round_cap),
            int(max_iter),
        )
        self.seed = int(seed)
        self.fit_cap = int(fit_cap)
        self.warm_levels = None if warm_levels is None else int(warm_levels)
        if int(block) < 1:
            raise ValueError("block must be at least 1")
        self.block = int(block)
        self.bias = float(bias)
        self.tile = tile
        if loss not in ("hinge", "dwd"):
            raise ValueError("loss must be 'hinge' or 'dwd'")
        self.loss = loss
        self.ridge_b = (
            1e-8  # n eps b^2 keeps the intercept step well defined (cvksvm's vareps)
        )

    # -- timing -------------------------------------------------------------
    def _now(self):
        if self.K.device.type == "cuda":
            torch.cuda.synchronize(self.K.device)
        return time.perf_counter()

    # -- objective pieces ---------------------------------------------------
    def _primal(self, Y, A, b, KA, lam):
        n = Y.shape[0]

        def rows(Y, A, KA):
            r = Y.double() * (KA.double() + b.double())
            return (_margin_loss(r, self.loss) * (Y != 0)).sum(0) / n + lam * (
                A.double() * KA.double()
            ).sum(0)

        return _row_sums(rows, Y, A, KA)

    def _dF(self, R, Y, A, KA, b, db, da, Kda, lam, delta):
        """Change of the smoothed objective along a step, in float64, from the
        step itself (no difference of two large sums)."""
        n = Y.shape[0]

        def rows(R, Y, KA, da, Kda):
            return _dF_rows(R, Y, KA, da, Kda, db, lam, delta, n, self.loss)

        db64 = db.double()
        return _row_sums(rows, R, Y, KA, da, Kda) + n * self.ridge_b * (
            2.0 * b.double() * db64 + db64 * db64
        )

    def _intercept(self, Y, A, b, KA, lam):
        """Brent search of the unsmoothed objective in b (as cvksvm)."""
        n, m = Y.shape
        aka = (A * KA).sum(0)

        def obj(points):
            bb = torch.as_tensor(points, dtype=KA.dtype, device=KA.device)

            def rows(Y, KA):
                return (_margin_loss(Y * (KA + bb), self.loss) * (Y != 0)).sum(0)

            return _row_sums(rows, Y, KA) / n + lam * aka

        old = obj(b.tolist())
        b_new, f_new = brent_minimize_batch(obj, -100.0, 100.0, m)
        b_new = torch.as_tensor(b_new, dtype=b.dtype, device=b.device)
        f_new = torch.as_tensor(f_new, dtype=old.dtype, device=old.device)
        return torch.where(f_new < old, b_new, b)

    # -- the smoothed objective at a batch of points ------------------------
    def _F(self, R, Y, A, KA, b, lam, delta):
        n = Y.shape[0]

        def rows(R, Y, A, KA):
            return _F_rows(R, Y, A, KA, lam, delta, n, self.loss)

        return _row_sums(rows, R, Y, A, KA) + n * self.ridge_b * b.double() ** 2

    def _F_at(self, Y, A, KA, b, lam, delta):
        """F at (A, b) itself (margins built row block by row block)."""
        n = Y.shape[0]

        def rows(Y, A, KA):
            return _F_rows(Y * (KA + b), Y, A, KA, lam, delta, n, self.loss)

        return _row_sums(rows, Y, A, KA) + n * self.ridge_b * b.double() ** 2

    def _band(self, Y, KA, b, delta):
        """Share of the labelled rows inside the smoothing band, per column."""
        n = Y.shape[0]

        def rows(Y, KA):
            r = Y * (KA + b)
            return (((r - 1.0).abs() <= delta) & (Y != 0)).double().sum(0)

        return _row_sums(rows, Y, KA) / n

    # -- iterations at one smoothing level ----------------------------------
    def _round(self, Y, A, b, KA, lam, delta, eps_round, budget):
        """Accelerated MM (FISTA momentum on the majorization step, with a
        function-value restart, so F never increases) at one delta, on the
        columns of A, b, KA in place. A column stops when its step's predicted
        decrease, -g'step, is at most eps_round times F (never when eps_round
        is 0), or after ``budget`` iterations. Each step is
        checked (``safeguard``) and replaced by the scalar-majorizer step when
        the check fails. One product with K per iteration. ``lam`` and
        ``eps_round`` are numbers or one value per column."""
        K, n, count = self.K, self.K.shape[0], self.count
        m = A.shape[1]
        dev, dt = A.device, A.dtype
        lam, eps_round = (
            torch.as_tensor(v, dtype=torch.float64, device=dev).reshape(-1).expand(m)
            for v in (lam, eps_round)
        )
        c = 4.0 * n * delta * lam
        # s, K s, v depend on the column only through lambda: one column per
        # distinct value (a wide block has block lambdas, not block x (F + 1))
        c_u, inv = torch.unique(c, return_inverse=True)
        s, Ks, v, den0 = self.backend.coupling(c_u, self.K1)
        den = (den0[inv] + 4.0 * n * delta * self.ridge_b).to(dt)
        ss64 = 1.0 / (self.backend.lmax + c)
        ss = ss64.to(dt)
        den_s = (
            n - ss64 * float(self.K1.double().sum()) + 4.0 * n * delta * self.ridge_b
        ).to(dt)
        t = torch.ones(m, dtype=torch.float64, device=dev)
        PA, Pb, PKA = A.clone(), b.clone(), KA.clone()
        Fx = self._F(Y * (KA + b), Y, A, KA, b, lam, delta)
        live = torch.ones(m, dtype=torch.bool, device=dev)
        every = torch.arange(m, device=dev)
        iters = 0
        for _ in range(budget):
            cols = torch.nonzero(live).squeeze(1)
            full = cols.numel() == m  # every column live: views, not copies

            def take(T):
                return T if full else T[:, cols]

            Yc, Ac, KAc = take(Y), take(A), take(KA)
            bc, lc, ssc, tc = b[cols], lam[cols], ss[cols], t[cols]
            # one-hot rows: column j of a block uses s[:, inv[j]] (thin products,
            # exact: every other term is a zero)
            onehot = torch.zeros(s.shape[1], cols.numel(), dtype=dt, device=dev)
            onehot[inv[cols], torch.arange(cols.numel(), device=dev)] = 1.0
            t_next = 0.5 + 0.5 * torch.sqrt(1.0 + 4.0 * tc * tc)
            beta = ((tc - 1.0) / t_next).to(dt)
            Ya = Ac + beta * (Ac - take(PA))
            Yb = bc + beta * (bc - Pb[cols])
            YKA = KAc + beta * (KAc - take(PKA))
            R = Yc * (YKA + Yb)
            Z = Yc * _loss_grad(R, delta, self.loss)
            l2 = (2.0 * n * lc).to(dt)
            KG = count.mm(K, Z) + l2 * YKA
            G = Z + l2 * Ya
            gb = Z.sum(0) + 2.0 * n * self.ridge_b * Yb
            del Z
            PG, KPG = self.backend.apply(G, KG, c[cols])
            vG = (v.T @ G).gather(0, inv[cols][None, :]).squeeze(0)
            db = -2.0 * delta * (gb - vG) / den[cols]
            da = PG.mul_(-2.0 * delta).sub_(s @ (onehot * db))
            Kda = KPG.mul_(-2.0 * delta).sub_(Ks @ (onehot * db))
            del PG, KPG
            dF = self._dF(R, Yc, Ya, YKA, Yb, db, da, Kda, lc, delta)
            d_f = (gb * db + (KG * da).sum(0)).double()
            if self.safeguard:
                db_s = -2.0 * delta * (gb - ssc * (self.K1 @ G)) / den_s[cols]
                da_s = -2.0 * delta * ssc * G - ssc * db_s
                Kda_s = -2.0 * delta * ssc * KG - ssc * self.K1[:, None] * db_s
                d_s = (gb * db_s + (KG * da_s).sum(0)).double()
                fast = (d_f <= self.eta * d_s) & (
                    (dF <= self.armijo * d_f) | (d_f.abs() <= 1e-12 * n)
                )
                if not bool(fast.all()):
                    self.counts["fallbacks"] += int((~fast).sum())
                    dF_s = self._dF(R, Yc, Ya, YKA, Yb, db_s, da_s, Kda_s, lc, delta)
                    db = torch.where(fast, db, db_s)
                    da = torch.where(fast, da, da_s)
                    Kda = torch.where(fast, Kda, Kda_s)
                    dF = torch.where(fast, dF, dF_s)
                del da_s, Kda_s
            Fnew = self._F(R, Yc, Ya, YKA, Yb, lc, delta) + dF
            keep = (
                Fnew <= Fx[cols]
            )  # else restart: stay at x, next step without momentum
            del R, G, KG
            if full:  # A, KA are Ac, KAc themselves: copy them out first
                PA.copy_(A)
                PKA.copy_(KA)
                Pb.copy_(b)
                torch.where(keep, Ya.add_(da), A, out=A)
                torch.where(keep, YKA.add_(Kda), KA, out=KA)
            else:
                PA[:, cols], Pb[cols], PKA[:, cols] = Ac, bc, KAc
                A[:, cols] = torch.where(keep, Ya.add_(da), Ac)
                KA[:, cols] = torch.where(keep, YKA.add_(Kda), KAc)
            b[cols] = torch.where(keep, Yb + db, bc)
            del Ya, YKA, da, Kda
            Fx[cols] = torch.where(keep, Fnew, Fx[cols])
            t[cols] = torch.where(keep, t_next, torch.ones_like(t_next))
            iters += 1
            self.counts["column_iterations"] += int(cols.numel())
            if iters % self.refresh == 0:  # tracked K alpha drifts by rounding
                both = count.mm(K, torch.cat([A[:, cols], PA[:, cols]], dim=1))
                KA[:, cols], PKA[:, cols] = (
                    both[:, : cols.numel()],
                    both[:, cols.numel() :],
                )
                Fx[cols] = self._F(
                    Y[:, cols] * (KA[:, cols] + b[cols]),
                    Y[:, cols],
                    A[:, cols],
                    KA[:, cols],
                    b[cols],
                    lc,
                    delta,
                )
            # scale-free stop: the step's predicted decrease against F
            small = -d_f <= eps_round[cols] * Fx[cols].abs()
            live[cols[keep & small]] = False
            if not bool(live.any()) or self.counts["column_iterations"] > self.max_iter:
                break
        return iters

    def _round_tiled(self, Y, A, b, KA, lam, delta, eps_round, budget):
        """``_round`` for a _Labels ``Y`` and the truncated spectrum, with every
        n x m quantity except A, K A, their previous values and the two
        buffers of the product with K built row tile by row tile (sweeps over
        the rows instead of stored temporaries). The same steps, safeguard,
        restarts and stopping rule; sums are accumulated over tiles."""
        K, n, count, be = self.K, self.K.shape[0], self.count, self.backend
        m = A.shape[1]
        dev, dt = A.device, A.dtype
        lam, eps_round = (
            torch.as_tensor(v, dtype=torch.float64, device=dev).reshape(-1).expand(m)
            for v in (lam, eps_round)
        )
        c = 4.0 * n * delta * lam
        c_u, inv = torch.unique(c, return_inverse=True)
        s, Ks, v, den0 = be.coupling(c_u, self.K1)
        den = (den0[inv] + 4.0 * n * delta * self.ridge_b).to(dt)
        ss64 = 1.0 / (be.lmax + c)
        ss = ss64.to(dt)
        den_s = (
            n - ss64 * float(self.K1.double().sum()) + 4.0 * n * delta * self.ridge_b
        ).to(dt)
        t = torch.ones(m, dtype=torch.float64, device=dev)
        PA, Pb, PKA = A.clone(), b.clone(), KA.clone()
        Fx = self._F_at(Y, A, KA, b, lam, delta)
        live = torch.ones(m, dtype=torch.bool, device=dev)
        ridge = 2.0 * n * self.ridge_b
        iters = 0
        for _ in range(budget):
            cols = torch.nonzero(live).squeeze(1)
            mc = cols.numel()
            full = mc == m
            Yl = Y if full else Y.cols(cols)
            bc, lc, ssc, tc, cc = b[cols], lam[cols], ss[cols], t[cols], c[cols]
            onehot = torch.zeros(s.shape[1], mc, dtype=dt, device=dev)
            onehot[inv[cols], torch.arange(mc, device=dev)] = 1.0
            t_next = 0.5 + 0.5 * torch.sqrt(1.0 + 4.0 * tc * tc)
            beta = ((tc - 1.0) / t_next).to(dt)
            Yb = bc + beta * (bc - Pb[cols])
            l2 = (2.0 * n * lc).to(dt)
            tiles = _tiles(n, mc)

            def take(T, i, j):
                return T[i:j] if full else T[i:j][:, cols]

            def point(i, j):
                Yt = Yl.rows(i, j)
                At, KAt = take(A, i, j), take(KA, i, j)
                Ya = At + beta * (At - take(PA, i, j))
                YKA = KAt + beta * (KAt - take(PKA, i, j))
                R = Yt * (YKA + Yb)
                Z = Yt * _loss_grad(R, delta, self.loss)
                return Yt, At, KAt, Ya, YKA, R, Z

            # sweep 1: the input of the product with K
            Zb = torch.empty(n, mc, dtype=dt, device=dev)
            zsum = torch.zeros(mc, dtype=dt, device=dev)
            for i, j in tiles:
                Z = point(i, j)[-1]
                Zb[i:j] = Z
                zsum += Z.sum(0)
            KZ = count.mm(K, Zb)
            del Zb
            gb = zsum + ridge * Yb
            # sweep 2: the thin reductions over the rows
            VtG = torch.zeros(be.V.shape[1], mc, dtype=dt, device=dev)
            vtG = torch.zeros(s.shape[1], mc, dtype=dt, device=dev)
            k1G = torch.zeros(mc, dtype=dt, device=dev)
            for i, j in tiles:
                _, _, _, Ya, _, _, Z = point(i, j)
                G = Z + l2 * Ya
                VtG += be.V[i:j].T @ G
                vtG += v[i:j].T @ G
                k1G += self.K1[i:j] @ G
            W = be.weights(VtG, cc)
            vG = vtG.gather(0, inv[cols][None, :]).squeeze(0)
            db = -2.0 * delta * (gb - vG) / den[cols]
            db_s = -2.0 * delta * (gb - ssc * k1G) / den_s[cols]
            sdb, Ksdb = onehot * db, onehot * db

            def steps(i, j):
                Yt, At, KAt, Ya, YKA, R, Z = point(i, j)
                G, KG = Z + l2 * Ya, KZ[i:j] + l2 * YKA
                PG, KPG = be.apply_rows(G, KG, W, cc, i, j)
                da = PG.mul_(-2.0 * delta).sub_(s[i:j] @ sdb)
                Kda = KPG.mul_(-2.0 * delta).sub_(Ks[i:j] @ Ksdb)
                da_s = -2.0 * delta * ssc * G - ssc * db_s
                Kda_s = -2.0 * delta * ssc * KG - ssc * self.K1[i:j, None] * db_s
                return Yt, At, KAt, Ya, YKA, R, KG, da, Kda, da_s, Kda_s

            # sweep 3: objective changes and predicted decreases of both steps
            dF = dF_s = Fn = None
            kda = kda_s = torch.zeros(mc, dtype=dt, device=dev)
            for i, j in tiles:
                Yt, _, _, Ya, YKA, R, KG, da, Kda, da_s, Kda_s = steps(i, j)
                parts = (
                    _dF_rows(R, Yt, YKA, da, Kda, db, lc, delta, n, self.loss),
                    _dF_rows(R, Yt, YKA, da_s, Kda_s, db_s, lc, delta, n, self.loss),
                    _F_rows(R, Yt, Ya, YKA, lc, delta, n, self.loss),
                )
                kda = kda + (KG * da).sum(0)
                kda_s = kda_s + (KG * da_s).sum(0)
                if dF is None:
                    dF, dF_s, Fn = parts
                else:
                    dF, dF_s, Fn = dF + parts[0], dF_s + parts[1], Fn + parts[2]
            Yb64 = Yb.double()
            dF = dF + n * self.ridge_b * (2.0 * Yb64 * db.double() + db.double() ** 2)
            dF_s = dF_s + n * self.ridge_b * (
                2.0 * Yb64 * db_s.double() + db_s.double() ** 2
            )
            d_f = (gb * db + kda).double()
            if self.safeguard:
                d_s = (gb * db_s + kda_s).double()
                fast = (d_f <= self.eta * d_s) & (
                    (dF <= self.armijo * d_f) | (d_f.abs() <= 1e-12 * n)
                )
                if not bool(fast.all()):
                    self.counts["fallbacks"] += int((~fast).sum())
                    db = torch.where(fast, db, db_s)
                    dF = torch.where(fast, dF, dF_s)
            else:
                fast = torch.ones(mc, dtype=torch.bool, device=dev)
            Fnew = Fn + n * self.ridge_b * Yb64**2 + dF
            keep = Fnew <= Fx[cols]  # else restart: stay at x, no momentum
            # sweep 4: the chosen step, applied tile by tile
            for i, j in tiles:
                _, At, KAt, Ya, YKA, _, _, da, Kda, da_s, Kda_s = steps(i, j)
                da = torch.where(fast, da, da_s)
                Kda = torch.where(fast, Kda, Kda_s)
                newA = torch.where(keep, Ya.add_(da), At)
                newKA = torch.where(keep, YKA.add_(Kda), KAt)
                if full:
                    PA[i:j].copy_(At)
                    PKA[i:j].copy_(KAt)
                    A[i:j].copy_(newA)
                    KA[i:j].copy_(newKA)
                else:
                    PA[i:j, cols], PKA[i:j, cols] = At, KAt
                    A[i:j, cols], KA[i:j, cols] = newA, newKA
            del KZ
            Pb[cols] = bc
            b[cols] = torch.where(keep, Yb + db, bc)
            Fx[cols] = torch.where(keep, Fnew, Fx[cols])
            t[cols] = torch.where(keep, t_next, torch.ones_like(t_next))
            iters += 1
            self.counts["column_iterations"] += mc
            if iters % self.refresh == 0:  # tracked K alpha drifts by rounding
                if full:
                    KA.copy_(count.mm(K, A))
                    PKA.copy_(count.mm(K, PA))
                else:
                    KA[:, cols] = count.mm(K, A[:, cols])
                    PKA[:, cols] = count.mm(K, PA[:, cols])
                Fx[cols] = self._F_at(
                    Yl, A if full else A[:, cols], KA if full else KA[:, cols],
                    b[cols], lc, delta,
                )
            small = -d_f <= eps_round[cols] * Fx[cols].abs()
            live[cols[keep & small]] = False
            if not bool(live.any()) or self.counts["column_iterations"] > self.max_iter:
                break
        return iters

    # -- one lambda, several columns (the whole data, or the folds) ---------
    def _fit_columns(self, Y, A, b, KA, lam, delta=1.0, cap=None):
        """Smoothing rounds delta, delta/8, ..., each solved loosely, until the
        smoothing bias is small: delta / 4 times the share of rows inside the
        smoothing band at most half the tolerance (times the objective). Then
        iterate at that delta in chunks, certifying after each, until every
        column's gap is at most ``gap_tol``; shrink delta when the gaps stop
        improving. All unfinished columns share delta (one batched product per
        step). ``lam`` is a number or one value per column; ``cap`` overrides
        ``fit_cap``."""
        n, m = A.shape
        K, count = self.K, self.count
        lam = torch.as_tensor(lam, dtype=torch.float64, device=A.device).reshape(-1)
        lam = lam.expand(m).clone() if lam.numel() == 1 else lam
        lam_label = float(lam.max())  # for the trace and messages: the largest
        cap = self.fit_cap if cap is None else int(cap)
        tiled = isinstance(Y, _Labels)
        round_fn = self._round_tiled if tiled else self._round
        gap = torch.full((m,), float("inf"), dtype=torch.float64, device=A.device)
        active = torch.ones(m, dtype=torch.bool, device=A.device)
        iters, chunk, certifying = 0, self.chunk, False
        if self.loss == "dwd":  # smooth already: one fixed width, certify at once
            delta, certifying = _DWD_DELTA, True
        eps_cert = self.eps * 1e-4  # early exit of certifying rounds; tightened
        for _ in range(self.max_rounds):
            cols = torch.nonzero(active).squeeze(1)
            full = cols.numel() == m  # every column active: views, not copies
            if full:
                Yc, Ac, KAc = Y, A, KA
            else:
                Yc = Y.cols(cols) if tiled else Y[:, cols]
                Ac, KAc = A[:, cols], KA[:, cols]
            bc, lc = b[cols], lam[cols]
            if not certifying:
                P = self._primal(Yc, Ac, bc, KAc, lc)
                band = self._band(Yc, KAc, bc, delta).clamp_min(1.0 / n)
                certifying = (
                    bool((delta * band / 4.0 <= self.bias * self.gap_tol * P).all())
                    or delta < self.min_delta
                )
            left = max(cap - iters, 0)
            certifying = certifying or left == 0  # out of budget: certify as is
            phase = "certify" if certifying else "loose"
            if not certifying:
                eps_round = torch.full_like(lc, self.eps)
                if self.knee and self.spectrum == "truncated":
                    knee = 4.0 * n * delta * lc < self.backend.tau
                    eps_round = torch.where(knee, 0.1 * eps_round, eps_round)
                used = round_fn(
                    Yc, Ac, bc, KAc, lc, delta, eps_round, min(self.round_cap, left)
                )
            else:
                eps_round = torch.full_like(lc, eps_cert)
                used = round_fn(
                    Yc, Ac, bc, KAc, lc, delta, eps_round, min(chunk, left)
                )
            iters += used
            self.trace.append((lam_label, phase, delta, int(cols.numel()), used))
            if not bool(torch.isfinite(Ac).all() and torch.isfinite(bc).all()):
                raise FloatingPointError(
                    f"non-finite coefficients at lambda {lam_label:g}, delta {delta:g}"
                )
            KAc = count.mm(K, Ac)
            bc = self._intercept(Yc, Ac, bc, KAc, lc)
            if full:
                KA.copy_(KAc)
                KAc = KA
            else:
                A[:, cols], KA[:, cols] = Ac, KAc
            b[cols] = bc
            if not certifying:
                delta /= 8.0
                continue
            t = self._now()
            g, _, _ = (_hinge_duality_gap_tiled if tiled else hinge_duality_gap)(
                _CountedK(K, self.counts["reads"]),
                Yc,
                Ac,
                bc,
                lc,
                Ka=KAc,
                delta=delta,
                refine=self.refine,
                loss=self.loss,
                lmax=self.backend.lmax,
                target=self.gap_tol,
            )
            self.timing["certificate"] += self._now() - t
            stalled = bool((g >= 0.9 * gap[cols]).all())
            if used < chunk:  # the round exited early without certifying
                eps_cert = eps_cert * 1e-2 if eps_cert > 1e-14 else 0.0
            gap[cols] = g
            active[cols] = g > self.gap_tol
            if (
                not bool(active.any())
                or self.counts["column_iterations"] > self.max_iter
                or iters >= cap
            ):
                break
            if stalled and self.loss == "hinge":
                # smooth less only if the smoothing bias could be what limits
                left = active[cols]
                band = self._band(Yc, KAc, bc, delta)
                P = self._primal(Yc, Ac, bc, KAc, lc)
                bias = delta * band.clamp_min(1.0 / n) / 4.0 > 0.5 * self.bias * self.gap_tol * P
                if bool((bias & left).any()) and delta / 8.0 >= self.min_delta:
                    delta, chunk = delta / 8.0, self.chunk
                    continue
            chunk = min(2 * chunk, 64 * self.chunk)
        return gap, ~active, iters, delta

    def _start(self, certified):
        """First smoothing level of a warm-started fit: ``warm_levels`` levels
        above the level where the fit it starts from was certified, or 1."""
        if self.warm_levels is None or certified is None:
            return 1.0
        return min(1.0, certified * 8.0**self.warm_levels)

    # -- wide scheduler: blocks of lambdas, whole data and folds together ----
    def _fit_wide(self, Yf, folds, fold_col):
        """Fit ``block`` lambdas at a time (see ``block``): columns are, per
        lambda, the whole data then the F folds. Path and fold fits share every
        product with K, so their time is all under timing["path"]; the
        held-out scoring is under timing["cross_validation"]. Iterations of a
        block are recorded at its first lambda."""
        K, y = self.K, self.y
        n, dev, dt = K.shape[0], K.device, K.dtype
        F, L = folds.shape[1], len(self.lambdas)
        tiled = self.spectrum == "truncated" and (
            self.tile
            if self.tile is not None
            else n * min(self.block, L) * (F + 1) > _ROW_BLOCK
        )
        if tiled:  # labels built on demand: y, each row's fold, each column's
            cf = torch.arange(-1, F, device=dev)
        else:
            Ycol = torch.cat([y[:, None], Yf], dim=1)  # the F + 1 problems of a lambda
        A = torch.zeros(n, F + 1, dtype=dt, device=dev)  # last solution per problem
        b = torch.zeros(F + 1, dtype=dt, device=dev)
        KA = torch.zeros(n, F + 1, dtype=dt, device=dev)
        d_last = None
        for j0 in range(0, L, self.block):
            js = list(range(j0, min(L, j0 + self.block)))
            B = len(js)
            lam = torch.tensor(
                [self.lambdas[j] for j in js], dtype=torch.float64, device=dev
            ).repeat_interleave(F + 1)
            Yb = _Labels(y, fold_col, cf.repeat(B)) if tiled else Ycol.repeat(1, B)
            Ab, bb, KAb = A.repeat(1, B), b.repeat(B), KA.repeat(1, B)
            t = self._now()
            gap, ok, it, d_last = self._fit_columns(
                Yb, Ab, bb, KAb, lam, self._start(d_last), cap=self.fit_cap * B
            )
            self.timing["path"] += self._now() - t
            t = self._now()
            Az = torch.cat(
                [Ab[:, i * (F + 1) + 1 : (i + 1) * (F + 1)] for i in range(B)], dim=1
            )
            scores = self.count.mm(K, Az.masked_fill(folds.repeat(1, B), 0.0))
            gap, ok = gap.cpu().reshape(B, F + 1), ok.cpu().reshape(B, F + 1)
            for i, j in enumerate(js):
                w = i * (F + 1)
                self.alphas[0, j], self.alphas[1:, j] = bb[w], Ab[:, w]
                self.gaps[j], self.converged[j] = float(gap[i, 0]), bool(ok[i, 0])
                sc = scores[:, i * F : (i + 1) * F] + bb[w + 1 : w + F + 1]
                self.cv_scores[:, j] = sc.gather(1, fold_col[:, None]).squeeze(1)
                self.fold_gaps[:, j], self.fold_converged[:, j] = gap[i, 1:], ok[i, 1:]
                self.path_iterations[j] = it if i == 0 else 0
                self.cv_iterations[j] = 0
            self.timing["cross_validation"] += self._now() - t
            w = (B - 1) * (F + 1)  # the next block starts from this block's last lambda
            A, b, KA = Ab[:, w:].clone(), bb[w:].clone(), KAb[:, w:].clone()

    # -- the whole fit ------------------------------------------------------
    def fit(self):
        K, y = self.K, self.y
        n, dev, dt = K.shape[0], K.device, K.dtype
        self.count = _Counter()
        self.counts = dict(column_iterations=0, fallbacks=0, reads=dict(certificate=0))
        self.trace = []  # (lambda, phase, delta, columns, iterations) per round
        self.timing = dict(
            factorization=0.0, path=0.0, cross_validation=0.0, certificate=0.0
        )
        gen = torch.Generator().manual_seed(self.seed)
        t = self._now()
        if self.spectrum == "full":
            self.backend = _FullSpectrum(K, self.count)
        else:
            self.backend = _TruncatedSpectrum(
                K,
                self.count,
                self.rank,
                self.passes,
                self.oversample,
                self.lanczos_steps,
                self.lanczos_eps,
                gen,
            )
        self.K1 = self.count.mm(K, torch.ones(n, dtype=dt, device=dev))
        self.timing["factorization"] = self._now() - t
        self.counts["reads"]["factorization"] = self.count.reads
        self.spectrum_info = dict(self.backend.info, spectrum=self.spectrum)

        L = len(self.lambdas)
        self.alphas = torch.zeros(n + 1, L, dtype=dt, device=dev)
        self.gaps = torch.full((L,), float("nan"), dtype=torch.float64)
        self.converged = torch.zeros(L, dtype=torch.bool)
        self.path_iterations = [0] * L
        folds = None
        if self.foldid is not None:
            ids = sorted(int(v) for v in torch.unique(self.foldid).tolist())
            folds = torch.stack(
                [self.foldid == f for f in ids], dim=1
            )  # held-out masks, n x F
            Yf = y[:, None].expand(n, len(ids)).clone()
            Yf[folds] = 0.0
            F = len(ids)
            self.cv_scores = torch.zeros(n, L, dtype=dt, device=dev)
            self.fold_gaps = torch.full((F, L), float("nan"), dtype=torch.float64)
            self.fold_converged = torch.zeros(F, L, dtype=torch.bool)
            self.cv_iterations = [0] * L
            fold_col = torch.argmax(folds.to(torch.int64), dim=1)
        reads0 = self.count.reads
        if folds is not None and self.block > 1:
            self._fit_wide(Yf, folds, fold_col)
            self.counts["reads"]["iterations"] = self.count.reads - reads0
            wrong = (torch.where(self.cv_scores > 0, 1.0, -1.0) != y[:, None]).to(
                torch.float64
            )
            self.cv_error = wrong.mean(0).cpu()
            return self
        A = torch.zeros(n, 1, dtype=dt, device=dev)
        b = torch.zeros(1, dtype=dt, device=dev)
        KA = torch.zeros(n, 1, dtype=dt, device=dev)
        d_last = None
        for j, lam in enumerate(self.lambdas):
            t = self._now()
            gap, ok, it, d_path = self._fit_columns(
                y[:, None], A, b, KA, lam, self._start(d_last)
            )
            d_last = d_path
            self.timing["path"] += self._now() - t
            self.alphas[0, j], self.alphas[1:, j] = b[0], A[:, 0]
            self.gaps[j], self.converged[j], self.path_iterations[j] = (
                float(gap[0]),
                bool(ok[0]),
                it,
            )
            if folds is None:
                continue
            t = self._now()
            Af = A.expand(n, F).clone()
            bf = b.expand(F).clone()
            KAf = KA.expand(n, F).clone()
            gap, ok, it, _ = self._fit_columns(
                Yf, Af, bf, KAf, lam, self._start(d_path)
            )
            Az = Af.masked_fill(folds, 0.0)
            scores = self.count.mm(K, Az) + bf
            self.cv_scores[:, j] = scores.gather(1, fold_col[:, None]).squeeze(1)
            self.fold_gaps[:, j], self.fold_converged[:, j] = gap.cpu(), ok.cpu()
            self.cv_iterations[j] = it
            self.timing["cross_validation"] += self._now() - t
        self.counts["reads"]["iterations"] = self.count.reads - reads0
        if folds is not None:
            wrong = (torch.where(self.cv_scores > 0, 1.0, -1.0) != y[:, None]).to(
                torch.float64
            )
            self.cv_error = wrong.mean(0).cpu()
        return self
