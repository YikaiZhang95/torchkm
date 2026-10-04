# SPDX-License-Identifier: MIT
import warnings

import torch

from .exceptions import ConvergenceWarning
from .functions import *
from .functions import brent_minimize
from .memory import kernel_eigh

# First smoothing bandwidth of the check loss. The cross-validation reuses the
# preconditioners the path built for each bandwidth, so both start here.
DELTA_INIT = 0.125


class cvkqr:
    """
    Kernel quantile regression with Regularization and Acceleration.

    This function initializes the optimization process for a kernel quantile regression model,
    supporting advanced features like GPU acceleration and iterative projection methods
    for large-scale data.

    Parameters
    ----------
    Kmat : ndarray or tensor
        The kernel matrix of shape (n_samples, n_samples).

    y : ndarray or tensor
        Target values for each sample, of shape (n_samples,).

    nlam : int
        The number of regularization parameters to consider in the optimization.

    ulam : ndarray or tensor
        User-specified regularization parameters, of shape (nlam,).

    tau : float or tensor
        Quantile level, in (0, 1).

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
        Regularization parameter for kernel methods.

    is_exact : int, default=0
        Indicates whether projection step is used (1 for exact, 0 for approximate).

    delta_len : int, default=4
        Length of delta vector used in projection steps.

    mproj : int, default=2
        Number of projection steps to perform for iterative optimization.

    KKTeps : float, default=1e-3
        Tolerance for KKT conditions in the primary optimization problem.

    KKTeps2 : float, default=1e-3
        Tolerance for KKT conditions in secondary checks.

        ``KKTeps`` and ``KKTeps2`` apply with ``is_exact=1`` only. With
        ``is_exact=0`` every lambda and every fold stops at the certified
        relative duality gap ``gap_tol`` instead (see Notes).

    gap_tol : float, default=1e-3
        ``is_exact=0``: a lambda (and each fold) is accepted when its relative
        duality gap ``(P - D) / P`` is at most this, where ``P`` is the
        unsmoothed objective and ``D`` the dual value of a feasible dual point
        built from the iterate, so ``(P - D) / P`` bounds the relative
        suboptimality. Otherwise the bandwidth is divided by 8 while the
        smoothing bias could be what limits the gap (at most ``delta_len``
        bandwidths), and a bandwidth whose solve did not reach ``gap_tol`` is
        solved again with ``eps`` divided by 100, up to ``max_tighten``
        times. Fits that end above ``gap_tol`` are listed in ``converged`` /
        ``fold_converged`` with their gaps (``gaps``, ``fold_gaps``) and
        reported in one ``ConvergenceWarning``.

    max_tighten : int, default=0
        ``is_exact=0``: re-solves with a 100 times smaller ``eps`` allowed per
        lambda (and per batch of folds) to reach ``gap_tol``. Each costs
        iterations; on cpusmall (n = 1,000) six of them certify every lambda
        but take about 20 times as long, with the same held-out predictions to
        three decimals, so the default reports the gaps instead.

    kkt_scaled : bool, default=False
        Scale-aware KKT stopping rule: compare ``n * sum(KKT**2)`` (the squared
        residual in units of its natural scale ``1/n``) with ``KKTeps`` instead of
        the absolute ``sum(KKT**2)``. With the default rule the threshold gets
        easier to meet as ``n`` grows; with ``kkt_scaled=True`` a given ``KKTeps``
        means the same relative accuracy at every ``n``.

    device : {'cuda', 'cpu'}, default=None
        Device to perform computations on. Defaults to 'cuda' if available, else 'cpu'.

    rebuild_kmat : callable, optional
        Returns ``Kmat`` again, with the same values. When given, the
        eigendecomposition overwrites ``Kmat``'s storage with the eigenvectors
        instead of factorizing a copy, which lowers the peak by one ``n x n``
        matrix, and ``Kmat`` is rebuilt with it afterwards. The fit is the same.
        The estimators pass their kernel construction here.

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
    Stopping (``is_exact=0``). The fit at each lambda minimises
    ``(lam / 2) a'Ka + mean(rho_tau(y - Ka - b))`` (plus ``1e-8 b^2``) through
    smoothed check losses of bandwidth ``delta``. Its dual is
    ``D(theta) = theta'y - theta'K theta / (2 lam)`` over
    ``(tau - 1) / n <= theta_i <= tau / n``, ``sum(theta) = 0`` (``theta_i = 0``
    on a fold's held-out rows); weak duality gives ``D <= P* <= P``. The dual
    point is the better of the smoothed loss's derivative and ``lam * a``,
    each projected onto that set. Before, the stopping test was a KKT residual
    of the unsmoothed loss with the subgradient of every row fixed by the sign
    of its residual; rows that the fit interpolates leave that residual
    nonzero at the optimum, so the test was never met, every lambda ran every
    bandwidth, and the solves stopped on an absolute step size that is loose
    when ``a`` is large (small ``lam``).

    This implementation is designed for large-scale data problems and leverages GPU
    acceleration for improved computational efficiency. Regularization is controlled
    through multiple hyperparameters, allowing fine-tuned trade-offs between accuracy
    and computational cost.

    Examples
    --------
    >>> from torchkm.cvkqr import cvkqr
    >>> from torchkm.functions import *
    >>> import torch
    >>> import numpy
    >>> nn = 1000 # Number of samples
    >>> pp = 10  # Number of features
    >>> sdn = 42  # Seed for reproducibility

    >>> nlam = 50
    >>> torch.manual_seed(sdn)
    >>> ulam = torch.logspace(3, -3, steps=nlam)

    >>> X_train = torch.randn(nn, pp)
    >>> y_train = X_train[:, 0] + 0.1 * torch.randn(nn)
    >>> X_train = standardize(X_train)

    >>> sig = sigest(X_train)
    >>> Kmat = rbf_kernel(X_train, sig)

    >>> torch.manual_seed(sdn)
    >>> nfolds = 10
    >>> if nfolds == nn:
    >>>     foldid = torch.arange(nn)
    >>> else:
    >>>     foldid = torch.randperm(nn) % nfolds + 1
    >>> model = cvkqr(Kmat=Kmat, y=y_train, nlam=nlam, ulam=ulam, tau=0.5, nfolds=nfolds, eps=1e-5, maxit=100000, gamma=1e-8, is_exact=0, device='cuda')
    >>> model.fit()
    """

    def __init__(
        self,
        Kmat,
        y,
        nlam,
        ulam,
        tau,
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
        device=None,
        kkt_scaled=False,
        rebuild_kmat=None,
        gap_tol=1e-3,
        max_tighten=0,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.rebuild_kmat = rebuild_kmat

        # --- Check Kmat ---
        if not isinstance(Kmat, torch.Tensor):
            raise TypeError("Kmat must be a torch.Tensor")
        Kmat = Kmat.double().to(self.device)
        self.Kmat = Kmat
        self.nobs = Kmat.shape[0]

        if not isinstance(y, torch.Tensor):
            raise TypeError("y must be a torch.Tensor")
        y = y.double().to(self.device)
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
                foldid = torch.arange(self.nobs)
            else:
                foldid = torch.randperm(self.nobs) % nfolds + 1
            foldid = foldid.to(self.device)

        # --- Shape check ---
        if Kmat.shape[0] != Kmat.shape[1]:
            raise ValueError("Kmat must be a square matrix")
        if Kmat.shape[0] != y.shape[0]:
            raise ValueError("Kmat and y size mismatch")

        self.nlam = nlam
        self.ulam = ulam.double()
        self.tau = tau
        self.eps = eps
        self.maxit = maxit
        self.gamma = gamma
        self.is_exact = is_exact
        self.delta_len = delta_len
        self.mproj = mproj
        self.KKTeps = KKTeps
        self.KKTeps2 = KKTeps2
        self.gap_tol = float(gap_tol)
        self.max_tighten = int(max_tighten)
        self.kkt_scaled = bool(kkt_scaled)
        # Each KKT entry has natural scale 1/n; the scale-aware rule compares
        # the squared norm in units of (1/n)^2 so KKTeps means the same relative
        # accuracy at every n (see docs/user_guide/model_selection.md).
        self._kkt_scale = float(self.nobs) if self.kkt_scaled else 1.0
        self.nfolds = nfolds
        self.nmaxit = self.nlam * self.maxit
        self.foldid = foldid

        # Initialize outputs
        self.alpmat = torch.zeros((self.nobs + 1, self.nlam), dtype=torch.double).to(
            self.device
        )
        self.anlam = 0
        self.npass = torch.zeros(self.nlam, dtype=torch.int32).to(self.device)
        self.cvnpass = torch.zeros(self.nlam, dtype=torch.int32).to(self.device)
        self.pred = torch.zeros((self.nobs, self.nlam), dtype=torch.double).to(
            self.device
        )
        self.jerr = 0

    def fit(self):
        nobs = self.nobs
        nlam = self.nlam
        y = self.y
        Kmat = self.Kmat
        nfolds = self.nfolds
        tau = self.tau

        r = torch.zeros(nobs, dtype=torch.double).to(self.device)
        alpmat = torch.zeros((nobs + 1, nlam), dtype=torch.double).to(self.device)
        # Iterations per lambda, counted on the host: reading a device counter
        # on every iteration would make the CPU wait for the GPU each time.
        npass = [0] * nlam
        cvnpass = [0] * nlam
        alpvec = torch.zeros(nobs + 1, dtype=torch.double).to(self.device)
        pred = torch.zeros((self.nobs, self.nlam), dtype=torch.double).to(self.device)
        jerr = 0
        eps2 = 1.0e-5
        one = torch.ones((), dtype=torch.double, device=self.device)
        step_buf = torch.empty(nobs + 1, dtype=torch.double, device=self.device)

        # Given rebuild_kmat, the eigenvectors overwrite Kmat's storage (one
        # n x n copy less at the peak) and Kmat is rebuilt for the rest of the fit.
        eigens, Umat = kernel_eigh(Kmat, overwrite=self.rebuild_kmat is not None)
        if self.rebuild_kmat is not None:
            Kmat = self.rebuild_kmat().double().to(self.device)
            self.Kmat = Kmat
        eigens = eigens.double().to(self.device)
        Umat = Umat.double().to(self.device)
        Kmat = Kmat.double().to(self.device)
        eigens += self.gamma
        Usum = torch.sum(Umat, dim=0)
        einv = 1 / eigens
        # K^{-1} = U diag(einv) U^T is applied on the fly in the projection
        # step; no extra n x n matrix is materialised.

        vareps = 1.0e-8

        lpUsum = torch.zeros(
            (nobs, self.delta_len), dtype=torch.double, device=self.device
        )
        lpinv = torch.zeros(
            (nobs, self.delta_len), dtype=torch.double, device=self.device
        )
        svec = torch.zeros(
            (nobs, self.delta_len), dtype=torch.double, device=self.device
        )
        vvec = torch.zeros(
            (nobs, self.delta_len), dtype=torch.double, device=self.device
        )
        gval = torch.zeros((self.delta_len), dtype=torch.double, device=self.device)
        # is_exact=0: bandwidth caches shared by the path and the folds, filled
        # per lambda on first use (saved[0] = levels filled), certified gaps
        saved = [0]
        caches = (eigens, Umat, Usum, lpinv, lpUsum, svec, vvec, gval, saved)
        gaps = torch.full((nlam,), float("nan"), dtype=torch.double)
        fold_gaps = torch.full((nfolds, nlam), float("nan"), dtype=torch.double)
        fold_conv = torch.zeros((nfolds, nlam), dtype=torch.bool)

        for l in range(nlam):
            al = self.ulam[l].item()
            delta = DELTA_INIT
            delta_id = 0
            delta_save = 0
            oldalpvec = torch.zeros(nobs + 1, dtype=torch.double).to(self.device)

            if self.is_exact == 0:  # stopped at the certified duality gap
                saved[0] = 0  # the caches depend on lambda
                # every lambda (and its folds) from DELTA_INIT: the steps scale
                # with delta, so the coarse bandwidths cover the distance from
                # a warm start fastest (starting the folds at the bandwidth
                # where the path ended took twice as long on cpusmall)
                alpvec, r, gaps[l], _ = self._path_lambda_gap(
                    l, al, alpvec, Kmat, y, caches, npass, step_buf
                )

            while self.is_exact != 0 and delta_id < self.delta_len:
                delta_id += 1

                if delta_id > delta_save:
                    lpinv[:, delta_id - 1] = 1.0 / (
                        eigens + 2.0 * float(nobs) * delta * al
                    )
                    lpUsum[:, delta_id - 1] = lpinv[:, delta_id - 1] * Usum
                    vvec[:, delta_id - 1] = torch.mv(
                        Umat, eigens * lpUsum[:, delta_id - 1]
                    )
                    svec[:, delta_id - 1] = torch.mv(Umat, lpUsum[:, delta_id - 1])
                    gval[delta_id - 1] = 1.0 / (
                        nobs + 4.0 * nobs * delta * vareps - vvec[:, delta_id - 1].sum()
                    )
                    delta_save = delta_id

                told = 1.0
                ka = torch.mv(Kmat, alpvec[1:])
                r = y - (alpvec[0] + ka)

                for iteration in range(self.maxit):
                    zvec = torch.where(
                        r < -delta,
                        -(tau - 1.0),
                        torch.where(r > delta, -tau, -r / (2.0 * delta) - tau + 0.5),
                    )
                    gamvec = zvec + float(nobs) * al * alpvec[1:]
                    rds = zvec.sum() + 2.0 * nobs * vareps * alpvec[0]
                    hval = rds - torch.dot(vvec[:, delta_id - 1], gamvec)

                    tnew = 0.5 + 0.5 * torch.sqrt(
                        torch.tensor(1.0, device=self.device) + 4.0 * told * told
                    )
                    mul = 1.0 + (told - 1.0) / tnew
                    told = tnew.item()

                    if delta_id > self.delta_len:
                        print("Exceeded maximum delta_id")
                        break

                    step_buf[0] = -2.0 * mul * delta * gval[delta_id - 1] * hval
                    step_buf[1:] = -step_buf[0] * svec[
                        :, delta_id - 1
                    ] - 2.0 * mul * delta * torch.mv(
                        Umat, gamvec @ Umat * lpinv[:, delta_id - 1]
                    )
                    alpvec += step_buf

                    ka = torch.mv(Kmat, alpvec[1:])
                    r = y - (alpvec[0] + ka)
                    npass[l] += 1

                    if torch.max(step_buf**2) < (self.eps * mul * mul):
                        break

                    if sum(npass) > self.maxit:
                        jerr = -l - 1
                        break

                # Check KKT conditions
                dif_step = oldalpvec - alpvec
                ka = torch.mv(Kmat, alpvec[1:])
                aka = torch.dot(ka, alpvec[1:])

                obj_value = self.objfun(alpvec[0], aka, ka, y, al, nobs, tau, 1e-9)
                golden_s = self.golden_section_search(
                    -100.0, 100.0, nobs, ka, aka, y, al, tau, 1e-9
                )
                int_new = golden_s[0]
                obj_value_new = golden_s[1]
                if obj_value_new < obj_value:
                    dif_step[0] = dif_step[0] + int_new - alpvec[0]
                    r = r - (int_new - alpvec[0])
                    alpvec[0] = int_new

                oldalpvec = alpvec.clone()

                zvec = torch.where(
                    r <= -1e-9,
                    -(tau - 1.0),
                    torch.where(r >= 1e-9, -tau, -r / (2.0 * 1e-9) - tau + 0.5),
                )
                cvec = torch.zeros((nobs + 1), dtype=torch.double, device=self.device)
                dvec = torch.zeros((nobs + 1), dtype=torch.double, device=self.device)
                cvec[0] = zvec.sum()
                cvec[1:] = torch.mv(Kmat, zvec)
                dvec[0] = 2 * vareps * alpvec[0]
                dvec[1:] = al * torch.mv(Kmat, alpvec[1:])
                KKT = cvec / float(nobs) + dvec
                uo = max(al, 1.0)
                KKT_norm = self._kkt_scale * torch.sum(KKT**2) / (uo**2)

                if KKT_norm < self.KKTeps:
                    dif_norm = torch.max(dif_step**2)
                    if dif_norm < float(nobs) * (self.eps * mul * mul):
                        if self.is_exact == 0:
                            break
                        else:
                            is_exit = False
                            alptmp = alpvec.clone()
                            for nn in range(self.mproj):
                                rmg = r
                                elbowid = torch.abs(rmg) < delta
                                elbchk = torch.all(rmg[elbowid] <= 1e-3).item()

                                if elbchk:
                                    break

                                told = 1.0
                                for _ in range(self.maxit):
                                    ka = torch.mv(Kmat, alptmp[1:])
                                    aKa = torch.dot(ka, alptmp[1:])

                                    obj_value = self.objfun(
                                        alptmp[0], aKa, ka, y, al, nobs, tau, 1e-9
                                    )
                                    golden_s = self.golden_section_search(
                                        -100.0, 100.0, nobs, ka, aKa, y, al, tau, 1e-9
                                    )
                                    int_new = golden_s[0]
                                    obj_value_new = golden_s[1]
                                    if obj_value_new < obj_value:
                                        dif_step[0] = dif_step[0] + int_new - alptmp[0]
                                        alptmp[0] = int_new

                                    r = y - (alptmp[0] + ka)
                                    zvec = torch.where(
                                        r < -delta,
                                        -(tau - 1.0),
                                        torch.where(
                                            r > delta,
                                            -tau,
                                            -r / (2.0 * delta) - tau + 0.5,
                                        ),
                                    )
                                    gamvec = zvec + float(nobs) * al * alptmp[1:]
                                    rds = zvec.sum() + 2.0 * nobs * vareps * alptmp[0]
                                    hval = rds - torch.dot(
                                        vvec[:, delta_id - 1], gamvec
                                    )

                                    tnew = 0.5 + 0.5 * torch.sqrt(
                                        torch.tensor(1.0, device=self.device)
                                        + 4.0 * told * told
                                    )
                                    mul = 1.0 + (told - 1.0) / tnew
                                    told = tnew.item()

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
                                    r = y - (alptmp[0] + ka)
                                    npass[l] += 1
                                    alp_old = alptmp.clone()

                                    if torch.sum(elbowid).item() > 1:
                                        theta = torch.mv(Kmat, alptmp[1:])
                                        theta[elbowid] += r[elbowid]
                                        alptmp[1:] = torch.mv(
                                            Umat, einv * torch.mv(Umat.T, theta)
                                        )

                                    dif_step = dif_step + alptmp - alp_old
                                    r = y - (alptmp[0] + torch.mv(Kmat, alptmp[1:]))
                                    mdd = torch.max(dif_step**2)
                                    if mdd < self.eps * mul**2:
                                        break
                                    elif mdd > nobs and npass[l] > 2:
                                        is_exit = True
                                        break
                                    if sum(npass) > self.maxit:
                                        is_exit = True
                                        break

                            if is_exit:
                                break
                            zvec = torch.where(
                                r <= -1e-9,
                                -(tau - 1.0),
                                torch.where(
                                    r >= 1e-9, -tau, -r / (2.0 * 1e-9) - tau + 0.5
                                ),
                            )
                            cvec[0] = zvec.sum()
                            cvec[1:] = torch.mv(Kmat, zvec)
                            dvec[0] = 2 * vareps * alptmp[0]
                            dvec[1:] = al * torch.mv(Kmat, alptmp[1:])
                            KKT = cvec / float(nobs) + dvec
                            uo = max(al, 1.0)

                            if (
                                self._kkt_scale * torch.sum(KKT**2) / (uo**2)
                                < self.KKTeps
                            ):
                                alpvec = alptmp.clone()
                                break

                if delta_id >= self.delta_len:
                    print(f"Exceeded maximum delta iterations for lambda {l}")
                    break
                delta *= 0.125

            # Save the alpha vector for current lambda
            alpmat[:, l] = alpvec
            self.anlam = l

            # Check if maximum iterations exceeded (is_exact=0 caps each
            # lambda at maxit instead and reports it as not converged)
            if self.is_exact != 0 and sum(npass) > self.maxit:
                self.jerr = -l - 1
                break

            ######### cross-validation
            if self.is_exact == 0:
                pred[:, l], fg, fc = self._cv_batched_lambda(
                    Kmat=Kmat,
                    y=y,
                    alpvec=alpvec,
                    al=al,
                    nobs=nobs,
                    nfolds=nfolds,
                    caches=caches,
                    cvnpass=cvnpass,
                    l=l,
                    one=one,
                    tau=tau,
                )
                fold_gaps[:, l], fold_conv[:, l] = fg.cpu(), fc.cpu()
                self.anlam = l
                continue

            for nf in range(nfolds):
                # Unlike a margin loss, the check loss of a row does not vanish
                # when its y is zeroed, so held-out rows are masked explicitly.
                held = self.foldid == (nf + 1)
                yn = y.clone()
                yn[held] = 0.0

                loor = r.clone()
                looalp = alpvec.clone()
                delta = DELTA_INIT
                delta_id = 0

                while True:
                    delta_id += 1

                    if delta_id > delta_save:
                        lpinv[:, delta_id - 1] = 1.0 / (
                            eigens + 2.0 * float(nobs) * delta * al
                        )
                        lpUsum[:, delta_id - 1] = lpinv[:, delta_id - 1] * Usum
                        vvec[:, delta_id - 1] = torch.mv(
                            Umat, eigens * lpUsum[:, delta_id - 1]
                        )
                        svec[:, delta_id - 1] = torch.mv(Umat, lpUsum[:, delta_id - 1])
                        gval[delta_id - 1] = 1.0 / (
                            nobs
                            + 4.0 * nobs * delta * vareps
                            - vvec[:, delta_id - 1].sum()
                        )
                        delta_save = delta_id

                    told = one
                    ka = torch.mv(Kmat, looalp[1:])
                    loor = yn - (looalp[0] + ka)

                    while sum(cvnpass) <= self.nmaxit:
                        zvec = torch.where(
                            loor < -delta,
                            -(tau - 1.0),
                            torch.where(
                                loor > delta,
                                -tau,
                                -loor / (2.0 * delta) - tau + 0.5,
                            ),
                        )
                        zvec[held] = 0.0
                        gamvec = zvec + float(nobs) * al * looalp[1:]
                        rds = zvec.sum() + 2.0 * nobs * vareps * looalp[0]
                        hval = rds - torch.dot(vvec[:, delta_id - 1], gamvec)

                        tnew = 0.5 + 0.5 * torch.sqrt(one + 4.0 * told * told)
                        mul = 1.0 + (told - 1.0) / tnew
                        told = tnew

                        step_buf[0] = -2.0 * mul * delta * gval[delta_id - 1] * hval
                        step_buf[1:] = -step_buf[0] * svec[
                            :, delta_id - 1
                        ] - 2.0 * mul * delta * torch.mv(
                            Umat, gamvec @ Umat * lpinv[:, delta_id - 1]
                        )
                        looalp += step_buf

                        loor = yn - (looalp[0] + torch.mv(Kmat, looalp[1:]))
                        cvnpass[l] += 1

                        if torch.max(step_buf**2) < eps2 * (mul**2):
                            break

                    if sum(cvnpass) > self.nmaxit:
                        break
                    dif_step = step_buf.clone()

                    ka = torch.mv(Kmat, looalp[1:])
                    aka = torch.dot(ka, looalp[1:])

                    obj_value = self.objfun(
                        looalp[0], aka, ka, yn, al, nobs, tau, 1e-9, held_out=held
                    )
                    golden_s = self.golden_section_search(
                        -100.0,
                        100.0,
                        nobs,
                        ka,
                        aka,
                        yn,
                        al,
                        tau,
                        1e-9,
                        held_out=held,
                    )
                    int_new = golden_s[0]
                    obj_value_new = golden_s[1]
                    if obj_value_new < obj_value:
                        dif_step[0] = dif_step[0] + int_new - looalp[0]
                        loor = loor - (int_new - looalp[0])
                        looalp[0] = int_new

                    oldalpvec = looalp.clone()

                    zvec = torch.where(
                        loor <= -1e-9,
                        -(tau - 1.0),
                        torch.where(
                            loor >= 1e-9,
                            -tau,
                            -loor / (2.0 * 1e-9) - tau + 0.5,
                        ),
                    )
                    zvec[held] = 0.0
                    cvec_cv = torch.zeros(
                        (nobs + 1), dtype=torch.double, device=self.device
                    )
                    dvec_cv = torch.zeros(
                        (nobs + 1), dtype=torch.double, device=self.device
                    )
                    cvec_cv[0] = zvec.sum()
                    cvec_cv[1:] = torch.mv(Kmat, zvec)
                    dvec_cv[0] = 2 * vareps * looalp[0]
                    dvec_cv[1:] = al * torch.mv(Kmat, looalp[1:])
                    KKT = cvec_cv / float(nobs) + dvec_cv
                    uo = max(al, 1.0)
                    KKT_norm = self._kkt_scale * torch.sum(KKT**2) / (uo**2)

                    if KKT_norm < self.KKTeps2:
                        if self.is_exact == 0:
                            break
                        else:
                            is_exit = False
                            alptmp = looalp.clone()
                            for nn in range(self.mproj):
                                rmg = loor
                                elbowid = (torch.abs(rmg) < delta) & ~held
                                elbchk = torch.all(rmg[elbowid] <= 1e-2).item()

                                if elbchk:
                                    break

                                told = one
                                for _ in range(self.maxit):
                                    ka = torch.mv(Kmat, alptmp[1:])
                                    aKa = torch.dot(ka, alptmp[1:])

                                    obj_value = self.objfun(
                                        alptmp[0],
                                        aKa,
                                        ka,
                                        yn,
                                        al,
                                        nobs,
                                        tau,
                                        1e-9,
                                        held_out=held,
                                    )
                                    golden_s = self.golden_section_search(
                                        -100.0,
                                        100.0,
                                        nobs,
                                        ka,
                                        aKa,
                                        yn,
                                        al,
                                        tau,
                                        1e-9,
                                        held_out=held,
                                    )
                                    int_new = golden_s[0]
                                    obj_value_new = golden_s[1]
                                    if obj_value_new < obj_value:
                                        dif_step[0] = dif_step[0] + int_new - alptmp[0]
                                        alptmp[0] = int_new

                                    loor = yn - (alptmp[0] + ka)
                                    zvec = torch.where(
                                        loor < -delta,
                                        -(tau - 1.0),
                                        torch.where(
                                            loor > delta,
                                            -tau,
                                            -loor / (2.0 * delta) - tau + 0.5,
                                        ),
                                    )
                                    zvec[held] = 0.0
                                    gamvec = zvec + float(nobs) * al * alptmp[1:]
                                    rds = zvec.sum() + 2.0 * nobs * vareps * alptmp[0]
                                    hval = rds - torch.dot(
                                        vvec[:, delta_id - 1], gamvec
                                    )

                                    tnew = 0.5 + 0.5 * torch.sqrt(
                                        one + 4.0 * told * told
                                    )
                                    mul = 1.0 + (told - 1.0) / tnew
                                    told = tnew

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
                                    loor = yn - (alptmp[0] + ka)
                                    cvnpass[l] += 1
                                    alp_old = alptmp.clone()

                                    if torch.sum(elbowid).item() > 1:
                                        theta = torch.mv(Kmat, alptmp[1:])
                                        theta[elbowid] += loor[elbowid]
                                        alptmp[1:] = torch.mv(
                                            Umat, einv * torch.mv(Umat.T, theta)
                                        )

                                    dif_step = dif_step + alptmp - alp_old
                                    loor = yn - (alptmp[0] + torch.mv(Kmat, alptmp[1:]))
                                    mdd = torch.max(dif_step**2)
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

                loo_ind = self.foldid == (nf + 1)
                looalp[1:][loo_ind] = 0.0
                pred[loo_ind, l] = looalp[1:] @ Kmat[:, loo_ind] + looalp[0]
            self.anlam = l

        self.alpmat = alpmat
        self.npass = torch.tensor(npass, dtype=torch.int32, device=self.device)
        self.cvnpass = torch.tensor(cvnpass, dtype=torch.int32, device=self.device)
        self.jerr = jerr
        self.pred = pred
        if self.is_exact == 0:
            self.gaps, self.fold_gaps = gaps, fold_gaps
            self.converged = gaps <= self.gap_tol
            self.fold_converged = fold_conv
            self._warn_not_converged()

    def _warn_not_converged(self):
        """One ConvergenceWarning for the uncertified fits (is_exact=0)."""
        bad = (~self.converged).nonzero().squeeze(1).tolist()
        bad_f = (~self.fold_converged.all(0)).nonzero().squeeze(1).tolist()
        if bad or bad_f:
            warnings.warn(
                f"cvkqr: {len(bad)} of {self.nlam} whole-data fits and the folds "
                f"of {len(bad_f)} lambda values ended above gap_tol="
                f"{self.gap_tol:g} (largest gaps {float(self.gaps.max()):.2e} and "
                f"{float(self.fold_gaps.max()):.2e}; lambda indices {sorted(set(bad) | set(bad_f))}). "
                "Their solutions, and CV losses, may be inaccurate; raise maxit "
                "or delta_len, or loosen gap_tol.",
                ConvergenceWarning,
            )

    # -- certified duality gap (is_exact=0) -------------------------------
    def _project_dual(self, theta, train):
        """Projection of each column of ``theta`` onto
        {(tau - 1) / n <= theta <= tau / n on train rows, 0 elsewhere,
        sum(theta) = 0}: clip(theta - mu) with mu by bisection."""
        n = self.nobs
        lo_b, hi_b = (self.tau - 1.0) / n, self.tau / n
        reach = theta.abs().amax(dim=0) + 1.0
        lo, hi = -reach, reach.clone()
        for _ in range(64):
            mu = 0.5 * (lo + hi)
            tot = (torch.clamp(theta - mu, lo_b, hi_b) * train).sum(dim=0)
            lo = torch.where(tot > 0, mu, lo)
            hi = torch.where(tot > 0, hi, mu)
        return torch.clamp(theta - 0.5 * (lo + hi), lo_b, hi_b) * train

    def _gap(self, Kmat, y, alp, R, al, delta, held=None):
        """Relative duality gap (P - D) / P of each column of ``alp``
        ((n + 1) x m, intercepts in row 0) with residuals R = y - b - K a,
        the unsmoothed primal P, and the share of training rows within
        ``delta`` of the elbow. ``held`` (n x m, bool) marks held-out rows."""
        n, tau = self.nobs, self.tau
        train = (
            torch.ones_like(R) if held is None else (~held).to(dtype=R.dtype)
        )
        a, b = alp[1:], alp[0]
        Ka = y[:, None] - R - b  # K a, from the residuals (no product)
        loss = (torch.maximum(tau * R, (tau - 1.0) * R) * train).sum(dim=0) / n
        P = (al / 2.0) * (a * Ka).sum(dim=0) + loss + 1e-8 * b * b
        z = torch.where(
            R < -delta,
            -(tau - 1.0),
            torch.where(R > delta, -tau, -R / (2.0 * delta) - tau + 0.5),
        )
        D = None
        sign = torch.where(R > 0, tau, tau - 1.0).to(R.dtype) / n
        for cand in (-z / n, al * a, sign):
            th = self._project_dual(cand * train, train)
            Dc = (th * y[:, None]).sum(dim=0) - (th * torch.mm(Kmat, th)).sum(dim=0) / (
                2.0 * al
            )
            D = Dc if D is None else torch.maximum(D, Dc)
        band = ((R.abs() <= delta).to(R.dtype) * train).sum(dim=0) / train.sum(dim=0)
        # relative to P, but not below an absolute 1e-12: a fit whose
        # objective is zero to rounding (y = 0) is certified, not divided by 0
        return (P - D) / P.abs().clamp_min(1e-12), P, band

    def _ensure_level(self, delta_id, delta, al, caches):
        """Bandwidth caches of level ``delta_id`` (1-based) for this lambda."""
        eigens, Umat, Usum, lpinv, lpUsum, svec, vvec, gval, saved = caches
        if delta_id > saved[0]:
            nobs, vareps = self.nobs, 1.0e-8
            lpinv[:, delta_id - 1] = 1.0 / (eigens + 2.0 * float(nobs) * delta * al)
            lpUsum[:, delta_id - 1] = lpinv[:, delta_id - 1] * Usum
            vvec[:, delta_id - 1] = torch.mv(Umat, eigens * lpUsum[:, delta_id - 1])
            svec[:, delta_id - 1] = torch.mv(Umat, lpUsum[:, delta_id - 1])
            gval[delta_id - 1] = 1.0 / (
                nobs + 4.0 * nobs * delta * vareps - vvec[:, delta_id - 1].sum()
            )
            saved[0] = delta_id

    def _next_level(self, gap, P, band, delta, delta_id, tight):
        """After a solve that left some gap above ``gap_tol``: 'delta' (smooth
        less) when the smoothing bias, delta / 4 times the band share, could be
        what limits it and a bandwidth is left; 'tighten' (eps / 100) when
        tightenings are left (``max_tighten``); else 'stop'."""
        bias = delta * band / 4.0 > 0.5 * self.gap_tol * P.abs()
        open_ = gap > self.gap_tol
        if bool((bias & open_).any()) and delta_id < self.delta_len:
            return "delta"
        return "tighten" if tight < self.max_tighten else "stop"

    def _path_lambda_gap(
        self, l, al, alpvec, Kmat, y, caches, npass, step_buf, start=1
    ):
        """Whole-data fit at one lambda, from bandwidth level ``start``
        (1 = DELTA_INIT), stopped at the certified gap. Returns the fit, its
        residuals, its gap and the level it ended at."""
        nobs, tau, vareps = self.nobs, self.tau, 1.0e-8
        eigens, Umat, Usum, lpinv, lpUsum, svec, vvec, gval, saved = caches
        delta_id, eps_in, tight = start, self.eps, 0
        delta = DELTA_INIT * 0.125 ** (start - 1)
        gap = torch.tensor([float("inf")], dtype=torch.double, device=self.device)
        while True:
            self._ensure_level(delta_id, delta, al, caches)
            told = 1.0
            r = y - (alpvec[0] + torch.mv(Kmat, alpvec[1:]))
            for _ in range(self.maxit):
                zvec = torch.where(
                    r < -delta,
                    -(tau - 1.0),
                    torch.where(r > delta, -tau, -r / (2.0 * delta) - tau + 0.5),
                )
                gamvec = zvec + float(nobs) * al * alpvec[1:]
                rds = zvec.sum() + 2.0 * nobs * vareps * alpvec[0]
                hval = rds - torch.dot(vvec[:, delta_id - 1], gamvec)
                tnew = 0.5 + 0.5 * (1.0 + 4.0 * told * told) ** 0.5
                mul = 1.0 + (told - 1.0) / tnew
                told = tnew
                step_buf[0] = -2.0 * mul * delta * gval[delta_id - 1] * hval
                step_buf[1:] = -step_buf[0] * svec[:, delta_id - 1] - 2.0 * mul * delta * (
                    torch.mv(Umat, gamvec @ Umat * lpinv[:, delta_id - 1])
                )
                alpvec += step_buf
                r = y - (alpvec[0] + torch.mv(Kmat, alpvec[1:]))
                npass[l] += 1
                if torch.max(step_buf**2) < eps_in * mul * mul or npass[l] >= self.maxit:
                    break
            ka = torch.mv(Kmat, alpvec[1:])
            aka = torch.dot(ka, alpvec[1:])
            obj = self.objfun(alpvec[0], aka, ka, y, al, nobs, tau, 1e-9)
            b_new, obj_new = self.golden_section_search(
                -100.0, 100.0, nobs, ka, aka, y, al, tau, 1e-9
            )
            if obj_new < obj:
                alpvec[0] = b_new
            r = y - (alpvec[0] + ka)
            gap, P, band = self._gap(
                Kmat, y, alpvec[:, None], r[:, None], al, delta
            )
            if float(gap[0]) <= self.gap_tol or npass[l] >= self.maxit:
                break
            step = self._next_level(gap, P, band, delta, delta_id, tight)
            if step == "stop":
                break
            if step == "delta":
                delta, delta_id = delta * 0.125, delta_id + 1
            else:
                eps_in, tight = eps_in * 1e-2, tight + 1
        return alpvec, r, float(gap[0]), delta_id

    def _cv_batched_lambda(
        self, *, Kmat, y, alpvec, al, nobs, nfolds, caches, cvnpass, l, one, tau, start=1
    ):
        """The folds at one lambda, batched, each warm-started from the
        whole-data fit at bandwidth level ``start`` (where that fit ended) and
        stopped at its certified gap (held-out rows take no part in the loss,
        the intercept search or the dual). Returns the held-out predictions,
        the fold gaps and whether each is certified."""
        eigens, Umat, Usum, lpinv, lpUsum, svec, vvec, gval, saved = caches
        vareps = 1.0e-8
        fold_ids = torch.arange(1, nfolds + 1, device=self.device)
        fold_masks = self.foldid.unsqueeze(1) == fold_ids.unsqueeze(0)
        fold_col_index = self.foldid.to(dtype=torch.long) - 1
        row_index = torch.arange(nobs, device=self.device)

        alp = alpvec.unsqueeze(1).expand(-1, nfolds).clone()
        step = torch.zeros((nobs + 1, nfolds), dtype=torch.double, device=self.device)
        gaps = torch.full((nfolds,), float("inf"), dtype=torch.double, device=self.device)
        active = torch.ones(nfolds, dtype=torch.bool, device=self.device)
        delta_id, eps_in, tight = start, self.eps, 0
        delta = DELTA_INIT * 0.125 ** (start - 1)
        cap = self.maxit * nfolds
        while True:
            self._ensure_level(delta_id, delta, al, caches)
            cols = torch.nonzero(active, as_tuple=False).squeeze(1)
            told = torch.ones(nfolds, dtype=torch.double, device=self.device)
            R = y.unsqueeze(1) - (alp[0, cols].unsqueeze(0) + torch.mm(Kmat, alp[1:, cols]))
            live = active.clone()
            while bool(live.any()) and cvnpass[l] < cap:
                ic = torch.nonzero(live, as_tuple=False).squeeze(1)
                pos = torch.searchsorted(cols, ic)  # columns of R for ic
                Ri, Ai = R[:, pos], alp[:, ic]
                zvec = torch.where(
                    Ri < -delta,
                    -(tau - 1.0),
                    torch.where(Ri > delta, -tau, -Ri / (2.0 * delta) - tau + 0.5),
                )
                zvec[fold_masks[:, ic]] = 0.0  # held-out rows add no loss
                gamvec = zvec + float(nobs) * al * Ai[1:, :]
                rds = zvec.sum(dim=0) + 2.0 * nobs * vareps * Ai[0, :]
                hval = rds - torch.matmul(vvec[:, delta_id - 1], gamvec)
                tnew = 0.5 + 0.5 * torch.sqrt(one + 4.0 * told[ic] * told[ic])
                mul = 1.0 + (told[ic] - 1.0) / tnew
                told[ic] = tnew
                step[0, ic] = -2.0 * mul * delta * gval[delta_id - 1] * hval
                spectral = torch.mm(Umat.T, gamvec)
                spectral.mul_(lpinv[:, delta_id - 1].unsqueeze(1))
                step[1:, ic] = -step[0, ic].unsqueeze(0) * svec[
                    :, delta_id - 1
                ].unsqueeze(1) - 2.0 * delta * mul.unsqueeze(0) * torch.mm(Umat, spectral)
                alp[:, ic] += step[:, ic]
                R[:, pos] = y.unsqueeze(1) - (
                    alp[0, ic].unsqueeze(0) + torch.mm(Kmat, alp[1:, ic])
                )
                cvnpass[l] += int(ic.numel())
                done = torch.max(step[:, ic] ** 2, dim=0).values < eps_in * mul**2
                live[ic[done]] = False
            # intercepts by the same search as the path, held-out rows excluded
            for j, nf in enumerate(cols.tolist()):
                held = fold_masks[:, nf]
                ka = y - R[:, j] - alp[0, nf]
                aka = torch.dot(ka, alp[1:, nf])
                obj = self.objfun(alp[0, nf], aka, ka, y, al, nobs, tau, 1e-9, held_out=held)
                b_new, obj_new = self.golden_section_search(
                    -100.0, 100.0, nobs, ka, aka, y, al, tau, 1e-9, held_out=held
                )
                if obj_new < obj:
                    R[:, j] -= b_new - alp[0, nf]
                    alp[0, nf] = b_new
            g, P, band = self._gap(
                Kmat, y, alp[:, cols], R, al, delta, held=fold_masks[:, cols]
            )
            gaps[cols] = g
            active[cols] = g > self.gap_tol
            if not bool(active.any()) or cvnpass[l] >= cap:
                break
            left = active[cols]
            nxt = self._next_level(g[left], P[left], band[left], delta, delta_id, tight)
            if nxt == "stop":
                break
            if nxt == "delta":
                delta, delta_id = delta * 0.125, delta_id + 1
            else:
                eps_in, tight = eps_in * 1e-2, tight + 1

        cv_alpha = alp[1:, :].clone()
        cv_alpha[fold_masks] = 0.0
        cv_scores = torch.mm(Kmat, cv_alpha) + alp[0, :].unsqueeze(0)
        return cv_scores[row_index, fold_col_index], gaps, gaps <= self.gap_tol

    def cv(self, pred, y):
        y_expanded = y[:, None]
        residuals = y_expanded - pred
        return cvkqr.check_loss(residuals, self.tau).mean(dim=0)

    @staticmethod
    def check_loss(u, tau):
        return torch.where(u >= 0, tau * u, (tau - 1) * u)

    def predict(self, Kmat_new, y_new, alp_b):
        result = torch.mv(Kmat_new, alp_b[1:]) + alp_b[0]
        return result

    def obj_value(self, alp_b, lam_b):
        intcpt = alp_b[0]
        alp = alp_b[1:]
        Kmat = self.Kmat.double().to(alp.device)
        ka = torch.mv(Kmat, alp)
        aka = torch.dot(alp, ka)
        y_train = self.y.to(alp.device)
        obj = self.objfun(intcpt, aka, ka, y_train, lam_b, self.nobs, self.tau, 1e-9)
        return obj

    def objfun(self, intcpt, aka, ka, y, lam, nobs, tau, delta, held_out=None):
        """
        Compute the objective function value for kernel quantile regression.

        Parameters:
        - intcpt (float): Intercept term.
        - aka (torch.Tensor): Regularization term (alpha * K * alpha).
        - ka (torch.Tensor): Kernel matrix dot alpha vector (K * alpha).
        - y (torch.Tensor): Target values of shape (nobs,).
        - lam (float): Regularization parameter.
        - nobs (int): Number of observations.
        - tau (float): Quantile level.
        - delta (float): Smoothing bandwidth for the quantile loss.
        - held_out (torch.Tensor, optional): Boolean mask of a held-out fold.
          Its rows add no loss; the mean is still over all rows, as in the
          cross-validation problem.

        Returns:
        - objval (float): Objective function value.
        """
        fh = ka + intcpt
        xi_tmp = y - fh
        ttau = tau - 1.0
        xi = torch.where(
            xi_tmp <= -delta,
            xi_tmp * ttau,
            torch.where(
                xi_tmp >= delta,
                xi_tmp * tau,
                xi_tmp**2 / (4.0 * delta) + (tau - 0.5) * xi_tmp + delta / 4.0,
            ),
        )
        if held_out is not None:
            xi = xi.masked_fill(held_out, 0.0)
        objval = (lam / 2.0) * aka + torch.mean(xi) + 1e-8 * intcpt**2
        return objval

    def golden_section_search(
        self, lmin, lmax, nobs, ka, aka, y, lam, tau, delta, held_out=None
    ):
        """Intercept minimising ``objfun`` on [lmin, lmax] by Brent's method
        (``functions.brent_minimize``); returns (intercept, objective) as
        floats."""
        return brent_minimize(
            lambda b: self.objfun(b, aka, ka, y, lam, nobs, tau, delta, held_out),
            lmin,
            lmax,
        )
