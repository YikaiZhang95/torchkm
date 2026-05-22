import torch

from .functions import *


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

    device : {'cuda', 'cpu'}, default=None
        Device to perform computations on. Defaults to 'cuda' if available, else 'cpu'.

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
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

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
        npass = torch.zeros(nlam, dtype=torch.int32).to(self.device)
        cvnpass = torch.zeros(nlam, dtype=torch.int32).to(self.device)
        alpvec = torch.zeros(nobs + 1, dtype=torch.double).to(self.device)
        pred = torch.zeros((self.nobs, self.nlam), dtype=torch.double).to(self.device)
        jerr = 0
        eps2 = 1.0e-5
        one = torch.ones((), dtype=torch.double, device=self.device)
        step_buf = torch.empty(nobs + 1, dtype=torch.double, device=self.device)

        # Precompute sum of Kmat along rows
        Ksum = torch.sum(Kmat, dim=1)

        eigens, Umat = torch.linalg.eigh(Kmat)
        eigens = eigens.double().to(self.device)
        Umat = Umat.double().to(self.device)
        Kmat = Kmat.double().to(self.device)
        eigens += self.gamma
        Usum = torch.sum(Umat, dim=0)
        einv = 1 / eigens
        eU = (einv * Umat).T

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

        for l in range(nlam):
            al = self.ulam[l].item()
            delta = 0.125
            delta_id = 0
            delta_save = 0
            oldalpvec = torch.zeros(nobs + 1, dtype=torch.double).to(self.device)

            while delta_id < self.delta_len:
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

                    if torch.sum(npass) > self.maxit:
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
                KKT_norm = torch.sum(KKT**2) / (uo**2)

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
                                        alptmp[1:] = torch.mv(Umat, torch.mv(eU, theta))

                                    dif_step = dif_step + alptmp - alp_old
                                    r = y - (alptmp[0] + torch.mv(Kmat, alptmp[1:]))
                                    mdd = torch.max(dif_step**2)
                                    if mdd < self.eps * mul**2:
                                        break
                                    elif mdd > nobs and npass[l] > 2:
                                        is_exit = True
                                        break
                                    if torch.sum(npass) > self.maxit:
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

                            if torch.sum(KKT**2) / (uo**2) < self.KKTeps:
                                alpvec = alptmp.clone()
                                break

                if delta_id >= self.delta_len:
                    print(f"Exceeded maximum delta iterations for lambda {l}")
                    break
                delta *= 0.125

            # Save the alpha vector for current lambda
            alpmat[:, l] = alpvec
            self.anlam = l

            # Check if maximum iterations exceeded
            if torch.sum(npass) > self.maxit:
                self.jerr = -l - 1
                break

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
                    l=l,
                    one=one,
                    tau=tau,
                )
                self.anlam = l
                continue

            for nf in range(nfolds):
                yn = y.clone()
                yn[self.foldid == (nf + 1)] = 0.0

                loor = r.clone()
                looalp = alpvec.clone()
                delta = 1.0
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

                    while torch.sum(cvnpass) <= self.nmaxit:
                        zvec = torch.where(
                            loor < -delta,
                            -(tau - 1.0),
                            torch.where(
                                loor > delta,
                                -tau,
                                -loor / (2.0 * delta) - tau + 0.5,
                            ),
                        )
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

                    if torch.sum(cvnpass) > self.nmaxit:
                        break
                    dif_step = step_buf.clone()

                    ka = torch.mv(Kmat, looalp[1:])
                    aka = torch.dot(ka, looalp[1:])

                    obj_value = self.objfun(looalp[0], aka, ka, yn, al, nobs, tau, 1e-9)
                    golden_s = self.golden_section_search(
                        -100.0, 100.0, nobs, ka, aka, yn, al, tau, 1e-9
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
                    KKT_norm = torch.sum(KKT**2) / (uo**2)

                    if KKT_norm < self.KKTeps2:
                        if self.is_exact == 0:
                            break
                        else:
                            is_exit = False
                            alptmp = looalp.clone()
                            for nn in range(self.mproj):
                                rmg = loor
                                elbowid = torch.abs(rmg) < delta
                                elbchk = torch.all(rmg[elbowid] <= 1e-2).item()

                                if elbchk:
                                    break

                                told = one
                                for _ in range(self.maxit):
                                    ka = torch.mv(Kmat, alptmp[1:])
                                    aKa = torch.dot(ka, alptmp[1:])

                                    obj_value = self.objfun(
                                        alptmp[0], aKa, ka, yn, al, nobs, tau, 1e-9
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
                                        alptmp[1:] = torch.mv(Umat, torch.mv(eU, theta))

                                    dif_step = dif_step + alptmp - alp_old
                                    loor = yn - (alptmp[0] + torch.mv(Kmat, alptmp[1:]))
                                    mdd = torch.max(dif_step**2)
                                    if mdd < nobs * eps2 * mul**2:
                                        break
                                    elif mdd > nobs and cvnpass[l] > 2:
                                        is_exit = True
                                        break
                                    if torch.sum(cvnpass) > self.nmaxit:
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
        self.npass = npass
        self.cvnpass = cvnpass
        self.jerr = jerr
        self.pred = pred

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
        l,
        one,
        tau,
    ):
        fold_ids = torch.arange(1, nfolds + 1, device=self.device)
        fold_masks = self.foldid.unsqueeze(1) == fold_ids.unsqueeze(0)
        fold_col_index = self.foldid.to(dtype=torch.long) - 1
        row_index = torch.arange(nobs, device=self.device)

        looalp_batch = alpvec.unsqueeze(1).expand(-1, nfolds).clone()
        loor_batch = r.unsqueeze(1).expand(-1, nfolds).clone()
        cv_step_buf = torch.zeros(
            (nobs + 1, nfolds), dtype=torch.double, device=self.device
        )

        active = torch.ones(nfolds, dtype=torch.bool, device=self.device)
        delta = 1.0
        delta_id = 0

        while torch.any(active):
            delta_id += 1

            if delta_id > delta_save:
                lpinv[:, delta_id - 1] = 1.0 / (eigens + 2.0 * float(nobs) * delta * al)
                lpUsum[:, delta_id - 1] = lpinv[:, delta_id - 1] * Usum
                vvec[:, delta_id - 1] = torch.mv(Umat, eigens * lpUsum[:, delta_id - 1])
                svec[:, delta_id - 1] = torch.mv(Umat, lpUsum[:, delta_id - 1])
                gval[delta_id - 1] = 1.0 / (
                    nobs + 4.0 * nobs * delta * vareps - vvec[:, delta_id - 1].sum()
                )
                delta_save = delta_id

            active_cols = torch.nonzero(active, as_tuple=False).squeeze(1)
            told = torch.ones(nfolds, dtype=torch.double, device=self.device)
            ka_batch = torch.mm(Kmat, looalp_batch[1:, active_cols])
            loor_batch[:, active_cols] = y.unsqueeze(1) - (
                looalp_batch[0, active_cols].unsqueeze(0) + ka_batch
            )

            active_iter = active.clone()
            while torch.any(active_iter):
                iter_cols = torch.nonzero(active_iter, as_tuple=False).squeeze(1)
                loor_iter = loor_batch[:, iter_cols]
                alp_iter = looalp_batch[:, iter_cols]
                told_iter = told[iter_cols]

                zvec = torch.where(
                    loor_iter < -delta,
                    -(tau - 1.0),
                    torch.where(
                        loor_iter > delta,
                        -tau,
                        -loor_iter / (2.0 * delta) - tau + 0.5,
                    ),
                )
                # Zero out fold members' gradient contributions
                zvec[fold_masks[:, iter_cols]] = 0.0
                gamvec = zvec + float(nobs) * al * alp_iter[1:, :]
                rds = zvec.sum(dim=0) + 2.0 * nobs * vareps * alp_iter[0, :]
                hval = rds - torch.matmul(vvec[:, delta_id - 1], gamvec)

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
                loor_batch[:, iter_cols] = y.unsqueeze(1) - (
                    looalp_batch[0, iter_cols].unsqueeze(0) + ka_batch
                )

                cvnpass[l] += iter_cols.numel()
                if torch.sum(cvnpass) > self.nmaxit:
                    break

                converged = torch.max(
                    cv_step_buf[:, iter_cols] ** 2, dim=0
                ).values < eps2 * (mul**2)
                active_iter[iter_cols[converged]] = False

            if torch.sum(cvnpass) > self.nmaxit:
                break

            current_cols = torch.nonzero(active, as_tuple=False).squeeze(1)
            for nf in current_cols.tolist():
                looalp = looalp_batch[:, nf]
                loor = loor_batch[:, nf].clone()
                yn = y.clone()
                yn[self.foldid == (nf + 1)] = 0.0
                dif_step = cv_step_buf[:, nf].clone()

                ka = torch.mv(Kmat, looalp[1:])
                aka = torch.dot(ka, looalp[1:])

                obj_value = self.objfun(looalp[0], aka, ka, yn, al, nobs, tau, 1e-9)
                golden_s = self.golden_section_search(
                    -100.0, 100.0, nobs, ka, aka, yn, al, tau, 1e-9
                )
                int_new = golden_s[0]
                obj_value_new = golden_s[1]
                if obj_value_new < obj_value:
                    dif_step[0] = dif_step[0] + int_new - looalp[0]
                    loor = loor - (int_new - looalp[0])
                    looalp[0] = int_new

                loor_batch[:, nf] = loor
                zvec = torch.where(
                    loor <= -1e-9,
                    -(tau - 1.0),
                    torch.where(loor >= 1e-9, -tau, -loor / (2.0 * 1e-9) - tau + 0.5),
                )
                fold_mask_nf = self.foldid == (nf + 1)
                zvec_kkt = zvec.clone()
                zvec_kkt[fold_mask_nf] = 0.0
                cvec_nf = torch.zeros(nobs + 1, dtype=torch.double, device=self.device)
                dvec_nf = torch.zeros(nobs + 1, dtype=torch.double, device=self.device)
                cvec_nf[0] = zvec_kkt.sum()
                cvec_nf[1:] = torch.mv(Kmat, zvec_kkt)
                dvec_nf[0] = 2 * vareps * looalp[0]
                dvec_nf[1:] = al * torch.mv(Kmat, looalp[1:])
                KKT = cvec_nf / float(nobs) + dvec_nf
                uo = max(al, 1.0)
                KKT_norm = torch.sum(KKT**2) / (uo**2)

                if KKT_norm < self.KKTeps2:
                    active[nf] = False

            if delta_id >= self.delta_len:
                print(f"Exceeded maximum delta iterations for lambda {l}")
                break
            delta *= 0.125

        cv_alpha = looalp_batch[1:, :].clone()
        cv_alpha[fold_masks] = 0.0
        cv_scores = torch.mm(Kmat, cv_alpha) + looalp_batch[0, :].unsqueeze(0)
        return cv_scores[row_index, fold_col_index]

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

    def objfun(self, intcpt, aka, ka, y, lam, nobs, tau, delta):
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
        objval = (lam / 2.0) * aka + torch.mean(xi) + 1e-8 * intcpt**2
        return objval

    def golden_section_search(self, lmin, lmax, nobs, ka, aka, y, lam, tau, delta):
        """
        Optimize the intercept using golden section search (Brent's method).

        Parameters:
        - lmin (float): Lower bound for the search interval.
        - lmax (float): Upper bound for the search interval.
        - nobs (int): Number of observations.
        - ka (torch.Tensor): Kernel matrix dot alpha vector (K * alpha).
        - aka (float): Regularization term (alpha * K * alpha).
        - y (torch.Tensor): Target values of shape (nobs,).
        - lam (float): Regularization parameter.
        - tau (float): Quantile level.
        - delta (float): Smoothing bandwidth for the quantile loss.

        Returns:
        - lhat (float): Optimized intercept value.
        - fx (float): Objective function value at the optimized intercept.
        """
        device = ka.device if isinstance(ka, torch.Tensor) else self.device
        eps = torch.tensor(
            torch.finfo(torch.float64).eps, dtype=torch.double, device=device
        )
        tol = eps**0.25
        tol1 = eps + 1.0
        eps = torch.sqrt(eps)

        # Golden ratio constant
        gold = (
            3.0 - torch.sqrt(torch.tensor(5.0, dtype=torch.double, device=device))
        ) * 0.5

        # Initialize variables
        a = lmin
        b = lmax
        v = a + gold * (b - a)
        w = v
        x = v
        d = 0.0
        e = 0.0

        # Evaluate the objective function at the initial x value
        fx = self.objfun(x, aka, ka, y, lam, nobs, tau, delta)
        fv = fx
        fw = fx
        tol3 = tol / 3.0
        # Main optimization loop
        while True:
            xm = (a + b) * 0.5
            tol1 = eps * abs(x) + tol3
            t2 = 2.0 * tol1

            # Check if the interval is small enough to exit
            if abs(x - xm) <= t2 - (b - a) * 0.5:
                break

            p = 0.0
            q = 0.0
            r = 0.0
            if abs(e) > tol1:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = 2.0 * (q - r)
                if q > 0.0:
                    p = -p
                else:
                    q = -q
                r = e
                e = d
            # Conditions to use golden section step
            if (abs(p) >= abs(0.5 * q * r)) or (p <= q * (a - x)) or (p >= q * (b - x)):
                if x < xm:
                    e = b - x
                else:
                    e = a - x
                d = gold * e
            else:
                # Parabolic interpolation step
                d = p / q
                u = x + d
                if (u - a < t2) or (b - u < t2):
                    d = tol1
                    if x >= xm:
                        d = -d

            # Set the new point u
            u = x + d if abs(d) >= tol1 else (x + tol1 if d > 0 else x - tol1)
            # Evaluate the objective function at u
            fu = self.objfun(u, aka, ka, y, lam, nobs, tau, delta)
            # Update the search bounds and objective values
            if fu <= fx:
                if u < x:
                    b = x
                else:
                    a = x
                v = w
                fv = fw
                w = x
                fw = fx
                x = u
                fx = fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v = w
                    fv = fw
                    w = u
                    fw = fu
                elif fu <= fv or v == x or v == w:
                    v = u
                    fv = fu
        # Return the optimal intercept and the objective value
        lhat = x
        res = self.objfun(x, aka, ka, y, lam, nobs, tau, delta)
        return lhat, res
