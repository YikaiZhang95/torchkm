import torch

from .functions import *


class cvknysqr:
    """
    Nyström-approximated kernel quantile regression with Regularization
    and Acceleration.

    Parameters mirror ``cvkqr`` (kernel quantile regression) but the
    n×n kernel matrix is replaced with low-rank Nyström features built
    from a random landmark subset of the training data, as in
    ``cvknyssvm``.

    Parameters
    ----------
    Xmat : torch.Tensor
        Training feature matrix of shape (n_samples, n_features).
    X_test : torch.Tensor
        Test feature matrix; transformed Nyström features are stored on
        the model after ``fit()``.
    y : torch.Tensor
        Continuous regression target of shape (n_samples,).
    nlam : int
        Number of regularization parameters along the path.
    ulam : torch.Tensor
        User-supplied regularization parameters of shape (nlam,).
    tau : float
        Quantile level in (0, 1).
    foldid : torch.Tensor, optional
        Fold assignment of length n_samples (values 1..nfolds).
    nfolds : int, default=5
        Number of CV folds.
    eps : float, default=1e-5
        Convergence tolerance.
    maxit : int, default=1000
        Maximum number of iterations.
    gamma : float, default=1.0
        Small regularizer added to the Nyström eigenvalues for stability.
    is_exact : int, default=0
        Whether to run the optional projection refinement (currently 0).
    delta_len : int, default=4
        Number of delta-annealing rounds.
    mproj : int, default=2
        Number of projection sub-steps (only used when is_exact=1).
    KKTeps : float, default=1e-3
        Tolerance for KKT condition (training).
    KKTeps2 : float, default=1e-3
        Tolerance for KKT condition (CV).
    num_landmarks : int, default=2000
        Number of Nyström landmarks.
    k : int, default=1000
        Rank of the Nyström feature map.
    device : {'cuda', 'cpu'}, default=None
        Device. Defaults to CUDA if available.

    Attributes
    ----------
    alpmat : torch.Tensor
        Coefficient matrix of shape (k_eff + 1, nlam); row 0 is intercept.
    pred : torch.Tensor
        Out-of-fold predictions of shape (n_samples, nlam).
    npass, cvnpass : torch.Tensor
        Iteration counts per lambda.
    jerr : int
        Error flag (0 = success).
    indices, landmarks_, sig_w_, M_, k_eff_ : Nyström state for transform().
    Z_test : Nyström-transformed test features.
    Z_train : Nyström-transformed training features.
    """

    def __init__(
        self,
        Xmat,
        X_test,
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
        num_landmarks=2000,
        k=1000,
        device=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.nobs = Xmat.shape[0]

        if not isinstance(Xmat, torch.Tensor):
            raise TypeError("Xmat must be a torch.Tensor")
        Xmat = Xmat.double().to(self.device)
        self.Xmat = Xmat

        if not isinstance(y, torch.Tensor):
            raise TypeError("y must be a torch.Tensor")
        y = y.double().to(self.device)
        self.y = y

        if not isinstance(ulam, torch.Tensor):
            raise TypeError("ulam must be a torch.Tensor")
        ulam = ulam.double().to(self.device)

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

        if Xmat.shape[0] != y.shape[0]:
            raise ValueError("Xmat and y size mismatch")

        self.X_test = X_test.double().to(self.device)
        self.np = Xmat.shape[1]
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
        self.num_landmarks = num_landmarks
        self.k = k
        self.nmaxit = self.nlam * self.maxit
        self.nfolds = nfolds
        self.foldid = foldid

        self.alpmat = torch.zeros((self.np + 1, self.nlam), dtype=torch.double).to(
            self.device
        )
        self.anlam = 0
        self.npass = torch.zeros(self.nlam, dtype=torch.int32).to(self.device)
        self.cvnpass = torch.zeros(self.nlam, dtype=torch.int32).to(self.device)
        self.pred = torch.zeros((self.nobs, self.nlam), dtype=torch.double).to(
            self.device
        )
        self.jerr = 0
        self.Z_test = torch.zeros(X_test.shape[0], dtype=torch.double).to(self.device)
        self.Z_train = torch.zeros(Xmat.shape[0], dtype=torch.double).to(self.device)
        self.indices = torch.zeros(self.num_landmarks, dtype=torch.double)
        self.landmarks_ = None
        self.sig_w_ = None
        self.M_ = None
        self.k_eff_ = None

    def fit(self):
        nobs = self.nobs
        nlam = self.nlam
        y = self.y
        Xmat = self.Xmat
        X_test = self.X_test
        num_landmarks = self.num_landmarks
        k = self.k
        nfolds = self.nfolds
        tau = self.tau

        torch.manual_seed(0)
        num_landmarks = min(num_landmarks, nobs)

        indices = torch.randperm(nobs)[:num_landmarks]
        Xmat_work = Xmat.float()
        landmarks = Xmat_work[indices]

        sig_w = sigest(landmarks)
        W = rbf_kernel(landmarks, sig_w)

        evals, evecs = torch.linalg.eigh(W)
        k = min(k, evals.numel())
        evals = evals[-k:].flip(0).clamp_min(torch.finfo(evals.dtype).eps)
        evecs = evecs[:, -k:].flip(1)

        M = evecs * torch.rsqrt(evals)
        self.indices = indices.detach().cpu().to(torch.int64)
        self.landmarks_ = landmarks.detach()
        self.sig_w_ = float(sig_w)
        self.M_ = M.detach()
        self.k_eff_ = int(k)

        Cmat = kernelMult(Xmat_work, landmarks, sig_w)
        Xmat = torch.mm(Cmat, M).double()

        C_test = kernelMult(X_test.float(), landmarks, sig_w)
        Z_test = torch.mm(C_test, M)

        np = Xmat.shape[1]
        r = y.clone()
        kz = torch.zeros(np + 1, dtype=torch.double).to(self.device)
        alpmat = torch.zeros((np + 1, nlam), dtype=torch.double).to(self.device)
        npass = torch.zeros(nlam, dtype=torch.int32).to(self.device)
        cvnpass = torch.zeros(nlam, dtype=torch.int32).to(self.device)
        alpvec = torch.zeros(np + 1, dtype=torch.double).to(self.device)
        pred = torch.zeros((self.nobs, self.nlam), dtype=torch.double).to(self.device)
        jerr = 0
        eps2 = 1.0e-5
        one = torch.ones((), dtype=torch.double, device=self.device)
        step_buf = torch.empty(np + 1, dtype=torch.double, device=self.device)

        Xsum = torch.sum(Xmat, dim=0)
        XX = torch.mm(Xmat.T, Xmat)

        Amat = torch.zeros((np + 1, np + 1), dtype=torch.double).to(self.device)
        Amat[0, 0] = nobs
        Amat[0, 1:] = Xsum
        Amat[1:, 0] = Xsum
        Amat[1:, 1:] = XX

        eigens, Umat = torch.linalg.eigh(Amat)
        eigens = eigens.double().to(self.device)
        Umat = Umat.double().to(self.device)
        eigens += self.gamma

        vareps = 1.0e-8

        cval = torch.zeros((self.delta_len), dtype=torch.double, device=self.device)
        pinv = torch.zeros(
            (np + 1, self.delta_len), dtype=torch.double, device=self.device
        )
        Aione = torch.zeros(
            (np + 1, self.delta_len), dtype=torch.double, device=self.device
        )
        gval = torch.zeros((self.delta_len), dtype=torch.double, device=self.device)

        for l in range(nlam):
            al = self.ulam[l].item()
            delta = 0.125
            delta_id = 0
            delta_save = 0
            oldalpvec = torch.zeros(np + 1, dtype=torch.double).to(self.device)

            while delta_id < self.delta_len:
                delta_id += 1

                if delta_id > delta_save:
                    cval[delta_id - 1] = 2.0 * float(nobs) * delta * al
                    pinv[:, delta_id - 1] = 1.0 / (eigens + cval[delta_id - 1])
                    Aione[:, delta_id - 1] = torch.mv(
                        Umat, pinv[:, delta_id - 1] * Umat[0, :]
                    )
                    gval[delta_id - 1] = cval[delta_id - 1] / (
                        1.0 - cval[delta_id - 1] * Aione[0, delta_id - 1]
                    )
                    delta_save = delta_id

                told = one
                r = y - (alpvec[0] + torch.mv(Xmat, alpvec[1:]))

                for iteration in range(self.maxit):
                    zvec = torch.where(
                        r < -delta,
                        torch.full_like(r, -(tau - 1.0)),
                        torch.where(
                            r > delta,
                            torch.full_like(r, -tau),
                            -r / (2.0 * delta) - tau + 0.5,
                        ),
                    )

                    tnew = 0.5 + 0.5 * torch.sqrt(one + 4.0 * told * told)
                    mul = 1.0 + (told - 1.0) / tnew
                    told = tnew

                    if delta_id > self.delta_len:
                        print("Exceeded maximum delta_id")
                        break

                    kz[0] = torch.sum(zvec)
                    kz[1:] = zvec @ Xmat + float(nobs) * al * alpvec[1:]
                    kz[0] = kz[0] + gval[delta_id - 1] * torch.dot(
                        Aione[:, delta_id - 1], kz
                    )

                    step_buf.copy_(
                        -2.0
                        * mul
                        * delta
                        * torch.mv(Umat, pinv[:, delta_id - 1] * (kz @ Umat))
                    )
                    alpvec += step_buf

                    r = r - (step_buf[0] + torch.mv(Xmat, step_buf[1:]))
                    npass[l] += 1

                    if torch.max(step_buf**2) < (self.eps * mul * mul):
                        break

                    if torch.sum(npass) > self.maxit:
                        jerr = -l - 1
                        break

                dif_step = oldalpvec - alpvec
                xa = torch.mv(Xmat, alpvec[1:])
                aa = torch.dot(alpvec[1:], alpvec[1:])
                obj_value = self.objfun(alpvec[0], aa, xa, y, al, nobs, tau, 1e-9)
                golden_s = self.golden_section_search(
                    -100.0, 100.0, nobs, xa, aa, y, al, tau, 1e-9
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
                    torch.full_like(r, -(tau - 1.0)),
                    torch.where(
                        r >= 1e-9,
                        torch.full_like(r, -tau),
                        -r / (2.0 * 1e-9) - tau + 0.5,
                    ),
                )
                KKT = zvec @ Xmat / float(nobs) + al * alpvec[1:]
                uo = max(al, 1.0)
                KKT_norm = torch.sum(KKT**2) / (uo**2)

                if KKT_norm < self.KKTeps:
                    dif_norm = torch.max(dif_step**2)
                    if dif_norm < float(nobs) * (self.eps * mul * mul):
                        break

                if delta_id >= self.delta_len:
                    print(f"Exceeded maximum delta iterations for lambda {l}")
                    break
                delta *= 0.125

            alpmat[:, l] = alpvec
            self.anlam = l

            if torch.sum(npass) > self.maxit:
                self.jerr = -l - 1
                break

            pred[:, l] = self._cv_batched_lambda(
                Xmat=Xmat,
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
                cval=cval,
                pinv=pinv,
                Aione=Aione,
                gval=gval,
                delta_save=delta_save,
                cvnpass=cvnpass,
                l=l,
                one=one,
            )
            self.anlam = l

        self.alpmat = alpmat
        self.npass = npass
        self.cvnpass = cvnpass
        self.jerr = jerr
        self.pred = pred
        self.Z_test = Z_test
        self.Z_train = Xmat
        self.indices = indices.detach().cpu().to(torch.int64)

    def _cv_batched_lambda(
        self,
        *,
        Xmat,
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
        cval,
        pinv,
        Aione,
        gval,
        delta_save,
        cvnpass,
        l,
        one,
    ):
        tau = self.tau
        np = Xmat.shape[1]

        fold_ids = torch.arange(1, nfolds + 1, device=self.device)
        fold_masks = self.foldid.unsqueeze(1) == fold_ids.unsqueeze(0)
        fold_col_index = self.foldid.to(dtype=torch.long) - 1
        row_index = torch.arange(nobs, device=self.device)

        yn_batch = y.unsqueeze(1).expand(-1, nfolds).clone()
        yn_batch[fold_masks] = 0.0

        looalp_batch = alpvec.unsqueeze(1).expand(-1, nfolds).clone()
        loor_batch = r.unsqueeze(1).expand(-1, nfolds).clone()
        cv_step_buf = torch.zeros(
            (np + 1, nfolds), dtype=torch.double, device=self.device
        )
        kz_batch = torch.zeros((np + 1, nfolds), dtype=torch.double, device=self.device)

        active = torch.ones(nfolds, dtype=torch.bool, device=self.device)
        delta = 0.125
        delta_id = 0

        while torch.any(active):
            delta_id += 1

            if delta_id > delta_save:
                cval[delta_id - 1] = 2.0 * float(nobs) * delta * al
                pinv[:, delta_id - 1] = 1.0 / (eigens + cval[delta_id - 1])
                Aione[:, delta_id - 1] = torch.mv(
                    Umat, pinv[:, delta_id - 1] * Umat[0, :]
                )
                gval[delta_id - 1] = cval[delta_id - 1] / (
                    1.0 - cval[delta_id - 1] * Aione[0, delta_id - 1]
                )
                delta_save = delta_id

            active_cols = torch.nonzero(active, as_tuple=False).squeeze(1)
            told = torch.ones(nfolds, dtype=torch.double, device=self.device)
            looalp_active = looalp_batch[:, active_cols]
            xa = torch.mm(Xmat, looalp_active[1:, :])
            loor_batch[:, active_cols] = (
                yn_batch[:, active_cols] - looalp_active[0, :].unsqueeze(0) - xa
            )

            active_iter = active.clone()
            while torch.any(active_iter):
                iter_cols = torch.nonzero(active_iter, as_tuple=False).squeeze(1)
                yn_iter = yn_batch[:, iter_cols]
                loor_iter = loor_batch[:, iter_cols]
                alp_iter = looalp_batch[:, iter_cols]
                told_iter = told[iter_cols]

                zvec = torch.where(
                    loor_iter < -delta,
                    torch.full_like(loor_iter, -(tau - 1.0)),
                    torch.where(
                        loor_iter > delta,
                        torch.full_like(loor_iter, -tau),
                        -loor_iter / (2.0 * delta) - tau + 0.5,
                    ),
                )

                tnew = 0.5 + 0.5 * torch.sqrt(one + 4.0 * told_iter * told_iter)
                mul = 1.0 + (told_iter - 1.0) / tnew
                told[iter_cols] = tnew

                kz_batch[0, iter_cols] = zvec.sum(dim=0)
                kz_batch[1:, iter_cols] = (
                    torch.mm(Xmat.T, zvec) + float(nobs) * al * alp_iter[1:, :]
                )
                kz_batch[0, iter_cols] = kz_batch[0, iter_cols] + gval[
                    delta_id - 1
                ] * torch.matmul(Aione[:, delta_id - 1], kz_batch[:, iter_cols])

                spectral = torch.mm(Umat.T, kz_batch[:, iter_cols])
                spectral.mul_(pinv[:, delta_id - 1].unsqueeze(1))
                cv_step_buf[:, iter_cols] = (
                    -2.0 * delta * mul.unsqueeze(0) * torch.mm(Umat, spectral)
                )
                looalp_batch[:, iter_cols] += cv_step_buf[:, iter_cols]

                loor_batch[:, iter_cols] = loor_batch[:, iter_cols] - (
                    cv_step_buf[0, iter_cols].unsqueeze(0)
                    + torch.mm(Xmat, cv_step_buf[1:, iter_cols])
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
                yn = yn_batch[:, nf]
                dif_step = cv_step_buf[:, nf].clone()

                xa = torch.mv(Xmat, looalp[1:])
                aa = torch.dot(looalp[1:], looalp[1:])
                obj_value = self.objfun(looalp[0], aa, xa, yn, al, nobs, tau, 1e-9)
                golden_s = self.golden_section_search(
                    -100.0, 100.0, nobs, xa, aa, yn, al, tau, 1e-9
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
                    torch.full_like(loor, -(tau - 1.0)),
                    torch.where(
                        loor >= 1e-9,
                        torch.full_like(loor, -tau),
                        -loor / (2.0 * 1e-9) - tau + 0.5,
                    ),
                )
                KKT = zvec @ Xmat / float(nobs) + al * looalp[1:]
                uo = max(al, 1.0)
                KKT_norm = torch.sum(KKT**2) / (uo**2)
                if KKT_norm < self.KKTeps2:
                    active[nf] = False

            if delta_id >= self.delta_len:
                break
            delta *= 0.125

        cv_scores = torch.mm(Xmat, looalp_batch[1:, :]) + looalp_batch[0, :].unsqueeze(0)
        return cv_scores[row_index, fold_col_index]

    def transform(self, X_new):
        """Transform raw features into the fitted Nyström feature space."""
        if self.landmarks_ is None or self.M_ is None or self.sig_w_ is None:
            raise RuntimeError("Call fit() before transform().")
        X_new_dev = X_new.float().to(device=self.device)
        C_new = kernelMult(X_new_dev, self.landmarks_, self.sig_w_)
        Z_new = torch.mm(C_new, self.M_)
        return Z_new.double()

    def cv(self, pred, y):
        y_expanded = y[:, None]
        residuals = y_expanded - pred
        return cvknysqr.check_loss(residuals, self.tau).mean(dim=0)

    @staticmethod
    def check_loss(u, tau):
        return torch.where(u >= 0, tau * u, (tau - 1) * u)

    def objfun(self, intcpt, aka, ka, y, lam, nobs, tau, delta):
        """Smoothed quantile-check objective for primal Nyström KQR."""
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
        """Optimize the intercept by Brent / golden-section search."""
        device = ka.device if isinstance(ka, torch.Tensor) else self.device
        eps = torch.tensor(
            torch.finfo(torch.float64).eps, dtype=torch.double, device=device
        )
        tol = eps**0.25
        tol1 = eps + 1.0
        eps = torch.sqrt(eps)

        gold = (
            3.0 - torch.sqrt(torch.tensor(5.0, dtype=torch.double, device=device))
        ) * 0.5

        a = lmin
        b = lmax
        v = a + gold * (b - a)
        w = v
        x = v
        d = 0.0
        e = 0.0

        fx = self.objfun(x, aka, ka, y, lam, nobs, tau, delta)
        fv = fx
        fw = fx
        tol3 = tol / 3.0
        while True:
            xm = (a + b) * 0.5
            tol1 = eps * abs(x) + tol3
            t2 = 2.0 * tol1

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
            if (abs(p) >= abs(0.5 * q * r)) or (p <= q * (a - x)) or (p >= q * (b - x)):
                if x < xm:
                    e = b - x
                else:
                    e = a - x
                d = gold * e
            else:
                d = p / q
                u = x + d
                if (u - a < t2) or (b - u < t2):
                    d = tol1
                    if x >= xm:
                        d = -d

            u = x + d if abs(d) >= tol1 else (x + tol1 if d > 0 else x - tol1)
            fu = self.objfun(u, aka, ka, y, lam, nobs, tau, delta)
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
        lhat = x
        res = self.objfun(x, aka, ka, y, lam, nobs, tau, delta)
        return lhat, res
