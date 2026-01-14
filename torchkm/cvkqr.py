import torch
import os
import numpy
import time

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
        Target labels for each sample, of shape (n_samples,). Typically, -1 or 1.

    nlam : int
        The number of regularization parameters to consider in the optimization.

    ulam : ndarray or tensor
        User-specified regularization parameters, of shape (nlam,).
    
    tau : tensor
        A user-supplied tau value for a quantile level.

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

    device : {'cuda', 'cpu'}, default='cuda'
        Device to perform computations on. Default is GPU ('cuda') for improved performance.

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
    >>> from torchsvm.cvkqr import cvkqr
    >>> from torchsvm.functions import *
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
    >>> model = cvkqr(Kmat=Kmat, y=y_train, nlam=nlam, ulam=ulam, nfolds=nfolds, eps=1e-5, maxit=100000, gamma=1e-8, is_exact=0, device='cuda')
    >>> model.fit()
    """

    def __init__(self, Kmat, y, nlam, ulam, tau, eps=1e-5, maxit=1000, gamma=1.0, is_exact=0, delta_len=4, mproj=2, KKTeps=1e-3, device='cuda'):
        self.device = device
        self.nobs = Kmat.shape[0]

        # --- Check Kmat ---
        if not isinstance(Kmat, torch.Tensor):
            raise TypeError("Kmat must be a torch.Tensor")
        Kmat = Kmat.double().to(self.device)
        self.Kmat = Kmat

        if not isinstance(y, torch.Tensor):
            raise TypeError("y must be a torch.Tensor")
        y = y.double().to(self.device)
        self.y = y

        # --- Check ulam ---
        if not isinstance(ulam, torch.Tensor):
            raise TypeError("ulam must be a torch.Tensor")
        ulam = ulam.double().to(self.device)   

        # --- Shape check ---
        if Kmat.shape[0] != Kmat.shape[1]:
            raise ValueError("Kmat must be a square matrix")
        if Kmat.shape[0] != y.shape[0]:
            raise ValueError("Kmat and y size mismatch")
        # self.Kmat = None
        # self.y = None
        
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
        self.nmaxit = self.nlam * self.maxit

        # Initialize outputs
        self.alpmat = torch.zeros((self.nobs + 1, self.nlam), dtype=torch.double).to(self.device)
        self.anlam = 0
        self.npass = torch.zeros(self.nlam, dtype=torch.int32).to(self.device)
        self.pred = torch.zeros((self.nobs, self.nlam), dtype=torch.double).to(self.device)
        self.jerr = 0

    def fit(self):
        nobs = self.nobs
        nlam = self.nlam
        y = self.y
        Kmat = self.Kmat
        tau = self.tau

        r = torch.zeros(nobs, dtype=torch.double).to(self.device)
        alpmat = torch.zeros((nobs + 1, nlam), dtype=torch.double).to(self.device)
        npass = torch.zeros(nlam, dtype=torch.int32).to(self.device)
        alpvec = torch.zeros(nobs + 1, dtype=torch.double).to(self.device)
        pred = torch.zeros((self.nobs, self.nlam), dtype=torch.double).to(self.device)
        jerr = 0

        # Precompute sum of Kmat along rows
        Ksum = torch.sum(Kmat, dim=1)
        # Kinv = torch.linalg.inv(Kmat)

        eigens, Umat = torch.linalg.eigh(Kmat)
        eigens = eigens.double().to(self.device)
        Umat = Umat.double().to(self.device)
        Kmat = Kmat.double().to(self.device)
        eigens += self.gamma
        Usum = torch.sum(Umat, dim = 0)
        einv = 1 / eigens
        # eU = torch.mm(torch.diag(einv), Umat.T)
        eU = (einv * Umat).T
        # Kinv1 = torch.mm(Umat, eU)
        r = y
        vareps = 1.0e-8

        lpUsum = torch.zeros((nobs, self.delta_len), dtype=torch.double, device=self.device)
        lpinv = torch.zeros((nobs, self.delta_len), dtype=torch.double, device=self.device)
        svec = torch.zeros((nobs, self.delta_len), dtype=torch.double, device=self.device)
        vvec = torch.zeros((nobs, self.delta_len), dtype=torch.double, device=self.device)
        gval = torch.zeros((self.delta_len), dtype=torch.double, device=self.device)


        for l in range(nlam):
            # start = time.time()
            al = self.ulam[l].item()
            delta = 0.125
            delta_id = 0
            delta_save = 0
            oldalpvec = torch.zeros(nobs + 1, dtype=torch.double).to(self.device)

            while delta_id < self.delta_len:
                delta_id += 1

                if delta_id > delta_save:
                    lpinv[:, delta_id - 1] = 1.0 / (eigens + 2.0 * float(nobs) * delta * al)
                    lpUsum[:, delta_id - 1] = lpinv[:, delta_id - 1] * Usum
                    vvec[:, delta_id - 1] = torch.mv(Umat, eigens * lpUsum[:, delta_id - 1])
                    svec[:, delta_id - 1] = torch.mv(Umat, lpUsum[:, delta_id - 1])
                    gval[delta_id - 1] = 1.0 / (nobs + 4.0 * nobs * delta * vareps - vvec[:, delta_id - 1].sum())
                    delta_save = delta_id

                # Compute residual r
                dif_step = torch.zeros((nobs + 1), dtype=torch.double, device=self.device)
                told = 1.0
                # ka = torch.mv(Kmat, alpvec[1:])
                # r = y - (alpvec[0] + ka)
                # Update alpha
                #alpha loop
                for iteration in range(self.maxit):
                    zvec = torch.where(r < - delta, -(tau - 1.0), torch.where(r > delta, -tau, - r / (2.0 * delta) - tau + 0.50))
                    gamvec = zvec + float(nobs) * al * alpvec[1:]##
                    rds = zvec.sum() + 2.0 * nobs * vareps * alpvec[0]
                    hval = rds - torch.dot(vvec[:, delta_id - 1], gamvec)

                    tnew = 0.5 + 0.5 * torch.sqrt(torch.tensor(1.0, device=self.device) + 4.0 * told * told)
                    mul = 1.0 + (told - 1.0) / tnew
                    told = tnew.item()
                    
                    # Update step using Pinv
                    if delta_id > self.delta_len:
                        print("Exceeded maximum delta_id")
                        break
                    
                    # Compute dif vector
                    
                    # dif_step = torch.zeros((nobs + 1), dtype=torch.double, device=self.device)
                    dif_step[0] = -2.0 * mul * delta * gval[delta_id - 1] * hval
                    dif_step[1:] = -dif_step[0] * svec[:, delta_id - 1] - 2.0 * mul * delta * torch.mv(Umat, gamvec @ Umat * lpinv[:, delta_id - 1])
                    alpvec += dif_step
                    # Update residual
                    ka = torch.mv(Kmat, alpvec[1:])
                    r = y - (alpvec[0] + ka)
                    npass[l] += 1
                    # print("##################################")
                    # print(torch.max(dif_step ** 2), self.eps * mul * mul)
                    # Check convergence
                    if torch.max(dif_step ** 2).item() < (self.eps * mul * mul):
                        # print("break the loop")
                        break


                    if torch.sum(npass).item() > self.maxit:
                        jerr = -l - 1
                        break

                # Check KKT conditions
                dif_step = oldalpvec - alpvec
                ka = torch.mv(Kmat, alpvec[1:])
                aka = torch.dot(ka, alpvec[1:])
                if self.device == 'cuda':
                    ka_cpu = ka.to('cpu')
                    aka_cpu = aka.to('cpu')
                    y_cpu = y.to('cpu')
                    alpvec0_cpu = alpvec[0].to('cpu')
                else:
                    ka_cpu = ka
                    aka_cpu = aka
                    y_cpu = y
                    alpvec0_cpu = alpvec[0]

                obj_value = self.objfun(alpvec0_cpu, aka_cpu, ka_cpu, y_cpu, al, nobs, tau, 1e-9)
                # eps_float64 = np.finfo(np.float64).eps
                # optimal_intercept = minimize_scalar(self.objfun, args=(aka, ka, y, al, nobs), bracket=(-100.0, 100.0), method="brent")
                # obj_value_new = self.objfun(optimal_intercept.x, aka, ka, y, al, nobs)
                golden_s = self.golden_section_search(-100.0, 100.0, nobs, ka_cpu, aka_cpu, y_cpu, al, tau, 1e-9)
                int_new = golden_s[0]
                obj_value_new = golden_s[1]
                if obj_value_new < obj_value:
                    dif_step[0] = dif_step[0] + int_new - alpvec[0]
                    r = r - (int_new - alpvec[0])
                    alpvec[0] = int_new

                oldalpvec = alpvec.clone()

                zvec = torch.where(r <= - 1e-9, -(tau - 1.0), torch.where(r >= 1e-9, -tau, - r / (2.0 * 1e-9) - tau + 0.5))
                cvec = torch.zeros((nobs + 1), dtype=torch.double, device=self.device)
                dvec = torch.zeros((nobs + 1), dtype=torch.double, device=self.device)
                cvec[0] = zvec.sum()
                cvec[1:] = torch.mv(Kmat,zvec)
                dvec[0] = 2 * vareps * alpvec[0]
                dvec[1:] = al * torch.mv(Kmat, alpvec[1:])
                KKT = cvec / float(nobs) + dvec
                uo = max(al, 1.0)
                KKT_norm = torch.sum(KKT ** 2) / (uo ** 2)
                # print("#######################")
                # print(KKT_norm)
                if KKT_norm.item() < self.KKTeps:
                    # Check convergence
                    dif_norm = torch.max(dif_step ** 2)
                    if dif_norm.item() < float(nobs) * (self.eps * mul * mul):
                        if self.is_exact == 0:
                            break
                        else:
                            is_exit = False
                            alptmp = alpvec.clone()
                            for nn in range(self.mproj):
                                elbowid = torch.zeros(nobs, dtype=torch.bool)
                                elbchk = True
                                # Compute rmg and check elbow condition
                                rmg = r
                                elbowid = torch.abs(rmg) < delta
                                elbchk = torch.all(rmg[elbowid] <= 1e-3).item()

                                if elbchk:
                                    break

                                # Projection update
                                told = 1.0
                                for _ in range(self.maxit):
                                    ka = torch.mv(Kmat, alptmp[1:])
                                    aKa = torch.dot(ka, alptmp[1:])
                                    if self.device == 'cuda':
                                        ka_cpu = ka.to('cpu')
                                        aka_cpu = aka.to('cpu')
                                        y_cpu = y.to('cpu')
                                        alptmp0_cpu = alptmp[0].to('cpu')
                                    else:
                                        ka_cpu = ka
                                        aka_cpu = aka
                                        y_cpu = y
                                        alptmp0_cpu = alptmp[0]

                                    obj_value = self.objfun(alptmp0_cpu, aka_cpu, ka_cpu, y_cpu, al, nobs, tau, 1e-9)

                                    
                                    # Optimize intercept
                                    # optimal_intercept = minimize_scalar(self.objfun, args=(aka, ka, y, al, nobs), bracket=(-100.0, 100.0), method = 'brent')
                                    # obj_value_new = self.objfun(optimal_intercept.x, aka, ka, y, al, nobs)
                                    golden_s = self.golden_section_search(-100.0, 100.0, nobs, ka_cpu, aka_cpu, y_cpu, al, tau, 1e-9)
                                    int_new = golden_s[0]
                                    obj_value_new = golden_s[1]
                                    print(obj_value_new, obj_value)
                                    if obj_value_new < obj_value:
                                        dif_step[0] = dif_step[0] + int_new - alptmp[0]
                                        alptmp[0] = int_new

                                    r = y - (alptmp[0] + ka)
                                    zvec = torch.where(r < -delta, -(tau - 1.0), torch.where(r > delta, -tau, - r / (2.0 * delta) - tau + 0.5))
                                    gamvec = zvec + float(nobs) * al * alptmp[1:]##
                                    rds = zvec.sum() + 2.0 * nobs * vareps * alptmp[0]
                                    hval = rds - torch.dot(vvec[:, delta_id - 1], gamvec)

                                    tnew = 0.5 + 0.5 * torch.sqrt(torch.tensor(1.0, device=self.device) + 4.0 * told * told)
                                    mul = 1.0 + (told - 1.0) / tnew
                                    told = tnew.item()
                                        
                                    # Compute dif vector
                                    
                                    # dif_step = torch.zeros((nobs + 1), dtype=torch.double, device=self.device)
                                    dif_step[0] = -2.0 * mul * delta * gval[delta_id - 1] * hval
                                    dif_step[1:] = -dif_step[0] * svec[:, delta_id - 1] - 2.0 * mul * delta * torch.mv(Umat, gamvec @ Umat * lpinv[:, delta_id - 1])
                                    alptmp += dif_step

                                    ka = torch.mv(Kmat, alptmp[1:])
                                    r = y - (alptmp[0] + ka)
                                    npass[l] += 1
                                    alp_old = alptmp.clone()

                                    if torch.sum(elbowid).item() > 1:
                                        theta = torch.mv(Kmat, alptmp[1:])
                                        theta = theta + r * (elbowid == 1).float()
                                        alptmp[1:] = torch.mv(Umat, torch.mv(eU, theta))

                                    dif_step = dif_step + alptmp - alp_old
                                    r = y - (alptmp[0] + torch.mv(Kmat, alptmp[1:]))
                                    mdd = torch.max(dif_step ** 2)
                                    # Check convergence
                                    if mdd < self.eps * mul ** 2:
                                        break
                                    elif mdd > nobs and npass[l] > 2:
                                        is_exit = True
                                        break
                                    if torch.sum(npass) > self.maxit:
                                        is_exit = True
                                        break

                            # Check KKT condition
                            if is_exit:
                                break
                            zvec = torch.where(r < -1e-9, -(tau-1.0), torch.where(r > 1e-9, -tau, - r / (2.0 * 1e-9) - tau + 0.5))
                            cvec[0] = zvec.sum()
                            cvec[1:] = torch.mv(Kmat,zvec)
                            dvec[0] = 2 * vareps * alpvec[0]
                            dvec[1:] = al * torch.mv(Kmat, alpvec[1:])
                            KKT = cvec / float(nobs) + dvec
                            uo = max(al, 1.0)

                            if torch.sum(KKT ** 2) / (uo ** 2) < self.KKTeps:
                                alpvec = alptmp.clone()
                                break
                # else:
                #     # Reduce delta
                #     delta *= 0.125
                if delta_id > self.delta_len:
                    print(f"Exceeded maximum delta iterations for lambda {l}")
                    break
                delta *= 0.125
            # Save the alpha vector for current lambda
            alpmat[:, l] = alpvec
            # Update anlam
            self.anlam = l

            # Check if maximum iterations exceeded
            if torch.sum(npass) > self.maxit:
                self.jerr = -l - 1
                break
            # print(f'Single fitting:{time.time() - start}')
            
            self.anlam = l

        self.alpmat = alpmat
        self.npass = npass
        self.jerr = jerr
        self.pred = pred
        
    def cross_validate(self, foldid=None, nfolds=5):
        """
        Perform K-fold cross-validation using the quantile check loss.

        Parameters
        ----------
        foldid : torch.Tensor or array-like, optional
            Fold assignments for each observation. If None, a deterministic split
            is created based on nfolds.
        nfolds : int, default=5
            Number of folds to use when foldid is not supplied.

        Returns
        -------
        tuple
            Tuple containing the mean loss per lambda (torch.Tensor) and the index
            of the lambda with the smallest loss.
        """
        nobs = self.nobs
        if foldid is None:
            if nfolds is None:
                raise ValueError("nfolds must be provided when foldid is None")
            if nfolds <= 1 or nfolds > nobs:
                raise ValueError("nfolds must be between 2 and the number of observations")
            foldid = torch.arange(nobs, device=self.Kmat.device, dtype=torch.long) % nfolds
        else:
            if not isinstance(foldid, torch.Tensor):
                foldid = torch.as_tensor(foldid, dtype=torch.long)
            else:
                foldid = foldid.long()
            if foldid.numel() != nobs:
                raise ValueError("foldid must have one entry per observation")
            foldid = foldid.to(self.Kmat.device)

        unique_folds = torch.unique(foldid)
        if nfolds is None:
            nfolds = unique_folds.numel()
        if unique_folds.numel() != nfolds:
            raise ValueError("foldid contains a different number of folds than nfolds")
        if nfolds < 2:
            raise ValueError("nfolds must be at least 2")

        cv_scores = torch.zeros(self.nlam, dtype=torch.double, device=self.Kmat.device)

        tau_arg = self.tau
        if isinstance(tau_arg, torch.Tensor):
            tau_arg = tau_arg.to(self.Kmat.device).double()
        else:
            tau_arg = float(tau_arg)

        for fold in unique_folds:
            val_mask = foldid == fold
            train_mask = ~val_mask
            if torch.sum(val_mask) == 0 or torch.sum(train_mask) == 0:
                raise ValueError("Each fold must have at least one training and one validation sample")

            train_idx = torch.nonzero(train_mask, as_tuple=False).flatten()
            val_idx = torch.nonzero(val_mask, as_tuple=False).flatten()

            K_train = self.Kmat.index_select(0, train_idx).index_select(1, train_idx)
            y_train = self.y.index_select(0, train_idx)

            fold_model = cvkqr(
                K_train,
                y_train,
                self.nlam,
                self.ulam.clone(),
                tau_arg,
                eps=self.eps,
                maxit=self.maxit,
                gamma=self.gamma,
                is_exact=self.is_exact,
                delta_len=self.delta_len,
                mproj=self.mproj,
                KKTeps=self.KKTeps,
                device=self.device,
            )
            fold_model.fit()

            coeffs = fold_model.alpmat[1:, :]
            intercept = fold_model.alpmat[0, :]

            K_val = self.Kmat.index_select(0, val_idx).index_select(1, train_idx)
            preds = torch.matmul(K_val, coeffs) + intercept.unsqueeze(0)

            y_val = self.y.index_select(0, val_idx).unsqueeze(1)
            residuals = y_val - preds

            loss = cvkqr.check_loss(residuals, tau_arg)
            cv_scores += loss.mean(dim=0)

        cv_scores /= float(nfolds)
        best_index = torch.argmin(cv_scores).item()

        self.cv_scores = cv_scores
        self.cv_best_index = best_index
        self.cv_best_lambda = self.ulam[best_index]

        return cv_scores, best_index
    
    def predict(self, Kmat_new, y_new, alp_b):
        result = torch.mv(Kmat_new, alp_b[1:]) + alp_b[0]
        return result

    def obj_value(self, alp_b, lam_b):
        intcpt = alp_b[0]
        alp = alp_b[1:]
        Kmat = self.Kmat.double().to('cpu')
        ka = torch.mv(Kmat, alp)
        aka = torch.dot(alp, ka)
        y_train = self.y.to('cpu')
        obj = self.objfun(intcpt, aka, ka, y_train, lam_b, self.nobs, tau, delta)
        return obj
    
    def check_loss(u, tau):
        # quantile check loss: ρ_τ(u) = u * (τ - 1(u < 0))
        return torch.where(u >= 0, tau * u, (tau - 1) * u)

        
    def objfun(self, intcpt, aka, ka, y, lam, nobs, tau, delta):
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
        # Initialize xi (hinge loss terms)
        xi = torch.zeros(nobs, dtype=torch.double)

        # Compute f_hat (fh) and the hinge loss xi
        fh = ka + intcpt
        xi_tmp = y - fh

        # smoothed quantile loss with torch.where
        ttau = tau - 1.0
        xi = torch.where(
            xi_tmp <= - delta,
            xi_tmp * ttau,
            torch.where(
                xi_tmp >= delta,
                xi_tmp * tau,
                xi_tmp**2 / (4.0 * delta)
                + (tau - 0.5) * xi_tmp
                + delta / 4.0,
            ),
        )

        # Compute the objective value
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
        - y (torch.Tensor): Labels vector of shape (nobs,).
        - lam (float): Regularization parameter.

        Returns:
        - lhat (float): Optimized intercept value.
        - fx (float): Objective function value at the optimized intercept.
        """
        eps = torch.tensor(torch.finfo(torch.float64).eps)
        tol = eps ** 0.25
        tol1 = eps + 1.0
        eps = torch.sqrt(eps)

        # Golden ratio constant
        gold = (3.0 - torch.sqrt(torch.tensor(5.0))) * 0.5

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
                        d = - d

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
