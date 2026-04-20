import torch
import os
import numpy
import time

from .functions import *

class cvknyssvm:
    def __init__(self, Xmat, X_test, y, nlam, ulam, foldid=None, nfolds = 5, eps=1e-5, maxit=1000, gamma=1.0, delta_len=8, KKTeps=1e-3, KKTeps2=1e-3,num_landmarks = 2000, k = 1000, device='cuda'):
        self.device = device
        self.nobs = Xmat.shape[0]

        # --- Check Kmat ---
        if not isinstance(Xmat, torch.Tensor):
            raise TypeError("Xmat must be a torch.Tensor")
        Xmat = Xmat.double().to(self.device)
        self.Xmat = Xmat

        if not isinstance(y, torch.Tensor):
            raise TypeError("y must be a torch.Tensor")
        y = y.double().to(self.device)

        # --- Label check ---
        unique_labels = torch.unique(y)
        if unique_labels.numel() > 2:
            raise ValueError(f"Multi-class detected: labels = {unique_labels.tolist()}. Only -1 and 1 allowed.")
        if not torch.all((unique_labels == -1) | (unique_labels == 1)):
            raise ValueError(f"Invalid labels: {unique_labels.tolist()}. Must be only -1 and 1.")
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
                foldid = torch.arange(self.nobs) # Each row gets its own fold ID
            else:
                # Randomly assign fold IDs across the rows
                # foldid = torch.tensor(np.random.permutation(np.repeat(np.arange(1, nfolds + 1), nn // nfolds + 1)[:nn]))
                foldid = torch.randperm(self.nobs) % nfolds + 1
            foldid = foldid.to(self.device) 

        # --- Shape check ---
        # if Xmat.shape[0] != Xmat.shape[1]:
        #     raise ValueError("Xmat must be a square matrix")
        if Xmat.shape[0] != y.shape[0]:
            raise ValueError("Xmat and y size mismatch")


        # self.Xmat = Xmat.double().to(self.device)
        self.X_test = X_test.double().to(self.device)
        self.y = y.double().to(self.device)
        self.nobs = Xmat.shape[0]
        self.np = Xmat.shape[1]
        self.nlam = nlam
        self.ulam = ulam.double()
        self.eps = eps
        self.maxit = maxit
        self.gamma = gamma
        self.delta_len = delta_len
        self.KKTeps = KKTeps
        self.KKTeps2 = KKTeps2
        self.num_landmarks = num_landmarks
        self.k = k
        self.nmaxit = self.nlam * self.maxit
        self.nfolds = nfolds
        self.foldid = foldid
        

        # Initialize outputs
        self.alpmat = torch.zeros((self.np + 1, self.nlam), dtype=torch.double).to(self.device)
        self.anlam = 0
        self.npass = torch.zeros(self.nlam, dtype=torch.int32).to(self.device)
        self.cvnpass = torch.zeros(self.nlam, dtype=torch.int32).to(self.device)
        self.pred = torch.zeros((self.nobs, self.nlam), dtype=torch.double).to(self.device)
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
        # store Nyström state for future transform/prediction
        self.indices = indices.detach().cpu().to(torch.int64)
        self.landmarks_ = landmarks.detach()
        self.sig_w_ = float(sig_w)
        self.M_ = M.detach()
        self.k_eff_ = int(k)

        Cmat = kernelMult(Xmat_work, landmarks, sig_w)  # Kernel matrix between X and landmarks
        Xmat = torch.mm(Cmat, M).double()

        C_test = kernelMult(X_test.float(), landmarks, sig_w)  # Kernel matrix between X and landmarks
        Z_test = torch.mm(C_test, M)  # Transformed training features
        
        np = Xmat.shape[1]
        r = torch.zeros(nobs, dtype=torch.double).to(self.device)
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
        # Precompute sum of Xmat along rows
        Xsum = torch.sum(Xmat, dim=0)
        XX = torch.mm(Xmat.T, Xmat)

        # Initialize Amat with zeros
        Amat = torch.zeros((np + 1, np + 1), dtype = torch.double).to(self.device)

        # Assign values to Amat
        Amat[0, 0] = nobs
        Amat[0, 1:] = Xsum
        Amat[1:, 0] = Xsum
        Amat[1:, 1:] = XX
        

        eigens, Umat = torch.linalg.eigh(Amat)
        eigens = eigens.double().to(self.device)
        Umat = Umat.double().to(self.device)
        eigens += self.gamma
        # Usum = torch.sum(Umat, dim = 0)
        # einv = 1 / eigens
        # eU = torch.mm(torch.diag(einv), Umat.T)
        # eU = (einv * Umat).T
        # Kinv1 = torch.mm(Umat, eU)

        vareps = 1.0e-8

        cval = torch.zeros((self.delta_len), dtype=torch.double, device=self.device)
        pinv = torch.zeros((np + 1, self.delta_len), dtype=torch.double, device=self.device)
        Aione = torch.zeros((np + 1, self.delta_len), dtype=torch.double, device=self.device)        
        gval = torch.zeros((self.delta_len), dtype=torch.double, device=self.device)


        for l in range(nlam):
            # start = time.time()
            al = self.ulam[l].item()
            delta = 1.0
            delta_id = 0
            delta_save = 0
            oldalpvec = torch.zeros(np + 1, dtype=torch.double).to(self.device)

            while delta_id < self.delta_len:
                delta_id += 1
                opdelta = 1.0 + delta
                omdelta = 1.0 - delta
                oddelta = 1.0 / delta

                if delta_id > delta_save:
                    cval[delta_id - 1] = 4.0 * float(nobs) * delta * al
                    pinv[:, delta_id - 1] = 1.0 / (eigens + cval[delta_id - 1])
                    Aione[:, delta_id - 1] = torch.mv(Umat, pinv[:, delta_id - 1] * Umat[0, :])
                    gval[delta_id - 1] = cval[delta_id - 1] / (1.0 - cval[delta_id - 1] * Aione[0, delta_id - 1])
                    delta_save = delta_id

                # Compute residual r
                told = one
            
                # Update alpha
                #alpha loop
                for iteration in range(self.maxit):
                    zvec = torch.where(r < omdelta, -y, torch.where(r > opdelta, torch.zeros(1, device=self.device), 0.5 * y * oddelta * (r - opdelta)))

                    tnew = 0.5 + 0.5 * torch.sqrt(one + 4.0 * told * told)
                    mul = 1.0 + (told - 1.0) / tnew
                    told = tnew
                    
                    # Update step using Pinv
                    if delta_id > self.delta_len:
                        print("Exceeded maximum delta_id")
                        break
                    
                    # Compute dif vector
                    kz[0] = torch.sum(zvec)
                    kz[1: ] = zvec @ Xmat + 2.0 * float(nobs) * al * alpvec[1:]
                    kz[0] = kz[0] + gval[delta_id - 1] * torch.dot(Aione[:, delta_id - 1], kz)

                    step_buf.copy_(-2.0 * mul * delta * torch.mv(Umat, pinv[:, delta_id - 1] * (kz @ Umat)))
                    alpvec += step_buf

                    # Update residual
                    r += y * (step_buf[0] + torch.mv(Xmat, step_buf[1:]))
                    npass[l] += 1

                    # Check convergence
                    if torch.max(step_buf ** 2) < (self.eps * mul * mul):
                        break


                    if torch.sum(npass) > self.maxit:
                        jerr = -l - 1
                        break

                # Check KKT conditions
                dif_step = oldalpvec - alpvec
                xa = torch.mv(Xmat, alpvec[1:])
                aa = torch.dot(alpvec[1:], alpvec[1:])
                obj_value = self.objfun(alpvec[0], aa, xa, y, al, nobs)
                # eps_float64 = np.finfo(np.float64).eps
                # optimal_intercept = minimize_scalar(self.objfun, args=(aka, ka, y, al, nobs), bracket=(-100.0, 100.0), method="brent")
                # obj_value_new = self.objfun(optimal_intercept.x, aka, ka, y, al, nobs)
                golden_s = self.golden_section_search(-100.0, 100.0, nobs, xa, aa, y, al)
                int_new = golden_s[0]
                obj_value_new = golden_s[1]
                if obj_value_new < obj_value:
                    dif_step[0] = dif_step[0] + int_new - alpvec[0]
                    r = r + y * (int_new - alpvec[0])
                    alpvec[0] = int_new

                oldalpvec = alpvec.clone()

                zvec = torch.where(r < 1.0, -y, torch.where(r > 1.0, torch.zeros(1).to(self.device), -0.5 * y))
                KKT = zvec @ Xmat / float(nobs) + 2.0 * al * alpvec[1:]
                # uo = max(al, 1.0)
                uo = 1.0
                KKT_norm = torch.sum(KKT ** 2) / (uo ** 2)
                # print(f'KKT:{KKT_norm}')
                if KKT_norm < self.KKTeps:
                    # Check convergence
                    dif_norm = torch.max(dif_step ** 2)
                    if dif_norm < float(nobs) * (self.eps * mul * mul):
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

            # Check if maximum iterations exceeded
            if torch.sum(npass) > self.maxit:
                self.jerr = -l - 1
                break
            # print(f'Single fitting:{time.time() - start}')

            ## Cross-validation
            for nf in range(nfolds):
                # start = time.time()
                yn = y.clone()

                # Set the current fold's labels to zero
                yn[self.foldid == (nf + 1)] = 0.0
                
                loor = r.clone() # Initial residuals
                looalp = alpvec.clone() # Initial alphas

                delta = 1.0
                delta_id = 0

                # while delta_id < self.delta_len:
                while True:
                    delta_id += 1
                    opdelta = 1.0 + delta
                    omdelta = 1.0 - delta
                    oddelta = 1.0 / delta

                    if delta_id > delta_save:
                        cval[delta_id - 1] = 4.0 * float(nobs) * delta * al
                        pinv[:, delta_id - 1] = 1.0 / (eigens + cval[delta_id - 1])
                        Aione[:, delta_id - 1] = torch.mv(Umat, pinv[:, delta_id - 1] * Umat[0, :])
                        gval[delta_id - 1] = cval[delta_id - 1] / (1.0 - cval[delta_id - 1] * Aione[0, delta_id - 1])
                        delta_save = delta_id

                    # Compute residual r
                    told = one

                    while torch.sum(cvnpass) <= self.nmaxit:
                        zvec = torch.where(loor < omdelta, -yn, torch.where(loor > opdelta, torch.zeros(1, device=self.device), 0.5 * yn * oddelta * (loor - opdelta)))

                        tnew = 0.5 + 0.5 * torch.sqrt(one + 4.0 * told * told)
                        mul = 1.0 + (told - 1.0) / tnew
                        told = tnew
                    
                        # Compute dif vector
                        kz[0] = torch.sum(zvec)
                        kz[1: ] = zvec @ Xmat + 2.0 * float(nobs) * al * looalp[1:]
                        kz[0] = kz[0] + gval[delta_id - 1] * torch.dot(Aione[:, delta_id - 1], kz)

                        step_buf.copy_(-2.0 * mul * delta * torch.mv(Umat, pinv[:, delta_id - 1] * (kz @ Umat)))
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
                        loor += yn * (step_buf[0] + torch.mv(Xmat, step_buf[1:]))

                        cvnpass[l] += 1

                        # Check convergence
                        if torch.max(step_buf ** 2) < eps2 * (mul ** 2):
                            break
                    if torch.sum(cvnpass) > self.nmaxit:
                        break
                    dif_step = step_buf.clone()
                    # dif_step = oldalpvec - alpvec
                    # print(f'Fitting alp time:{time.time() - start}')

                    xa = torch.mv(Xmat, looalp[1:])
                    aa = torch.dot(looalp[1:], looalp[1:])
                    obj_value = self.objfun(looalp[0], aa, xa, yn, al, nobs)
                    
                    # optimal_intercept = minimize_scalar(self.objfun, args=(aka, ka, yn, al, nobs), bracket=(-100.0, 100.0), method="brent")
                    # obj_value_new = self.objfun(optimal_intercept.x, aka, ka, yn, al, nobs)
                    golden_s = self.golden_section_search(-100.0, 100.0, nobs, xa, aa, yn, al)
                    int_new = golden_s[0]
                    obj_value_new = golden_s[1]
                    if obj_value_new < obj_value:
                        dif_step[0] = dif_step[0] + int_new - looalp[0]
                        loor = loor + y * (int_new - looalp[0])
                        looalp[0] = int_new

                    oldalpvec = alpvec.clone()
                    
                    zvec = torch.where(loor < 1.0, -yn, torch.where(loor > 1.0, torch.zeros(1).to(self.device), -0.5 * yn))
                    KKT = zvec @ Xmat / float(nobs) + 2.0 * al * looalp[1:]
                    # uo = max(al, 1.0)
                    uo = 1.0
                    KKT_norm = torch.sum(KKT ** 2) / (uo ** 2)


                    if KKT_norm < self.KKTeps2:
                        dif_norm = torch.max(dif_step ** 2)
                        if dif_norm < float(nobs) * (self.eps * mul * mul):
                            break
                        elif dif_norm > nobs and cvnpass[l] > 2:
                            break
                        if torch.sum(cvnpass) > self.nmaxit:
                            break
                    
                    if delta_id >= self.delta_len:
                        print(f"Exceeded maximum delta iterations for lambda {l}")
                        break
                    delta *= 0.125

                # for j in range(nobs):
                #     if self.foldid[j] == (nf + 1):
                #         looalp[j + 1] = 0.0
                loo_ind = (self.foldid == (nf + 1))
                # looalp[1:][loo_ind] = 0.0
                # pred[loo_ind, l] = looalp[1:] @ Xmat[loo_ind, :]  + looalp[0]
                pred[loo_ind, l] = torch.mv(Xmat[loo_ind, :].double(), looalp[1: ])  + looalp[0]
                # print(pred[loo_ind, l][:10])
                # for j in range(nobs):
                #     if self.foldid[j] == (nf + 1):
                #         pred[j, l] = torch.sum(Kmat[:, j] * looalp[1:]) + looalp[0]
                # print(pred[loo_ind, l][:10])
                # print(f'{nf}-fold: {time.time() - start}')
            self.anlam = l
            
        self.alpmat = alpmat
        self.npass = npass
        self.cvnpass = cvnpass
        self.jerr = jerr
        self.pred = pred
        self.Z_test = Z_test
        self.Z_train = Xmat
        self.indices = indices.detach().cpu().to(torch.int64)
    
    def transform(self, X_new):
        """
        Transform new raw features into the fitted Nyström feature space.
        Returns a tensor on self.device with shape (n_new, k_eff).
        """
        if self.landmarks_ is None or self.M_ is None or self.sig_w_ is None:
            raise RuntimeError("Call fit() before transform().")

        X_new_dev = X_new.float().to(device=self.device)
        C_new = kernelMult(X_new_dev, self.landmarks_, self.sig_w_)
        Z_new = torch.mm(C_new, self.M_)
        return Z_new.double()
    
    def cv(self, pred, y):
        pred_label = torch.where(pred > 0, 1, -1).to(device = 'cpu')
        y_expanded = y[:, None]
        misclass_matrix = (pred_label != y_expanded).float()
        misclass_rate = misclass_matrix.mean(dim=0)
        return misclass_rate
        
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
        objval = lam * aka + torch.sum(xi) / nobs

        return objval

    def golden_section_search(self, lmin, lmax, nobs, ka, aka, y, lam):
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
        fx = self.objfun(x, aka, ka, y, lam, nobs)
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
            fu = self.objfun(u, aka, ka, y, lam, nobs)
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
        res = self.objfun(x, aka, ka, y, lam, nobs)


        return lhat, res
