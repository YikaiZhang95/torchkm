# SPDX-License-Identifier: MIT
import math

import torch


def data_gen(nn, nm, pp, p1, p2, mu, ro, sdn=None, means=None):
    """
    Generate synthetic data with positive and negative centers.

    Parameters:
    - nn (int): Number of samples (total observations).
    - nm (int): Number of clusters (per class).
    - pp (int): Number of features.
    - p1 (int): First set of positive centers.
    - p2 (int): Second set of positive centers (unused in original code).
    - mu (float): Mean shift for positive and negative centers.
    - ro (float): Standard deviation for normal distribution.
    - sdn (int, optional): Seed for reproducibility.
    - means (torch.Tensor, optional): Predefined cluster means (if any).

    Returns:
    - X (torch.Tensor): Feature matrix.
    - y (torch.Tensor): Labels vector.
    - means (torch.Tensor): Cluster centers.
    """

    # Set seed if provided
    if sdn is not None and means is None:
        torch.manual_seed(sdn)
        means = torch.randn(nm * 2, pp)
        # Negative centers: Shift the first `p1` features
        means[:nm, :p1] += mu
        # Positive centers: Shift the remaining features
        means[nm:, p1:pp] += mu

    # Generate binary labels (randomly assign 1 and -1)
    id_pos = torch.bernoulli(torch.full((nn,), 0.5)).bool()
    size_pos = torch.sum(id_pos).item()
    size_neg = nn - size_pos

    # Initialize labels
    y = torch.full((nn,), -1.0)
    y[id_pos] = 1.0

    # Generate random features from normal distribution
    X = torch.randn(nn, pp) * ro

    # Assign random cluster IDs for negative and positive samples
    ids = torch.empty(nn).long()
    ids[~id_pos] = torch.randint(0, nm, (size_neg,))
    ids[id_pos] = torch.randint(nm, nm * 2, (size_pos,))

    # Adjust features based on the cluster centers
    X += means[ids]

    return X, y, means


def sigest(x, frac=0.5, generator=None):
    """
    PyTorch equivalent of the R function sigest.

    Parameters:
    - x (torch.Tensor): Input tensor of shape (m, n), where m is the number of samples and n is the number of features.
    - frac (float): Fraction of samples to use for computing the distance.
    - generator (torch.Generator, optional): CPU generator used to draw the
      random pairs. Pass one for a reproducible estimate that leaves the global
      torch RNG untouched; ``None`` uses the global RNG.

    Returns:
    - sigma_estimate (float): Estimated sigma based on quantiles of squared distances.
    """

    # Number of samples (m)
    m = x.shape[0]

    # Number of random samples to take for the distance calculation
    n = int(frac * m)

    # Randomly sample `n` indices (two sets)
    index1 = torch.randint(0, m, (n,), dtype=torch.long, generator=generator)
    index2 = torch.randint(0, m, (n,), dtype=torch.long, generator=generator)

    # Compute the squared differences between the randomly paired rows
    temp = x[index1] - x[index2]
    dist = torch.sum(temp**2, dim=1)

    # Exclude zero distances (self-pairs)
    non_zero_dist = dist[dist != 0]

    # Compute quantiles (0.9, 0.5, 0.1)
    q = torch.tensor(
        [0.9, 0.5, 0.1], dtype=non_zero_dist.dtype, device=non_zero_dist.device
    )
    srange = 1.0 / torch.quantile(non_zero_dist, q)

    # Return the mean of the 90th and 10th quantiles
    sigma_estimate = torch.mean(srange[[0, 2]]).item()

    return sigma_estimate


def rbf_kernel(x, sigma):
    """
    Compute the RBF (Gaussian) kernel matrix in PyTorch.

    Parameters:
    - x (torch.Tensor): Input tensor of shape (n_samples, n_features).
    - sigma (float): The standard deviation parameter for the RBF kernel (Gaussian width).

    Returns:
    - K (torch.Tensor): RBF kernel matrix of shape (n_samples, n_samples).
    """
    # Compute pairwise squared Euclidean distances
    x_norm = torch.sum(x * x, dim=1, keepdim=True)
    pairwise_dists = x_norm + x_norm.t()
    pairwise_dists.addmm_(x, x.t(), beta=1.0, alpha=-2.0)
    pairwise_dists.clamp_min_(0.0)

    # Compute the RBF kernel matrix in place: the distance buffer becomes K,
    # so no second n x n (or n x m) temporary is allocated.
    K = pairwise_dists.mul_(-2.0 * sigma).exp_()

    return K


def standardize(x):
    """
    Standardizes the input tensor (feature-wise standardization).

    Args:
    - x (torch.Tensor): Input tensor (matrix) of shape (n_samples, n_features).

    Returns:
    - x_standardized (torch.Tensor): Standardized tensor where each feature has mean 0 and standard deviation 1.
    """
    # Compute column-wise means and standard deviations
    mean = torch.mean(x, dim=0)
    std = torch.std(x, dim=0)

    # Replace zeros in std with 1 to avoid division by zero
    std[std == 0] = 1

    # Standardize: subtract the mean and divide by the standard deviation
    x_standardized = (x - mean) / std
    return x_standardized


def kernelMult(X, X_new, sigma):
    """
    Compute the RBF (Gaussian) kernel matrix between X and X_new in PyTorch.

    Parameters:
    - X (torch.Tensor): Input tensor of shape (n_samples_X, n_features).
    - X_new (torch.Tensor): Input tensor of shape (n_samples_X_new, n_features).
    - sigma (float): The standard deviation parameter for the RBF kernel (Gaussian width).

    Returns:
    - K (torch.Tensor): RBF kernel matrix of shape (n_samples_X, n_samples_X_new).
    """
    # Compute squared L2 norms
    X_norm = torch.sum(X * X, dim=1, keepdim=True)
    X_new_norm = torch.sum(X_new * X_new, dim=1).view(1, -1)

    # Compute pairwise squared Euclidean distances
    pairwise_dists = X_norm + X_new_norm
    pairwise_dists.addmm_(X, X_new.t(), beta=1.0, alpha=-2.0)
    pairwise_dists.clamp_min_(0.0)

    # Compute the RBF kernel matrix in place: the distance buffer becomes K,
    # so no second n x n (or n x m) temporary is allocated.
    K = pairwise_dists.mul_(-2.0 * sigma).exp_()

    return K


def brent_minimize(f, lmin, lmax):
    """
    Minimise a function of one variable on [lmin, lmax] by Brent's method
    (golden-section steps with parabolic interpolation), as the solvers do for
    their intercept.

    The search's own arithmetic is in Python floats: ``f`` returns a scalar
    (a one-element tensor or a float) and is read once per evaluation. On a
    GPU that read is the only wait for the device per step; the solvers' old
    per-method copies kept the search's state in device tensors, so every
    comparison and update was a kernel launch and most of them a wait.

    Parameters:
    - f (callable): Objective of the intercept.
    - lmin, lmax (float): Search interval.

    Returns:
    - (x, fx) (float, float): Minimiser and objective value there.
    """
    eps = torch.finfo(torch.float64).eps
    tol3 = eps**0.25 / 3.0
    eps = math.sqrt(eps)
    gold = (3.0 - math.sqrt(5.0)) * 0.5
    a, b = float(lmin), float(lmax)
    x = w = v = a + gold * (b - a)
    fx = fw = fv = float(f(x))
    d = e = 0.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = 2.0 * tol1
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
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
        if abs(p) >= abs(0.5 * q * r) or p <= q * (a - x) or p >= q * (b - x):
            # golden-section step
            e = b - x if x < xm else a - x
            d = gold * e
        else:
            # parabolic step
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        u = x + d if abs(d) >= tol1 else (x + tol1 if d > 0 else x - tol1)
        fu = float(f(u))
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv, w, fw, x, fx = w, fw, x, fx, u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x, fx
