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
    x, fx = brent_minimize_batch(lambda b: [float(f(b[0]))], lmin, lmax, 1)
    return x[0], fx[0]


def brent_minimize_batch(f, lmin, lmax, k):
    """
    ``k`` independent searches of :func:`brent_minimize` run in step, for
    problems whose objectives are cheapest to evaluate together (the fold fits
    of cross-validation): ``f`` maps a list of ``k`` points to their ``k``
    objective values (a tensor or a sequence), read once per step for all of
    them. Each search takes exactly the steps it would take alone.

    Returns:
    - (x, fx) (list, list): Minimisers and objective values, ``k`` each.
    """

    def floats(values):
        if isinstance(values, torch.Tensor):
            return [float(v) for v in values.reshape(-1).tolist()]
        return [float(v) for v in values]

    eps = torch.finfo(torch.float64).eps
    tol3 = eps**0.25 / 3.0
    eps = math.sqrt(eps)
    gold = (3.0 - math.sqrt(5.0)) * 0.5
    a, b = [float(lmin)] * k, [float(lmax)] * k
    x = [a[i] + gold * (b[i] - a[i]) for i in range(k)]
    w, v, u = x[:], x[:], x[:]
    fx = floats(f(x))
    fw, fv = fx[:], fx[:]
    d, e = [0.0] * k, [0.0] * k
    done = [False] * k
    while True:
        for i in range(k):
            if done[i]:
                continue
            xm = (a[i] + b[i]) * 0.5
            tol1 = eps * abs(x[i]) + tol3
            t2 = 2.0 * tol1
            if abs(x[i] - xm) <= t2 - (b[i] - a[i]) * 0.5:
                done[i] = True
                continue
            p = q = r = 0.0
            if abs(e[i]) > tol1:
                r = (x[i] - w[i]) * (fx[i] - fv[i])
                q = (x[i] - v[i]) * (fx[i] - fw[i])
                p = (x[i] - v[i]) * q - (x[i] - w[i]) * r
                q = 2.0 * (q - r)
                if q > 0.0:
                    p = -p
                else:
                    q = -q
                r = e[i]
                e[i] = d[i]
            if (
                abs(p) >= abs(0.5 * q * r)
                or p <= q * (a[i] - x[i])
                or p >= q * (b[i] - x[i])
            ):
                # golden-section step
                e[i] = b[i] - x[i] if x[i] < xm else a[i] - x[i]
                d[i] = gold * e[i]
            else:
                # parabolic step
                d[i] = p / q
                ui = x[i] + d[i]
                if ui - a[i] < t2 or b[i] - ui < t2:
                    d[i] = tol1 if x[i] < xm else -tol1
            if abs(d[i]) >= tol1:
                u[i] = x[i] + d[i]
            else:
                u[i] = x[i] + tol1 if d[i] > 0 else x[i] - tol1
        if all(done):
            return x, fx
        fu = floats(f(u))
        for i in range(k):
            if done[i]:
                continue
            if fu[i] <= fx[i]:
                if u[i] < x[i]:
                    b[i] = x[i]
                else:
                    a[i] = x[i]
                v[i], fv[i], w[i], fw[i], x[i], fx[i] = (
                    w[i],
                    fw[i],
                    x[i],
                    fx[i],
                    u[i],
                    fu[i],
                )
            else:
                if u[i] < x[i]:
                    a[i] = u[i]
                else:
                    b[i] = u[i]
                if fu[i] <= fw[i] or w[i] == x[i]:
                    v[i], fv[i], w[i], fw[i] = w[i], fw[i], u[i], fu[i]
                elif fu[i] <= fv[i] or v[i] == x[i] or v[i] == w[i]:
                    v[i], fv[i] = u[i], fu[i]
