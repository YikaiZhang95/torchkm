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
    if non_zero_dist.numel() == 0:
        # every sampled pair was a self-pair (possible when frac * m is a
        # handful of rows): use every pair of distinct rows instead
        all_dist = torch.cdist(x, x).pow(2).flatten()
        non_zero_dist = all_dist[all_dist != 0]
        if non_zero_dist.numel() == 0:
            raise ValueError("sigest needs at least two distinct rows")

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


# A kernel matrix that is never stored.
#
# :class:`RBFKernelOperator` stands in for the n x n RBF kernel matrix wherever
# only products ``K @ B`` are needed, as in
# :class:`torchkm.cvksvm.SpectralSVMPath` with the truncated spectrum. Each
# product recomputes K in blocks of rows, uses each block once and frees it.
# Memory is then O(n p + block + n k) for k right-hand columns instead of n^2,
# and each product costs the kernel's arithmetic again (about 2 n^2 p flops for
# the distances) instead of one read of a stored K.
#
# With ``fused=True`` (CUDA, float32) a product runs in one fused kernel per 128
# columns and never writes a block of K to memory. With
# a_i = [4 sigma x_i, -2 sigma |x_i|^2, 1] and b_j = [x_j, 1, -2 sigma |x_j|^2],
# a_i'b_j = -2 sigma |x_i - x_j|^2, so K B = exp(A B') V: attention without the
# softmax normalization. PyTorch's memory-efficient attention computes it in
# float32 and returns the log-sum-exp of each row, which undoes the
# normalization. Building the blocks of K instead writes and reads each entry
# about ten times, which dominates the block path's time.
# (The same code as torchkm.experimental.kernels, kept there for exploring.)


class RBFKernelOperator:
    """K = exp(-2 sigma |x_i - x_j|^2) over the rows of ``X``, as an operator.

    ``K @ B`` equals ``torchkm.functions.rbf_kernel(X, sigma) @ B`` (the same
    formula, block by block, so the entries agree to rounding). ``B`` may be a
    vector or a matrix in ``X``'s dtype and on its device.

    Duplicate rows of ``X`` have identical kernel rows and columns, so
    K B = P K_u (P' B) with K_u the kernel of the unique rows and P the n x u
    indicator of each row's copy: the products then cost u^2 instead of n^2
    (binary or categorical features often repeat rows; w8a: u = 0.70 n, half
    the work), with the same kernel entries.

    Parameters
    ----------
    X : tensor (n, p)
        Training rows, on the device and in the dtype to compute in.
    sigma : float
        Bandwidth, as in ``rbf_kernel``.
    block_bytes : int, default 2**30
        Size of one block of kernel rows; the rows per block follow from it.
    fused : bool, default False
        Products in one fused kernel per ``FUSED_MAX_COLUMNS`` columns (CUDA,
        float32 only) instead of blocks of K. Entries agree
        with the block path to float32 rounding (relative error about 4e-5
        against float64, 2e-5 for the blocks), not bitwise.
    dedup : bool or "auto", default "auto"
        Work on the unique rows of ``X`` (above). ``"auto"``: when at least
        ``DEDUP_MIN_SHARE`` of the rows are duplicates. The result is the same
        to rounding (duplicates' coefficients are summed before the product).
    cache_bytes : int, optional
        Memory for storing kernel rows (of the unique rows), read in every
        product instead of recomputed: the first ``cached_rows`` rows that fit,
        all u when u x u fits (``stored``). A product then recomputes only the
        other rows, so its cost falls in proportion.
    """

    FUSED_MAX_COLUMNS = 128
    DEDUP_MIN_SHARE = 0.05

    def __init__(
        self,
        X: torch.Tensor,
        sigma: float,
        block_bytes: int = 2**30,
        fused: bool = False,
        dedup="auto",
        cache_bytes=None,
    ):
        n = X.shape[0]
        self.sigma = float(sigma)
        self.shape = (n, n)
        self.dtype, self.device = X.dtype, X.device
        self._inv = None  # row -> its unique row, when deduplicated
        if dedup:
            U, inv = torch.unique(X, dim=0, return_inverse=True)
            if dedup is True or n - U.shape[0] >= self.DEDUP_MIN_SHARE * n:
                X, self._inv = U, inv
        self.X = X  # the rows the kernel is computed on (unique when deduplicated)
        self.n_unique = X.shape[0]
        self.x_norm = (X * X).sum(dim=1)
        u = X.shape[0]
        self.block_rows = max(1, min(u, int(block_bytes) // (u * X.element_size())))
        self.fused = bool(fused)
        if self.fused:
            if X.device.type != "cuda" or X.dtype != torch.float32:
                raise ValueError("fused=True needs float32 rows on a CUDA device")
            s, nx = self.sigma, self.x_norm[:, None]
            one = torch.ones_like(nx)
            self._q = _pad8(torch.cat([4.0 * s * X, -2.0 * s * nx, one], dim=1))
            self._k = _pad8(torch.cat([X, one, -2.0 * s * nx], dim=1))
        # stored kernel rows 0 .. cached_rows - 1 (against every row), as many
        # as fit cache_bytes; the rest are recomputed in each product
        c = 0 if cache_bytes is None else min(u, int(cache_bytes) // (u * X.element_size()))
        self.cached_rows = c
        self._K = None
        if c > 0:
            self._K = torch.empty(c, u, dtype=X.dtype, device=X.device)
            # filled in small blocks (64 MiB) so that building it adds little
            # to the peak beyond the cache itself
            step = max(1, min(self.block_rows, 2**26 // (u * X.element_size())))
            for i in range(0, c, step):
                j = min(c, i + step)
                D = self.x_norm[i:j, None] + self.x_norm[None, :]
                D.addmm_(X[i:j], X.T, beta=1.0, alpha=-2.0)
                self._K[i:j] = D.clamp_min_(0.0).mul_(-2.0 * self.sigma).exp_()
                del D
        self.stored = c == u

    def _collapse(self, B2: torch.Tensor) -> torch.Tensor:
        """P' B: the rows of B summed per unique row."""
        if self._inv is None:
            return B2
        out = torch.zeros(self.n_unique, B2.shape[1], dtype=B2.dtype, device=B2.device)
        return out.index_add_(0, self._inv, B2)

    def _fused(self, B: torch.Tensor, q: torch.Tensor = None) -> torch.Tensor:
        q = self._q if q is None else q
        n, w = q.shape[0], B.shape[1]
        out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(
            q[None, None],
            self._k[None, None],
            _pad8(B)[None, None],
            None,
            True,  # return the log-sum-exp of each row
            0.0,
            False,
            scale=1.0,
        )[:2]
        return out[0, 0, :, :w] * lse[0, 0, :n, None].exp()

    def _rows_times(self, Xq: torch.Tensor, q, Bu: torch.Tensor) -> torch.Tensor:
        """K(Xq, X) Bu over the operator's (unique) rows X; ``q`` the fused
        query rows of Xq (None for the block path)."""
        s = self.sigma
        if self.fused:
            w = self.FUSED_MAX_COLUMNS
            return torch.cat(
                [self._fused(Bu[:, j : j + w], q) for j in range(0, Bu.shape[1], w)],
                dim=1,
            )
        rows = self.block_rows
        out = torch.empty(Xq.shape[0], Bu.shape[1], dtype=Bu.dtype, device=Bu.device)
        for i in range(0, Xq.shape[0], rows):
            # rbf_kernel's arithmetic on rows i .. i + rows
            Xb = Xq[i : i + rows]
            D = (Xb * Xb).sum(dim=1)[:, None] + self.x_norm[None, :]
            D.addmm_(Xb, self.X.T, beta=1.0, alpha=-2.0)
            D.clamp_min_(0.0)
            out[i : i + rows] = D.mul_(-2.0 * s).exp_() @ Bu
        return out

    def cross(self, Xq: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """K(Xq, X) @ B for other rows ``Xq`` (prediction), in blocks of query
        rows, or fused (no block of the kernel is formed) with ``fused``."""
        vector = B.dim() == 1
        B2 = B.unsqueeze(1) if vector else B
        q = None
        if self.fused:
            s = self.sigma
            nq = (Xq * Xq).sum(dim=1, keepdim=True)
            q = _pad8(torch.cat([4.0 * s * Xq, -2.0 * s * nq, torch.ones_like(nq)], 1))
        out = self._rows_times(Xq, q, self._collapse(B2))
        return out.squeeze(1) if vector else out

    def __matmul__(self, B: torch.Tensor) -> torch.Tensor:
        vector = B.dim() == 1
        B2 = B.unsqueeze(1) if vector else B
        Bu = self._collapse(B2)
        c = self.cached_rows
        if c == self.n_unique:
            out = self._K @ Bu
        elif c > 0:  # stored rows read, the others recomputed
            out = torch.cat(
                [self._K @ Bu,
                 self._rows_times(self.X[c:], self._q[c:] if self.fused else None, Bu)]
            )
        else:
            out = self._rows_times(self.X, self._q if self.fused else None, Bu)
        if self._inv is not None:
            out = out[self._inv]  # P (K_u P'B): each row its unique row's
        return out.squeeze(1) if vector else out


def _pad8(M: torch.Tensor) -> torch.Tensor:
    """Columns zero-padded to a multiple of 8 (the fused kernel's alignment)."""
    return torch.nn.functional.pad(M, (0, (-M.shape[1]) % 8)).contiguous()


class UniqueRowsKernel:
    """K = P K_u P' as an operator: ``K_u`` the stored kernel of the unique
    rows, ``inverse`` each row's unique row (``torch.unique(..., dim=0,
    return_inverse=True)``). Duplicate rows have identical kernel rows and
    columns, so the products are exact and cost a read of K_u (u^2) instead
    of K (n^2)."""

    def __init__(self, K_unique: torch.Tensor, inverse: torch.Tensor):
        self.K_unique, self.inverse = K_unique, inverse
        n = inverse.shape[0]
        self.shape = (n, n)
        self.dtype, self.device = K_unique.dtype, K_unique.device

    def __matmul__(self, B: torch.Tensor) -> torch.Tensor:
        vector = B.dim() == 1
        B2 = B.unsqueeze(1) if vector else B
        Bu = torch.zeros(
            self.K_unique.shape[0], B2.shape[1], dtype=B2.dtype, device=B2.device
        ).index_add_(0, self.inverse, B2)
        out = (self.K_unique @ Bu)[self.inverse]
        return out.squeeze(1) if vector else out
