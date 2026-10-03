# SPDX-License-Identifier: MIT
"""Exact-kernel SVM path with a spectral majorizer and a certified stop.

See EIGENDECOMPOSITION_OPTIONS.md (section 4) and
EIGENDECOMPOSITION_REVIEW_REPLY.md. For each lambda the solver minimises the
smoothed-hinge objective

    F(b, alpha) = sum_i phi_delta(y_i (b + K_i alpha)) + n lam alpha'K alpha + n eps b^2

over a decreasing smoothing schedule delta = 1, 1/8, ..., and accepts the
lambda (and each cross-validation fold) only when a duality gap for the
unsmoothed hinge problem is below ``gap_tol``. The curvature matrix of the
majorization steps is either K's full eigendecomposition or a truncated
spectrum: the top-r Ritz pairs plus a flat tail that bounds the rest.
"""

import math
import time

import torch

from ..functions import brent_minimize_batch
from ..memory import kernel_eigh


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
    K, y, alpha, b, lam, *, Ka=None, delta=None, refine=10, lmax=None, target=None
):
    """Certified relative gap of the unsmoothed kernel SVM, one column per problem.

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
    P = (torch.clamp(1.0 - r, min=0.0) * on).sum(dim=0) / n + lam * _row_sums(
        lambda a, ka: (a.double() * ka.double()).sum(dim=0), alpha, Ka
    )
    upper = on.double() / n

    def score(beta, yy, lam):
        by = beta * yy
        Kby = (K @ by.to(K.dtype)).double()
        return beta.sum(dim=0) - (by * Kby).sum(dim=0) / (4.0 * lam), Kby

    # candidates one at a time (one n x m block each, not all c stacked); per
    # column the first best is kept
    def candidates():
        yield 2.0 * lam * y64 * alpha.double()
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
    K, Y, alpha, b, lam, *, Ka, delta=None, refine=10, lmax=None, target=None
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
        return (torch.clamp(1.0 - r, min=0.0) * (Yt != 0)).sum(0) / n + lam * (
            a.double() * ka.double()
        ).sum(0)

    P = _row_sums(P_rows, Y, alpha, Ka)

    def score(beta):
        by = torch.empty(n, m, dtype=K.dtype, device=dev)
        bsum = torch.zeros(m, dtype=torch.float64, device=dev)
        for i, j in tiles:
            by[i:j] = (beta[i:j] * Y.rows(i, j).double()).to(K.dtype)
            bsum += beta[i:j].sum(0)
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
            if k <= 3:
                tt = (1e-2, 1e-3, 1e-4)[k - 1]
                mid = torch.clamp(base, min=0.0).minimum(upper)
                return torch.where(r < 1.0 - tt, upper, torch.where(r > 1.0 + tt, 0.0, mid))
            return -smoothed_hinge_grad(r, delta) * upper

        return src

    work = torch.empty(n, m, dtype=torch.float64, device=dev)
    beta = torch.empty(n, m, dtype=torch.float64, device=dev)
    D = None
    for k in range(5 if delta is not None else 4):
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


def _dF_rows(R, Y, KA, da, Kda, db, lam, delta, n):
    """Rows' share of the smoothed objective's change along a step (float64)."""
    R64 = R.double()
    dR = (Y * (db + Kda)).double()
    dloss = (
        (smoothed_hinge(R64 + dR, delta) - smoothed_hinge(R64, delta)) * (Y != 0)
    ).sum(0)
    da64, Kda64 = da.double(), Kda.double()
    return dloss + n * lam * (2.0 * (da64 * KA.double()).sum(0) + (da64 * Kda64).sum(0))


def _F_rows(R, Y, A, KA, lam, delta, n):
    """Rows' share of the smoothed objective, without the intercept's ridge."""
    loss = (smoothed_hinge(R.double(), delta) * (Y != 0)).sum(0)
    return loss + n * lam * (A.double() * KA.double()).sum(0)


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
        from ..cvksvm import _factorization_error

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
        :class:`torchkm.experimental.RBFKernelOperator` (matrix-free).
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
            return (torch.clamp(1.0 - r, min=0.0) * (Y != 0)).sum(0) / n + lam * (
                A.double() * KA.double()
            ).sum(0)

        return _row_sums(rows, Y, A, KA)

    def _dF(self, R, Y, A, KA, b, db, da, Kda, lam, delta):
        """Change of the smoothed objective along a step, in float64, from the
        step itself (no difference of two large sums)."""
        n = Y.shape[0]

        def rows(R, Y, KA, da, Kda):
            return _dF_rows(R, Y, KA, da, Kda, db, lam, delta, n)

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
                return (torch.clamp(1.0 - Y * (KA + bb), min=0.0) * (Y != 0)).sum(0)

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
            return _F_rows(R, Y, A, KA, lam, delta, n)

        return _row_sums(rows, R, Y, A, KA) + n * self.ridge_b * b.double() ** 2

    def _F_at(self, Y, A, KA, b, lam, delta):
        """F at (A, b) itself (margins built row block by row block)."""
        n = Y.shape[0]

        def rows(Y, A, KA):
            return _F_rows(Y * (KA + b), Y, A, KA, lam, delta, n)

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
            Z = Yc * smoothed_hinge_grad(R, delta)
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
                Z = Yt * smoothed_hinge_grad(R, delta)
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
                    _dF_rows(R, Yt, YKA, da, Kda, db, lc, delta, n),
                    _dF_rows(R, Yt, YKA, da_s, Kda_s, db_s, lc, delta, n),
                    _F_rows(R, Yt, Ya, YKA, lc, delta, n),
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
            if stalled:
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
