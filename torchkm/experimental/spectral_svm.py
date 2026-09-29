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

    Returns (gap, P, D) as float64 tensors with one entry per column.
    """
    squeeze = alpha.dim() == 1
    if squeeze:
        y, alpha = y.unsqueeze(1), alpha.unsqueeze(1)
        Ka = None if Ka is None else Ka.unsqueeze(1)
    n, m = alpha.shape
    b = torch.as_tensor(b, dtype=torch.float64, device=alpha.device).reshape(-1)
    if Ka is None:
        Ka = K @ alpha
    y64, a64, Ka64 = y.double(), alpha.double(), Ka.double()
    on = (y64 != 0).double()
    r = y64 * (Ka64 + b)
    P = (torch.clamp(1.0 - r, min=0.0) * on).sum(dim=0) / n + lam * (a64 * Ka64).sum(
        dim=0
    )
    upper = on / n

    cands = [2.0 * lam * y64 * a64]
    for t in (1e-2, 1e-3, 1e-4):
        mid = torch.clamp(2.0 * lam * y64 * a64, min=0.0).minimum(upper)
        cands.append(
            torch.where(r < 1.0 - t, upper, torch.where(r > 1.0 + t, 0.0, mid))
        )
    if delta is not None:
        cands.append(-smoothed_hinge_grad(r, delta) * upper)
    c = len(cands)
    beta = project_dual(torch.cat(cands, dim=1), y64.repeat(1, c), upper.repeat(1, c))

    def score(beta, yy):
        by = beta * yy
        Kby = (K @ by.to(K.dtype)).double()
        return beta.sum(dim=0) - (by * Kby).sum(dim=0) / (4.0 * lam), Kby

    D_all, _ = score(beta, y64.repeat(1, c))
    D_all = D_all.reshape(c, m)
    best = D_all.argmax(dim=0)
    D = D_all.gather(0, best.unsqueeze(0)).squeeze(0)
    beta = beta.reshape(n, c, m)[:, best, torch.arange(m, device=beta.device)]

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
            Dp, Kby = score(point, y64)
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
        m = G.shape[1]
        inv = 1.0 / (self.et + c)
        UG = self.count.mm(self.U.T, G)
        out = self.count.mm(
            self.U, torch.cat([UG * inv[:, None], UG * (self.e * inv)[:, None]], dim=1)
        )
        return out[:, :m], out[:, m:]

    def coupling(self, c, K1):
        """s = P 1, K s, v = P K 1 and n - 1'v (in a cancellation-free form)."""
        inv = 1.0 / (self.et.double() + c)
        e = self.e.double()
        s = self.count.mm(self.U, (self.u1 * inv).to(self.U.dtype))
        Ks = self.count.mm(self.U, (self.u1 * e * inv).to(self.U.dtype))
        den0 = float((self.u1**2 * (self.et.double() - e + c) * inv).sum())
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
        V, KV = Q @ S, KQ @ S
        Rk = KV - V * th
        rho = math.sqrt(
            max(float(torch.linalg.eigvalsh((Rk.T @ Rk).double())[-1]), 0.0)
        )
        del Rk
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
        dc = 1.0 / (self.th + c) - 1.0 / (self.tau + c)
        W = dc[:, None] * (self.V.T @ G)
        PG = G / (self.tau + c) + self.V @ W
        KPG = None if KG is None else KG / (self.tau + c) + self.KV @ W
        return PG, KPG

    def coupling(self, c, K1):
        one = torch.ones_like(K1)[:, None]
        s, Ks = self.apply(one, K1[:, None], c)
        v, _ = self.apply(K1[:, None], None, c)
        s, Ks, v = s[:, 0], Ks[:, 0], v[:, 0]
        s64 = s.double()
        Kt1 = (
            self.tau
            + (
                self.V @ ((self.th.double() - self.tau) * self.v1).to(self.V.dtype)
            ).double()
        )
        den0 = float(c * s64.sum() + s64 @ (Kt1 - K1.double()))
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
    K : tensor (n, n)
        Kernel matrix; its dtype and device are the working ones.
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
        Iterations after which one lambda (or one batch of folds) stops
        certifying; unfinished columns are reported as not converged.
    warm_levels : int or None
        A warm-started fit (the next lambda, or the folds) starts smoothing
        ``warm_levels`` levels above the delta at which its starting point was
        certified; None starts every fit at delta = 1 (as cvksvm).

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
        seed=0,
    ):
        if spectrum not in ("truncated", "full"):
            raise ValueError("spectrum must be 'truncated' or 'full'")
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
        on = Y != 0
        r = Y.double() * (KA.double() + b.double())
        n = Y.shape[0]
        return (torch.clamp(1.0 - r, min=0.0) * on).sum(0) / n + lam * (
            A.double() * KA.double()
        ).sum(0)

    def _dF(self, R, Y, A, KA, b, db, da, Kda, lam, delta):
        """Change of the smoothed objective along a step, in float64, from the
        step itself (no difference of two large sums)."""
        n = Y.shape[0]
        on = Y != 0
        R64 = R.double()
        dR = (Y * (db + Kda)).double()
        dloss = (
            (smoothed_hinge(R64 + dR, delta) - smoothed_hinge(R64, delta)) * on
        ).sum(0)
        da64, Kda64 = da.double(), Kda.double()
        dpen = n * lam * (2.0 * (da64 * KA.double()).sum(0) + (da64 * Kda64).sum(0))
        db64 = db.double()
        return dloss + dpen + n * self.ridge_b * (2.0 * b.double() * db64 + db64 * db64)

    def _intercept(self, Y, A, b, KA, lam):
        """Brent search of the unsmoothed objective in b (as cvksvm)."""
        n, m = Y.shape
        on = Y != 0
        aka = (A * KA).sum(0)

        def obj(points):
            bb = torch.as_tensor(points, dtype=KA.dtype, device=KA.device)
            return (torch.clamp(1.0 - Y * (KA + bb), min=0.0) * on).sum(
                0
            ) / n + lam * aka

        old = obj(b.tolist())
        b_new, f_new = brent_minimize_batch(obj, -100.0, 100.0, m)
        b_new = torch.as_tensor(b_new, dtype=b.dtype, device=b.device)
        f_new = torch.as_tensor(f_new, dtype=old.dtype, device=old.device)
        return torch.where(f_new < old, b_new, b)

    # -- the smoothed objective at a batch of points ------------------------
    def _F(self, R, Y, A, KA, b, lam, delta):
        n = Y.shape[0]
        on = Y != 0
        loss = (smoothed_hinge(R.double(), delta) * on).sum(0)
        return (
            loss
            + n * lam * (A.double() * KA.double()).sum(0)
            + n * self.ridge_b * b.double() ** 2
        )

    # -- iterations at one smoothing level ----------------------------------
    def _round(self, Y, A, b, KA, lam, delta, eps_round, budget):
        """Accelerated MM (FISTA momentum on the majorization step, with a
        function-value restart, so F never increases) at one delta, on the
        columns of A, b, KA in place. A column stops when its step's predicted
        decrease, -g'step, is at most eps_round times F (never when eps_round
        is 0), or after ``budget`` iterations. Each step is
        checked (``safeguard``) and replaced by the scalar-majorizer step when
        the check fails. One product with K per iteration."""
        K, n, count = self.K, self.K.shape[0], self.count
        c = 4.0 * n * delta * lam
        s, Ks, v, den0 = self.backend.coupling(c, self.K1)
        den = den0 + 4.0 * n * delta * self.ridge_b
        ss = 1.0 / (self.backend.lmax + c)
        den_s = n - ss * float(self.K1.double().sum()) + 4.0 * n * delta * self.ridge_b
        m = A.shape[1]
        dev, dt = A.device, A.dtype
        t = torch.ones(m, dtype=torch.float64, device=dev)
        PA, Pb, PKA = A.clone(), b.clone(), KA.clone()
        Fx = self._F(Y * (KA + b), Y, A, KA, b, lam, delta)
        live = torch.ones(m, dtype=torch.bool, device=dev)
        iters = 0
        for _ in range(budget):
            cols = torch.nonzero(live).squeeze(1)
            Yc, Ac, bc, KAc = Y[:, cols], A[:, cols], b[cols], KA[:, cols]
            tc = t[cols]
            t_next = 0.5 + 0.5 * torch.sqrt(1.0 + 4.0 * tc * tc)
            beta = ((tc - 1.0) / t_next).to(dt)
            Ya = Ac + beta * (Ac - PA[:, cols])
            Yb = bc + beta * (bc - Pb[cols])
            YKA = KAc + beta * (KAc - PKA[:, cols])
            R = Yc * (YKA + Yb)
            Z = Yc * smoothed_hinge_grad(R, delta)
            KG = count.mm(K, Z) + 2.0 * n * lam * YKA
            G = Z + 2.0 * n * lam * Ya
            gb = Z.sum(0) + 2.0 * n * self.ridge_b * Yb
            PG, KPG = self.backend.apply(G, KG, c)
            db = -2.0 * delta * (gb - v @ G) / den
            da = -2.0 * delta * PG - s[:, None] * db
            Kda = -2.0 * delta * KPG - Ks[:, None] * db
            dF = self._dF(R, Yc, Ya, YKA, Yb, db, da, Kda, lam, delta)
            d_f = (gb * db + (KG * da).sum(0)).double()
            if self.safeguard:
                db_s = -2.0 * delta * (gb - ss * (self.K1 @ G)) / den_s
                da_s = -2.0 * delta * ss * G - ss * db_s
                Kda_s = -2.0 * delta * ss * KG - ss * self.K1[:, None] * db_s
                d_s = (gb * db_s + (KG * da_s).sum(0)).double()
                fast = (d_f <= self.eta * d_s) & (
                    (dF <= self.armijo * d_f) | (d_f.abs() <= 1e-12 * n)
                )
                if not bool(fast.all()):
                    self.counts["fallbacks"] += int((~fast).sum())
                    dF_s = self._dF(R, Yc, Ya, YKA, Yb, db_s, da_s, Kda_s, lam, delta)
                    f = fast.to(dt)
                    db, da, Kda = (
                        f * db + (1 - f) * db_s,
                        f * da + (1 - f) * da_s,
                        f * Kda + (1 - f) * Kda_s,
                    )
                    dF = torch.where(fast, dF, dF_s)
            Fnew = self._F(R, Yc, Ya, YKA, Yb, lam, delta) + dF
            keep = (
                Fnew <= Fx[cols]
            )  # else restart: stay at x, next step without momentum
            kf = keep.to(dt)
            PA[:, cols], Pb[cols], PKA[:, cols] = Ac, bc, KAc
            A[:, cols] = kf * (Ya + da) + (1 - kf) * Ac
            b[cols] = kf * (Yb + db) + (1 - kf) * bc
            KA[:, cols] = kf * (YKA + Kda) + (1 - kf) * KAc
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
                    lam,
                    delta,
                )
            # scale-free stop: the step's predicted decrease against F
            small = -d_f <= eps_round * Fx[cols].abs()
            live[cols[keep & small]] = False
            if not bool(live.any()) or self.counts["column_iterations"] > self.max_iter:
                break
        return iters

    # -- one lambda, several columns (the whole data, or the folds) ---------
    def _fit_columns(self, Y, A, b, KA, lam, delta=1.0):
        """Smoothing rounds delta, delta/8, ..., each solved loosely, until the
        smoothing bias is small: delta / 4 times the share of rows inside the
        smoothing band at most half the tolerance (times the objective). Then
        iterate at that delta in chunks, certifying after each, until every
        column's gap is at most ``gap_tol``; shrink delta when the gaps stop
        improving. All unfinished columns share delta (one batched product per
        step)."""
        n, m = A.shape
        K, count = self.K, self.count
        gap = torch.full((m,), float("inf"), dtype=torch.float64, device=A.device)
        active = torch.ones(m, dtype=torch.bool, device=A.device)
        iters, chunk, certifying = 0, self.chunk, False
        eps_cert = self.eps * 1e-4  # early exit of certifying rounds; tightened
        for _ in range(self.max_rounds):
            cols = torch.nonzero(active).squeeze(1)
            Yc, Ac, bc, KAc = Y[:, cols], A[:, cols], b[cols], KA[:, cols]
            if not certifying:
                P = self._primal(Yc, Ac, bc, KAc, lam)
                r = Yc * (KAc + bc)
                band = (
                    (((r - 1.0).abs() <= delta) & (Yc != 0))
                    .double()
                    .mean(0)
                    .clamp_min(1.0 / n)
                )
                certifying = (
                    bool((delta * band / 4.0 <= 0.5 * self.gap_tol * P).all())
                    or delta < self.min_delta
                )
            phase = "certify" if certifying else "loose"
            if not certifying:
                eps_round = self.eps
                if (
                    self.knee
                    and self.spectrum == "truncated"
                    and 4.0 * n * delta * lam < self.backend.tau
                ):
                    eps_round *= 0.1
                used = self._round(
                    Yc, Ac, bc, KAc, lam, delta, eps_round, self.round_cap
                )
            else:
                used = self._round(Yc, Ac, bc, KAc, lam, delta, eps_cert, chunk)
            iters += used
            self.trace.append((lam, phase, delta, int(cols.numel()), used))
            if not bool(torch.isfinite(Ac).all() and torch.isfinite(bc).all()):
                raise FloatingPointError(
                    f"non-finite coefficients at lambda {lam:g}, delta {delta:g}"
                )
            KAc = count.mm(K, Ac)
            bc = self._intercept(Yc, Ac, bc, KAc, lam)
            A[:, cols], b[cols], KA[:, cols] = Ac, bc, KAc
            if not certifying:
                delta /= 8.0
                continue
            t = self._now()
            g, _, _ = hinge_duality_gap(
                _CountedK(K, self.counts["reads"]),
                Yc,
                Ac,
                bc,
                lam,
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
                or iters >= self.fit_cap
            ):
                break
            if stalled:
                # smooth less only if the smoothing bias could be what limits
                left = active[cols]
                r = Yc * (KAc + bc)
                band = (((r - 1.0).abs() <= delta) & (Yc != 0)).double().mean(0)
                P = self._primal(Yc, Ac, bc, KAc, lam)
                bias = delta * band.clamp_min(1.0 / n) / 4.0 > 0.25 * self.gap_tol * P
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
        A = torch.zeros(n, 1, dtype=dt, device=dev)
        b = torch.zeros(1, dtype=dt, device=dev)
        KA = torch.zeros(n, 1, dtype=dt, device=dev)
        reads0 = self.count.reads
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
