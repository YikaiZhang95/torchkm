# SPDX-License-Identifier: MIT
"""The experimental certified SVM path (torchkm.experimental): the duality-gap
certificate bounds the true suboptimality, both spectra reach every lambda and
fold at the requested gap, and the step safeguard handles the reviewer's
counterexample."""

import math

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from torchkm import rbf_kernel, sigest
from torchkm.experimental import (
    RBFKernelOperator,
    SpectralSVMPath,
    hinge_duality_gap,
)
from torchkm.experimental.spectral_svm import (
    _Counter,
    _TruncatedSpectrum,
    project_dual,
)


def _problem(n=200, seed=0, dtype=torch.float64):
    X, y = make_classification(
        n_samples=n, n_features=8, n_informative=5, flip_y=0.05, random_state=seed
    )
    Xt = torch.as_tensor(X)
    torch.manual_seed(seed)
    K = rbf_kernel(Xt, float(sigest(Xt))).to(dtype)
    return K, torch.as_tensor(np.where(y == 0, -1.0, 1.0), dtype=dtype)


def _libsvm(K, y, lam):
    """Optimum of mean hinge + lam a'K a (C = 1 / (2 n lam))."""
    n = len(y)
    s = SVC(kernel="precomputed", C=1.0 / (2 * n * lam), tol=1e-10, max_iter=-1)
    s.fit(K.double().numpy(), y.double().numpy())
    a = np.zeros(n)
    a[s.support_] = s.dual_coef_[0]
    return torch.as_tensor(a), float(s.intercept_[0])


def _primal(K, y, a, b, lam):
    Ka = K.double() @ a.double()
    return float(
        torch.clamp(1 - y.double() * (Ka + b), min=0).mean() + lam * a.double() @ Ka
    )


def test_projection_is_feasible():
    g = torch.Generator().manual_seed(0)
    beta = torch.randn(50, 3, generator=g, dtype=torch.float64)
    y = torch.where(torch.rand(50, 3, generator=g) < 0.4, -1.0, 1.0).double()
    upper = torch.full((50, 3), 0.02, dtype=torch.float64)
    p = project_dual(beta, y, upper)
    assert float(p.min()) >= 0.0 and float((p - upper).max()) <= 0.0
    assert float((p * y).sum(0).abs().max()) < 1e-12


def test_certificate_bounds_the_true_gap():
    K, y = _problem()
    lmax = float(torch.linalg.eigvalsh(K)[-1])
    for lam in (1.0, 0.03, 0.002):
        a, b = _libsvm(K, y, lam)
        pstar = _primal(K, y, a, b, lam)
        # at the optimum the certificate closes
        g, P, D = hinge_duality_gap(K, y, a, b, lam, refine=50, lmax=lmax)
        assert float(g) < 1e-5
        # at a perturbed point: D <= P* <= P and the gap bounds the excess
        a2 = a * 0.9 + 1e-3
        g2, P2, D2 = hinge_duality_gap(K, y, a2, b + 0.05, lam, refine=20, lmax=lmax)
        assert float(D2) <= pstar + 1e-10 <= float(P2) + 1e-10
        assert float(g2) >= (float(P2) - pstar) / float(P2) - 1e-12


def test_certificate_columns_match_single_problems():
    K, y = _problem(n=120)
    lmax = float(torch.linalg.eigvalsh(K)[-1])
    lam = 0.01
    a, b = _libsvm(K, y, lam)
    Y = y[:, None].repeat(1, 3)
    Y[:40, 0] = 0.0  # a held-out block per column
    Y[40:80, 1] = 0.0
    A = (a * 0.95)[:, None].repeat(1, 3)
    B = torch.tensor([b, b + 0.01, b - 0.02])
    g, P, D = hinge_duality_gap(K, Y, A, B, lam, refine=10, lmax=lmax)
    for j in range(3):
        gj, Pj, Dj = hinge_duality_gap(
            K, Y[:, j], A[:, j], float(B[j]), lam, refine=10, lmax=lmax
        )
        assert math.isclose(float(P[j]), float(Pj), rel_tol=1e-12)
        assert math.isclose(float(D[j]), float(Dj), rel_tol=1e-9, abs_tol=1e-12)


@pytest.mark.parametrize("spectrum", ["full", "truncated"])
def test_path_is_certified_and_near_libsvm(spectrum):
    K, y = _problem()
    lams = np.logspace(0, -3, 6)
    fold = torch.arange(len(y)) % 4 + 1
    m = SpectralSVMPath(
        K, y, lams, fold, spectrum=spectrum, rank=40, gap_tol=1e-3
    ).fit()
    assert bool(m.converged.all()) and bool(m.fold_converged.all())
    assert float(m.gaps.max()) <= 1e-3 and float(m.fold_gaps.max()) <= 1e-3
    for j, lam in enumerate(lams):
        a, b = _libsvm(K, y, lam)
        pstar = _primal(K, y, a, b, lam)
        P = _primal(K, y, m.alphas[1:, j], float(m.alphas[0, j]), lam)
        assert pstar - 1e-9 <= P <= pstar + 1e-3 * P  # what the certificate promises
    assert m.cv_scores.shape == (len(y), len(lams)) and m.cv_error.shape == (len(lams),)
    assert m.counts["fallbacks"] == 0
    if spectrum == "truncated":
        info = m.spectrum_info
        assert (
            info["rank"] == 40 and info["tau"] >= info["tau0"] > 0 and info["rho"] > 0
        )


def test_truncated_and_full_agree():
    K, y = _problem(n=120, seed=1)
    lams = np.logspace(-0.5, -2.5, 3)
    fold = torch.arange(len(y)) % 3 + 1
    fits = {
        s: SpectralSVMPath(K, y, lams, fold, spectrum=s, rank=30, gap_tol=1e-4).fit()
        for s in ("full", "truncated")
    }
    for j, lam in enumerate(lams):
        P = [
            _primal(K, y, f.alphas[1:, j], float(f.alphas[0, j]), lam)
            for f in fits.values()
        ]
        assert abs(P[0] - P[1]) <= 2e-4 * max(P)
    # At large lambda a fold whose training labels balance has an objective
    # flat in the intercept, so held-out signs there are arbitrary; compare
    # the selected lambda and the error at the smallest lambda.
    err = [f.cv_error for f in fits.values()]
    assert int(err[0].argmin()) == int(err[1].argmin())
    assert abs(float(err[0][-1]) - float(err[1][-1])) <= 2.0 / len(y)


def test_float32_reaches_the_gap():
    K, y = _problem(n=150, dtype=torch.float32)
    lams = np.logspace(0, -2, 4)
    for spectrum in ("full", "truncated"):
        m = SpectralSVMPath(
            K, y, lams, None, spectrum=spectrum, rank=30, gap_tol=1e-3
        ).fit()
        assert bool(m.converged.all()), spectrum
        assert m.alphas.dtype == torch.float32


def test_safeguard_turns_the_counterexample_into_descent():
    # the reviewer's 2 x 2 example: Ritz vector e1 with the rho shift, where the
    # plain step raises F from 674.02 to 976.63
    K = torch.tensor(
        [[99.0, math.sqrt(98.0)], [math.sqrt(98.0), 2.0]], dtype=torch.float64
    )
    y = torch.tensor([1.0, -1.0], dtype=torch.float64)
    lam, delta = 0.0125, 1.0
    rho = math.sqrt(98.0)

    def fitted(safeguard):
        m = SpectralSVMPath(
            K, y, [lam], spectrum="truncated", rank=1, safeguard=safeguard
        )
        m.count, m.K1 = _Counter(), K @ torch.ones(2, dtype=torch.float64)
        m.counts = dict(column_iterations=0, fallbacks=0, reads=dict(certificate=0))
        be = _TruncatedSpectrum.__new__(_TruncatedSpectrum)
        be.V = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        be.KV, be.count = K @ be.V, m.count
        be.th = torch.tensor([99.0 + rho], dtype=torch.float64)
        be.tau = 2.0 + rho
        be.lmax = 99.0 + rho
        be.v1 = be.V.T @ torch.ones(2, dtype=torch.float64)
        m.backend = be
        A = torch.tensor([[20.0], [-40.0]], dtype=torch.float64)
        b = torch.tensor([-200.0], dtype=torch.float64)
        KA = K @ A
        Y = y[:, None]
        F0 = float(m._F(Y * (KA + b), Y, A, KA, b, lam, delta))
        m._round(Y, A, b, KA, lam, delta, 0.0, 1)
        F1 = float(m._F(Y * (KA + b), Y, A, KA, b, lam, delta))
        return F0, F1, m.counts["fallbacks"]

    F0, F1, fallbacks = fitted(True)
    assert math.isclose(F0, 674.021002536, rel_tol=1e-9)
    assert F1 < F0 and fallbacks == 1
    F0, F1, fallbacks = fitted(
        False
    )  # no safeguard: the uphill step is refused, no progress
    assert F1 == F0 and fallbacks == 0


def test_bad_spectrum_is_refused():
    K, y = _problem(n=20)
    with pytest.raises(ValueError):
        SpectralSVMPath(K, y, [0.1], spectrum="nystrom")


def test_torchkmsvc_truncated_spectrum_reaches_the_gap():
    from torchkm.estimators import TorchKMSVC

    X, y01 = make_classification(
        n_samples=150, n_features=6, n_informative=4, flip_y=0.05, random_state=0
    )
    Cs = np.logspace(-2, 1, 4)  # C ascending: lambda from large to small
    clf = TorchKMSVC(
        kernel="rbf",
        rbf_sigma=0.3,
        Cs=Cs,
        cv=3,
        device="cpu",
        spectrum="truncated",
        spectrum_rank=30,
        gap_tol=1e-3,
        random_state=0,
        store_path=True,
    ).fit(X, y01)
    assert clf.converged_.all()
    assert clf.cv_mis_.shape == (4,) and clf.best_C_ in Cs
    assert {"factorization", "path", "cross_validation"} <= set(clf.fit_timing_)
    assert clf.predict(X).shape == (150,)
    # every lambda of the path is within the gap of libsvm's optimum
    K = rbf_kernel(torch.as_tensor(X), 0.3)
    y = torch.as_tensor(np.where(y01 == 1, 1.0, -1.0))
    path = clf.alpmat_path_.double()
    for j, C in enumerate(Cs):
        lam = 1.0 / (2 * len(y) * C)
        a, b = _libsvm(K, y, lam)
        pstar = _primal(K, y, a, b, lam)
        P = _primal(K, y, path[1:, j], float(path[0, j]), lam)
        assert pstar - 1e-9 <= P <= pstar + 1e-3 * P


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_kernel_operator_matches_the_stored_kernel(dtype):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(70, 5, generator=g, dtype=torch.float64).to(dtype)
    K = rbf_kernel(X, 0.4)
    op = RBFKernelOperator(X, 0.4, block_bytes=70 * X.element_size() * 16)
    assert op.block_rows == 16 and op.shape == (70, 70) and op.dtype == dtype
    B = torch.randn(70, 3, generator=g, dtype=torch.float64).to(dtype)
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    torch.testing.assert_close(op @ B, K @ B, rtol=tol, atol=tol)
    torch.testing.assert_close(op @ B[:, 0], K @ B[:, 0], rtol=tol, atol=tol)


def test_matrix_free_path_matches_the_stored_kernel():
    X, y01 = make_classification(
        n_samples=120, n_features=6, n_informative=4, flip_y=0.05, random_state=2
    )
    Xt = torch.as_tensor(X)
    y = torch.as_tensor(np.where(y01 == 1, 1.0, -1.0))
    lams = np.logspace(-0.5, -2.5, 3)
    fold = torch.arange(120) % 3 + 1
    kw = dict(spectrum="truncated", rank=30, gap_tol=1e-4)
    K = rbf_kernel(Xt, 0.3)
    stored = SpectralSVMPath(K, y, lams, fold, **kw).fit()
    op = RBFKernelOperator(Xt, 0.3, block_bytes=120 * 8 * 32)  # 32 rows per block
    free = SpectralSVMPath(op, y, lams, fold, **kw).fit()
    assert bool(free.converged.all()) and bool(free.fold_converged.all())
    for j, lam in enumerate(lams):
        P = [
            _primal(K, y, m.alphas[1:, j], float(m.alphas[0, j]), lam)
            for m in (stored, free)
        ]
        assert abs(P[0] - P[1]) <= 2e-4 * max(P)
    assert float((free.cv_error - stored.cv_error).abs().max()) <= 2.0 / 120


def test_full_spectrum_needs_the_matrix():
    X = torch.randn(20, 3, dtype=torch.float64)
    with pytest.raises(ValueError):
        SpectralSVMPath(
            RBFKernelOperator(X, 0.5), torch.ones(20), [0.1], spectrum="full"
        )


def test_truncated_spectrum_option_is_checked():
    from torchkm.estimators import TorchKMDWD, TorchKMSVC

    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    for bad in (
        TorchKMSVC(spectrum="nystrom", device="cpu"),
        TorchKMSVC(spectrum="truncated", low_rank=True, device="cpu"),
        TorchKMSVC(spectrum="truncated", is_exact=1, device="cpu"),
        TorchKMDWD(spectrum="truncated", device="cpu"),
    ):
        with pytest.raises(ValueError):
            bad.fit(X, y)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_wide_blocks_certify_the_same_problems(dtype):
    # whole-data fits and folds of several lambdas share each product with K;
    # every fit is still certified, so objectives agree within the tolerance
    K, y = _problem(n=300, dtype=dtype)
    lams = np.logspace(-2, -4.5, 12)
    foldid = torch.as_tensor(np.arange(300) % 5 + 1)
    serial = SpectralSVMPath(K, y, lams, foldid, rank=40, block=1).fit()
    wide = SpectralSVMPath(K, y, lams, foldid, rank=40, block=4).fit()
    assert bool(wide.converged.all()) and bool(wide.fold_converged.all())
    assert float(np.nanmax(wide.gaps)) <= 1e-3
    assert float(np.nanmax(wide.fold_gaps)) <= 1e-3
    assert wide.count.reads < serial.count.reads
    for j, lam in enumerate(lams):
        P_w = _primal(K, y, wide.alphas[1:, j], float(wide.alphas[0, j]), lam)
        P_s = _primal(K, y, serial.alphas[1:, j], float(serial.alphas[0, j]), lam)
        assert abs(P_w - P_s) <= 2e-3 * max(P_w, P_s)
    assert wide.cv_scores.shape == serial.cv_scores.shape
    # in float32 the serial folds can stop uncertified at the smallest lambda
    # (fit_cap); compare CV errors where both are certified
    both = serial.fold_converged.all(0) & wide.fold_converged.all(0)
    assert int(both.sum()) >= len(lams) - 1
    diff = (wide.cv_error - serial.cv_error).abs()[both]
    assert float(diff.max()) <= 0.05


def test_block_must_be_positive():
    K, y = _problem(n=20)
    with pytest.raises(ValueError):
        SpectralSVMPath(K, y, [1e-2], block=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused path needs CUDA")
def test_fused_kernel_operator_matches_the_stored_kernel():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(300, 7, generator=g, dtype=torch.float64)
    K = rbf_kernel(X, 0.4)
    op = RBFKernelOperator(X.float().cuda(), 0.4, fused=True)
    for w in (1, 3, 130):  # 130 > FUSED_MAX_COLUMNS: two fused chunks
        B = torch.randn(300, w, generator=g, dtype=torch.float64)
        got = (op @ B.float().cuda()).double().cpu()
        torch.testing.assert_close(got, K @ B, rtol=1e-4, atol=1e-4)
    v = torch.randn(300, generator=g, dtype=torch.float64)
    torch.testing.assert_close(
        (op @ v.float().cuda()).double().cpu(), K @ v, rtol=1e-4, atol=1e-4
    )


def test_fused_kernel_operator_needs_cuda_float32():
    with pytest.raises(ValueError):
        RBFKernelOperator(torch.randn(10, 3, dtype=torch.float64), 0.5, fused=True)


def test_row_blocked_sums_give_the_same_fit(monkeypatch):
    # the float64 objective sums run in row blocks at large n; force blocks
    # of a few rows and compare with the unblocked fit
    import torchkm.experimental.spectral_svm as ss

    K, y = _problem(n=200)
    lams = np.logspace(-2, -3.5, 4)
    foldid = torch.as_tensor(np.arange(200) % 4 + 1)
    whole = SpectralSVMPath(K, y, lams, foldid, rank=30, block=2, tile=False).fit()
    monkeypatch.setattr(ss, "_ROW_BLOCK", 7 * 6)  # 7 rows per block at 6 columns
    blocked = SpectralSVMPath(K, y, lams, foldid, rank=30, block=2, tile=False).fit()
    # both certified; blocked float32 sums move the intercept search inside its
    # flat minimum, so compare objectives, decisions and CV errors, not alpha
    for j, lam in enumerate(lams):
        P_b = _primal(K, y, blocked.alphas[1:, j], float(blocked.alphas[0, j]), lam)
        P_w = _primal(K, y, whole.alphas[1:, j], float(whole.alphas[0, j]), lam)
        assert abs(P_b - P_w) <= 1e-3 * P_w
    f_b = K @ blocked.alphas[1:] + blocked.alphas[0]
    f_w = K @ whole.alphas[1:] + whole.alphas[0]
    assert float((f_b - f_w).abs().max()) <= 1e-2 * float(f_w.abs().max())
    assert float((blocked.cv_error - whole.cv_error).abs().max()) <= 2.0 / 200


@pytest.mark.parametrize(
    "fused", [False, pytest.param(True, marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="fused path needs CUDA"))]
)
def test_kernel_operator_cross_product(fused):
    g = torch.Generator().manual_seed(1)
    X = torch.randn(120, 5, generator=g, dtype=torch.float64)
    Xq = torch.randn(37, 5, generator=g, dtype=torch.float64)
    B = torch.randn(120, 3, generator=g, dtype=torch.float64)
    ref = torch.exp(-2 * 0.3 * torch.cdist(Xq, X) ** 2) @ B
    dev, dt = ("cuda", torch.float32) if fused else ("cpu", torch.float64)
    op = RBFKernelOperator(X.to(dev, dt), 0.3, block_bytes=120 * 8 * 10, fused=fused)
    got = op.cross(Xq.to(dev, dt), B.to(dev, dt)).double().cpu()
    tol = 1e-4 if fused else 1e-10
    torch.testing.assert_close(got, ref, rtol=tol, atol=tol)
    torch.testing.assert_close(
        op.cross(Xq.to(dev, dt), B[:, 0].to(dev, dt)).double().cpu(), ref[:, 0],
        rtol=tol, atol=tol,
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_tiled_wide_fit_matches_the_dense_one(monkeypatch, dtype):
    # row-tiled steps and certificate (labels never stored) against the dense
    # path; tiny tiles so that every quantity spans several tiles. Both stop at
    # certified gaps, and rounding moves alpha along near-null directions of K,
    # so the comparison is on what is identified: objective, gaps, decisions
    import torchkm.experimental.spectral_svm as ss

    K, y = _problem(n=240, dtype=dtype)
    lams = np.logspace(-2, -3.5, 6)
    foldid = torch.as_tensor(np.arange(240) % 4 + 1)
    dense = SpectralSVMPath(K, y, lams, foldid, rank=30, block=3, tile=False).fit()
    monkeypatch.setattr(ss, "_ROW_BLOCK", 15 * 37)  # 37 rows per tile at 15 columns
    tiled = SpectralSVMPath(K, y, lams, foldid, rank=30, block=3, tile=True).fit()
    assert bool(tiled.converged.all()) and bool(tiled.fold_converged.all())
    assert float(tiled.gaps.max()) <= 1e-3 and float(np.nanmax(tiled.fold_gaps)) <= 1e-3
    for j, lam in enumerate(lams):
        P_t = _primal(K, y, tiled.alphas[1:, j], float(tiled.alphas[0, j]), lam)
        P_d = _primal(K, y, dense.alphas[1:, j], float(dense.alphas[0, j]), lam)
        assert abs(P_t - P_d) <= 1e-3 * P_d
    f_t = K.double() @ tiled.alphas[1:].double() + tiled.alphas[0].double()
    f_d = K.double() @ dense.alphas[1:].double() + dense.alphas[0].double()
    assert float((f_t - f_d).abs().max()) <= 1e-2 * float(f_d.abs().max())
    assert float((tiled.cv_error - dense.cv_error).abs().max()) <= 2.0 / 240


def test_tiled_step_and_certificate_match_the_dense_ones(monkeypatch):
    # one step from the same state, and the certificate (with refinement) of
    # the same point: equal to rounding
    import torchkm.experimental.spectral_svm as ss

    K, y = _problem(n=240)
    foldid = torch.as_tensor(np.arange(240) % 4 + 1)
    m = SpectralSVMPath(K, y, [1e-2, 5e-3], foldid, rank=30, block=2, tile=False).fit()
    lab = ss._Labels(y, torch.as_tensor(np.arange(240) % 4), torch.arange(-1, 4).repeat(2))
    Yd = lab.rows(0, 240)
    lam = torch.tensor([1e-2] * 5 + [5e-3] * 5, dtype=torch.float64)
    A0 = torch.cat([m.alphas[1:, :1].expand(-1, 5), m.alphas[1:, 1:].expand(-1, 5)], 1)
    b0 = torch.cat([m.alphas[0, :1].expand(5), m.alphas[0, 1:].expand(5)])
    A0 = A0 + 1e-3 * torch.randn(A0.shape, generator=torch.Generator().manual_seed(0), dtype=A0.dtype)
    monkeypatch.setattr(ss, "_ROW_BLOCK", 10 * 37)
    zero = torch.zeros(10, dtype=torch.float64)
    A, b, KA = A0.clone(), b0.clone(), K @ A0
    A2, b2, KA2 = A0.clone(), b0.clone(), K @ A0
    m._round(Yd, A, b, KA, lam, 0.125, zero, 3)
    m._round_tiled(lab, A2, b2, KA2, lam, 0.125, zero, 3)
    torch.testing.assert_close(A2, A, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(KA2, KA, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(b2, b, rtol=1e-10, atol=1e-12)
    lmax = float(torch.linalg.matrix_norm(K, 2))
    kw = dict(Ka=K @ A0, delta=1e-3, refine=20, lmax=lmax, target=4e-5)
    g1, P1, D1 = ss.hinge_duality_gap(K, Yd, A0, b0, lam, **kw)
    g2, P2, D2 = ss._hinge_duality_gap_tiled(K, lab, A0, b0, lam, **kw)
    torch.testing.assert_close(g2, g1, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(D2, D1, rtol=1e-10, atol=1e-12)
