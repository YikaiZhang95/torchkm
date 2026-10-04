# SPDX-License-Identifier: MIT
"""The truncated-spectrum SVM path in the package API (torchkm.cvksvm and
torchkm.functions), which the estimators use. torchkm.experimental keeps its
own copy for exploring; these tests do not depend on it."""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from torchkm.cvksvm import SpectralSVMPath, hinge_duality_gap
from torchkm.functions import RBFKernelOperator, rbf_kernel, sigest


def _problem(n=200, seed=0, dtype=torch.float64):
    X, y = make_classification(
        n_samples=n, n_features=8, n_informative=5, flip_y=0.05, random_state=seed
    )
    Xt = torch.as_tensor(X)
    torch.manual_seed(seed)
    sig = float(sigest(Xt))
    return Xt, rbf_kernel(Xt, sig).to(dtype), torch.as_tensor(
        np.where(y == 0, -1.0, 1.0), dtype=dtype
    ), sig


def _primal(K, y, a, b, lam):
    Ka = K.double() @ a.double()
    return float(
        torch.clamp(1 - y.double() * (Ka + b), min=0).mean() + lam * a.double() @ Ka
    )


def _libsvm(K, y, lam):
    n = len(y)
    s = SVC(kernel="precomputed", C=1.0 / (2 * n * lam), tol=1e-10, max_iter=-1)
    s.fit(K.double().numpy(), y.double().numpy())
    a = np.zeros(n)
    a[s.support_] = s.dual_coef_[0]
    return torch.as_tensor(a), float(s.intercept_[0])


@pytest.mark.parametrize("spectrum", ["truncated", "full"])
def test_path_is_certified_and_near_libsvm(spectrum):
    _, K, y, _ = _problem()
    lams = [1e-1, 1e-2, 1e-3]
    m = SpectralSVMPath(K, y, lams, spectrum=spectrum, rank=40).fit()
    assert bool(m.converged.all()) and float(m.gaps.max()) <= 1e-3
    for j, lam in enumerate(lams):
        a, b = _libsvm(K, y, lam)
        P_ref = _primal(K, y, a, b, lam)
        P = _primal(K, y, m.alphas[1:, j], float(m.alphas[0, j]), lam)
        assert P - P_ref <= 1e-3 * P_ref + 1e-9


def test_wide_blocks_certify_every_fit():
    _, K, y, _ = _problem(n=240)
    foldid = torch.as_tensor(np.arange(240) % 4 + 1)
    m = SpectralSVMPath(K, y, np.logspace(-2, -3.5, 6), foldid, rank=30, block=3).fit()
    assert bool(m.converged.all()) and bool(m.fold_converged.all())
    assert float(np.nanmax(m.fold_gaps)) <= 1e-3
    assert m.cv_scores.shape == (240, 6)


def test_never_stored_kernel_gives_the_stored_fit():
    X, K, y, sig = _problem(n=150)
    foldid = torch.as_tensor(np.arange(150) % 3 + 1)
    lams = [1e-1, 1e-2]
    op = RBFKernelOperator(X, sig, block_bytes=150 * 8 * 16)  # several row blocks
    B = torch.randn(150, 3, dtype=torch.float64)
    torch.testing.assert_close(op @ B, K @ B, rtol=1e-12, atol=1e-12)
    a = SpectralSVMPath(K, y, lams, foldid, rank=30, block=2, seed=1).fit()
    b = SpectralSVMPath(op, y, lams, foldid, rank=30, block=2, seed=1).fit()
    # rounding of the blocked products moves alpha along near-null directions
    # of K only: decisions, objectives and CV errors agree
    f_a = K @ a.alphas[1:] + a.alphas[0]
    f_b = K @ b.alphas[1:] + b.alphas[0]
    assert float((f_a - f_b).abs().max()) <= 1e-2 * float(f_a.abs().max())
    for j, lam in enumerate(lams):
        P_a = _primal(K, y, a.alphas[1:, j], float(a.alphas[0, j]), lam)
        P_b = _primal(K, y, b.alphas[1:, j], float(b.alphas[0, j]), lam)
        assert abs(P_a - P_b) <= 1e-3 * P_a
    assert float((a.cv_error - b.cv_error).abs().max()) <= 2.0 / 150


def test_certificate_bounds_the_true_gap():
    _, K, y, _ = _problem(n=150)
    lam = 1e-2
    a, b = _libsvm(K, y, lam)
    P_ref = _primal(K, y, a, b, lam)
    noisy = a + 1e-3 * torch.randn(len(a), dtype=a.dtype)
    gap, P, D = hinge_duality_gap(K, y, noisy, b, lam, refine=20)
    assert float(D) <= P_ref + 1e-9 <= float(P) + 2e-9
    assert float(gap) >= (float(P) - P_ref) / float(P) - 1e-9


def test_estimators_do_not_use_the_experimental_copy(monkeypatch):
    # TorchKMSVC's truncated and matrix-free modes run on the package API
    import torchkm.experimental as exp
    from torchkm import TorchKMSVC

    def refuse(*args, **kwargs):
        raise AssertionError("the estimator used torchkm.experimental")

    monkeypatch.setattr(exp, "SpectralSVMPath", refuse)
    monkeypatch.setattr(exp, "RBFKernelOperator", refuse)
    X, _, y, _ = _problem(n=120)
    kw = dict(Cs=np.array([1.0, 0.1]), nC=2, cv=3, device="cpu", random_state=0)
    for extra in (dict(spectrum="truncated"), dict(low_rank=True)):
        clf = TorchKMSVC(**kw, **extra).fit(X.numpy(), y.numpy())
        assert np.isfinite(clf.decision_function(X.numpy()[:5])).all()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_tiled_wide_fit_matches_the_dense_one(monkeypatch, dtype):
    # the large-n path: labels never stored, steps and certificate in row
    # tiles (tiny tiles here so that every quantity spans several)
    import torchkm.cvksvm as ck

    _, K, y, _ = _problem(n=240, dtype=dtype)
    lams = np.logspace(-2, -3.5, 6)
    foldid = torch.as_tensor(np.arange(240) % 4 + 1)
    dense = SpectralSVMPath(K, y, lams, foldid, rank=30, block=3, tile=False).fit()
    monkeypatch.setattr(ck, "_TILE_BLOCK", 15 * 37)  # 37 rows per tile, 15 columns
    tiled = SpectralSVMPath(K, y, lams, foldid, rank=30, block=3, tile=True).fit()
    assert bool(tiled.converged.all()) and bool(tiled.fold_converged.all())
    assert float(np.nanmax(tiled.fold_gaps)) <= 1e-3
    for j, lam in enumerate(lams):
        P_t = _primal(K, y, tiled.alphas[1:, j], float(tiled.alphas[0, j]), lam)
        P_d = _primal(K, y, dense.alphas[1:, j], float(dense.alphas[0, j]), lam)
        assert abs(P_t - P_d) <= 1e-3 * P_d
    assert float((tiled.cv_error - dense.cv_error).abs().max()) <= 2.0 / 240


def test_tiled_step_and_certificate_match_the_dense_ones(monkeypatch):
    import torchkm.cvksvm as ck

    _, K, y, _ = _problem(n=240)
    foldid = torch.as_tensor(np.arange(240) % 4 + 1)
    m = SpectralSVMPath(K, y, [1e-2, 5e-3], foldid, rank=30, block=2, tile=False).fit()
    lab = ck._Labels(y, torch.as_tensor(np.arange(240) % 4), torch.arange(-1, 4).repeat(2))
    Yd = lab.rows(0, 240)
    lam = torch.tensor([1e-2] * 5 + [5e-3] * 5, dtype=torch.float64)
    A0 = torch.cat([m.alphas[1:, :1].expand(-1, 5), m.alphas[1:, 1:].expand(-1, 5)], 1)
    b0 = torch.cat([m.alphas[0, :1].expand(5), m.alphas[0, 1:].expand(5)])
    A0 = A0 + 1e-3 * torch.randn(A0.shape, generator=torch.Generator().manual_seed(0), dtype=A0.dtype)
    monkeypatch.setattr(ck, "_TILE_BLOCK", 10 * 37)
    zero = torch.zeros(10, dtype=torch.float64)
    A, b, KA = A0.clone(), b0.clone(), K @ A0
    A2, b2, KA2 = A0.clone(), b0.clone(), K @ A0
    m._round(Yd, A, b, KA, lam, 0.125, zero, 3)
    m._round_tiled(lab, A2, b2, KA2, lam, 0.125, zero, 3)
    torch.testing.assert_close(A2, A, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(b2, b, rtol=1e-10, atol=1e-12)
    lmax = float(torch.linalg.matrix_norm(K, 2))
    kw = dict(Ka=K @ A0, delta=1e-3, refine=20, lmax=lmax, target=4e-5)
    g1, _, D1 = ck.hinge_duality_gap(K, Yd, A0, b0, lam, **kw)
    g2, _, D2 = ck._hinge_duality_gap_tiled(K, lab, A0, b0, lam, **kw)
    torch.testing.assert_close(g2, g1, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(D2, D1, rtol=1e-10, atol=1e-12)
