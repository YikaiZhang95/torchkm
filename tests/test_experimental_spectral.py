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
from torchkm.experimental import SpectralSVMPath, hinge_duality_gap
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
