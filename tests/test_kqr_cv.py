# SPDX-License-Identifier: MIT
"""Cross-validation of the exact kernel quantile regression solver.

Each fold is warm-started from the full-data fit and reuses the preconditioners
the path built for each smoothing bandwidth, so the fold fits must follow the
path's bandwidth schedule, and the held-out rows must stay out of every loss
term, including the intercept search. A zeroed response removes a row from a
margin loss, but not from the check loss.
"""

import numpy as np
import pytest
import torch

from torchkm.cvkqr import cvkqr
from torchkm.functions import rbf_kernel, sigest


def _problem(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] + 0.3 * rng.standard_t(3, size=n)
    X = torch.as_tensor((X - X.mean(0)) / X.std(0))
    y = torch.as_tensor((y - y.mean()) / y.std())
    torch.manual_seed(seed)
    return rbf_kernel(X, float(sigest(X))).double(), y


def _fit(K, y, lams, tau, foldid, **kwargs):
    model = cvkqr(
        Kmat=K,
        y=y,
        nlam=len(lams),
        ulam=torch.as_tensor(lams, dtype=torch.double),
        tau=tau,
        foldid=foldid,
        nfolds=int(foldid.max()),
        gamma=1e-8,
        device="cpu",
        **kwargs,
    )
    model.fit()
    return model


def _pinball(u, tau):
    return np.mean(np.maximum(tau * u, (tau - 1.0) * u), axis=0)


def test_held_out_rows_add_no_loss():
    rng = np.random.default_rng(1)
    n, b, lam, tau = 30, 0.2, 0.1, 0.3
    ka = torch.as_tensor(rng.normal(size=n))
    y = torch.as_tensor(rng.normal(size=n))
    held = torch.arange(n) % 3 == 0
    aka = torch.tensor(1.5, dtype=torch.double)
    got = cvkqr.objfun(None, b, aka, ka, y, lam, n, tau, 1e-9, held_out=held)
    u = (y - ka - b).numpy()[~held.numpy()]
    want = lam / 2 * 1.5 + np.sum(np.maximum(tau * u, (tau - 1) * u)) / n
    assert float(got) == pytest.approx(want + 1e-8 * b**2, rel=1e-12)


@pytest.mark.parametrize("is_exact", [0, 1])
def test_cv_converges(is_exact):
    K, y = _problem()
    foldid = torch.arange(len(y)) % 4 + 1
    maxit = 20000
    model = _fit(
        K, y, [1e-1, 1e-2], 0.5, foldid, eps=1e-5, maxit=maxit, is_exact=is_exact
    )
    assert torch.isfinite(model.pred).all()
    # Every lambda's fold fits finish well inside the shared iteration budget.
    assert int(model.cvnpass.max()) < maxit


@pytest.mark.parametrize("tau", [0.1, 0.5])
def test_cv_loss_matches_direct_fold_fits(tau):
    K, y = _problem()
    n = len(y)
    foldid = torch.arange(n) % 4 + 1
    lams = np.array([1e-1, 1e-2])
    model = _fit(K, y, lams, tau, foldid, eps=1e-5, maxit=20000)

    # Fold k solves (lam/2) a'Ka + (1/n) sum_{i not in k} loss_i; rescaled to
    # the solver's mean over the fold's own rows this is lam * n / n_train.
    direct = np.zeros((n, len(lams)))
    for k in range(1, 5):
        tr = torch.nonzero(foldid != k).squeeze(1)
        te = torch.nonzero(foldid == k).squeeze(1)
        fit = _fit(
            K[tr][:, tr],
            y[tr],
            lams * n / len(tr),
            tau,
            torch.arange(len(tr)) % 3 + 1,
            eps=1e-9,
            maxit=200000,
            KKTeps=1e-6,
            delta_len=8,
        )
        direct[te.numpy()] = (K[te][:, tr] @ fit.alpmat[1:] + fit.alpmat[0]).numpy()

    # The fold fits start from the full-data fit and stop at the CV tolerance,
    # so the CV loss is close to, not equal to, the exact one.
    yv = y.numpy()[:, None]
    np.testing.assert_allclose(
        _pinball(yv - model.pred.numpy(), tau), _pinball(yv - direct, tau), rtol=0.1
    )


def _primal(K, y, alp, lam, tau):
    a, b = alp[1:], alp[0]
    r = y - K @ a - b
    return float((lam / 2) * a @ (K @ a) + torch.maximum(tau * r, (tau - 1) * r).mean())


def test_gap_stop_prints_nothing_and_reports_gaps(capsys):
    # the old KKT test could not be met (rows the fit interpolates keep a
    # nonzero subgradient residual), so every lambda printed a message
    K, y = _problem(n=120)
    foldid = torch.as_tensor(np.arange(120) % 3 + 1)
    model = _fit(K, y, [1e-1, 1e-2], 0.3, foldid, eps=1e-5, maxit=20000)
    assert capsys.readouterr().out == ""
    assert model.gaps.shape == (2,) and model.fold_gaps.shape == (3, 2)
    assert bool(torch.isfinite(model.gaps).all())


def test_reported_gap_bounds_the_suboptimality():
    # the gap of a default fit bounds its distance to a much tighter solve
    K, y = _problem(n=80)
    foldid = torch.as_tensor(np.arange(80) % 2 + 1)
    lams = [1e-1, 1e-2]
    loose = _fit(K, y, lams, 0.3, foldid, eps=1e-5, maxit=20000)
    tight = _fit(
        K, y, lams, 0.3, foldid, eps=1e-5, maxit=50000, max_tighten=4, gap_tol=1e-6,
        delta_len=6,
    )
    for j, lam in enumerate(lams):
        P = _primal(K, y, loose.alpmat[:, j], lam, 0.3)
        P_ref = _primal(K, y, tight.alpmat[:, j], lam, 0.3)
        assert P_ref <= P + 1e-12  # the tighter solve is at least as good
        assert P - P_ref <= float(loose.gaps[j]) * P + 1e-12


def test_cross_validation_uses_eps():
    # the folds stopped on a hard-coded 1e-5 before, whatever eps was
    K, y = _problem(n=120)
    foldid = torch.as_tensor(np.arange(120) % 3 + 1)
    coarse = _fit(K, y, [1e-2], 0.3, foldid, eps=1e-3, maxit=20000)
    fine = _fit(K, y, [1e-2], 0.3, foldid, eps=1e-9, maxit=20000)
    assert int(fine.cvnpass.sum()) > int(coarse.cvnpass.sum())
