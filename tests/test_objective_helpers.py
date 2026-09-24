# SPDX-License-Identifier: MIT
"""The exact solvers' objective helpers, used by the intercept line search.

Each solver refines the intercept with a golden-section search over its
``objfun``. If ``objfun`` is not the solver's own objective, the search returns
an intercept that is optimal for a different function. These tests pin every
helper to its loss formula and check that a fitted intercept cannot be improved
on the true objective.
"""

import numpy as np
import pytest
import torch
from scipy.optimize import minimize_scalar
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from torchkm.cvkdwd import cvkdwd
from torchkm.cvklogit import cvklogit
from torchkm.cvksqsvm import cvksqsvm
from torchkm.cvksvm import cvksvm
from torchkm.functions import rbf_kernel


def _dwd(u):
    return np.where(u <= 0.5, 1.0 - u, 1.0 / (4.0 * np.maximum(u, 1e-300)))


LOSSES = {
    cvkdwd: _dwd,
    cvklogit: lambda u: np.logaddexp(0.0, -u),
    cvksqsvm: lambda u: np.maximum(0.0, 1.0 - u) ** 2,
    cvksvm: lambda u: np.maximum(0.0, 1.0 - u),
}


def _problem(n=120, seed=1):
    X, y = make_classification(
        n_samples=n,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        flip_y=0.05,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X)
    y = np.where(y == 0, -1.0, 1.0)
    K = rbf_kernel(torch.as_tensor(X), 0.15)
    return K, y


def _true_objective(loss, Kn, y, alpha, b, lam):
    return float(np.mean(loss(y * (Kn @ alpha + b))) + lam * alpha @ Kn @ alpha)


@pytest.mark.parametrize("cls", list(LOSSES))
def test_helper_is_the_solver_objective(cls):
    """objfun = lam * a'Ka + mean(loss(y f)) at arbitrary (alpha, b)."""
    K, y = _problem()
    Kn = K.numpy()
    rng = np.random.default_rng(0)
    for _ in range(5):
        alpha = rng.normal(scale=0.3, size=len(y))
        b = float(rng.normal())
        lam = float(10 ** rng.uniform(-4, -1))
        ka = torch.as_tensor(Kn @ alpha)
        aka = torch.dot(ka, torch.as_tensor(alpha))
        got = float(
            cls.objfun(
                None,
                torch.tensor(b, dtype=torch.double),
                aka,
                ka,
                torch.as_tensor(y),
                lam,
                len(y),
            )
        )
        want = _true_objective(LOSSES[cls], Kn, y, alpha, b, lam)
        assert got == pytest.approx(want, rel=1e-10)


def test_logistic_helper_is_finite_for_large_margins():
    ka = torch.tensor([800.0, -800.0], dtype=torch.double)
    val = cvklogit.objfun(
        None, torch.tensor(0.0), torch.tensor(0.0), ka, torch.ones(2), 0.0, 2
    )
    assert torch.isfinite(val)
    assert float(val) == pytest.approx(400.0)  # (0 + 800) / 2


@pytest.mark.parametrize("cls", [cvkdwd, cvklogit, cvksqsvm])
def test_fitted_intercept_cannot_be_improved(cls):
    K, y = _problem()
    Kn = K.numpy()
    lams = np.array([1e-1, 1e-2, 1e-3])
    model = cls(
        Kmat=K,
        y=torch.as_tensor(y),
        nlam=len(lams),
        ulam=torch.as_tensor(lams),
        foldid=torch.arange(len(y)) % 4 + 1,
        nfolds=4,
        eps=1e-8,
        maxit=500000,
        gamma=1e-8,
        device="cpu",
    )
    model.fit()
    loss = LOSSES[cls]
    for j, lam in enumerate(lams):
        alpha = model.alpmat[1:, j].numpy()
        b = float(model.alpmat[0, j])
        at_fit = _true_objective(loss, Kn, y, alpha, b, lam)
        best_b = minimize_scalar(
            lambda bb: _true_objective(loss, Kn, y, alpha, bb, lam),
            bounds=(-10.0, 10.0),
            method="bounded",
            options={"xatol": 1e-10},
        ).fun
        assert at_fit - best_b <= 1e-6 * max(1.0, abs(best_b))
