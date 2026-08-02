# SPDX-License-Identifier: MIT
import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")

import torch
from sklearn.datasets import make_circles
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from torchkm import ConvergenceWarning as TorchKMConvergenceWarning
from torchkm.estimators import TorchKMDWD, TorchKMLogit, TorchKMSVC


def _hard_data(n_samples=120, random_state=0):
    X, y = make_circles(
        n_samples=n_samples, factor=0.4, noise=0.08, random_state=random_state
    )
    X = StandardScaler().fit_transform(X)
    return X, y


def test_exported_warning_is_sklearn_convergence_warning():
    assert TorchKMConvergenceWarning is ConvergenceWarning


@pytest.mark.parametrize("Estimator", [TorchKMSVC, TorchKMDWD, TorchKMLogit])
def test_low_max_iter_raises_convergence_warning(Estimator):
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = _hard_data()

    clf = Estimator(
        kernel="rbf",
        Cs=np.logspace(2, -2, num=4),
        cv=5,
        device="cpu",
        random_state=0,
        max_iter=5,  # deliberately too low to converge
    )
    with pytest.warns(ConvergenceWarning):
        clf.fit(X, y)

    # per-lambda convergence status is exposed on the fitted estimator
    assert clf.converged_ is not None
    assert clf.converged_.shape == (4,)
    assert not clf.converged_.all()


@pytest.mark.parametrize("Estimator", [TorchKMSVC, TorchKMDWD, TorchKMLogit])
def test_converged_fit_does_not_warn(Estimator):
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = _hard_data(n_samples=60)

    clf = Estimator(
        kernel="rbf",
        nC=2,
        cv=3,
        device="cpu",
        random_state=0,
        max_iter=100000,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        clf.fit(X, y)

    assert clf.converged_ is not None
    assert clf.converged_.all()
