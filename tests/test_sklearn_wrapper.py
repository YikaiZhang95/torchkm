import numpy as np
import pytest

pytest.importorskip("sklearn")

import torch
from sklearn.base import clone
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torchkm.sklearn_wrapper import TorchKMSVC


def _toy_data(n_samples=120, random_state=0):
    X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=random_state)
    # y is {0,1} -> wrapper should internally map to {-1,+1} as needed
    return X.astype(np.float32), y


@pytest.mark.parametrize("probability", [False, True])
def test_pipeline_fit_predict(probability):
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)

    X, y = _toy_data(n_samples=120, random_state=0)

    # Keep runtime small: small nlam, small nfolds
    clf = make_pipeline(
        StandardScaler(),
        TorchKMSVC(
            kernel="rbf",
            nlam=5,
            nfolds=3,
            device="cpu",
            random_state=123,
            maxit=80,
            probability=probability,
        ),
    )

    clf.fit(X, y)
    pred = clf.predict(X[:10])

    assert pred.shape == (10,)
    # predictions must be in original label set {0,1}
    assert set(np.unique(pred)).issubset(set(np.unique(y)))


def test_clone_works():
    X, y = _toy_data(n_samples=100, random_state=1)

    est = TorchKMSVC(
        kernel="rbf",
        nlam=3,
        nfolds=3,
        device="cpu",
        random_state=7,
        maxit=60,
        probability=False,
    )

    est2 = clone(est)
    # sklearn clone() requires params round-trip cleanly
    assert est.get_params() == est2.get_params()

    est2.fit(X, y)
    pred = est2.predict(X[:5])
    assert pred.shape == (5,)


def test_decision_function_shape():
    X, y = _toy_data(n_samples=120, random_state=2)

    est = TorchKMSVC(
        kernel="rbf",
        nlam=3,
        nfolds=3,
        device="cpu",
        random_state=0,
        maxit=60,
        probability=False,
    ).fit(X, y)

    scores = est.decision_function(X[:11])
    assert scores.shape == (11,)
    assert np.isfinite(scores).all()


def test_predict_proba_sums_to_one():
    X, y = _toy_data(n_samples=120, random_state=3)

    est = TorchKMSVC(
        kernel="rbf",
        nlam=5,
        nfolds=3,
        device="cpu",
        random_state=0,
        maxit=80,
        probability=True,
    ).fit(X, y)

    proba = est.predict_proba(X[:17])
    assert proba.shape == (17, 2)
    assert np.all(proba >= -1e-8) and np.all(proba <= 1 + 1e-8)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_random_state_deterministic_folds():
    X, y = _toy_data(n_samples=150, random_state=4)

    est1 = TorchKMSVC(
        kernel="rbf",
        nlam=3,
        nfolds=5,
        device="cpu",
        random_state=999,
        maxit=60,
        probability=False,
    ).fit(X, y)

    est2 = TorchKMSVC(
        kernel="rbf",
        nlam=3,
        nfolds=5,
        device="cpu",
        random_state=999,
        maxit=60,
        probability=False,
    ).fit(X, y)

    # This requires your wrapper to store foldid_ during fit()
    assert hasattr(est1, "foldid_") and hasattr(est2, "foldid_")
    assert np.array_equal(est1.foldid_, est2.foldid_)

    # Different random_state should usually produce different folds
    est3 = TorchKMSVC(
        kernel="rbf",
        nlam=3,
        nfolds=5,
        device="cpu",
        random_state=1000,
        maxit=60,
        probability=False,
    ).fit(X, y)

    assert not np.array_equal(est1.foldid_, est3.foldid_)
