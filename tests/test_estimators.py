# SPDX-License-Identifier: MIT
import numpy as np
import pytest

pytest.importorskip("sklearn")

import torch
from sklearn.base import clone
from sklearn.datasets import make_moons, make_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from torchkm.estimators import TorchKMSVC, TorchKMKQR


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

    # Keep runtime small: small nC, small cv
    clf = make_pipeline(
        StandardScaler(),
        TorchKMSVC(
            kernel="rbf",
            nC=5,
            cv=3,
            device="cpu",
            random_state=123,
            max_iter=80,
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
        nC=3,
        cv=3,
        device="cpu",
        random_state=7,
        max_iter=60,
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
        nC=3,
        cv=3,
        device="cpu",
        random_state=0,
        max_iter=60,
        probability=False,
    ).fit(X, y)

    scores = est.decision_function(X[:11])
    assert scores.shape == (11,)
    assert np.isfinite(scores).all()


def test_predict_proba_sums_to_one():
    X, y = _toy_data(n_samples=120, random_state=3)

    est = TorchKMSVC(
        kernel="rbf",
        nC=5,
        cv=3,
        device="cpu",
        random_state=0,
        max_iter=80,
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
        nC=3,
        cv=5,
        device="cpu",
        random_state=999,
        max_iter=60,
        probability=False,
    ).fit(X, y)

    est2 = TorchKMSVC(
        kernel="rbf",
        nC=3,
        cv=5,
        device="cpu",
        random_state=999,
        max_iter=60,
        probability=False,
    ).fit(X, y)

    # This requires your wrapper to store foldid_ during fit()
    assert hasattr(est1, "foldid_") and hasattr(est2, "foldid_")
    assert np.array_equal(est1.foldid_, est2.foldid_)

    # Different random_state should usually produce different folds
    est3 = TorchKMSVC(
        kernel="rbf",
        nC=3,
        cv=5,
        device="cpu",
        random_state=1000,
        max_iter=60,
        probability=False,
    ).fit(X, y)

    assert not np.array_equal(est1.foldid_, est3.foldid_)


def _toy_reg_data(n_samples=120, random_state=0):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=5,
        noise=0.5,
        random_state=random_state,
    )
    return X.astype(np.float32), y.astype(np.float64)


def test_kqr_pipeline_fit_predict():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)

    X, y = _toy_reg_data(n_samples=120, random_state=0)

    reg = make_pipeline(
        StandardScaler(),
        TorchKMKQR(
            kernel="rbf",
            nC=5,
            cv=3,
            tau=0.5,
            device="cpu",
            random_state=123,
            max_iter=200,
        ),
    )

    reg.fit(X, y)
    pred = reg.predict(X[:10])

    assert pred.shape == (10,)
    assert np.isfinite(pred).all()


def test_kqr_clone_works():
    X, y = _toy_reg_data(n_samples=100, random_state=1)

    est = TorchKMKQR(
        kernel="rbf",
        nC=3,
        cv=3,
        tau=0.5,
        device="cpu",
        random_state=7,
        max_iter=200,
    )

    est2 = clone(est)
    assert est.get_params() == est2.get_params()

    est2.fit(X, y)
    pred = est2.predict(X[:5])
    assert pred.shape == (5,)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("tau", [0.25, 0.5, 0.75])
def test_kqr_quantile_levels(tau):
    X, y = _toy_reg_data(n_samples=150, random_state=2)

    est = TorchKMKQR(
        kernel="rbf",
        nC=3,
        cv=3,
        tau=tau,
        device="cpu",
        random_state=0,
        max_iter=200,
    ).fit(X, y)

    pred = est.predict(X)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()

    below = float(np.mean(y <= pred))
    assert abs(below - tau) < 0.25


def test_kqr_random_state_deterministic_folds():
    X, y = _toy_reg_data(n_samples=150, random_state=4)

    common = dict(
        kernel="rbf",
        nC=3,
        cv=5,
        tau=0.5,
        device="cpu",
        max_iter=200,
    )

    est1 = TorchKMKQR(random_state=999, **common).fit(X, y)
    est2 = TorchKMKQR(random_state=999, **common).fit(X, y)
    est3 = TorchKMKQR(random_state=1000, **common).fit(X, y)

    assert hasattr(est1, "foldid_") and hasattr(est2, "foldid_")
    assert np.array_equal(est1.foldid_, est2.foldid_)
    assert not np.array_equal(est1.foldid_, est3.foldid_)


def test_nys_kqr_pipeline_fit_predict():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)

    X, y = _toy_reg_data(n_samples=200, random_state=0)

    reg = make_pipeline(
        StandardScaler(),
        TorchKMKQR(
            kernel="rbf",
            low_rank=True,
            nC=5,
            cv=3,
            tau=0.5,
            device="cpu",
            random_state=123,
            max_iter=200,
            num_landmarks=50,
            nys_k=30,
        ),
    )

    reg.fit(X, y)
    pred = reg.predict(X[:10])

    assert pred.shape == (10,)
    assert np.isfinite(pred).all()


def test_nys_kqr_clone_works():
    X, y = _toy_reg_data(n_samples=150, random_state=1)

    est = TorchKMKQR(
        kernel="rbf",
        low_rank=True,
        nC=3,
        cv=3,
        tau=0.5,
        device="cpu",
        random_state=7,
        max_iter=200,
        num_landmarks=40,
        nys_k=20,
    )

    est2 = clone(est)
    assert est.get_params() == est2.get_params()

    est2.fit(X, y)
    pred = est2.predict(X[:5])
    assert pred.shape == (5,)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("tau", [0.25, 0.5, 0.75])
def test_nys_kqr_quantile_levels(tau):
    X, y = _toy_reg_data(n_samples=200, random_state=2)

    est = TorchKMKQR(
        kernel="rbf",
        low_rank=True,
        nC=3,
        cv=3,
        tau=tau,
        device="cpu",
        random_state=0,
        max_iter=200,
        num_landmarks=60,
        nys_k=30,
    ).fit(X, y)

    pred = est.predict(X)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()

    below = float(np.mean(y <= pred))
    assert abs(below - tau) < 0.25


def test_nys_kqr_random_state_deterministic_folds():
    X, y = _toy_reg_data(n_samples=200, random_state=4)

    common = dict(
        nC=3,
        cv=5,
        tau=0.5,
        device="cpu",
        max_iter=200,
        num_landmarks=50,
        nys_k=25,
        low_rank=True,
    )

    est1 = TorchKMKQR(random_state=999, **common).fit(X, y)
    est2 = TorchKMKQR(random_state=999, **common).fit(X, y)
    est3 = TorchKMKQR(random_state=1000, **common).fit(X, y)

    assert hasattr(est1, "foldid_") and hasattr(est2, "foldid_")
    assert np.array_equal(est1.foldid_, est2.foldid_)
    assert not np.array_equal(est1.foldid_, est3.foldid_)
