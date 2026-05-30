# SPDX-License-Identifier: MIT
import numpy as np
import pytest
from sklearn.base import clone

from torchkm.estimators import TorchKMKQR


def _make_regression(seed=0, n=60, p=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = np.sin(X[:, 0]) + 0.25 * X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def test_torchkmkqr_exact_smoke_cpu():
    X, y = _make_regression()
    Cs = np.logspace(1, -1, 3)

    model = TorchKMKQR(
        kernel="rbf",
        Cs=Cs,
        nC=len(Cs),
        cv=3,
        tau=0.5,
        device="cpu",
        max_iter=20,
        random_state=0,
    )
    clone(model)
    model.fit(X, y)
    pred = model.predict(X[:5])

    assert pred.shape == (5,)
    assert np.all(np.isfinite(pred))
    assert hasattr(model, "best_C_")
    assert hasattr(model, "cv_loss_")


def test_torchkmkqr_low_rank_smoke_cpu():
    X, y = _make_regression(n=80)
    Cs = np.logspace(1, -1, 3)

    model = TorchKMKQR(
        kernel="rbf",
        Cs=Cs,
        nC=len(Cs),
        cv=3,
        tau=0.5,
        low_rank=True,
        num_landmarks=20,
        nys_k=10,
        device="cpu",
        max_iter=20,
        random_state=0,
    )
    clone(model)
    model.fit(X, y)
    pred = model.predict(X[:5])

    assert pred.shape == (5,)
    assert np.all(np.isfinite(pred))
    assert hasattr(model, "best_C_")
    assert hasattr(model, "cv_loss_")
    assert hasattr(model, "_low_rank_backend_")
    assert model._low_rank_backend_.__class__.__name__ == "cvknyqr"


def test_torchkmkqr_rejects_invalid_tau():
    X, y = _make_regression()
    model = TorchKMKQR(tau=1.5, device="cpu", max_iter=5)
    with pytest.raises(ValueError, match="tau"):
        model.fit(X, y)


def test_torchkmkqr_score_is_negative_pinball_loss():
    X, y = _make_regression(n=50)
    model = TorchKMKQR(
        kernel="rbf",
        nC=2,
        cv=2,
        tau=0.5,
        device="cpu",
        max_iter=15,
        random_state=0,
    ).fit(X, y)

    score = model.score(X[:10], y[:10])
    assert np.isfinite(score)
    assert score <= 0.0


def test_no_torchkmnyskqr_exported():
    import torchkm

    assert not hasattr(torchkm, "TorchKMNysKQR")
