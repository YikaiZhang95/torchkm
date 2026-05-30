# SPDX-License-Identifier: MIT
import numpy as np
import torch
from sklearn.datasets import make_circles

import torchkm
from torchkm.estimators import TorchKMKQR, TorchKMSVC


def _make_paper_data():
    return make_circles(n_samples=120, factor=0.4, noise=0.08, random_state=0)


def test_svm_paper_style_snippet():
    X, y = _make_paper_data()
    Cs = np.logspace(2, -2, num=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clf = TorchKMSVC(
        kernel="rbf",
        Cs=Cs,
        nC=len(Cs),
        cv=3,
        device=device,
        probability=True,
        max_iter=20,
        num_landmarks=30,
        nys_k=15,
    )

    clf.fit(X, y)
    pred = clf.predict(X[:5])
    proba = clf.predict_proba(X[:5])

    assert pred.shape == (5,)
    assert proba.shape == (5, 2)

    clf.fit(X, y, low_rank=True)
    pred_nys = clf.predict(X[:5])

    assert pred_nys.shape == (5,)
    assert clf.low_rank is True


def test_svm_fit_time_low_rank_switching_modes():
    X, y = _make_paper_data()
    Cs = np.logspace(2, -2, num=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clf = TorchKMSVC(
        kernel="rbf",
        Cs=Cs,
        nC=len(Cs),
        cv=3,
        device=device,
        probability=True,
        max_iter=20,
        num_landmarks=30,
        nys_k=15,
    )

    clf.fit(X, y, low_rank=True)
    pred1 = clf.predict(X[:5])

    clf.fit(X, y, low_rank=False)
    pred2 = clf.predict(X[:5])

    assert pred1.shape == (5,)
    assert pred2.shape == (5,)
    assert clf.low_rank is False


def test_canonical_quantile_regression_import():
    from torchkm import TorchKMKQR as TopLevelTorchKMKQR

    assert TopLevelTorchKMKQR is TorchKMKQR


def test_kqr_fit_time_low_rank_switching_modes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=60)
    Cs = np.logspace(1, -1, num=3)

    reg = TorchKMKQR(
        kernel="rbf",
        Cs=Cs,
        nC=len(Cs),
        cv=3,
        tau=0.5,
        device="cpu",
        max_iter=20,
        num_landmarks=20,
        nys_k=10,
        random_state=0,
    )

    reg.fit(X, y, low_rank=True)
    pred1 = reg.predict(X[:5])

    reg.fit(X, y, low_rank=False)
    pred2 = reg.predict(X[:5])

    assert pred1.shape == (5,)
    assert pred2.shape == (5,)
    assert np.isfinite(pred1).all()
    assert np.isfinite(pred2).all()
    assert reg.low_rank is False
    assert not hasattr(reg, "_low_rank_backend_")


def test_minimal_top_level_import():
    assert hasattr(torchkm, "TorchKMSVC")
    assert hasattr(torchkm, "TorchKMKQR")
