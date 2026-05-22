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


def test_minimal_top_level_import():
    assert hasattr(torchkm, "TorchKMSVC")
    assert hasattr(torchkm, "TorchKMKQR")
