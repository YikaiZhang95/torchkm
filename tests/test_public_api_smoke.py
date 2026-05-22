import numpy as np
import pytest
import torch
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

from torchkm.estimators import TorchKMSVC
from torchkm.platt import PlattScalerTorch


def test_torchkmsvc_cpu_fit_predict_and_proba_smoke():
    X, y = make_circles(n_samples=48, factor=0.45, noise=0.05, random_state=0)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = np.where(y == 0, -1, 1)

    clf = TorchKMSVC(
        kernel="rbf",
        Cs=np.array([1.0, 0.1]),
        nC=2,
        cv=2,
        device="cpu",
        probability=True,
        max_iter=15,
        random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:5])
    proba = clf.predict_proba(X[:5])

    assert pred.shape == (5,)
    assert proba.shape == (5, 2)
    assert np.isfinite(pred).all()
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_platt_scaler_tiny_tensor_smoke():
    scores = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.double)
    labels = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.double)

    scaler = PlattScalerTorch(max_iter=20, device="cpu").fit(scores, labels)
    proba = scaler.predict_proba(scores)
    pred = scaler.predict(scores)

    assert proba.shape == (4, 2)
    assert pred.shape == (4,)
    assert torch.isfinite(proba).all()
    assert torch.allclose(proba.sum(dim=1), torch.ones(4, dtype=torch.double))
    assert set(pred.tolist()).issubset({-1.0, 1.0})


def test_platt_scaler_predict_proba_requires_fit():
    scaler = PlattScalerTorch(device="cpu")

    with pytest.raises(RuntimeError, match="fit"):
        scaler.predict_proba(torch.tensor([0.0], dtype=torch.double))
