import numpy as np
import pytest
import torch
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

from torchkm.estimators import TorchKMSVC

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
]


def test_torchkmsvc_cuda_smoke():
    X, y = make_circles(n_samples=64, factor=0.4, noise=0.05, random_state=0)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = np.where(y == 0, -1, 1)

    clf = TorchKMSVC(
        kernel="rbf",
        Cs=np.array([1.0, 0.1]),
        nC=2,
        cv=2,
        device="cuda",
        max_iter=20,
        probability=False,
        random_state=0,
    )
    clf.fit(X, y)
    pred = clf.predict(X[:8])

    assert torch.cuda.is_available()
    assert pred.shape == (8,)
    assert np.isfinite(pred).all()
    assert getattr(clf, "_device_str_", None) == "cuda"
