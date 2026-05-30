# SPDX-License-Identifier: MIT
import pytest
import torch

from torchkm.cvkqr import cvkqr
from torchkm.cvksvm import cvksvm


def _tiny_kernel_data(n=10):
    torch.manual_seed(0)
    X = torch.linspace(-1.0, 1.0, steps=n, dtype=torch.double).reshape(n, 1)
    K = X @ X.T + 0.1 * torch.eye(n, dtype=torch.double)
    foldid = torch.arange(n, dtype=torch.int64) % 2 + 1
    ulam = torch.tensor([1.0, 0.1], dtype=torch.double)
    return X, K, foldid, ulam


def test_cvksvm_tiny_direct_backend_smoke_cpu():
    _, K, foldid, ulam = _tiny_kernel_data()
    y = torch.tensor([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1], dtype=torch.double)

    model = cvksvm(
        Kmat=K,
        y=y,
        nlam=2,
        ulam=ulam,
        foldid=foldid,
        nfolds=2,
        eps=1e-4,
        maxit=20,
        gamma=1e-6,
        is_exact=0,
        device="cpu",
    )
    model.fit()

    assert model.jerr == 0
    assert model.alpmat.shape == (K.shape[0] + 1, 2)
    assert model.pred.shape == (K.shape[0], 2)
    assert torch.isfinite(model.alpmat).all()
    assert torch.isfinite(model.pred).all()
    assert model.cv(model.pred, y).shape == (2,)


def test_cvkqr_tiny_direct_backend_smoke_cpu():
    X, K, foldid, ulam = _tiny_kernel_data()
    y = torch.sin(2.0 * X[:, 0])

    model = cvkqr(
        Kmat=K,
        y=y,
        nlam=2,
        ulam=ulam,
        tau=0.5,
        foldid=foldid,
        nfolds=2,
        eps=1e-4,
        maxit=20,
        gamma=1e-6,
        is_exact=0,
        device="cpu",
    )
    model.fit()

    assert model.jerr == 0
    assert model.alpmat.shape == (K.shape[0] + 1, 2)
    assert model.pred.shape == (K.shape[0], 2)
    assert torch.isfinite(model.alpmat).all()
    assert torch.isfinite(model.pred).all()
    assert model.cv(model.pred, y).shape == (2,)


def test_cvkqr_validation_rejects_non_square_kernel():
    y = torch.zeros(4, dtype=torch.double)
    with pytest.raises(ValueError, match="square"):
        cvkqr(
            Kmat=torch.zeros((4, 3), dtype=torch.double),
            y=y,
            nlam=1,
            ulam=torch.ones(1, dtype=torch.double),
            tau=0.5,
            nfolds=2,
            device="cpu",
        )
