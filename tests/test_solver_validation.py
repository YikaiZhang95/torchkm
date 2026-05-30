# SPDX-License-Identifier: MIT
import pytest
import torch

from torchkm.cvkqr import cvkqr
from torchkm.cvksvm import cvksvm


def _tiny_kernel_inputs(n=8):
    torch.manual_seed(123)
    x = torch.linspace(-1.0, 1.0, steps=n, dtype=torch.double).reshape(n, 1)
    K = x @ x.T + 0.2 * torch.eye(n, dtype=torch.double)
    y_cls = torch.where(x[:, 0] >= 0, 1.0, -1.0).to(torch.double)
    y_reg = torch.sin(2.0 * x[:, 0]).to(torch.double)
    foldid = torch.arange(n, dtype=torch.int64) % 2 + 1
    ulam = torch.tensor([1.0, 0.2], dtype=torch.double)
    return K, y_cls, y_reg, foldid, ulam


def test_cvksvm_default_device_and_auto_foldid_cpu():
    K, y_cls, _, _, ulam = _tiny_kernel_inputs()

    model = cvksvm(
        Kmat=K,
        y=y_cls,
        nlam=2,
        ulam=ulam,
        foldid=None,
        nfolds=2,
        eps=1e-4,
        maxit=30,
        gamma=1e-6,
        is_exact=0,
        device="cpu",
    )
    model.fit()

    assert model.foldid.shape == (K.shape[0],)
    assert model.alpmat.shape == (K.shape[0] + 1, 2)
    assert model.pred.shape == (K.shape[0], 2)
    assert torch.isfinite(model.alpmat).all()
    assert torch.isfinite(model.pred).all()


def test_cvkqr_default_foldid_cpu():
    K, _, y_reg, _, ulam = _tiny_kernel_inputs()

    model = cvkqr(
        Kmat=K,
        y=y_reg,
        nlam=2,
        ulam=ulam,
        tau=0.25,
        foldid=None,
        nfolds=2,
        eps=1e-4,
        maxit=30,
        gamma=1e-6,
        is_exact=0,
        device="cpu",
    )
    model.fit()

    assert model.foldid.shape == (K.shape[0],)
    assert model.alpmat.shape == (K.shape[0] + 1, 2)
    assert model.pred.shape == (K.shape[0], 2)
    assert torch.isfinite(model.alpmat).all()
    assert torch.isfinite(model.pred).all()


def test_cvksvm_leave_one_out_foldid_none_smoke_cpu():
    K, y_cls, _, _, _ = _tiny_kernel_inputs()
    ulam = torch.tensor([0.5], dtype=torch.double)

    model = cvksvm(
        Kmat=K,
        y=y_cls,
        nlam=1,
        ulam=ulam,
        foldid=None,
        nfolds=K.shape[0],
        eps=1e-4,
        maxit=30,
        gamma=1e-6,
        is_exact=0,
        device="cpu",
    )
    model.fit()

    assert model.alpmat.shape == (K.shape[0] + 1, 1)
    assert model.pred.shape == (K.shape[0], 1)
    assert torch.isfinite(model.alpmat).all()
    assert torch.isfinite(model.pred).all()


def test_cvkqr_leave_one_out_foldid_none_smoke_cpu():
    K, _, y_reg, _, _ = _tiny_kernel_inputs()
    ulam = torch.tensor([0.5], dtype=torch.double)

    model = cvkqr(
        Kmat=K,
        y=y_reg,
        nlam=1,
        ulam=ulam,
        tau=0.5,
        foldid=None,
        nfolds=K.shape[0],
        eps=1e-4,
        maxit=30,
        gamma=1e-6,
        is_exact=0,
        device="cpu",
    )
    model.fit()

    assert model.alpmat.shape == (K.shape[0] + 1, 1)
    assert model.pred.shape == (K.shape[0], 1)
    assert torch.isfinite(model.alpmat).all()
    assert torch.isfinite(model.pred).all()


def test_cvksvm_exact_projection_smoke_cpu():
    K, y_cls, _, foldid, _ = _tiny_kernel_inputs(n=6)
    ulam = torch.tensor([0.5], dtype=torch.double)

    model = cvksvm(
        Kmat=K,
        y=y_cls,
        nlam=1,
        ulam=ulam,
        foldid=foldid,
        nfolds=2,
        eps=1e-4,
        maxit=20,
        gamma=1e-6,
        is_exact=1,
        delta_len=3,
        mproj=1,
        device="cpu",
    )
    model.fit()

    assert model.alpmat.shape == (K.shape[0] + 1, 1)
    assert model.pred.shape == (K.shape[0], 1)
    assert torch.isfinite(model.alpmat).all()
    assert torch.isfinite(model.pred).all()


def test_cvkqr_exact_projection_smoke_cpu():
    K, _, y_reg, foldid, _ = _tiny_kernel_inputs(n=6)
    ulam = torch.tensor([0.5], dtype=torch.double)

    model = cvkqr(
        Kmat=K,
        y=y_reg,
        nlam=1,
        ulam=ulam,
        tau=0.5,
        foldid=foldid,
        nfolds=2,
        eps=1e-4,
        maxit=20,
        gamma=1e-6,
        is_exact=1,
        delta_len=3,
        mproj=1,
        device="cpu",
    )
    model.fit()

    assert model.alpmat.shape == (K.shape[0] + 1, 1)
    assert model.pred.shape == (K.shape[0], 1)
    assert torch.isfinite(model.alpmat).all()
    assert torch.isfinite(model.pred).all()


@pytest.mark.parametrize(
    "backend_cls, y_name", [(cvksvm, "classification"), (cvkqr, "regression")]
)
def test_exact_backends_reject_non_tensor_kmat(backend_cls, y_name):
    K, y_cls, y_reg, foldid, ulam = _tiny_kernel_inputs()
    y = y_cls if y_name == "classification" else y_reg

    kwargs = dict(
        Kmat=K.numpy(),
        y=y,
        nlam=2,
        ulam=ulam,
        foldid=foldid,
        nfolds=2,
        device="cpu",
    )
    if backend_cls is cvkqr:
        kwargs["tau"] = 0.5

    with pytest.raises(TypeError, match="Kmat"):
        backend_cls(**kwargs)


@pytest.mark.parametrize(
    "backend_cls, y_name", [(cvksvm, "classification"), (cvkqr, "regression")]
)
def test_exact_backends_reject_non_tensor_y(backend_cls, y_name):
    K, y_cls, y_reg, foldid, ulam = _tiny_kernel_inputs()
    y = y_cls if y_name == "classification" else y_reg

    kwargs = dict(
        Kmat=K,
        y=y.tolist(),
        nlam=2,
        ulam=ulam,
        foldid=foldid,
        nfolds=2,
        device="cpu",
    )
    if backend_cls is cvkqr:
        kwargs["tau"] = 0.5

    with pytest.raises(TypeError, match="y"):
        backend_cls(**kwargs)


@pytest.mark.parametrize(
    "backend_cls, y_name", [(cvksvm, "classification"), (cvkqr, "regression")]
)
def test_exact_backends_reject_non_tensor_ulam(backend_cls, y_name):
    K, y_cls, y_reg, foldid, _ = _tiny_kernel_inputs()
    y = y_cls if y_name == "classification" else y_reg

    kwargs = dict(
        Kmat=K,
        y=y,
        nlam=2,
        ulam=[1.0, 0.2],
        foldid=foldid,
        nfolds=2,
        device="cpu",
    )
    if backend_cls is cvkqr:
        kwargs["tau"] = 0.5

    with pytest.raises(TypeError, match="ulam"):
        backend_cls(**kwargs)


@pytest.mark.parametrize(
    "backend_cls, y_name", [(cvksvm, "classification"), (cvkqr, "regression")]
)
def test_exact_backends_reject_non_tensor_foldid(backend_cls, y_name):
    K, y_cls, y_reg, foldid, ulam = _tiny_kernel_inputs()
    y = y_cls if y_name == "classification" else y_reg

    kwargs = dict(
        Kmat=K,
        y=y,
        nlam=2,
        ulam=ulam,
        foldid=foldid.tolist(),
        nfolds=2,
        device="cpu",
    )
    if backend_cls is cvkqr:
        kwargs["tau"] = 0.5

    with pytest.raises(TypeError, match="foldid"):
        backend_cls(**kwargs)


@pytest.mark.parametrize(
    "backend_cls, y_name", [(cvksvm, "classification"), (cvkqr, "regression")]
)
def test_exact_backends_reject_non_square_kernel(backend_cls, y_name):
    K, y_cls, y_reg, foldid, ulam = _tiny_kernel_inputs()
    y = y_cls if y_name == "classification" else y_reg

    kwargs = dict(
        Kmat=K[:, :-1],
        y=y,
        nlam=2,
        ulam=ulam,
        foldid=foldid,
        nfolds=2,
        device="cpu",
    )
    if backend_cls is cvkqr:
        kwargs["tau"] = 0.5

    with pytest.raises(ValueError, match="square"):
        backend_cls(**kwargs)


@pytest.mark.parametrize(
    "backend_cls, y_name", [(cvksvm, "classification"), (cvkqr, "regression")]
)
def test_exact_backends_reject_k_y_size_mismatch(backend_cls, y_name):
    K, y_cls, y_reg, foldid, ulam = _tiny_kernel_inputs()
    y = y_cls if y_name == "classification" else y_reg

    kwargs = dict(
        Kmat=K,
        y=y[:-1],
        nlam=2,
        ulam=ulam,
        foldid=foldid,
        nfolds=2,
        device="cpu",
    )
    if backend_cls is cvkqr:
        kwargs["tau"] = 0.5

    with pytest.raises(ValueError, match="size mismatch"):
        backend_cls(**kwargs)


def test_cvksvm_rejects_multiclass_labels():
    K, y_cls, _, foldid, ulam = _tiny_kernel_inputs()
    y_bad = y_cls.clone()
    y_bad[0] = 0.0

    with pytest.raises(ValueError, match="Multi-class"):
        cvksvm(
            Kmat=K,
            y=y_bad,
            nlam=2,
            ulam=ulam,
            foldid=foldid,
            nfolds=2,
            device="cpu",
        )


def test_cvksvm_rejects_invalid_binary_label_values():
    K, y_cls, _, foldid, ulam = _tiny_kernel_inputs()
    y_bad = torch.where(y_cls > 0, 2.0, 0.0).to(torch.double)

    with pytest.raises(ValueError, match="Invalid labels"):
        cvksvm(
            Kmat=K,
            y=y_bad,
            nlam=2,
            ulam=ulam,
            foldid=foldid,
            nfolds=2,
            device="cpu",
        )
