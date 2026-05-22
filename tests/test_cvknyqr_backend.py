import pytest
import torch

from torchkm.cvknyqr import cvknyqr


def _tiny_kqr_data(n=10):
    torch.manual_seed(123)
    x0 = torch.linspace(-1.0, 1.0, steps=n, dtype=torch.double)
    X = torch.stack([x0, x0**2], dim=1)
    y = torch.sin(2.0 * x0).to(torch.double)
    ulam = torch.tensor([1.0, 0.2], dtype=torch.double)
    foldid = torch.arange(n, dtype=torch.int64) % 2 + 1
    return X, y, ulam, foldid


def _make_backend(**overrides):
    X, y, ulam, foldid = _tiny_kqr_data()
    kwargs = dict(
        Xmat=X,
        y=y,
        nlam=2,
        ulam=ulam,
        tau=0.5,
        foldid=foldid,
        nfolds=2,
        eps=1e-4,
        maxit=20,
        gamma=1e-6,
        num_landmarks=5,
        k=3,
        sigma=1.0,
        device="cpu",
    )
    kwargs.update(overrides)
    return cvknyqr(**kwargs)


def test_cvknyqr_direct_fit_predict_and_helpers_cpu():
    X, y, _, _ = _tiny_kqr_data()
    model = _make_backend()
    model.fit()

    alp_b = model.alpmat[:, 0]
    pred = model.predict(X[:3], alp_b)
    Z = model.transform(X[:3])
    K_new = model.approx_kernel_to_train(X[:3])
    cv_loss = model.cv(model.pred, y)

    assert pred.shape == (3,)
    assert Z.shape == (3, model.k_eff_)
    assert K_new.shape == (3, X.shape[0])
    assert cv_loss.shape == (2,)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(Z).all()
    assert torch.isfinite(K_new).all()
    assert torch.isfinite(cv_loss).all()


def test_cvknyqr_cv_before_fit_uses_check_loss():
    X, y, _, _ = _tiny_kqr_data()
    model = _make_backend()
    pred = torch.zeros((X.shape[0], 2), dtype=torch.double)
    loss = model.cv(pred, y)

    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()


def test_cvknyqr_leave_one_out_foldid_cpu():
    X, y, ulam, _ = _tiny_kqr_data(n=6)
    model = cvknyqr(
        Xmat=X,
        y=y,
        nlam=1,
        ulam=ulam[:1],
        tau=0.5,
        foldid=None,
        nfolds=X.shape[0],
        maxit=10,
        num_landmarks=4,
        k=2,
        device="cpu",
    )
    model.fit()

    assert model.foldid.shape == (X.shape[0],)
    assert model.foldid.min().item() == 1
    assert model.foldid.max().item() == X.shape[0]
    assert torch.isfinite(model.pred).all()


def test_cvknyqr_constructs_random_foldid_cpu():
    X, y, ulam, _ = _tiny_kqr_data(n=8)
    model = cvknyqr(
        Xmat=X,
        y=y,
        nlam=1,
        ulam=ulam[:1],
        tau=0.5,
        foldid=None,
        nfolds=2,
        maxit=10,
        num_landmarks=4,
        k=2,
        random_state=123,
        device="cpu",
    )
    model.fit()

    assert model.foldid.shape == (X.shape[0],)
    assert model.foldid.min().item() >= 1
    assert model.foldid.max().item() <= 2
    assert torch.isfinite(model.pred).all()


def test_cvknyqr_unfit_helpers_raise():
    X, _, _, _ = _tiny_kqr_data()
    model = _make_backend()

    with pytest.raises(RuntimeError, match="fit"):
        model.transform(X[:2])
    with pytest.raises(RuntimeError, match="fit"):
        model.approx_kernel_to_train(X[:2])
    with pytest.raises(ValueError, match="one-dimensional"):
        model.predict(X[:2], torch.zeros((2, 2), dtype=torch.double))


def test_cvknyqr_transform_rejects_non_tensor_after_fit():
    model = _make_backend()
    model.fit()

    with pytest.raises(TypeError, match="X_new"):
        model.transform([[0.0, 0.0]])


def test_cvknyqr_rejects_invalid_constructor_inputs():
    X, y, ulam, foldid = _tiny_kqr_data()

    with pytest.raises(TypeError, match="Xmat"):
        cvknyqr(Xmat=X.numpy(), y=y, ulam=ulam, device="cpu")
    with pytest.raises(ValueError, match="y is required"):
        cvknyqr(Xmat=X, y=None, ulam=ulam, device="cpu")
    with pytest.raises(TypeError, match="y"):
        cvknyqr(Xmat=X, y=y.numpy(), ulam=ulam, device="cpu")
    with pytest.raises(ValueError, match="ulam is required"):
        cvknyqr(Xmat=X, y=y, ulam=None, device="cpu")
    with pytest.raises(TypeError, match="ulam"):
        cvknyqr(Xmat=X, y=y, ulam=[1.0, 0.2], device="cpu")
    with pytest.raises(ValueError, match="tau"):
        cvknyqr(Xmat=X, y=y, ulam=ulam, tau=1.0, device="cpu")
    with pytest.raises(ValueError, match="shape"):
        cvknyqr(Xmat=X, y=y[:-1], ulam=ulam, device="cpu")

    model = cvknyqr(Xmat=X, y=y, ulam=ulam, foldid=foldid.tolist(), device="cpu")
    with pytest.raises(TypeError, match="foldid"):
        model._make_foldid()

    model = cvknyqr(Xmat=X, y=y, ulam=ulam, foldid=foldid[:-1], device="cpu")
    with pytest.raises(ValueError, match="length"):
        model._make_foldid()
