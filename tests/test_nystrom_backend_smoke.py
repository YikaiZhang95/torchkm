import pytest
import torch

from torchkm.cvknysdwd import cvknysdwd
from torchkm.cvknyslogit import cvknyslogit

BACKENDS = [cvknysdwd, cvknyslogit]


def _tiny_nystrom_classification_data(n=12):
    torch.manual_seed(123)
    x0 = torch.linspace(-1.0, 1.0, steps=n, dtype=torch.double)
    X = torch.stack([x0, x0**2], dim=1)
    y = torch.where(x0 >= 0, 1.0, -1.0).to(torch.double)
    X_test = X[:4].clone()
    foldid = torch.arange(n, dtype=torch.int64) % 2 + 1
    ulam = torch.tensor([1.0, 0.2], dtype=torch.double)
    return X, X_test, y, foldid, ulam


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_nystrom_backend_fit_transform_cv_cpu(backend_cls):
    X, X_test, y, foldid, ulam = _tiny_nystrom_classification_data()

    model = backend_cls(
        Xmat=X,
        X_test=X_test,
        y=y,
        nlam=2,
        ulam=ulam,
        foldid=foldid,
        nfolds=2,
        eps=1e-4,
        maxit=30,
        gamma=1e-6,
        num_landmarks=6,
        k=3,
        device="cpu",
    )
    model.fit()

    assert model.alpmat.shape[1] == 2
    assert model.pred.shape == (X.shape[0], 2)
    assert torch.isfinite(model.alpmat).all()
    assert torch.isfinite(model.pred).all()

    cv = model.cv(model.pred, y)
    assert cv.shape == (2,)
    assert torch.isfinite(cv).all()

    assert model.landmarks_ is not None
    assert model.M_ is not None
    assert model.sig_w_ is not None
    assert model.k_eff_ is not None
    assert model.k_eff_ <= 3

    Z = model.transform(X_test[:2])
    assert Z.shape == (2, model.k_eff_)
    assert torch.isfinite(Z).all()


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_nystrom_backend_transform_before_fit_raises(backend_cls):
    X, X_test, y, foldid, ulam = _tiny_nystrom_classification_data()

    model = backend_cls(
        Xmat=X,
        X_test=X_test,
        y=y,
        nlam=2,
        ulam=ulam,
        foldid=foldid,
        nfolds=2,
        maxit=2,
        num_landmarks=4,
        k=2,
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="fit"):
        model.transform(X_test)


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_nystrom_backend_rejects_non_tensor_xmat(backend_cls):
    X, X_test, y, foldid, ulam = _tiny_nystrom_classification_data()

    with pytest.raises(TypeError, match="Xmat"):
        backend_cls(
            Xmat=X.numpy(),
            X_test=X_test,
            y=y,
            nlam=2,
            ulam=ulam,
            foldid=foldid,
            nfolds=2,
            device="cpu",
        )


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_nystrom_backend_rejects_non_tensor_xtest(backend_cls):
    X, X_test, y, foldid, ulam = _tiny_nystrom_classification_data()

    with pytest.raises(TypeError, match="X_test"):
        backend_cls(
            Xmat=X,
            X_test=X_test.numpy(),
            y=y,
            nlam=2,
            ulam=ulam,
            foldid=foldid,
            nfolds=2,
            device="cpu",
        )


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_nystrom_backend_rejects_non_tensor_y(backend_cls):
    X, X_test, y, foldid, ulam = _tiny_nystrom_classification_data()

    with pytest.raises(TypeError, match="y"):
        backend_cls(
            Xmat=X,
            X_test=X_test,
            y=y.tolist(),
            nlam=2,
            ulam=ulam,
            foldid=foldid,
            nfolds=2,
            device="cpu",
        )


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_nystrom_backend_rejects_non_tensor_ulam(backend_cls):
    X, X_test, y, foldid, ulam = _tiny_nystrom_classification_data()

    with pytest.raises(TypeError, match="ulam"):
        backend_cls(
            Xmat=X,
            X_test=X_test,
            y=y,
            nlam=2,
            ulam=[1.0, 0.2],
            foldid=foldid,
            nfolds=2,
            device="cpu",
        )


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_nystrom_backend_rejects_non_tensor_foldid(backend_cls):
    X, X_test, y, foldid, ulam = _tiny_nystrom_classification_data()

    with pytest.raises(TypeError, match="foldid"):
        backend_cls(
            Xmat=X,
            X_test=X_test,
            y=y,
            nlam=2,
            ulam=ulam,
            foldid=foldid.tolist(),
            nfolds=2,
            device="cpu",
        )


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_nystrom_backend_rejects_multiclass_labels(backend_cls):
    X, X_test, y, foldid, ulam = _tiny_nystrom_classification_data()
    y_bad = y.clone()
    y_bad[0] = 0.0

    with pytest.raises(ValueError, match="Multi-class"):
        backend_cls(
            Xmat=X,
            X_test=X_test,
            y=y_bad,
            nlam=2,
            ulam=ulam,
            foldid=foldid,
            nfolds=2,
            device="cpu",
        )


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_nystrom_backend_rejects_invalid_binary_label_values(backend_cls):
    X, X_test, y, foldid, ulam = _tiny_nystrom_classification_data()
    y_bad = torch.where(y > 0, 2.0, 0.0).to(torch.double)

    with pytest.raises(ValueError, match="Invalid labels"):
        backend_cls(
            Xmat=X,
            X_test=X_test,
            y=y_bad,
            nlam=2,
            ulam=ulam,
            foldid=foldid,
            nfolds=2,
            device="cpu",
        )


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_nystrom_backend_rejects_x_y_length_mismatch(backend_cls):
    X, X_test, y, foldid, ulam = _tiny_nystrom_classification_data()

    with pytest.raises(ValueError, match="size mismatch"):
        backend_cls(
            Xmat=X,
            X_test=X_test,
            y=y[:-1],
            nlam=2,
            ulam=ulam,
            foldid=foldid,
            nfolds=2,
            device="cpu",
        )


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_nystrom_backend_constructs_foldid_when_none(backend_cls):
    X, X_test, y, _, ulam = _tiny_nystrom_classification_data()

    model = backend_cls(
        Xmat=X,
        X_test=X_test,
        y=y,
        nlam=2,
        ulam=ulam,
        foldid=None,
        nfolds=2,
        maxit=2,
        num_landmarks=4,
        k=2,
        device="cpu",
    )

    assert model.foldid.shape == (X.shape[0],)
    assert model.foldid.min().item() >= 1
    assert model.foldid.max().item() <= 2
