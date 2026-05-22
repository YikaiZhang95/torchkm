import numpy as np
import pytest
import torch

from torchkm.estimators import (
    TorchKMKQR,
    TorchKMSVC,
    _as_numpy,
    _check_binary_y,
    _make_foldid,
    _make_ulam,
    _pick_device_str,
)


def test_as_numpy_accepts_numpy_torch_and_list_inputs():
    arr = np.array([[1.0, 2.0]])
    tensor = torch.tensor([[3.0, 4.0]])
    values = [[5.0, 6.0]]

    assert _as_numpy(arr) is arr
    assert np.array_equal(_as_numpy(tensor), tensor.numpy())
    assert np.array_equal(_as_numpy(values), np.array(values))


def test_pick_device_str_cpu_paths_and_cuda_fallback():
    assert _pick_device_str(None) in {"cpu", "cuda"}
    assert _pick_device_str("cpu") == "cpu"
    assert _pick_device_str(torch.device("cpu")) == "cpu"
    if not torch.cuda.is_available():
        assert _pick_device_str("cuda") == "cpu"


def test_make_ulam_explicit_and_generated_values():
    explicit = _make_ulam(nC=2, Cs=[10.0, 1.0], C_max=100.0, C_min=0.1)
    generated = _make_ulam(nC=3, Cs=None, C_max=100.0, C_min=0.01)

    assert torch.allclose(explicit, torch.tensor([10.0, 1.0], dtype=torch.double))
    assert torch.allclose(generated, torch.logspace(2, -2, steps=3, dtype=torch.double))


def test_make_foldid_explicit_invalid_and_seeded_random_state():
    explicit = _make_foldid(n=4, nfolds=2, foldid=[1, 2, 1, 2], random_state=None)
    first = _make_foldid(n=8, nfolds=2, foldid=None, random_state=123)
    second = _make_foldid(n=8, nfolds=2, foldid=None, random_state=123)

    assert torch.equal(explicit, torch.tensor([1, 2, 1, 2], dtype=torch.int64))
    assert torch.equal(first, second)
    with pytest.raises(ValueError, match="foldid"):
        _make_foldid(n=4, nfolds=2, foldid=[1, 2, 1], random_state=None)


def test_check_binary_y_valid_and_multiclass_error():
    y_pm1, neg_label, pos_label = _check_binary_y(np.array(["no", "yes", "yes"]))

    assert neg_label == "no"
    assert pos_label == "yes"
    assert np.array_equal(y_pm1, np.array([-1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="Only binary classification"):
        _check_binary_y(np.array([0, 1, 2]))


def test_torchkmsvc_tiny_cpu_fit_predict():
    X = np.array([[-1.0, -1.0], [-0.8, -1.2], [1.0, 1.0], [1.2, 0.8]], dtype=np.float64)
    y = np.array([-1, -1, 1, 1])

    clf = TorchKMSVC(
        kernel="rbf",
        Cs=np.array([1.0, 0.1]),
        nC=2,
        cv=2,
        device="cpu",
        max_iter=10,
        random_state=0,
    ).fit(X, y)

    pred = clf.predict(X)
    assert pred.shape == (4,)
    assert np.isfinite(pred).all()
    assert hasattr(clf, "best_C_")


def test_torchkmkqr_tiny_cpu_fit_predict():
    X = np.array(
        [[-1.0, 0.0], [-0.5, 0.2], [0.5, 0.4], [1.0, 0.8], [1.5, 1.0]],
        dtype=np.float64,
    )
    y = np.array([-1.0, -0.4, 0.2, 0.9, 1.4])

    reg = TorchKMKQR(
        kernel="rbf",
        Cs=np.array([1.0, 0.1]),
        nC=2,
        cv=2,
        tau=0.5,
        device="cpu",
        max_iter=10,
        random_state=0,
    ).fit(X, y)

    pred = reg.predict(X[:3])
    assert pred.shape == (3,)
    assert np.isfinite(pred).all()
    assert hasattr(reg, "best_C_")
