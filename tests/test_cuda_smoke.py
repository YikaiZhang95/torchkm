# SPDX-License-Identifier: MIT
"""GPU smoke tests.

Every test in this module is gated by ``pytest.mark.cuda`` and skipped
when CUDA is not available, so it is a no-op on the CPU CI runner. The
purpose is to make sure that:

  * every public estimator can be ``fit`` / predicted on a GPU,
  * the alternative kernels (linear, poly, precomputed) work on GPU,
  * the low-rank / Nyström path works on GPU,
  * probability calibration (Platt) survives a round trip through GPU,
  * the low-level ``cv*`` solver classes write their outputs to CUDA
    tensors,
  * a model fit on CUDA produces predictions consistent with the same
    model fit on CPU.

Run on a GPU workstation with ``scripts/run_cuda_tests.sh`` — see
``docs/developer/cuda_testing.md``.
"""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_circles, make_regression
from sklearn.preprocessing import StandardScaler

from torchkm.cvkdwd import cvkdwd
from torchkm.cvklogit import cvklogit
from torchkm.cvknysdwd import cvknysdwd
from torchkm.cvknyslogit import cvknyslogit
from torchkm.cvknyqr import cvknyqr
from torchkm.cvknyssvm import cvknyssvm
from torchkm.cvkqr import cvkqr
from torchkm.cvksvm import cvksvm
from torchkm.estimators import TorchKMDWD, TorchKMKQR, TorchKMLogit, TorchKMSVC
from torchkm.functions import rbf_kernel, sigest, standardize

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
]


# ----------------------------------------------------------------------
# Shared data helpers
# ----------------------------------------------------------------------


def _binary_features(n=80, random_state=0):
    X, y = make_circles(n_samples=n, factor=0.4, noise=0.05, random_state=random_state)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = np.where(y == 0, -1, 1)
    return X, y


def _regression_features(n=80, random_state=0):
    X, y = make_regression(
        n_samples=n, n_features=4, noise=0.5, random_state=random_state
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    return X, y.astype(np.float64)


def _binary_tensors(n=40, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, 4, dtype=torch.float64)
    w = torch.randn(4, dtype=torch.float64)
    y = torch.where(X @ w > 0, torch.tensor(1.0), torch.tensor(-1.0))
    return standardize(X), y


def _regression_tensors(n=40, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, 4, dtype=torch.float64)
    y = X[:, 0] + 0.1 * torch.randn(n, dtype=torch.float64)
    return standardize(X), y


# ----------------------------------------------------------------------
# sklearn-style estimators
# ----------------------------------------------------------------------


def test_torchkmsvc_cuda_smoke():
    """``TorchKMSVC`` should fit and predict end-to-end on CUDA."""
    X, y = _binary_features(n=64)

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

    assert pred.shape == (8,)
    assert np.isfinite(pred).all()
    assert clf._device_str_ == "cuda"


def test_torchkmdwd_cuda_smoke():
    """``TorchKMDWD`` should fit and predict end-to-end on CUDA."""
    X, y = _binary_features(n=64)

    clf = TorchKMDWD(
        kernel="rbf",
        Cs=np.array([1.0, 0.1]),
        nC=2,
        cv=2,
        device="cuda",
        max_iter=20,
        random_state=0,
    )
    clf.fit(X, y)
    pred = clf.predict(X[:8])

    assert pred.shape == (8,)
    assert np.isfinite(pred).all()
    assert clf._device_str_ == "cuda"


def test_torchkmlogit_cuda_smoke():
    """``TorchKMLogit`` should fit and predict end-to-end on CUDA."""
    X, y = _binary_features(n=64)

    clf = TorchKMLogit(
        kernel="rbf",
        Cs=np.array([1.0, 0.1]),
        nC=2,
        cv=2,
        device="cuda",
        max_iter=20,
        random_state=0,
    )
    clf.fit(X, y)
    pred = clf.predict(X[:8])

    assert pred.shape == (8,)
    assert np.isfinite(pred).all()
    assert clf._device_str_ == "cuda"


def test_torchkmkqr_cuda_smoke():
    """``TorchKMKQR`` should fit and predict end-to-end on CUDA."""
    X, y = _regression_features(n=80)

    reg = TorchKMKQR(
        kernel="rbf",
        Cs=np.array([1.0, 0.1]),
        nC=2,
        cv=2,
        tau=0.5,
        device="cuda",
        max_iter=50,
        random_state=0,
    )
    reg.fit(X, y)
    pred = reg.predict(X[:8])

    assert pred.shape == (8,)
    assert np.isfinite(pred).all()
    assert reg._device_str_ == "cuda"


# ----------------------------------------------------------------------
# Alternative kernels on CUDA
# ----------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["linear", "poly"])
def test_torchkmsvc_alternative_kernels_cuda(kernel):
    """Non-default kernels (``linear`` / ``poly``) should also work on CUDA."""
    X, y = _binary_features(n=48)

    clf = TorchKMSVC(
        kernel=kernel,
        Cs=np.array([1.0, 0.1]),
        nC=2,
        cv=2,
        device="cuda",
        max_iter=20,
        random_state=0,
        poly_degree=2,
        poly_coef0=1.0,
        poly_gamma=0.5,
    )
    clf.fit(X, y)
    pred = clf.predict(X[:8])

    assert pred.shape == (8,)
    assert np.isfinite(pred).all()


def test_torchkmsvc_precomputed_kernel_cuda():
    """A precomputed kernel matrix should be accepted on CUDA."""
    X, y = _binary_features(n=48)
    K = (X @ X.T + 0.1 * np.eye(X.shape[0])).astype(np.float64)

    clf = TorchKMSVC(
        kernel="precomputed",
        Cs=np.array([1.0, 0.1]),
        nC=2,
        cv=2,
        device="cuda",
        max_iter=20,
        random_state=0,
    )
    clf.fit(K, y)
    pred = clf.predict(K[:6, :])

    assert pred.shape == (6,)
    assert np.isfinite(pred).all()


# ----------------------------------------------------------------------
# Low-rank / Nyström paths
# ----------------------------------------------------------------------


def test_torchkmsvc_low_rank_cuda():
    """The low-rank Nyström SVM path should run on CUDA."""
    X, y = _binary_features(n=64)

    clf = TorchKMSVC(
        kernel="rbf",
        Cs=np.array([1.0, 0.1]),
        nC=2,
        cv=2,
        device="cuda",
        max_iter=20,
        random_state=0,
        low_rank=True,
        num_landmarks=16,
        nys_k=8,
    )
    clf.fit(X, y)
    pred = clf.predict(X[:8])

    assert pred.shape == (8,)
    assert np.isfinite(pred).all()
    assert clf.low_rank is True


def test_torchkmkqr_low_rank_cuda():
    """The low-rank Nyström KQR path should run on CUDA."""
    X, y = _regression_features(n=80)

    reg = TorchKMKQR(
        kernel="rbf",
        Cs=np.array([1.0, 0.1]),
        nC=2,
        cv=2,
        tau=0.5,
        device="cuda",
        max_iter=50,
        random_state=0,
        low_rank=True,
        num_landmarks=20,
        nys_k=10,
    )
    reg.fit(X, y)
    pred = reg.predict(X[:8])

    assert pred.shape == (8,)
    assert np.isfinite(pred).all()
    assert reg.low_rank is True


# ----------------------------------------------------------------------
# Probability calibration on CUDA
# ----------------------------------------------------------------------


def test_torchkmsvc_predict_proba_cuda():
    """With ``probability=True`` the Platt scaler should be fitted and
    ``predict_proba`` should return calibrated probabilities on CUDA.
    """
    X, y = _binary_features(n=64)

    clf = TorchKMSVC(
        kernel="rbf",
        Cs=np.array([1.0, 0.1]),
        nC=2,
        cv=2,
        device="cuda",
        max_iter=20,
        probability=True,
        random_state=0,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X[:8])

    assert proba.shape == (8, 2)
    assert np.all(proba >= -1e-8)
    assert np.all(proba <= 1.0 + 1e-8)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ----------------------------------------------------------------------
# Low-level dense solvers
# ----------------------------------------------------------------------


def test_cvksvm_cuda_solver():
    """The low-level ``cvksvm`` solver should land its outputs on CUDA."""
    X, y = _binary_tensors(n=48)
    sig = sigest(X)
    Kmat = rbf_kernel(X, sig)
    torch.manual_seed(0)
    foldid = torch.randperm(X.shape[0]) % 3 + 1

    m = cvksvm(
        Kmat=Kmat,
        y=y,
        nlam=2,
        ulam=torch.tensor([1.0, 0.1], dtype=torch.float64),
        foldid=foldid,
        nfolds=3,
        eps=1e-4,
        maxit=80,
        gamma=1e-6,
        is_exact=0,
        device="cuda",
    )
    m.fit()

    assert m.alpmat.device.type == "cuda"
    assert m.pred.device.type == "cuda"
    assert torch.isfinite(m.alpmat).all()
    assert torch.isfinite(m.pred).all()


def test_cvkdwd_cuda_solver():
    """The low-level ``cvkdwd`` solver should land its outputs on CUDA."""
    X, y = _binary_tensors(n=48)
    sig = sigest(X)
    Kmat = rbf_kernel(X, sig)
    torch.manual_seed(0)
    foldid = torch.randperm(X.shape[0]) % 3 + 1

    m = cvkdwd(
        Kmat=Kmat,
        y=y,
        nlam=2,
        ulam=torch.tensor([1.0, 0.1], dtype=torch.float64),
        foldid=foldid,
        nfolds=3,
        eps=1e-4,
        maxit=80,
        gamma=1e-6,
        device="cuda",
    )
    m.fit()

    assert m.alpmat.device.type == "cuda"
    assert m.pred.device.type == "cuda"
    assert torch.isfinite(m.alpmat).all()


def test_cvklogit_cuda_solver():
    """The low-level ``cvklogit`` solver should land its outputs on CUDA."""
    X, y = _binary_tensors(n=48)
    sig = sigest(X)
    Kmat = rbf_kernel(X, sig)
    torch.manual_seed(0)
    foldid = torch.randperm(X.shape[0]) % 3 + 1

    m = cvklogit(
        Kmat=Kmat,
        y=y,
        nlam=2,
        ulam=torch.tensor([1.0, 0.1], dtype=torch.float64),
        foldid=foldid,
        nfolds=3,
        eps=1e-4,
        maxit=80,
        gamma=1e-6,
        device="cuda",
    )
    m.fit()

    assert m.alpmat.device.type == "cuda"
    assert m.pred.device.type == "cuda"
    assert torch.isfinite(m.alpmat).all()


def test_cvkqr_cuda_solver():
    """The low-level ``cvkqr`` solver should land its outputs on CUDA."""
    X, y = _regression_tensors(n=48)
    sig = sigest(X)
    Kmat = rbf_kernel(X, sig)
    torch.manual_seed(0)
    foldid = torch.randperm(X.shape[0]) % 3 + 1

    m = cvkqr(
        Kmat=Kmat,
        y=y,
        nlam=2,
        ulam=torch.tensor([1.0, 0.1], dtype=torch.float64),
        tau=0.5,
        foldid=foldid,
        nfolds=3,
        eps=1e-4,
        maxit=200,
        gamma=1e-6,
        is_exact=0,
        device="cuda",
    )
    m.fit()

    assert m.alpmat.device.type == "cuda"
    assert m.pred.device.type == "cuda"
    assert torch.isfinite(m.alpmat).all()


# ----------------------------------------------------------------------
# Low-level Nyström solvers
# ----------------------------------------------------------------------


def test_cvknyssvm_cuda_solver():
    """The Nyström SVM solver should land its outputs on CUDA."""
    X, y = _binary_tensors(n=40)

    m = cvknyssvm(
        Xmat=X,
        X_test=X[:8],
        y=y,
        nlam=2,
        ulam=torch.tensor([1.0, 0.1], dtype=torch.float64),
        foldid=None,
        nfolds=3,
        eps=1e-4,
        maxit=80,
        gamma=1e-6,
        num_landmarks=10,
        k=5,
        device="cuda",
    )
    m.fit()

    assert m.alpmat.device.type == "cuda"
    assert m.pred.device.type == "cuda"
    assert torch.isfinite(m.alpmat).all()


def test_cvknysdwd_cuda_solver():
    """The Nyström DWD solver should land its outputs on CUDA."""
    X, y = _binary_tensors(n=40)

    m = cvknysdwd(
        Xmat=X,
        X_test=X[:8],
        y=y,
        nlam=2,
        ulam=torch.tensor([1.0, 0.1], dtype=torch.float64),
        foldid=None,
        nfolds=3,
        eps=1e-4,
        maxit=80,
        gamma=1e-6,
        num_landmarks=10,
        k=5,
        device="cuda",
    )
    m.fit()

    assert m.alpmat.device.type == "cuda"
    assert m.pred.device.type == "cuda"
    assert torch.isfinite(m.alpmat).all()


def test_cvknyslogit_cuda_solver():
    """The Nyström logistic solver should land its outputs on CUDA."""
    X, y = _binary_tensors(n=40)

    m = cvknyslogit(
        Xmat=X,
        X_test=X[:8],
        y=y,
        nlam=2,
        ulam=torch.tensor([1.0, 0.1], dtype=torch.float64),
        foldid=None,
        nfolds=3,
        eps=1e-4,
        maxit=80,
        gamma=1e-6,
        num_landmarks=10,
        k=5,
        device="cuda",
    )
    m.fit()

    assert m.alpmat.device.type == "cuda"
    assert m.pred.device.type == "cuda"
    assert torch.isfinite(m.alpmat).all()


def test_cvknyqr_cuda_solver():
    """The Nyström quantile-regression solver should land its outputs on CUDA."""
    X, y = _regression_tensors(n=40)

    m = cvknyqr(
        Xmat=X,
        y=y,
        nlam=2,
        ulam=torch.tensor([1.0, 0.1], dtype=torch.float64),
        tau=0.5,
        foldid=None,
        nfolds=3,
        eps=1e-4,
        maxit=200,
        gamma=1e-6,
        num_landmarks=10,
        k=5,
        device="cuda",
    )
    m.fit()

    assert m.alpmat.device.type == "cuda"
    assert torch.isfinite(m.alpmat).all()


# ----------------------------------------------------------------------
# CPU / CUDA cross-device agreement
# ----------------------------------------------------------------------


def test_torchkmsvc_cpu_cuda_predictions_agree():
    """The same ``TorchKMSVC`` configuration on CPU and CUDA should produce
    predictions that agree on the great majority of points, and decision
    scores that are numerically close.
    """
    X, y = _binary_features(n=80, random_state=0)

    common = dict(
        kernel="rbf",
        Cs=np.array([1.0, 0.1]),
        nC=2,
        cv=2,
        max_iter=30,
        random_state=42,
        rbf_sigma=1.0,
    )
    cpu = TorchKMSVC(device="cpu", **common).fit(X, y)
    gpu = TorchKMSVC(device="cuda", **common).fit(X, y)

    cpu_pred = cpu.predict(X[:16])
    gpu_pred = gpu.predict(X[:16])
    cpu_score = cpu.decision_function(X[:16])
    gpu_score = gpu.decision_function(X[:16])

    # Allow at most one label flip out of 16 across the two devices, and
    # require the raw decision scores to agree to a few decimal places.
    label_match = int(np.sum(cpu_pred == gpu_pred))
    assert label_match >= 15, (cpu_pred.tolist(), gpu_pred.tolist())
    assert np.allclose(cpu_score, gpu_score, atol=1e-3, rtol=1e-3), (
        cpu_score.tolist(),
        gpu_score.tolist(),
    )


def test_torchkmkqr_cpu_cuda_predictions_agree():
    """Same agreement check for ``TorchKMKQR`` regression."""
    X, y = _regression_features(n=80, random_state=0)

    common = dict(
        kernel="rbf",
        Cs=np.array([1.0, 0.1]),
        nC=2,
        cv=2,
        tau=0.5,
        max_iter=80,
        random_state=42,
        rbf_sigma=1.0,
    )
    cpu = TorchKMKQR(device="cpu", **common).fit(X, y)
    gpu = TorchKMKQR(device="cuda", **common).fit(X, y)

    cpu_pred = cpu.predict(X[:16])
    gpu_pred = gpu.predict(X[:16])

    assert cpu_pred.shape == gpu_pred.shape
    assert np.allclose(cpu_pred, gpu_pred, atol=1e-2, rtol=1e-2), (
        cpu_pred.tolist(),
        gpu_pred.tolist(),
    )
