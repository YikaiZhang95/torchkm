# SPDX-License-Identifier: MIT
"""Where the exact-mode eigendecomposition runs (``eigh_backend``).

The backends differ in time and device memory only; the tests check that the
option reaches every exact solver, that the fitted model does not depend on
it, that ``"auto"`` falls back in the right order when a backend runs out of
memory, and that the memory model uses the per-backend peaks.
"""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression

from torchkm import functions, linalg
from torchkm.estimators import TorchKMDWD, TorchKMKQR, TorchKMLogit, TorchKMSVC
from torchkm.memory import (
    EXACT_MODE_COPIES,
    EXACT_MODE_COPIES_BY_BACKEND,
    LOW_MEMORY_COPIES,
    exact_mode_memory_estimate,
    exact_mode_oom_message,
    max_exact_n,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _binary(n=90, p=5, seed=0):
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=3, n_redundant=0, random_state=seed
    )
    return X, np.where(y == 0, -1, 1)


def _spd(n=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, dtype=torch.double)
    return A @ A.T + n * torch.eye(n, dtype=torch.double)


# --------------------------------------------------------------------------
# torchkm.linalg
# --------------------------------------------------------------------------


def test_backend_names_are_checked():
    assert linalg.check_eigh_backend("MAGMA") == "magma"
    for name in linalg.EIGH_BACKENDS:
        assert linalg.check_eigh_backend(name) == name
    with pytest.raises(ValueError, match="eigh_backend"):
        linalg.check_eigh_backend("rocsolver")


@pytest.mark.parametrize("backend", linalg.EIGH_BACKENDS)
def test_cpu_tensor_gives_lapack_result_for_every_backend(backend):
    K = _spd()
    w, U, info = linalg.kernel_eigh(K, backend, return_info=True)
    w_ref, U_ref = torch.linalg.eigh(K)
    torch.testing.assert_close(w, w_ref)
    torch.testing.assert_close(U.abs(), U_ref.abs())
    assert info["used"] == "lapack"
    assert info["requested"] == backend
    assert info["failed"] == []
    assert info["seconds"] >= 0.0
    assert len(linalg.kernel_eigh(K, backend)) == 2


def test_attempt_order(monkeypatch):
    monkeypatch.setattr(linalg, "has_magma", lambda: True)
    assert linalg._attempt_order("auto") == ["cusolver", "magma", "cpu"]
    assert linalg._attempt_order("magma") == ["magma"]
    assert linalg.low_memory_backend() == "magma"
    monkeypatch.setattr(linalg, "has_magma", lambda: False)
    assert linalg._attempt_order("auto") == ["cusolver", "cpu"]
    assert linalg._attempt_order("magma") == ["cpu"]
    assert linalg._attempt_order("cusolver") == ["cusolver"]
    assert linalg.low_memory_backend() == "cpu"


def _fake_backends(monkeypatch, failures):
    """Replace the single-backend call: names in ``failures`` raise that error."""

    def fake(K, name):
        if name in failures:
            raise failures[name]
        return torch.linalg.eigh(K)

    monkeypatch.setattr(linalg, "_eigh_once", fake)


def test_out_of_memory_falls_back_to_magma(monkeypatch):
    _fake_backends(monkeypatch, {"cusolver": torch.cuda.OutOfMemoryError("oom")})
    (w, _), used, failed = linalg._run_attempts(_spd(), ["cusolver", "magma", "cpu"])
    assert used == "magma" and failed == ["cusolver"]
    torch.testing.assert_close(w, torch.linalg.eigh(_spd())[0])


def test_magma_error_falls_back_to_host(monkeypatch):
    _fake_backends(
        monkeypatch,
        {
            "cusolver": torch.cuda.OutOfMemoryError("oom"),
            "magma": RuntimeError("linalg.eigh: error code -113"),
        },
    )
    _, used, failed = linalg._run_attempts(_spd(), ["cusolver", "magma", "cpu"])
    assert used == "cpu" and failed == ["cusolver", "magma"]


def test_other_cusolver_errors_propagate(monkeypatch):
    _fake_backends(monkeypatch, {"cusolver": RuntimeError("not a memory problem")})
    with pytest.raises(RuntimeError, match="not a memory problem"):
        linalg._run_attempts(_spd(), ["cusolver", "magma", "cpu"])


def test_last_backend_out_of_memory_propagates(monkeypatch):
    oom = torch.cuda.OutOfMemoryError("oom")
    _fake_backends(monkeypatch, {"cusolver": oom, "cpu": oom})
    with pytest.raises(torch.cuda.OutOfMemoryError):
        linalg._run_attempts(_spd(), ["cusolver", "cpu"])


# --------------------------------------------------------------------------
# Solvers and estimators
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [TorchKMSVC, TorchKMDWD, TorchKMLogit])
def test_classifiers_report_the_eigendecomposition(cls):
    X, y = _binary()
    clf = cls(kernel="rbf", nC=3, cv=3, device="cpu", max_iter=50).fit(X, y)
    assert clf.eigh_backend == "auto"
    assert clf.eigh_backend_ == "lapack"
    assert clf.eigh_seconds_ >= 0.0


def test_kqr_reports_the_eigendecomposition():
    X, y = make_regression(n_samples=60, n_features=4, noise=0.3, random_state=0)
    reg = TorchKMKQR(kernel="rbf", nC=2, cv=2, device="cpu", max_iter=30).fit(X, y)
    assert reg.eigh_backend_ == "lapack"
    assert reg.eigh_seconds_ >= 0.0


def test_fitted_model_does_not_depend_on_the_backend():
    X, y = _binary()
    # a fixed bandwidth: sigest draws random pairs, which would differ per fit
    kw = dict(kernel="rbf", rbf_sigma=0.3, nC=4, cv=3, device="cpu", max_iter=200)
    a = TorchKMSVC(**kw).fit(X, y)
    b = TorchKMSVC(eigh_backend="cpu", **kw).fit(X, y)
    assert a.best_C_ == b.best_C_
    np.testing.assert_allclose(
        a.decision_function(X), b.decision_function(X), rtol=0, atol=1e-12
    )


def test_unknown_backend_is_rejected_at_fit():
    X, y = _binary()
    with pytest.raises(ValueError, match="eigh_backend"):
        TorchKMSVC(nC=2, cv=2, device="cpu", eigh_backend="rocsolver").fit(X, y)


def test_nystrom_path_has_no_kernel_eigendecomposition():
    X, y = _binary(n=120)
    clf = TorchKMSVC(
        low_rank=True, num_landmarks=30, nys_k=10, nC=2, cv=2, device="cpu"
    ).fit(X, y)
    assert clf.eigh_backend_ is None and clf.eigh_seconds_ is None


def test_solver_records_eigh_info():
    from torchkm.cvksvm import cvksvm

    X, y = _binary(n=60, p=4)
    K = functions.rbf_kernel(torch.as_tensor(X, dtype=torch.double), 0.5)
    model = cvksvm(
        Kmat=K,
        y=torch.as_tensor(y, dtype=torch.double),
        nlam=2,
        ulam=torch.logspace(-1, -2, 2, dtype=torch.double),
        nfolds=3,
        maxit=100,
        device="cpu",
        eigh_backend="magma",
    )
    assert model.eigh_info is None
    model.fit()
    assert model.eigh_info["requested"] == "magma"
    assert model.eigh_info["used"] == "lapack"


# --------------------------------------------------------------------------
# Memory model
# --------------------------------------------------------------------------


def test_memory_model_uses_the_backend_peaks():
    n = 20_000
    fast = exact_mode_memory_estimate(n)
    assert fast == exact_mode_memory_estimate(n, backend="cusolver")
    assert fast == exact_mode_memory_estimate(n, backend="auto")
    assert EXACT_MODE_COPIES == EXACT_MODE_COPIES_BY_BACKEND["cusolver"]
    low = exact_mode_memory_estimate(n, backend="magma")
    assert low == exact_mode_memory_estimate(n, backend="cpu")
    ratio = (fast - 2 * n * 50 * 8) / (low - 2 * n * 50 * 8)
    assert ratio == pytest.approx(EXACT_MODE_COPIES / LOW_MEMORY_COPIES)
    assert exact_mode_memory_estimate(n, n_copies=3.0, backend="magma") == (
        exact_mode_memory_estimate(n, n_copies=3.0)
    )
    assert max_exact_n(48e9, backend="cpu") > max_exact_n(48e9)
    with pytest.raises(ValueError, match="backend"):
        exact_mode_memory_estimate(n, backend="rocsolver")


def test_oom_message_names_both_eigendecompositions():
    msg = exact_mode_oom_message(32_561, "cpu")
    assert "eigh_backend" in msg
    assert "cuSOLVER" in msg
    assert "low_rank=True" in msg


# --------------------------------------------------------------------------
# On a GPU
# --------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("backend", ["cusolver", "magma", "cpu", "auto"])
def test_gpu_backends_agree(backend):
    K = _spd(200).cuda()
    w, U, info = linalg.kernel_eigh(K, backend, return_info=True)
    w_ref, _ = torch.linalg.eigh(K.cpu())
    assert w.device == K.device and U.device == K.device
    torch.testing.assert_close(w.cpu(), w_ref, rtol=1e-10, atol=1e-8)
    expected = "cusolver" if backend == "auto" else backend
    if backend == "magma" and not linalg.has_magma():
        expected = "cpu"
    assert info["used"] == expected
    # the process-wide library preference is restored
    assert torch.backends.cuda.preferred_linalg_library() is not None
