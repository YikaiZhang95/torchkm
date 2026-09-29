# SPDX-License-Identifier: MIT
"""Memory-envelope helpers, peak-memory reporting, Nyström seeding, and the
solver changes that reduce exact-mode memory (no ``eU`` copy, in-place kernel)."""

import importlib

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression

import torchkm
from torchkm import functions, memory
from torchkm.estimators import TorchKMKQR, TorchKMSVC
from torchkm.memory import (
    EXACT_MODE_COPIES,
    EXACT_MODE_MAX_N_CUDA,
    exact_mode_memory_estimate,
    exact_mode_oom_message,
    format_bytes,
    kernel_eigh,
    max_exact_n,
)


def _binary(n=90, p=6, seed=0):
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=4, n_redundant=0, random_state=seed
    )
    return X, np.where(y == 0, -1, 1)


# --------------------------------------------------------------------------
# torchkm.memory
# --------------------------------------------------------------------------


def test_estimate_follows_the_quadratic_model():
    n, nlam = 10_000, 50
    expected = EXACT_MODE_COPIES * 8 * n * n + 2 * n * nlam * 8
    assert exact_mode_memory_estimate(n, nlam=nlam) == pytest.approx(expected, rel=1e-9)
    assert exact_mode_memory_estimate(2 * n) > 3.9 * exact_mode_memory_estimate(n)
    assert exact_mode_memory_estimate(
        n, dtype=torch.float32
    ) < exact_mode_memory_estimate(n)


def test_max_exact_n_inverts_the_estimate_with_headroom():
    budget = 48e9
    n = max_exact_n(budget, size_limit=None)
    assert exact_mode_memory_estimate(n) <= budget
    assert exact_mode_memory_estimate(int(n * 1.1)) > 0.9 * budget
    assert max_exact_n(0) == 0
    assert max_exact_n(budget, usable_fraction=1.0, size_limit=None) > n


def test_max_exact_n_stops_at_the_eigensolver_size_limit():
    # memory alone would allow n = 42,426 in 80 GB; cuSOLVER refuses more
    # than EXACT_MODE_MAX_N_CUDA
    assert max_exact_n(80e9, size_limit=None) > EXACT_MODE_MAX_N_CUDA
    assert max_exact_n(80e9) == EXACT_MODE_MAX_N_CUDA == 32_768
    assert max_exact_n(8e9) == max_exact_n(8e9, size_limit=None) == 13_416


def test_estimate_rejects_negative_n():
    with pytest.raises(ValueError):
        exact_mode_memory_estimate(-1)


def test_format_bytes():
    assert format_bytes(12.34e9) == "12.3 GB"
    assert format_bytes(512e6) == "512 MB"


def test_oom_message_names_the_size_and_the_remedy():
    msg = exact_mode_oom_message(30_000, "cpu")
    assert "n_samples=30,000" in msg
    assert "low_rank=True" in msg
    assert "GB" in msg
    # No CUDA device in a CPU test: the device-total clause is omitted.
    assert "device reports" not in msg


_REFUSAL = (
    "cusolver error: CUSOLVER_STATUS_INVALID_VALUE, when calling "
    "`cusolverDnXsyevd_bufferSize( handle, params, jobz, uplo, n, ...)`"
)


def _refusing_eigh(monkeypatch, message=_REFUSAL, limit=10):
    """eigh fails above ``limit`` as cuSOLVER does above 32,768."""
    real = torch.linalg.eigh

    def eigh(A, *args, **kwargs):
        if A.shape[-1] > limit:
            raise torch.linalg.LinAlgError(message)
        return real(A, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", eigh)
    monkeypatch.setattr(memory, "EXACT_MODE_MAX_N_CUDA", limit)


def test_kernel_eigh_names_the_size_limit(monkeypatch):
    _refusing_eigh(monkeypatch)
    with pytest.raises(torch.linalg.LinAlgError, match="n_samples=20") as info:
        kernel_eigh(torch.eye(20, dtype=torch.float64))
    assert "low_rank=True" in str(info.value)
    assert "cusolver" in str(info.value.__cause__)
    e, _ = kernel_eigh(torch.eye(5, dtype=torch.float64))  # within the limit
    torch.testing.assert_close(e, torch.ones(5, dtype=torch.float64))


def test_kernel_eigh_passes_other_errors_through(monkeypatch):
    _refusing_eigh(monkeypatch, message="linalg.eigh: failed to converge")
    with pytest.raises(torch.linalg.LinAlgError, match="failed to converge"):
        kernel_eigh(torch.eye(20, dtype=torch.float64))


_EXACT = [
    ("cvksvm", {}),
    ("cvkdwd", {}),
    ("cvklogit", {}),
    ("cvksqsvm", {}),
    ("cvkqr", {"tau": 0.5}),
    ("cvkhuber", {"delta": 1.0}),
]


def _exact_solver(name, extra, rebuild=False):
    """One of the six exact solvers on a 40-row RBF problem; with ``rebuild`` it
    factorizes the kernel in place and rebuilds it."""
    X, y = _binary(n=40, p=4)
    Xt = torch.as_tensor(X, dtype=torch.double)
    return getattr(importlib.import_module(f"torchkm.{name}"), name)(
        Kmat=functions.rbf_kernel(Xt, 0.5),
        y=torch.as_tensor(y, dtype=torch.double),
        nlam=2,
        ulam=torch.tensor([0.1, 0.01], dtype=torch.double),
        foldid=torch.arange(40) % 2 + 1,
        nfolds=2,
        maxit=100,
        device="cpu",
        rebuild_kmat=(lambda: functions.rbf_kernel(Xt, 0.5)) if rebuild else None,
        **extra,
    )


@pytest.mark.parametrize("name, extra", _EXACT)
def test_exact_solvers_name_the_size_limit(monkeypatch, name, extra):
    _refusing_eigh(monkeypatch)
    with pytest.raises(torch.linalg.LinAlgError, match="n_samples=40"):
        _exact_solver(name, extra).fit()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_kernel_eigh_in_place_gives_the_same_eigenpairs(dtype):
    torch.manual_seed(0)
    A = torch.randn(30, 30, dtype=torch.float64)
    K = (A @ A.T).to(dtype)
    w, U = kernel_eigh(K.clone())
    K2 = K.clone()
    w2, U2 = kernel_eigh(K2, overwrite=True)
    assert torch.equal(w, w2) and torch.equal(U, U2)
    assert U2.data_ptr() == K2.data_ptr()  # the eigenvectors took K's storage


@pytest.mark.parametrize("name, extra", _EXACT)
def test_exact_solvers_factorize_in_place_with_the_same_fit(name, extra):
    copy = _exact_solver(name, extra)
    copy.fit()
    inplace = _exact_solver(name, extra, rebuild=True)
    inplace.fit()
    assert torch.equal(inplace.alpmat, copy.alpmat)
    assert torch.equal(inplace.pred, copy.pred)


@pytest.mark.parametrize("estimator", [TorchKMSVC, TorchKMKQR])
def test_estimators_factorize_in_place_but_not_a_precomputed_kernel(
    monkeypatch, estimator
):
    module = importlib.import_module(
        "torchkm.cvksvm" if estimator is TorchKMSVC else "torchkm.cvkqr"
    )
    real = module.kernel_eigh
    seen = []

    def spy(K, overwrite=False):
        seen.append(overwrite)
        return real(K, overwrite=overwrite)

    X, y = _binary()
    kw = dict(nC=2, cv=2, device="cpu", max_iter=20)
    monkeypatch.setattr(module, "kernel_eigh", spy)
    a = estimator(kernel="rbf", rbf_sigma=0.5, **kw).fit(X, y)
    # the same fit through the copying path
    monkeypatch.setattr(module, "kernel_eigh", lambda K, overwrite=False: real(K))
    b = estimator(kernel="rbf", rbf_sigma=0.5, **kw).fit(X, y)
    assert seen == [True]
    np.testing.assert_array_equal(a.alpha_, b.alpha_)
    assert a.intercept_ == b.intercept_
    # a precomputed kernel is the caller's array: factorized as a copy
    K = functions.rbf_kernel(torch.as_tensor(X, dtype=torch.double), 0.5).numpy()
    K0 = K.copy()
    monkeypatch.setattr(module, "kernel_eigh", spy)
    estimator(kernel="precomputed", **kw).fit(K, y)
    assert seen[-1] is False
    np.testing.assert_array_equal(K, K0)


def test_estimator_names_the_size_limit(monkeypatch):
    _refusing_eigh(monkeypatch)
    X, y = _binary()
    with pytest.raises(torch.linalg.LinAlgError, match="n_samples=90"):
        TorchKMSVC(kernel="rbf", nC=2, cv=2, device="cpu", max_iter=20).fit(X, y)


def test_memory_helpers_are_exported():
    assert torchkm.exact_mode_memory_estimate is exact_mode_memory_estimate
    assert torchkm.max_exact_n is max_exact_n


# --------------------------------------------------------------------------
# peak_gpu_memory_bytes_
# --------------------------------------------------------------------------


def test_peak_memory_attribute_is_none_after_cpu_fit():
    X, y = _binary()
    clf = TorchKMSVC(kernel="rbf", nC=3, cv=3, device="cpu", max_iter=30).fit(X, y)
    assert clf.peak_gpu_memory_bytes_ is None

    Xr, yr = make_regression(n_samples=60, n_features=4, noise=0.3, random_state=0)
    reg = TorchKMKQR(kernel="rbf", nC=2, cv=2, device="cpu", max_iter=30).fit(Xr, yr)
    assert reg.peak_gpu_memory_bytes_ is None


def test_peak_memory_attribute_is_cleared_on_refit():
    X, y = _binary()
    clf = TorchKMSVC(kernel="rbf", nC=2, cv=2, device="cpu", max_iter=20).fit(X, y)
    clf.peak_gpu_memory_bytes_ = 123
    clf.fit(X, y)
    assert clf.peak_gpu_memory_bytes_ is None


# --------------------------------------------------------------------------
# Nyström seeding
# --------------------------------------------------------------------------


def _nys(seed, **kw):
    return TorchKMSVC(
        kernel="rbf",
        low_rank=True,
        num_landmarks=30,
        nys_k=10,
        nC=3,
        cv=3,
        device="cpu",
        max_iter=30,
        random_state=seed,
        **kw,
    )


def test_nystrom_landmarks_follow_random_state():
    X, y = _binary(n=120)
    a = _nys(1).fit(X, y)
    b = _nys(1).fit(X, y)
    c = _nys(2).fit(X, y)
    np.testing.assert_array_equal(
        a.low_rank_landmark_indices_, b.low_rank_landmark_indices_
    )
    np.testing.assert_allclose(a.decision_function(X), b.decision_function(X))
    assert not np.array_equal(
        a.low_rank_landmark_indices_, c.low_rank_landmark_indices_
    )


def test_nystrom_fit_leaves_the_global_rng_alone():
    X, y = _binary(n=120)
    torch.manual_seed(7)
    before = torch.get_rng_state()
    _nys(3).fit(X, y)
    after = torch.get_rng_state()
    assert torch.equal(before, after)


def test_nystrom_without_random_state_draws_from_the_global_rng():
    X, y = _binary(n=120)
    torch.manual_seed(11)
    a = _nys(None).fit(X, y)
    torch.manual_seed(11)
    b = _nys(None).fit(X, y)
    np.testing.assert_array_equal(
        a.low_rank_landmark_indices_, b.low_rank_landmark_indices_
    )


def test_nystrom_honours_an_explicit_bandwidth():
    X, y = _binary(n=120)
    clf = _nys(0, rbf_sigma=0.7).fit(X, y)
    assert clf._low_rank_backend_.sig_w_ == pytest.approx(0.7)


def test_sigest_generator_is_reproducible_and_local():
    x = torch.randn(200, 5)
    g1 = torch.Generator().manual_seed(5)
    g2 = torch.Generator().manual_seed(5)
    torch.manual_seed(0)
    state = torch.get_rng_state()
    s1 = functions.sigest(x, generator=g1)
    s2 = functions.sigest(x, generator=g2)
    assert s1 == pytest.approx(s2)
    assert torch.equal(state, torch.get_rng_state())


# --------------------------------------------------------------------------
# Kernel construction and the projection identity
# --------------------------------------------------------------------------


def test_rbf_kernel_matches_definition_and_leaves_input_untouched():
    x = torch.randn(40, 3, dtype=torch.double)
    x_copy = x.clone()
    sig = 0.3
    K = functions.rbf_kernel(x, sig)
    d2 = torch.cdist(x, x) ** 2
    torch.testing.assert_close(K, torch.exp(-2.0 * sig * d2), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(x, x_copy)
    z = torch.randn(7, 3, dtype=torch.double)
    Kz = functions.kernelMult(z, x, sig)
    torch.testing.assert_close(
        Kz, torch.exp(-2.0 * sig * torch.cdist(z, x) ** 2), rtol=1e-10, atol=1e-10
    )


def test_projection_without_materialised_inverse_matches_solve():
    """U diag(1/e) U^T theta, applied as in the solvers, equals (K + gamma I)^-1 theta."""
    torch.manual_seed(0)
    A = torch.randn(25, 25, dtype=torch.double)
    K = A @ A.T + 25 * torch.eye(25, dtype=torch.double)
    gamma = 1e-8
    eigens, U = torch.linalg.eigh(K)
    einv = 1.0 / (eigens + gamma)
    theta = torch.randn(25, dtype=torch.double)
    applied = torch.mv(U, einv * torch.mv(U.T, theta))
    expected = torch.linalg.solve(K + gamma * torch.eye(25, dtype=torch.double), theta)
    torch.testing.assert_close(applied, expected, rtol=1e-8, atol=1e-8)


def test_exact_projection_path_still_runs():
    """``is_exact=1`` exercises the projection that used the removed matrix."""
    from torchkm.cvksvm import cvksvm

    X, y = _binary(n=80, p=4)
    Xt = torch.as_tensor(X, dtype=torch.double)
    K = functions.rbf_kernel(Xt, 0.5)
    model = cvksvm(
        Kmat=K,
        y=torch.as_tensor(y, dtype=torch.double),
        nlam=3,
        ulam=torch.logspace(-1, -3, 3, dtype=torch.double),
        nfolds=4,
        eps=1e-4,
        maxit=200,
        gamma=1e-8,
        is_exact=1,
        KKTeps=1e-1,
        device="cpu",
    )
    model.fit()
    assert torch.isfinite(model.alpmat).all()
