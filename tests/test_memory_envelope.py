# SPDX-License-Identifier: MIT
"""Memory-envelope helpers, peak-memory reporting, Nyström seeding, and the
solver changes that reduce exact-mode memory (no ``eU`` copy, in-place kernel)."""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression

import torchkm
from torchkm import functions
from torchkm.estimators import TorchKMKQR, TorchKMSVC
from torchkm.memory import (
    EXACT_MODE_COPIES,
    exact_mode_memory_estimate,
    exact_mode_oom_message,
    format_bytes,
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
    n = max_exact_n(budget)
    assert exact_mode_memory_estimate(n) <= budget
    assert exact_mode_memory_estimate(int(n * 1.1)) > 0.9 * budget
    assert max_exact_n(0) == 0
    assert max_exact_n(budget, usable_fraction=1.0) > n


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
