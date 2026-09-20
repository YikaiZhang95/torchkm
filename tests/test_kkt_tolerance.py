# SPDX-License-Identifier: MIT
"""The KKT stopping tolerance is exposed on the classifiers and tightening it
moves the SVM solution towards the exact optimum at weak regularization."""

import numpy as np
import torch
from sklearn.datasets import make_classification

from torchkm import rbf_kernel, sigest
from torchkm.estimators import TorchKMDWD, TorchKMLogit, TorchKMSVC


def _objective(K, y, alpha, b, lam):
    Ka = K @ alpha
    hinge = torch.clamp(1.0 - y * (Ka + b), min=0.0)
    return float(hinge.mean() + lam * torch.dot(alpha, Ka))


def test_kkteps_is_passed_to_the_backends():
    X, y = make_classification(n_samples=80, n_features=5, random_state=0)
    y = np.where(y == 0, -1, 1)
    for cls in (TorchKMSVC, TorchKMDWD, TorchKMLogit):
        clf = cls(kernel="rbf", nC=2, cv=2, device="cpu", max_iter=20, KKTeps=1e-4)
        clf.fit(X, y)
        assert clf.KKTeps == 1e-4
        assert clf.get_params()["KKTeps"] == 1e-4


def test_tighter_kkteps_lowers_the_svm_objective_at_weak_regularization():
    # The squared KKT residual scales like 1/n, so the loose default only bites
    # once n is in the low thousands; n=2500 keeps the eigendecomposition cheap.
    X, y = make_classification(
        n_samples=2500, n_features=20, n_informative=10, random_state=1
    )
    y = np.where(y == 0, -1.0, 1.0)
    torch.manual_seed(0)
    Xt = torch.as_tensor(X, dtype=torch.double)
    sig = float(sigest(Xt))
    K = rbf_kernel(Xt, sig)
    y_t = torch.as_tensor(y, dtype=torch.double)
    lam = 1e-3
    C = 1.0 / (2.0 * X.shape[0] * lam)

    def fit(kkteps):
        clf = TorchKMSVC(
            kernel="rbf",
            rbf_sigma=sig,
            Cs=[C],
            nC=1,
            cv=2,
            device="cpu",
            max_iter=100_000,
            KKTeps=kkteps,
            random_state=0,
        ).fit(X, y)
        alpha = torch.as_tensor(clf.alpha_, dtype=torch.double)
        return _objective(K, y_t, alpha, clf.intercept_, lam)

    loose, tight = fit(1e-3), fit(1e-6)
    assert tight <= loose + 1e-12
    # The loose default stops well short of the optimum in this regime.
    assert loose - tight > 1e-3


def test_scale_aware_rule_reaches_the_optimum_with_the_default_tolerance():
    """``kkt_scaled=True`` at KKTeps=1e-3 behaves like a tight absolute tolerance."""
    X, y = make_classification(
        n_samples=2500, n_features=20, n_informative=10, random_state=1
    )
    y = np.where(y == 0, -1.0, 1.0)
    torch.manual_seed(0)
    Xt = torch.as_tensor(X, dtype=torch.double)
    sig = float(sigest(Xt))
    K = rbf_kernel(Xt, sig)
    y_t = torch.as_tensor(y, dtype=torch.double)
    lam = 1e-3
    C = 1.0 / (2.0 * X.shape[0] * lam)

    def fit(**kw):
        clf = TorchKMSVC(
            kernel="rbf",
            rbf_sigma=sig,
            Cs=[C],
            nC=1,
            cv=2,
            device="cpu",
            max_iter=100_000,
            random_state=0,
            **kw,
        ).fit(X, y)
        alpha = torch.as_tensor(clf.alpha_, dtype=torch.double)
        return _objective(K, y_t, alpha, clf.intercept_, lam)

    loose = fit()
    scaled = fit(kkt_scaled=True)
    tight = fit(KKTeps=1e-6)
    assert scaled <= loose + 1e-12
    assert loose - scaled > 1e-3
    # Same relative accuracy as the tight absolute rule (which equals
    # KKTeps_scaled = n * 1e-6 = 2.5e-3 here), within solver noise.
    assert abs(scaled - tight) < 5e-3


def test_kkt_scaled_is_passed_to_the_kqr_backends():
    from sklearn.datasets import make_regression

    from torchkm.estimators import TorchKMKQR

    Xr, yr = make_regression(n_samples=60, n_features=4, noise=0.3, random_state=0)
    for low_rank in (False, True):
        reg = TorchKMKQR(
            kernel="rbf",
            nC=2,
            cv=2,
            device="cpu",
            max_iter=30,
            kkt_scaled=True,
            low_rank=low_rank,
            num_landmarks=20,
            nys_k=10,
            random_state=0,
        ).fit(Xr, yr)
        assert reg.kkt_scaled is True
        assert np.isfinite(reg.predict(Xr[:3])).all()
