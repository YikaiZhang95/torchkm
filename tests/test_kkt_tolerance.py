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
