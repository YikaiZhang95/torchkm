# SPDX-License-Identifier: MIT
"""``is_exact`` in the Nystrom SVM solver (``cvknyssvm``).

With ``is_exact=1`` the solver puts the points in the smoothing band exactly on
the margin once the band is narrow enough for the rank-k feature model, and
keeps the result only if the hinge-loss KKT condition holds there.
"""

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from torchkm.cvknyssvm import cvknyssvm
from torchkm.estimators import TorchKMSVC


def _data(n=300, seed=0):
    X, y = make_classification(
        n_samples=n, n_features=6, n_informative=4, flip_y=0.05, random_state=seed
    )
    X = StandardScaler().fit_transform(X)
    return X, np.where(y == 0, -1.0, 1.0)


def _fit(X, y, lams, is_exact):
    model = cvknyssvm(
        Xmat=torch.as_tensor(X),
        X_test=torch.as_tensor(X[:20]),
        y=torch.as_tensor(y),
        nlam=len(lams),
        ulam=torch.as_tensor(lams),
        foldid=torch.arange(len(y)) % 3 + 1,
        nfolds=3,
        eps=1e-5,
        maxit=20000,
        gamma=1e-8,
        num_landmarks=60,
        k=15,
        device="cpu",
        random_state=0,
        is_exact=is_exact,
    )
    model.fit()
    return model


def _objective(model, y, lam, j):
    Z = model.Z_train.numpy()
    alp = model.alpmat[:, j].numpy()
    f = Z @ alp[1:] + alp[0]
    return lam * alp[1:] @ alp[1:] + np.mean(np.maximum(0.0, 1.0 - y * f))


def test_is_exact_fit_is_finite_and_as_good_as_the_default():
    X, y = _data()
    lams = np.array([1e-1, 1e-2])
    plain, exact = _fit(X, y, lams, 0), _fit(X, y, lams, 1)
    assert torch.isfinite(exact.alpmat).all()
    assert torch.isfinite(exact.pred).all()
    for j, lam in enumerate(lams):
        assert _objective(exact, y, lam, j) <= _objective(plain, y, lam, j) * 1.01


def test_estimator_passes_is_exact_to_the_nystrom_backend():
    X, y = _data(n=200)
    clf = TorchKMSVC(
        low_rank=True,
        is_exact=1,
        num_landmarks=40,
        nys_k=10,
        nC=3,
        cv=3,
        device="cpu",
        random_state=0,
    ).fit(X, y)
    assert clf._low_rank_backend_.is_exact == 1
    assert np.isfinite(clf.decision_function(X)).all()
