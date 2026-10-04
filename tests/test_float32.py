# SPDX-License-Identifier: MIT
"""The exact SVM solver in float32 reaches the float64 solution to single
precision, and the option is refused where it is not supported."""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification

from torchkm import rbf_kernel, sigest
from torchkm.cvksvm import cvksvm
from torchkm.estimators import TorchKMDWD, TorchKMSVC


def _problem(n=300, seed=0):
    X, y = make_classification(
        n_samples=n, n_features=8, n_informative=5, flip_y=0.05, random_state=seed
    )
    return X, np.where(y == 0, -1.0, 1.0)


def _objective(K, y, alp, lam):
    Ka = K @ alp[1:]
    hinge = torch.clamp(1.0 - y * (Ka + alp[0]), min=0.0)
    return float(hinge.mean() + lam * torch.dot(alp[1:], Ka))


def test_cvksvm_float32_path_matches_float64():
    # KKTeps=1e-6: at the loose default both precisions stop short of the
    # optimum, at different points, and differ by more than their precision
    X, y = _problem()
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y)
    torch.manual_seed(0)
    sig = float(sigest(Xt))
    K = rbf_kernel(Xt, sig)
    ulam = torch.logspace(0, -3, 8, dtype=torch.double)
    foldid = torch.arange(len(y)) % 3 + 1
    fits = {}
    for dt in (torch.float64, torch.float32):
        m = cvksvm(
            Kmat=rbf_kernel(Xt.to(dt), sig),
            y=yt,
            nlam=len(ulam),
            ulam=ulam,
            foldid=foldid,
            nfolds=3,
            eps=1e-5,
            maxit=100_000,
            gamma=1e-8,
            KKTeps=1e-6,
            device="cpu",
            dtype=dt,
        )
        m.fit()
        assert m.alpmat.dtype == dt and m.pred.dtype == dt
        assert m.jerr == 0
        fits[dt] = m
    a64, a32 = fits[torch.float64], fits[torch.float32]
    for j, lam in enumerate(ulam.tolist()):
        o64 = _objective(K, yt, a64.alpmat[:, j], lam)
        o32 = _objective(K, yt, a32.alpmat[:, j].double(), lam)
        # the float64 fit is itself only as close to the optimum as its stopping
        # rule allows (float32 can land below it): float32 must not land above
        assert o32 - o64 <= 1e-3 * abs(o64)
    np.testing.assert_allclose(
        a32.cv(a32.pred, yt).numpy(), a64.cv(a64.pred, yt).numpy(), atol=2.0 / len(y)
    )


def test_torchkmsvc_float32_predicts_like_float64():
    X, y = _problem(seed=1)
    Xte, _ = _problem(n=200, seed=2)
    kw = dict(kernel="rbf", nC=6, C_max=1e2, C_min=1e-1, cv=3, device="cpu")
    kw.update(max_iter=100_000, KKTeps=1e-6, random_state=0, rbf_sigma=0.1)
    f64 = TorchKMSVC(**kw).fit(X, y)
    f32 = TorchKMSVC(dtype="float32", **kw).fit(X, y)
    assert f32.best_C_ == f64.best_C_
    s64, s32 = f64.decision_function(Xte), f32.decision_function(Xte)
    np.testing.assert_allclose(s32, s64, atol=5e-3 * np.abs(s64).max())
    assert np.mean(f32.predict(Xte) == f64.predict(Xte)) >= 0.99


def test_dtype_is_validated():
    X, y = _problem(n=60)
    with pytest.raises(ValueError, match="dtype"):
        TorchKMSVC(dtype="float16", nC=2, cv=2, device="cpu").fit(X, y)
    with pytest.raises(ValueError, match="TorchKMSVC"):  # exact DWD is float64
        TorchKMDWD(dtype="float32", nC=2, cv=2, device="cpu").fit(X, y)
    # the matrix-free SVM computes in float32 too (fused products on CUDA)
    clf = TorchKMSVC(dtype="float32", low_rank=True, nC=2, cv=2, device="cpu")
    assert np.isfinite(clf.fit(X, y).decision_function(X[:3])).all()
    K = torch.eye(4)
    with pytest.raises(ValueError, match="dtype"):
        cvksvm(K, torch.tensor([1.0, -1, 1, -1]), 1, torch.ones(1), dtype=torch.float16)
