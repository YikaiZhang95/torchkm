# SPDX-License-Identifier: MIT
"""fit_timing_ and n_passes_ report where an exact SVM fit's time goes."""

from sklearn.datasets import make_classification

from torchkm.estimators import TorchKMSVC

PHASES = (
    "kernel",
    "eigendecomposition",
    "factorization_error",
    "path",
    "cross_validation",
)


def test_fit_timing_accounts_for_the_exact_fit():
    X, y = make_classification(n_samples=120, n_features=5, random_state=0)
    clf = TorchKMSVC(
        Cs=[0.1, 1.0, 10.0], nC=3, cv=3, device="cpu", max_iter=100_000, random_state=0
    )
    clf.fit(X, y)
    t = clf.fit_timing_
    assert set(t) == set(PHASES) | {"total"}
    assert all(v >= 0.0 for v in t.values())
    assert sum(t[k] for k in PHASES) <= t["total"]
    assert clf.n_passes_["path"] > 0 and clf.n_passes_["cross_validation"] > 0
    fp = clf.fit_profile_
    assert len(fp["lambdas"]) == len(fp["cv_seconds"]) == len(fp["path_passes"]) == 3
    assert len(fp["cv_passes"]) == 3 and all(len(f) == 3 for f in fp["cv_passes"])
    assert abs(sum(fp["cv_seconds"]) - t["cross_validation"]) < 1e-9
    assert sum(map(sum, fp["cv_passes"])) == clf.n_passes_["cross_validation"]
    assert sum(fp["path_passes"]) == clf.n_passes_["path"]


def test_fit_timing_low_rank_has_kernel_and_total():
    X, y = make_classification(n_samples=120, n_features=5, random_state=0)
    clf = TorchKMSVC(low_rank=True, nC=3, cv=3, device="cpu").fit(X, y)
    assert {"kernel", "factorization", "path", "total"} <= set(clf.fit_timing_)
    assert "eigendecomposition" not in clf.fit_timing_  # no full factorization
