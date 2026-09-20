# SPDX-License-Identifier: MIT
"""The binary estimators compose with scikit-learn's multiclass wrappers."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from torchkm.estimators import TorchKMSVC


def _three_classes():
    X, y = make_classification(
        n_samples=150,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=0,
    )
    return X, y


def test_one_vs_rest_trains_one_tuned_model_per_class():
    X, y = _three_classes()
    clf = OneVsRestClassifier(
        TorchKMSVC(kernel="rbf", nC=3, cv=3, device="cpu", max_iter=50, random_state=0)
    )
    clf.fit(X, y)
    assert len(clf.estimators_) == 3
    pred = clf.predict(X)
    assert set(np.unique(pred)) <= {0, 1, 2}
    assert (pred == y).mean() > 0.8
    assert clf.decision_function(X).shape == (150, 3)


def test_one_vs_rest_probabilities_sum_to_one():
    X, y = _three_classes()
    clf = OneVsRestClassifier(
        TorchKMSVC(
            kernel="rbf",
            nC=3,
            cv=3,
            device="cpu",
            max_iter=50,
            probability=True,
            random_state=0,
        )
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (150, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_one_vs_one_works_too():
    X, y = _three_classes()
    clf = OneVsOneClassifier(
        TorchKMSVC(kernel="rbf", nC=3, cv=3, device="cpu", max_iter=50, random_state=0)
    )
    clf.fit(X, y)
    assert (clf.predict(X) == y).mean() > 0.8
