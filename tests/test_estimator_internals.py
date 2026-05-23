"""Internal-contract tests for the scikit-learn-style estimator wrappers.

These cover code paths that aren't visible from the public ``fit`` /
``predict`` flow: helper-function input validation, backend dispatch,
shape checks for the ``precomputed`` kernel, and the score-shape
contract on regressors.
"""

import unittest

import numpy as np
import torch


def _toy_binary(n=80):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64) * 2 - 1
    return X, y


def _toy_regression(n=80):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (X[:, 0] + 0.1 * rng.normal(size=n)).astype(np.float64)
    return X, y


class TestUlamConstruction(unittest.TestCase):
    """``_make_ulam`` should reject a non-1-D ``Cs`` argument."""

    def test_multi_dimensional_cs_raises(self):
        from torchkm.estimators import _make_ulam

        with self.assertRaisesRegex(ValueError, "Cs must be 1D"):
            _make_ulam(nC=5, Cs=np.zeros((3, 3)), C_max=1.0, C_min=1e-3)


class TestUnsupportedKernelGuard(unittest.TestCase):
    """The kernel helpers should report unknown kernel names clearly.

    Mutating ``self.kernel`` after construction bypasses the constructor
    type check but lets us exercise the guard inside ``_compute_K_train``
    / ``_compute_K_test``.
    """

    def test_svc_unsupported_kernel_on_compute_K_train(self):
        from torchkm.estimators import TorchKMSVC

        X, _ = _toy_binary(40)
        est = TorchKMSVC(
            kernel="rbf",
            nC=2,
            cv=2,
            device="cpu",
            random_state=0,
            max_iter=20,
        )
        est.kernel = "unknown-kernel"
        with self.assertRaisesRegex(ValueError, "Unsupported kernel"):
            est._compute_K_train(torch.as_tensor(X, dtype=torch.double))

    def test_svc_unsupported_kernel_on_compute_K_test(self):
        from torchkm.estimators import TorchKMSVC

        X, _ = _toy_binary(40)
        est = TorchKMSVC(
            kernel="rbf",
            nC=2,
            cv=2,
            device="cpu",
            random_state=0,
            max_iter=20,
        )
        est.kernel = "unknown-kernel"
        with self.assertRaisesRegex(ValueError, "Unsupported kernel"):
            est._compute_K_test(
                torch.as_tensor(X, dtype=torch.double),
                torch.as_tensor(X, dtype=torch.double),
                {},
            )

    def test_kqr_unsupported_kernel_on_compute_K_train_and_test(self):
        from torchkm.estimators import TorchKMKQR

        X, _ = _toy_regression(40)
        est = TorchKMKQR(
            kernel="rbf",
            nC=2,
            cv=2,
            tau=0.5,
            device="cpu",
            random_state=0,
            max_iter=20,
        )
        est.kernel = "unknown-kernel"
        with self.assertRaisesRegex(ValueError, "Unsupported kernel"):
            est._compute_K_train(torch.as_tensor(X, dtype=torch.double))
        with self.assertRaisesRegex(ValueError, "Unsupported kernel"):
            est._compute_K_test(
                torch.as_tensor(X, dtype=torch.double),
                torch.as_tensor(X, dtype=torch.double),
                {},
            )


class TestUnknownBackendGuard(unittest.TestCase):
    """An unrecognised ``_BACKEND`` token should be flagged immediately."""

    def test_make_backend_rejects_unknown_label(self):
        from torchkm.estimators import TorchKMSVC

        est = TorchKMSVC(
            kernel="rbf",
            nC=2,
            cv=2,
            device="cpu",
            random_state=0,
            max_iter=20,
        )
        est._BACKEND = "not-a-real-backend"
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            est._make_backend(
                low_rank=False,
                dev="cpu",
                X_train_t=torch.zeros((4, 4), dtype=torch.double),
                K_train=torch.eye(4, dtype=torch.double),
                y_backend=torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.double),
                nlam=2,
                ulam_backend=torch.tensor([1.0, 0.1], dtype=torch.double),
                foldid_backend=torch.tensor([1, 2, 1, 2]),
            )


class TestPrecomputedKernelShape(unittest.TestCase):
    """``kernel='precomputed'`` should require a square training kernel."""

    def test_svc_rejects_non_square_precomputed_train(self):
        from torchkm.estimators import TorchKMSVC

        K = np.zeros((10, 6), dtype=np.float64)
        y = (np.arange(10) % 2) * 2 - 1
        est = TorchKMSVC(
            kernel="precomputed",
            nC=2,
            cv=2,
            device="cpu",
            random_state=0,
            max_iter=20,
        )
        with self.assertRaisesRegex(ValueError, "must be a square"):
            est.fit(K, y)

    def test_kqr_rejects_non_square_precomputed_train(self):
        from torchkm.estimators import TorchKMKQR

        K = np.zeros((10, 6), dtype=np.float64)
        y = np.arange(10, dtype=np.float64)
        est = TorchKMKQR(
            kernel="precomputed",
            nC=2,
            cv=2,
            tau=0.5,
            device="cpu",
            random_state=0,
            max_iter=20,
        )
        with self.assertRaisesRegex(ValueError, "must be a square"):
            est.fit(K, y)


class TestKqrScore(unittest.TestCase):
    """``TorchKMKQR.score`` should refuse mismatched ``X`` / ``y`` lengths."""

    def test_score_rejects_mismatched_lengths(self):
        from torchkm.estimators import TorchKMKQR

        X, y = _toy_regression(60)
        est = TorchKMKQR(
            kernel="rbf",
            nC=2,
            cv=2,
            tau=0.5,
            device="cpu",
            random_state=0,
            max_iter=50,
        ).fit(X, y)
        with self.assertRaisesRegex(ValueError, "incompatible lengths"):
            est.score(X, y[:10])


class TestLowRankFitTimeOptions(unittest.TestCase):
    """Passing low-rank options into ``fit`` should update the estimator's
    state in place so subsequent calls observe the new configuration.
    """

    def test_apply_fit_low_rank_options_updates(self):
        from torchkm.estimators import TorchKMKQR

        est = TorchKMKQR(
            kernel="rbf",
            nC=2,
            cv=2,
            tau=0.5,
            device="cpu",
            low_rank=False,
            num_landmarks=10,
            nys_k=5,
            max_iter=20,
        )
        est._apply_fit_low_rank_options(low_rank=True, num_landmarks=33, nys_k=7)
        self.assertTrue(est.low_rank)
        self.assertEqual(est.num_landmarks, 33)
        self.assertEqual(est.nys_k, 7)


if __name__ == "__main__":
    unittest.main()
