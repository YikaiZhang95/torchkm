# SPDX-License-Identifier: MIT
"""Convergence-related edge cases for the kernel-machine solvers.

These tests probe the solvers under iteration-budget exhaustion, the
exact-projection refinement (``is_exact=1``) for a few corner regimes,
and a handful of helper-method contracts (e.g. ``transform`` before
``fit``).
"""

import unittest

import torch

from torchkm.cvkdwd import cvkdwd
from torchkm.cvkhuber import cvkhuber
from torchkm.cvklogit import cvklogit
from torchkm.cvknysdwd import cvknysdwd
from torchkm.cvknyslogit import cvknyslogit
from torchkm.cvknyssvm import cvknyssvm
from torchkm.cvkqr import cvkqr
from torchkm.cvksqsvm import cvksqsvm
from torchkm.cvksvm import cvksvm
from torchkm.functions import rbf_kernel, sigest, standardize


def _binary_data(nn=120, pp=4, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(nn, pp, dtype=torch.float64)
    w = torch.randn(pp, dtype=torch.float64)
    y = torch.where(X @ w > 0, torch.tensor(1.0), torch.tensor(-1.0))
    return standardize(X), y


def _reg_data(nn=80, pp=4, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(nn, pp, dtype=torch.float64)
    y = X[:, 0] + 0.1 * torch.randn(nn, dtype=torch.float64)
    return standardize(X), y


def _kernel_and_folds(X, nfolds=3, seed=0):
    sig = sigest(X)
    K = rbf_kernel(X, sig)
    torch.manual_seed(seed)
    foldid = torch.randperm(X.shape[0]) % nfolds + 1
    return K, foldid


class TestSolverIterationBudget(unittest.TestCase):
    """``maxit=1`` should cause every backend to exit cleanly and set a
    non-positive ``jerr`` rather than hanging or returning garbage.
    """

    def _make_kernel(self, X):
        sig = sigest(X)
        return rbf_kernel(X, sig)

    def test_cvkhuber_finishes_under_tight_budget(self):
        X, y = _reg_data(nn=60)
        Kmat = self._make_kernel(X)
        ulam = torch.tensor([10.0, 1.0, 0.1], dtype=torch.float64)
        torch.manual_seed(0)
        foldid = torch.randperm(60) % 3 + 1
        m = cvkhuber(
            delta=0.1,
            Kmat=Kmat,
            y=y,
            nlam=3,
            ulam=ulam,
            foldid=foldid,
            nfolds=3,
            eps=1e-12,
            maxit=1,
            gamma=1e-7,
            device="cpu",
        )
        m.fit()
        # A failed-to-converge run should be reported via a non-positive
        # error flag, not by silently returning a half-finished solution.
        self.assertTrue(m.jerr <= 0)

    def test_cvksqsvm_finishes_under_tight_budget(self):
        X, y = _binary_data(nn=60)
        Kmat = self._make_kernel(X)
        ulam = torch.tensor([10.0, 1.0, 0.1], dtype=torch.float64)
        torch.manual_seed(0)
        foldid = torch.randperm(60) % 3 + 1
        m = cvksqsvm(
            Kmat=Kmat,
            y=y,
            nlam=3,
            ulam=ulam,
            foldid=foldid,
            nfolds=3,
            eps=1e-12,
            maxit=1,
            gamma=1e-7,
            device="cpu",
        )
        m.fit()
        self.assertTrue(m.jerr <= 0)

    def test_cvklogit_finishes_under_tight_budget(self):
        X, y = _binary_data(nn=60)
        Kmat = self._make_kernel(X)
        ulam = torch.tensor([10.0, 1.0, 0.1], dtype=torch.float64)
        torch.manual_seed(0)
        foldid = torch.randperm(60) % 3 + 1
        m = cvklogit(
            Kmat=Kmat,
            y=y,
            nlam=3,
            ulam=ulam,
            foldid=foldid,
            nfolds=3,
            eps=1e-12,
            maxit=1,
            gamma=1e-7,
            device="cpu",
        )
        m.fit()
        self.assertTrue(m.jerr <= 0)

    def test_cvkqr_finishes_under_tight_budget(self):
        X, y = _reg_data(nn=40)
        Kmat, foldid = _kernel_and_folds(X, nfolds=3)
        ulam = torch.tensor([1.0, 0.1, 0.01], dtype=torch.float64)
        m = cvkqr(
            Kmat=Kmat,
            y=y,
            nlam=3,
            ulam=ulam,
            tau=0.5,
            foldid=foldid,
            nfolds=3,
            eps=1e-12,
            maxit=1,
            gamma=1e-7,
            is_exact=0,
            device="cpu",
        )
        m.fit()
        self.assertTrue(m.jerr <= 0)

    def test_cvkdwd_finishes_under_tight_budget(self):
        X, y = _binary_data(nn=40)
        Kmat, foldid = _kernel_and_folds(X, nfolds=3)
        ulam = torch.tensor([10.0, 1.0, 0.1], dtype=torch.float64)
        m = cvkdwd(
            Kmat=Kmat,
            y=y,
            nlam=3,
            ulam=ulam,
            foldid=foldid,
            nfolds=3,
            eps=1e-12,
            maxit=1,
            gamma=1e-7,
            device="cpu",
        )
        m.fit()
        self.assertTrue(m.jerr <= 0 or torch.sum(m.cvnpass).item() > 0)

    def test_cvknysdwd_finishes_under_tight_budget(self):
        X, y = _binary_data(nn=30)
        ulam = torch.tensor([10.0, 1.0, 0.1], dtype=torch.float64)
        m = cvknysdwd(
            Xmat=X,
            X_test=X[:5],
            y=y,
            nlam=3,
            ulam=ulam,
            foldid=None,
            nfolds=3,
            eps=1e-12,
            maxit=1,
            gamma=1e-7,
            num_landmarks=6,
            k=4,
            device="cpu",
        )
        m.fit()
        self.assertTrue(m.jerr <= 0)

    def test_cvknyslogit_finishes_under_tight_budget(self):
        X, y = _binary_data(nn=30)
        ulam = torch.tensor([10.0, 1.0, 0.1], dtype=torch.float64)
        m = cvknyslogit(
            Xmat=X,
            X_test=X[:5],
            y=y,
            nlam=3,
            ulam=ulam,
            foldid=None,
            nfolds=3,
            eps=1e-12,
            maxit=1,
            gamma=1e-7,
            num_landmarks=6,
            k=4,
            device="cpu",
        )
        m.fit()
        self.assertTrue(m.jerr <= 0)

    def test_cvknyssvm_finishes_under_tight_budget(self):
        X, y = _binary_data(nn=30)
        ulam = torch.tensor([10.0, 1.0, 0.1], dtype=torch.float64)
        m = cvknyssvm(
            Xmat=X,
            X_test=X[:5],
            y=y,
            nlam=3,
            ulam=ulam,
            foldid=None,
            nfolds=3,
            eps=1e-12,
            maxit=1,
            gamma=1e-7,
            num_landmarks=6,
            k=4,
            device="cpu",
        )
        m.fit()
        self.assertTrue(m.jerr <= 0)


class TestNystromLooFoldid(unittest.TestCase):
    """Leave-one-out fold assignment for the Nyström-based solvers."""

    def test_cvknysdwd_loo_foldid(self):
        X, y = _binary_data(nn=30)
        ulam = torch.tensor([10.0, 1.0, 0.1], dtype=torch.float64)
        m = cvknysdwd(
            Xmat=X,
            X_test=X[:5],
            y=y,
            nlam=3,
            ulam=ulam,
            foldid=None,
            nfolds=X.shape[0],
            eps=1e-4,
            maxit=200,
            gamma=1e-7,
            num_landmarks=6,
            k=4,
            device="cpu",
        )
        self.assertEqual(m.foldid.numel(), X.shape[0])

    def test_cvknyslogit_loo_foldid(self):
        X, y = _binary_data(nn=30)
        ulam = torch.tensor([10.0, 1.0, 0.1], dtype=torch.float64)
        m = cvknyslogit(
            Xmat=X,
            X_test=X[:5],
            y=y,
            nlam=3,
            ulam=ulam,
            foldid=None,
            nfolds=X.shape[0],
            eps=1e-4,
            maxit=200,
            gamma=1e-7,
            num_landmarks=6,
            k=4,
            device="cpu",
        )
        self.assertEqual(m.foldid.numel(), X.shape[0])


class TestCvknyssvmTransform(unittest.TestCase):
    """``transform`` should refuse to map new features before ``fit``."""

    def test_transform_before_fit_raises(self):
        X, y = _binary_data(nn=20)
        m = cvknyssvm(
            Xmat=X,
            X_test=X[:5],
            y=y,
            nlam=2,
            ulam=torch.tensor([1.0, 0.1], dtype=torch.float64),
            foldid=None,
            nfolds=2,
            num_landmarks=5,
            k=3,
            device="cpu",
        )
        with self.assertRaisesRegex(RuntimeError, "Call fit"):
            m.transform(X[:3])


class TestCvklogitPredictionHelpers(unittest.TestCase):
    """``cvklogit.predict`` and ``cvklogit.obj_value`` should return sensible
    outputs once the solver has been fitted.
    """

    def test_predict_returns_pm1_labels_and_accuracy(self):
        X, y = _binary_data(nn=80)
        Kmat, foldid = _kernel_and_folds(X, nfolds=3)
        ulam = torch.logspace(0, -2, steps=3, dtype=torch.float64)
        m = cvklogit(
            Kmat=Kmat,
            y=y,
            nlam=3,
            ulam=ulam,
            foldid=foldid,
            nfolds=3,
            eps=1e-4,
            maxit=200,
            gamma=1e-7,
            device="cpu",
        )
        m.fit()

        alpmat = m.alpmat.to("cpu")
        ypred, acc = m.predict(Kmat.double(), y, alpmat[:, 0])
        self.assertEqual(ypred.shape, (80,))
        self.assertTrue(0.0 <= float(acc) <= 1.0)
        self.assertTrue(set(torch.unique(ypred).tolist()).issubset({-1, 1}))

    def test_obj_value_is_finite(self):
        X, y = _binary_data(nn=80)
        Kmat, foldid = _kernel_and_folds(X, nfolds=3)
        ulam = torch.logspace(0, -2, steps=3, dtype=torch.float64)
        m = cvklogit(
            Kmat=Kmat,
            y=y,
            nlam=3,
            ulam=ulam,
            foldid=foldid,
            nfolds=3,
            eps=1e-4,
            maxit=200,
            gamma=1e-7,
            device="cpu",
        )
        m.fit()
        obj = m.obj_value(m.alpmat.to("cpu")[:, 0], ulam[0].item())
        self.assertTrue(torch.isfinite(obj).item())


class TestExactProjectionConvergence(unittest.TestCase):
    """A few corner regimes for the ``is_exact=1`` projection refinement.

    A trivial target (constant or zero ``y``) keeps the smooth-quantile
    FISTA from oscillating, so the post-FISTA KKT check and the
    delta-refinement schedule both run to completion. Together these
    cases stand in for behaviour we would otherwise only be able to
    observe on a much larger, slower-to-converge problem.
    """

    def test_cvksvm_is_exact_per_fold_cv_runs(self):
        # On a larger sample with loose KKT tolerances, ``is_exact=1``
        # should drive both the main fit and the per-fold CV through the
        # post-FISTA projection refinement and write valid coefficients
        # back for every lambda.
        torch.manual_seed(0)
        X = torch.randn(40, 4, dtype=torch.float64)
        w = torch.randn(4, dtype=torch.float64)
        y = torch.where(X @ w > 0, torch.tensor(1.0), torch.tensor(-1.0))
        X = standardize(X)
        sig = sigest(X)
        Kmat = rbf_kernel(X, sig)
        torch.manual_seed(0)
        foldid = torch.randperm(40) % 3 + 1
        m = cvksvm(
            Kmat=Kmat,
            y=y,
            nlam=2,
            ulam=torch.tensor([1.0, 0.1], dtype=torch.float64),
            foldid=foldid,
            nfolds=3,
            eps=1e-3,
            maxit=100,
            gamma=1e-7,
            is_exact=1,
            mproj=1,
            delta_len=2,
            KKTeps=1e-2,
            KKTeps2=1e-2,
            device="cpu",
        )
        m.fit()
        self.assertEqual(m.alpmat.shape, (41, 2))
        self.assertEqual(m.pred.shape, (40, 2))
        self.assertTrue(torch.isfinite(m.alpmat).all())

    def test_cvkqr_is_exact_converges_with_zero_target(self):
        # A zero target lets the smooth-quantile FISTA converge in one
        # sweep per fold, so the per-fold CV reaches the post-FISTA
        # KKT-check and projection refinement that would otherwise be
        # blocked by iteration-budget exhaustion.
        torch.manual_seed(0)
        X = torch.randn(15, 3, dtype=torch.float64)
        y = torch.zeros(15, dtype=torch.float64)
        X = standardize(X)
        sig = sigest(X)
        Kmat = rbf_kernel(X, sig)
        torch.manual_seed(0)
        foldid = torch.randperm(15) % 3 + 1
        m = cvkqr(
            Kmat=Kmat,
            y=y,
            nlam=1,
            ulam=torch.tensor([0.5], dtype=torch.float64),
            tau=0.5,
            foldid=foldid,
            nfolds=3,
            eps=1e-3,
            maxit=500,
            gamma=1e-7,
            is_exact=1,
            mproj=2,
            delta_len=3,
            KKTeps=1.0,
            KKTeps2=1.0,
            device="cpu",
        )
        m.fit()
        self.assertEqual(int(m.cvnpass[0]), 3)

    def test_cvkqr_batched_cv_converges_with_zero_target(self):
        # Same trivial target, but ``is_exact=0`` — the batched per-fold
        # KKT loop should run to completion instead of being skipped
        # because the FISTA stage exhausted its iteration budget.
        torch.manual_seed(0)
        X = torch.randn(15, 3, dtype=torch.float64)
        y = torch.zeros(15, dtype=torch.float64)
        X = standardize(X)
        sig = sigest(X)
        Kmat = rbf_kernel(X, sig)
        torch.manual_seed(0)
        foldid = torch.randperm(15) % 3 + 1
        m = cvkqr(
            Kmat=Kmat,
            y=y,
            nlam=1,
            ulam=torch.tensor([0.5], dtype=torch.float64),
            tau=0.5,
            foldid=foldid,
            nfolds=3,
            eps=1e-3,
            maxit=500,
            gamma=1e-7,
            is_exact=0,
            delta_len=3,
            device="cpu",
        )
        m.fit()
        self.assertEqual(int(m.cvnpass[0]), 3)

    def test_cvkqr_batched_cv_refines_delta_beyond_main_fit(self):
        # A tiny constant target means the main fit needs only one
        # smoothing level, but the batched CV still steps through every
        # delta level, refining the per-fold cache.
        torch.manual_seed(0)
        X = torch.randn(10, 2, dtype=torch.float64)
        y = torch.full((10,), 0.001, dtype=torch.float64)
        X = standardize(X)
        sig = sigest(X)
        Kmat = rbf_kernel(X, sig)
        torch.manual_seed(0)
        foldid = torch.randperm(10) % 2 + 1
        m = cvkqr(
            Kmat=Kmat,
            y=y,
            nlam=1,
            ulam=torch.tensor([1.0], dtype=torch.float64),
            tau=0.5,
            foldid=foldid,
            nfolds=2,
            eps=1e-3,
            maxit=500,
            gamma=1e-7,
            is_exact=0,
            delta_len=4,
            device="cpu",
        )
        m.fit()
        self.assertEqual(m.alpmat.shape, (11, 1))

    def test_cvkqr_per_fold_cv_refines_delta_with_tight_kkt(self):
        # When the per-fold KKT tolerance is essentially impossible to
        # satisfy, the per-fold CV outer loop must walk through every
        # smoothing level before giving up.
        torch.manual_seed(0)
        X = torch.randn(10, 2, dtype=torch.float64)
        y = torch.full((10,), 0.001, dtype=torch.float64)
        X = standardize(X)
        sig = sigest(X)
        Kmat = rbf_kernel(X, sig)
        torch.manual_seed(0)
        foldid = torch.randperm(10) % 2 + 1
        m = cvkqr(
            Kmat=Kmat,
            y=y,
            nlam=1,
            ulam=torch.tensor([1.0], dtype=torch.float64),
            tau=0.5,
            foldid=foldid,
            nfolds=2,
            eps=1e-3,
            maxit=500,
            gamma=1e-7,
            is_exact=1,
            mproj=1,
            delta_len=4,
            KKTeps=1.0,
            KKTeps2=1e-30,
            device="cpu",
        )
        m.fit()
        self.assertEqual(m.alpmat.shape, (11, 1))

    def test_cvkqr_is_exact_main_fit_runs_theta_update(self):
        # A skewed target with an asymmetric quantile level forces the
        # exact projection's elbow-residual theta update to fire — the
        # deepest part of the ``is_exact=1`` code path on the main-fit
        # side.
        torch.manual_seed(7)
        X = torch.randn(30, 4, dtype=torch.float64)
        y = X[:, 0] ** 2 + 0.5 * torch.randn(30, dtype=torch.float64).abs()
        X = standardize(X)
        sig = sigest(X)
        Kmat = rbf_kernel(X, sig)
        torch.manual_seed(7)
        foldid = torch.randperm(30) % 3 + 1
        m = cvkqr(
            Kmat=Kmat,
            y=y,
            nlam=1,
            ulam=torch.tensor([1.0], dtype=torch.float64),
            tau=0.75,
            foldid=foldid,
            nfolds=3,
            eps=1e-3,
            maxit=5000,
            gamma=1e-7,
            is_exact=1,
            mproj=2,
            delta_len=3,
            KKTeps=1.0,
            KKTeps2=1.0,
            device="cpu",
        )
        m.fit()
        self.assertEqual(m.alpmat.shape, (31, 1))


if __name__ == "__main__":
    unittest.main()
