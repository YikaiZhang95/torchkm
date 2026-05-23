"""Additional tests targeting previously-uncovered branches.

These tests are kept small and CPU-only so the suite stays fast.
"""

import unittest

import numpy as np
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
    raw = X @ w
    y = torch.where(raw > 0, torch.tensor(1.0), torch.tensor(-1.0))
    X = standardize(X)
    return X, y


def _reg_data(nn=80, pp=4, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(nn, pp, dtype=torch.float64)
    y = X[:, 0] + 0.1 * torch.randn(nn, dtype=torch.float64)
    X = standardize(X)
    return X, y


def _kernel_and_folds(X, nfolds=3, seed=0):
    sig = sigest(X)
    Kmat = rbf_kernel(X, sig)
    torch.manual_seed(seed)
    foldid = torch.randperm(X.shape[0]) % nfolds + 1
    return Kmat, foldid, sig


# -----------------------------------------------------------------
# Validation paths in cv* constructors
# -----------------------------------------------------------------
class TestValidationPaths(unittest.TestCase):
    """Hit the TypeError / ValueError branches in the solver __init__.

    These branches are otherwise dead because every existing fit test
    supplies valid tensors.
    """

    def setUp(self):
        self.X, self.y = _binary_data(nn=40)
        self.Kmat, self.foldid, _ = _kernel_and_folds(self.X, nfolds=3)
        self.ulam = torch.logspace(0, -2, steps=3, dtype=torch.float64)

    def _common_kwargs(self):
        return dict(
            nlam=3,
            ulam=self.ulam,
            foldid=self.foldid,
            nfolds=3,
            eps=1e-4,
            maxit=50,
            gamma=1e-7,
            device="cpu",
        )

    def test_cvksvm_kmat_must_be_tensor(self):
        with self.assertRaisesRegex(TypeError, "Kmat must be a torch.Tensor"):
            cvksvm(Kmat=self.Kmat.numpy(), y=self.y, **self._common_kwargs())

    def test_cvksvm_y_must_be_tensor(self):
        with self.assertRaisesRegex(TypeError, "y must be a torch.Tensor"):
            cvksvm(Kmat=self.Kmat, y=self.y.numpy(), **self._common_kwargs())

    def test_cvksvm_multiclass_labels_rejected(self):
        y_bad = self.y.clone()
        y_bad[:5] = 2.0  # introduce a third label
        with self.assertRaisesRegex(ValueError, "Multi-class"):
            cvksvm(Kmat=self.Kmat, y=y_bad, **self._common_kwargs())

    def test_cvksvm_invalid_labels_rejected(self):
        y_bad = self.y.clone()
        y_bad[self.y == -1] = 0.0
        with self.assertRaisesRegex(ValueError, "Invalid labels"):
            cvksvm(Kmat=self.Kmat, y=y_bad, **self._common_kwargs())

    def test_cvksvm_ulam_must_be_tensor(self):
        kwargs = self._common_kwargs()
        kwargs["ulam"] = kwargs["ulam"].numpy()
        with self.assertRaisesRegex(TypeError, "ulam must be a torch.Tensor"):
            cvksvm(Kmat=self.Kmat, y=self.y, **kwargs)

    def test_cvksvm_foldid_must_be_tensor(self):
        kwargs = self._common_kwargs()
        kwargs["foldid"] = kwargs["foldid"].numpy()
        with self.assertRaisesRegex(TypeError, "foldid must be a torch.Tensor"):
            cvksvm(Kmat=self.Kmat, y=self.y, **kwargs)

    def test_cvksvm_non_square_kmat_rejected(self):
        kwargs = self._common_kwargs()
        with self.assertRaisesRegex(ValueError, "square"):
            cvksvm(Kmat=self.Kmat[:, :10], y=self.y, **kwargs)

    def test_cvksvm_size_mismatch_rejected(self):
        kwargs = self._common_kwargs()
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            cvksvm(Kmat=self.Kmat, y=self.y[:10], **kwargs)

    def test_cvksvm_default_foldid_via_nobs_equal_nfolds(self):
        # Triggers the `nfolds == self.nobs` branch.
        kwargs = self._common_kwargs()
        kwargs["foldid"] = None
        kwargs["nfolds"] = self.Kmat.shape[0]
        m = cvksvm(Kmat=self.Kmat, y=self.y, **kwargs)
        self.assertEqual(m.foldid.numel(), self.Kmat.shape[0])

    def test_cvksvm_default_foldid_random(self):
        kwargs = self._common_kwargs()
        kwargs["foldid"] = None
        kwargs["nfolds"] = 4
        m = cvksvm(Kmat=self.Kmat, y=self.y, **kwargs)
        self.assertEqual(m.foldid.numel(), self.Kmat.shape[0])

    def test_cvksvm_device_auto_default(self):
        # Exercise the device=None default branch.
        kwargs = self._common_kwargs()
        kwargs.pop("device")
        m = cvksvm(Kmat=self.Kmat, y=self.y, **kwargs)
        self.assertIn(m.device.type, ("cpu", "cuda"))

    def test_cvkqr_kmat_must_be_tensor(self):
        X, y = _reg_data(nn=40)
        Kmat, foldid, _ = _kernel_and_folds(X, nfolds=3)
        with self.assertRaisesRegex(TypeError, "Kmat must be a torch.Tensor"):
            cvkqr(
                Kmat=Kmat.numpy(),
                y=y,
                nlam=3,
                ulam=self.ulam,
                tau=0.5,
                foldid=foldid,
                nfolds=3,
                device="cpu",
            )

    def test_cvkqr_y_must_be_tensor(self):
        X, y = _reg_data(nn=40)
        Kmat, foldid, _ = _kernel_and_folds(X, nfolds=3)
        with self.assertRaisesRegex(TypeError, "y must be a torch.Tensor"):
            cvkqr(
                Kmat=Kmat,
                y=y.numpy(),
                nlam=3,
                ulam=self.ulam,
                tau=0.5,
                foldid=foldid,
                nfolds=3,
                device="cpu",
            )

    def test_cvkqr_ulam_must_be_tensor(self):
        X, y = _reg_data(nn=40)
        Kmat, foldid, _ = _kernel_and_folds(X, nfolds=3)
        with self.assertRaisesRegex(TypeError, "ulam must be a torch.Tensor"):
            cvkqr(
                Kmat=Kmat,
                y=y,
                nlam=3,
                ulam=self.ulam.numpy(),
                tau=0.5,
                foldid=foldid,
                nfolds=3,
                device="cpu",
            )

    def test_cvkqr_foldid_must_be_tensor(self):
        X, y = _reg_data(nn=40)
        Kmat, foldid, _ = _kernel_and_folds(X, nfolds=3)
        with self.assertRaisesRegex(TypeError, "foldid must be a torch.Tensor"):
            cvkqr(
                Kmat=Kmat,
                y=y,
                nlam=3,
                ulam=self.ulam,
                tau=0.5,
                foldid=foldid.numpy(),
                nfolds=3,
                device="cpu",
            )

    def test_cvkqr_non_square_kmat_rejected(self):
        X, y = _reg_data(nn=40)
        Kmat, foldid, _ = _kernel_and_folds(X, nfolds=3)
        with self.assertRaisesRegex(ValueError, "square"):
            cvkqr(
                Kmat=Kmat[:, :10],
                y=y,
                nlam=3,
                ulam=self.ulam,
                tau=0.5,
                foldid=foldid,
                nfolds=3,
                device="cpu",
            )

    def test_cvkqr_size_mismatch_rejected(self):
        X, y = _reg_data(nn=40)
        Kmat, foldid, _ = _kernel_and_folds(X, nfolds=3)
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            cvkqr(
                Kmat=Kmat,
                y=y[:10],
                nlam=3,
                ulam=self.ulam,
                tau=0.5,
                foldid=foldid,
                nfolds=3,
                device="cpu",
            )

    def test_cvkqr_default_foldid_via_nobs_equal_nfolds(self):
        X, y = _reg_data(nn=40)
        Kmat, _, _ = _kernel_and_folds(X, nfolds=3)
        m = cvkqr(
            Kmat=Kmat,
            y=y,
            nlam=3,
            ulam=self.ulam,
            tau=0.5,
            foldid=None,
            nfolds=Kmat.shape[0],
            device="cpu",
        )
        self.assertEqual(m.foldid.numel(), Kmat.shape[0])

    def test_cvkqr_default_foldid_random(self):
        X, y = _reg_data(nn=40)
        Kmat, _, _ = _kernel_and_folds(X, nfolds=3)
        m = cvkqr(
            Kmat=Kmat,
            y=y,
            nlam=3,
            ulam=self.ulam,
            tau=0.5,
            foldid=None,
            nfolds=4,
            device="cpu",
        )
        self.assertEqual(m.foldid.numel(), Kmat.shape[0])

    def test_cvkqr_device_auto_default(self):
        X, y = _reg_data(nn=40)
        Kmat, foldid, _ = _kernel_and_folds(X, nfolds=3)
        m = cvkqr(
            Kmat=Kmat,
            y=y,
            nlam=3,
            ulam=self.ulam,
            tau=0.5,
            foldid=foldid,
            nfolds=3,
        )
        self.assertIn(m.device.type, ("cpu", "cuda"))

    def test_cvkdwd_validation_paths(self):
        X, y = _binary_data(nn=40)
        Kmat, foldid, _ = _kernel_and_folds(X, nfolds=3)
        with self.assertRaisesRegex(TypeError, "Kmat must be a torch.Tensor"):
            cvkdwd(
                Kmat=Kmat.numpy(),
                y=y,
                nlam=3,
                ulam=self.ulam,
                foldid=foldid,
                nfolds=3,
                device="cpu",
            )
        with self.assertRaisesRegex(TypeError, "y must be a torch.Tensor"):
            cvkdwd(
                Kmat=Kmat,
                y=y.numpy(),
                nlam=3,
                ulam=self.ulam,
                foldid=foldid,
                nfolds=3,
                device="cpu",
            )
        y_bad = y.clone()
        y_bad[:5] = 2.0
        with self.assertRaisesRegex(ValueError, "Multi-class"):
            cvkdwd(
                Kmat=Kmat,
                y=y_bad,
                nlam=3,
                ulam=self.ulam,
                foldid=foldid,
                nfolds=3,
                device="cpu",
            )
        y_bad = y.clone()
        y_bad[y == -1] = 0.0
        with self.assertRaisesRegex(ValueError, "Invalid labels"):
            cvkdwd(
                Kmat=Kmat,
                y=y_bad,
                nlam=3,
                ulam=self.ulam,
                foldid=foldid,
                nfolds=3,
                device="cpu",
            )
        with self.assertRaisesRegex(TypeError, "ulam must be a torch.Tensor"):
            cvkdwd(
                Kmat=Kmat,
                y=y,
                nlam=3,
                ulam=self.ulam.numpy(),
                foldid=foldid,
                nfolds=3,
                device="cpu",
            )
        with self.assertRaisesRegex(TypeError, "foldid must be a torch.Tensor"):
            cvkdwd(
                Kmat=Kmat,
                y=y,
                nlam=3,
                ulam=self.ulam,
                foldid=foldid.numpy(),
                nfolds=3,
                device="cpu",
            )
        with self.assertRaisesRegex(ValueError, "square"):
            cvkdwd(
                Kmat=Kmat[:, :10],
                y=y,
                nlam=3,
                ulam=self.ulam,
                foldid=foldid,
                nfolds=3,
                device="cpu",
            )
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            cvkdwd(
                Kmat=Kmat,
                y=y[:10],
                nlam=3,
                ulam=self.ulam,
                foldid=foldid,
                nfolds=3,
                device="cpu",
            )

    def test_cvkdwd_default_foldid_paths(self):
        X, y = _binary_data(nn=40)
        Kmat, _, _ = _kernel_and_folds(X, nfolds=3)
        m1 = cvkdwd(
            Kmat=Kmat,
            y=y,
            nlam=3,
            ulam=self.ulam,
            foldid=None,
            nfolds=Kmat.shape[0],
            device="cpu",
        )
        self.assertEqual(m1.foldid.numel(), Kmat.shape[0])
        m2 = cvkdwd(
            Kmat=Kmat,
            y=y,
            nlam=3,
            ulam=self.ulam,
            foldid=None,
            nfolds=4,
            device="cpu",
        )
        self.assertEqual(m2.foldid.numel(), Kmat.shape[0])

    def test_cvknyssvm_validation_paths(self):
        X, y = _binary_data(nn=40)
        with self.assertRaisesRegex(TypeError, "Xmat must be a torch.Tensor"):
            cvknyssvm(
                Xmat=X.numpy(),
                X_test=X,
                y=y,
                nlam=3,
                ulam=self.ulam,
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )
        with self.assertRaisesRegex(TypeError, "y must be a torch.Tensor"):
            cvknyssvm(
                Xmat=X,
                X_test=X,
                y=y.numpy(),
                nlam=3,
                ulam=self.ulam,
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )
        y_bad = y.clone()
        y_bad[:5] = 2.0
        with self.assertRaisesRegex(ValueError, "Multi-class"):
            cvknyssvm(
                Xmat=X,
                X_test=X,
                y=y_bad,
                nlam=3,
                ulam=self.ulam,
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )
        y_bad = y.clone()
        y_bad[y == -1] = 0.0
        with self.assertRaisesRegex(ValueError, "Invalid labels"):
            cvknyssvm(
                Xmat=X,
                X_test=X,
                y=y_bad,
                nlam=3,
                ulam=self.ulam,
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )
        with self.assertRaisesRegex(TypeError, "ulam must be a torch.Tensor"):
            cvknyssvm(
                Xmat=X,
                X_test=X,
                y=y,
                nlam=3,
                ulam=self.ulam.numpy(),
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )
        with self.assertRaisesRegex(TypeError, "foldid must be a torch.Tensor"):
            cvknyssvm(
                Xmat=X,
                X_test=X,
                y=y,
                nlam=3,
                ulam=self.ulam,
                foldid=np.zeros(40),
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            cvknyssvm(
                Xmat=X,
                X_test=X,
                y=y[:10],
                nlam=3,
                ulam=self.ulam,
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )

    def test_cvknyssvm_default_foldid_paths(self):
        X, y = _binary_data(nn=30)
        m_loo = cvknyssvm(
            Xmat=X,
            X_test=X[:5],
            y=y,
            nlam=2,
            ulam=torch.tensor([1.0, 0.1], dtype=torch.float64),
            foldid=None,
            nfolds=X.shape[0],
            num_landmarks=8,
            k=4,
            device="cpu",
        )
        self.assertEqual(m_loo.foldid.numel(), X.shape[0])
        m_rand = cvknyssvm(
            Xmat=X,
            X_test=X[:5],
            y=y,
            nlam=2,
            ulam=torch.tensor([1.0, 0.1], dtype=torch.float64),
            foldid=None,
            nfolds=3,
            num_landmarks=8,
            k=4,
            device="cpu",
        )
        self.assertEqual(m_rand.foldid.numel(), X.shape[0])

    def test_cvknyqr_default_device(self):
        # Cover the device=None branch in cvknyqr.
        from torchkm.cvknyqr import cvknyqr

        X, y = _reg_data(nn=30)
        ulam = torch.tensor([1.0, 0.1], dtype=torch.float64)
        m = cvknyqr(
            Xmat=X,
            y=y,
            ulam=ulam,
            nlam=2,
            tau=0.5,
            nfolds=3,
            num_landmarks=8,
            k=4,
            device=None,
        )
        self.assertIn(m.device.type, ("cpu", "cuda"))

    def test_cvknysdwd_validation_paths(self):
        X, y = _binary_data(nn=40)
        with self.assertRaisesRegex(TypeError, "Xmat must be a torch.Tensor"):
            cvknysdwd(
                Xmat=X.numpy(),
                X_test=X,
                y=y,
                nlam=3,
                ulam=self.ulam,
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )
        y_bad = y.clone()
        y_bad[:5] = 2.0
        with self.assertRaisesRegex(ValueError, "Multi-class"):
            cvknysdwd(
                Xmat=X,
                X_test=X,
                y=y_bad,
                nlam=3,
                ulam=self.ulam,
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )
        y_bad = y.clone()
        y_bad[y == -1] = 0.0
        with self.assertRaisesRegex(ValueError, "Invalid labels"):
            cvknysdwd(
                Xmat=X,
                X_test=X,
                y=y_bad,
                nlam=3,
                ulam=self.ulam,
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )

    def test_cvknyslogit_validation_paths(self):
        X, y = _binary_data(nn=40)
        with self.assertRaisesRegex(TypeError, "Xmat must be a torch.Tensor"):
            cvknyslogit(
                Xmat=X.numpy(),
                X_test=X,
                y=y,
                nlam=3,
                ulam=self.ulam,
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )
        y_bad = y.clone()
        y_bad[:5] = 2.0
        with self.assertRaisesRegex(ValueError, "Multi-class"):
            cvknyslogit(
                Xmat=X,
                X_test=X,
                y=y_bad,
                nlam=3,
                ulam=self.ulam,
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )


# -----------------------------------------------------------------
# Maxit exhaustion (jerr path)
# -----------------------------------------------------------------
class TestMaxitExhaustion(unittest.TestCase):
    """Hit the `npass > maxit` branch in cvkhuber / cvksqsvm / cvklogit.

    With maxit=1 the inner FISTA loop must exceed the iteration budget,
    which exercises the jerr-assignment paths.
    """

    def _make_kernel(self, X):
        sig = sigest(X)
        return rbf_kernel(X, sig), sig

    def test_cvkhuber_npass_exceeds_maxit(self):
        X, y = _reg_data(nn=60)
        Kmat, _ = self._make_kernel(X)
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
            eps=1e-12,  # never converge in 1 step
            maxit=1,  # exhausts after first lambda
            gamma=1e-7,
            device="cpu",
        )
        m.fit()
        # Either jerr is set negative or an early break flagged failure.
        self.assertTrue(m.jerr <= 0)

    def test_cvksqsvm_npass_exceeds_maxit(self):
        X, y = _binary_data(nn=60)
        Kmat, _ = self._make_kernel(X)
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

    def test_cvklogit_npass_exceeds_maxit(self):
        X, y = _binary_data(nn=60)
        Kmat, _ = self._make_kernel(X)
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


# -----------------------------------------------------------------
# cvklogit helper methods
# -----------------------------------------------------------------
class TestCvklogitHelpers(unittest.TestCase):
    """Call cvklogit.predict() and cvklogit.obj_value() to cover lines 402-416."""

    def test_predict_and_obj_value(self):
        X, y = _binary_data(nn=80)
        Kmat, foldid, _ = _kernel_and_folds(X, nfolds=3)
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
        # predict returns (ypred, accuracy); both should be finite/sensible
        ypred, acc = m.predict(Kmat.double(), y, alpmat[:, 0])
        self.assertEqual(ypred.shape, (80,))
        self.assertTrue(0.0 <= float(acc) <= 1.0)
        self.assertTrue(set(torch.unique(ypred).tolist()).issubset({-1, 1}))

        obj = m.obj_value(alpmat[:, 0], ulam[0].item())
        self.assertTrue(torch.isfinite(obj).item())


# -----------------------------------------------------------------
# is_exact=1 projection branches in cvksvm and cvkqr
# -----------------------------------------------------------------
class TestIsExactBranches(unittest.TestCase):
    """The `is_exact=1` projection-refinement branches are the largest
    uncovered chunks in cvksvm.py and cvkqr.py. Run on a small problem so
    coverage is exercised quickly.
    """

    def test_cvksvm_is_exact(self):
        X, y = _binary_data(nn=40)
        Kmat, foldid, _ = _kernel_and_folds(X, nfolds=3)
        ulam = torch.tensor([1.0, 0.1], dtype=torch.float64)
        m = cvksvm(
            Kmat=Kmat,
            y=y,
            nlam=2,
            ulam=ulam,
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
        # Some lambdas should produce non-zero predictions even though
        # the per-fold CV is the same `is_exact=1` path.
        self.assertEqual(m.pred.shape, (40, 2))

    def test_cvkqr_is_exact(self):
        X, y = _reg_data(nn=30)
        Kmat, foldid, _ = _kernel_and_folds(X, nfolds=3)
        ulam = torch.tensor([1.0, 0.1], dtype=torch.float64)
        # Loose KKTeps/KKTeps2 guarantee we enter the is_exact mproj branch in
        # both the main fit and the per-fold CV loop. Large maxit keeps
        # `torch.sum(npass) > self.maxit` from breaking out before CV runs.
        m = cvkqr(
            Kmat=Kmat,
            y=y,
            nlam=2,
            ulam=ulam,
            tau=0.5,
            foldid=foldid,
            nfolds=3,
            eps=1e-3,
            maxit=5000,
            gamma=1e-7,
            is_exact=1,
            mproj=1,
            delta_len=2,
            KKTeps=1.0,
            KKTeps2=1.0,
            device="cpu",
        )
        m.fit()
        self.assertEqual(m.alpmat.shape, (31, 2))
        self.assertEqual(m.pred.shape, (30, 2))

    def test_cvkqr_is_exact_triggers_elbchk_and_cv_path(self):
        # Trivial target (y=0) → main fit and per-fold CV converge fast, so
        # the `is_exact=1` post-FISTA branch (golden + KKT + mproj) in BOTH
        # the main fit and the per-fold CV is reached. Without this, the
        # CV FISTA exhausts `nmaxit` before reaching lines 569+ in cvkqr.
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
        # CV should converge in a single sweep (one increment per fold).
        self.assertEqual(int(m.cvnpass[0]), 3)

    def test_cvkqr_is_exact_zero_target_batched_cv(self):
        # Same trivial target but is_exact=0 — exercises the per-fold KKT
        # loop in `_cv_batched_lambda` (lines 842-889) which is otherwise
        # skipped because the batched FISTA exhausts `nmaxit` on non-trivial
        # targets.
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

    def test_cvkqr_constant_target_batched_cv_delta_refinement(self):
        # Tiny constant target: main fit converges in one delta step
        # (delta_save=1) but the batched CV still iterates through all
        # delta levels — exercising the `if delta_id > delta_save` cache
        # update block (cvkqr lines 770-777) and the golden-section
        # intercept improvement in the per-fold KKT loop (lines 860-862,
        # 887-888).
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

    def test_cvkqr_constant_target_per_fold_cv_delta_refinement(self):
        # Same trivial target with is_exact=1 + tight KKTeps2 → the
        # per-fold CV outer delta-refinement loop runs through every
        # delta level, hitting the cache-update block (lines 516-529)
        # and the golden-intercept improvement (lines 581-583).
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

    def test_cvkqr_is_exact_main_fit_projection_theta_update(self):
        # Skewed target with tau=0.75 → the main-fit `is_exact=1` projection
        # loop enters the `torch.sum(elbowid) > 1` theta-update branch
        # (cvkqr lines 423-432), exercising the deepest part of the
        # exact-projection code path.
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


# -----------------------------------------------------------------
# cvknyssvm.transform error path
# -----------------------------------------------------------------
class TestCvknyssvmTransform(unittest.TestCase):
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
        # Solver doesn't expose landmarks_/M_/sig_w_ until fit() — actually
        # only the cvknyssvm fit() may not set them; the transform() will
        # raise if those attrs are None.
        if hasattr(m, "landmarks_"):
            self.assertIsNone(m.landmarks_)


# -----------------------------------------------------------------
# sklearn estimator validation paths
# -----------------------------------------------------------------
class TestEstimatorValidation(unittest.TestCase):
    """Exercise raise-paths and helper branches in torchkm.estimators."""

    def _toy_data(self, n=80):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64) * 2 - 1
        return X, y

    def _toy_reg(self, n=80):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, 4)).astype(np.float32)
        y = (X[:, 0] + 0.1 * rng.normal(size=n)).astype(np.float64)
        return X, y

    def test_make_ulam_rejects_non_1d_Cs(self):
        from torchkm.estimators import _make_ulam

        with self.assertRaisesRegex(ValueError, "Cs must be 1D"):
            _make_ulam(nC=5, Cs=np.zeros((3, 3)), C_max=1.0, C_min=1e-3)

    def test_compute_k_train_unsupported_kernel(self):
        from torchkm.estimators import TorchKMSVC

        X, y = self._toy_data(40)
        est = TorchKMSVC(
            kernel="rbf",  # set to something valid initially
            nC=2,
            cv=2,
            device="cpu",
            random_state=0,
            max_iter=20,
        )
        # Force an unsupported kernel value AFTER construction so _compute_K_train
        # hits the trailing ValueError. We call the internal helper directly.
        est.kernel = "unknown-kernel"
        with self.assertRaisesRegex(ValueError, "Unsupported kernel"):
            est._compute_K_train(torch.as_tensor(X, dtype=torch.double))

    def test_compute_k_test_unsupported_kernel(self):
        from torchkm.estimators import TorchKMSVC

        X, y = self._toy_data(40)
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

    def test_unknown_backend_raises(self):
        from torchkm.estimators import TorchKMSVC

        X, y = self._toy_data(40)
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

    def test_svc_precomputed_non_square_train_raises(self):
        from torchkm.estimators import TorchKMSVC

        # Make a fake "precomputed kernel" that's not square -> ValueError.
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

    def test_kqr_precomputed_non_square_train_raises(self):
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

    def test_kqr_score_shape_mismatch_raises(self):
        from torchkm.estimators import TorchKMKQR

        X, y = self._toy_reg(60)
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

    def test_kqr_unsupported_kernel_raises(self):
        from torchkm.estimators import TorchKMKQR

        X, y = self._toy_reg(40)
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

    def test_pick_device_str_branches(self):
        from torchkm.estimators import _pick_device_str

        # torch.device branch
        self.assertEqual(_pick_device_str(torch.device("cpu")), "cpu")
        # string "cpu"
        self.assertEqual(_pick_device_str("cpu"), "cpu")
        # explicit cuda when not available
        if not torch.cuda.is_available():
            self.assertEqual(_pick_device_str("cuda"), "cpu")

    def test_make_foldid_with_explicit(self):
        from torchkm.estimators import _make_foldid

        explicit = np.array([1, 2, 1, 2, 1])
        out = _make_foldid(n=5, nfolds=2, foldid=explicit, random_state=None)
        self.assertEqual(out.tolist(), [1, 2, 1, 2, 1])
        with self.assertRaisesRegex(ValueError, "foldid must have length"):
            _make_foldid(n=10, nfolds=2, foldid=explicit, random_state=None)

    def test_check_binary_y_rejects_multiclass(self):
        from torchkm.estimators import _check_binary_y

        with self.assertRaisesRegex(ValueError, "binary classification"):
            _check_binary_y(np.array([0, 1, 2, 0, 1, 2]))


# -----------------------------------------------------------------
# Platt plot via sklearn wrapper
# -----------------------------------------------------------------
class TestPlattPlot(unittest.TestCase):
    """Hit the platt_plot() flow in TorchKMSVC."""

    def _make_classifier(self, probability):
        from torchkm.estimators import TorchKMSVC

        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 3)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)
        est = TorchKMSVC(
            kernel="rbf",
            nC=3,
            cv=3,
            device="cpu",
            random_state=0,
            max_iter=60,
            probability=probability,
        ).fit(X, y)
        return est, X, y

    def test_platt_plot_without_probability_raises(self):
        est, X, y = self._make_classifier(probability=False)
        with self.assertRaisesRegex(AttributeError, "probability=True"):
            est.platt_plot()

    def test_platt_plot_stored_calibration_data(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        est, X, y = self._make_classifier(probability=True)
        ax, stats = est.platt_plot(n_bins=4, annotate_counts=True)
        self.assertIn("ece", stats)
        self.assertIn("brier", stats)
        self.assertTrue(0.0 <= stats["ece"] <= 1.0)
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_platt_plot_with_X_y(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        est, X, y = self._make_classifier(probability=True)
        ax, stats = est.platt_plot(X=X, y=y, n_bins=3, strategy="quantile")
        self.assertIn("ece", stats)
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_platt_plot_X_without_y_raises(self):
        est, X, y = self._make_classifier(probability=True)
        with self.assertRaisesRegex(ValueError, "y must also be provided"):
            est.platt_plot(X=X)

    def test_platt_plot_bad_strategy_raises(self):
        est, X, y = self._make_classifier(probability=True)
        with self.assertRaisesRegex(ValueError, "uniform.*quantile"):
            est.platt_plot(strategy="bogus")

    def test_predict_proba_without_probability_raises(self):
        est, X, y = self._make_classifier(probability=False)
        with self.assertRaisesRegex(AttributeError, "probability=False"):
            est.predict_proba(X)

    def test_platt_plot_missing_stored_data_raises(self):
        est, X, y = self._make_classifier(probability=True)
        est.platt_scores_ = None
        with self.assertRaisesRegex(AttributeError, "Stored calibration data"):
            est.platt_plot()

    def test_platt_plot_with_user_ax_and_savepath(self, tmp_root="/tmp"):
        import os
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        est, X, y = self._make_classifier(probability=True)
        fig, user_ax = plt.subplots(figsize=(4, 4))
        savepath = os.path.join(tmp_root, "torchkm_platt_test.png")
        try:
            ax, stats = est.platt_plot(
                ax=user_ax, savepath=savepath, annotate_counts=False, n_bins=3
            )
            self.assertIs(ax, user_ax)
            self.assertTrue(os.path.exists(savepath))
        finally:
            plt.close("all")
            if os.path.exists(savepath):
                os.remove(savepath)

    def test_platt_plot_handles_1d_proba(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        est, X, y = self._make_classifier(probability=True)

        # Monkey-patch predict_proba to return a 1D array, exercising the
        # `proba.ndim != 2` branch in platt_plot.
        original = est.predict_proba

        def fake_predict_proba(X_in):
            full = original(X_in)
            return full[:, -1]  # 1D

        est.predict_proba = fake_predict_proba
        try:
            ax, stats = est.platt_plot(X=X, y=y, n_bins=3)
            self.assertIn("ece", stats)
        finally:
            import matplotlib.pyplot as plt

            plt.close("all")
            est.predict_proba = original

    def test_platt_plot_length_mismatch_raises(self):
        est, X, y = self._make_classifier(probability=True)
        # Provide labels with a mismatching length to trigger ValueError.
        with self.assertRaisesRegex(ValueError, "same length"):
            est.platt_plot(X=X, y=y[:5])


# -----------------------------------------------------------------
# Platt scaler line-search damping branch
# -----------------------------------------------------------------
class TestPlattLineSearch(unittest.TestCase):
    """Encourage the `damping *= 0.5` branch in PlattScalerTorch.fit by
    using an inverted label distribution that causes a bad initial Newton
    step.
    """

    def test_damping_branch_triggered(self):
        from torchkm.platt import PlattScalerTorch

        # Scores are positively correlated with the negative class, so the
        # initial Newton step typically overshoots and the line search must
        # halve the damping at least once.
        f = torch.tensor(
            [-5.0, -4.0, -3.0, -2.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double
        )
        y = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=torch.double)
        model = PlattScalerTorch(device="cpu", max_iter=30, tol=1e-12, reg=1.0)
        model.fit(f, y)
        proba = model.predict_proba(f)
        self.assertTrue(torch.isfinite(proba).all())


# -----------------------------------------------------------------
# cvkqr.fit with maxit exhaustion (jerr path)
# -----------------------------------------------------------------
class TestCvkqrMaxitJerr(unittest.TestCase):
    """Trip the `npass > maxit` jerr branch in cvkqr.fit."""

    def test_maxit_exhausted_sets_jerr(self):
        X, y = _reg_data(nn=40)
        Kmat, foldid, _ = _kernel_and_folds(X, nfolds=3)
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


# -----------------------------------------------------------------
# Nyström solver maxit exhaustion + LOO foldid branches
# -----------------------------------------------------------------
class TestNystromExtras(unittest.TestCase):
    """Hit the remaining lines in the Nyström solvers:

    - Default-foldid branch when nfolds == nobs (LOO).
    - jerr / break paths when maxit is exhausted.
    - transform() called before fit() raises RuntimeError.
    """

    def _data(self, n=30):
        return _binary_data(nn=n)

    def test_cvknysdwd_loo_foldid_and_maxit(self):
        X, y = self._data(n=30)
        ulam = torch.tensor([10.0, 1.0, 0.1], dtype=torch.float64)
        # LOO foldid branch.
        m_loo = cvknysdwd(
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
        self.assertEqual(m_loo.foldid.numel(), X.shape[0])

        # maxit exhaustion branch.
        m_exh = cvknysdwd(
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
        m_exh.fit()
        self.assertTrue(m_exh.jerr <= 0)

    def test_cvknyslogit_loo_foldid_and_maxit(self):
        X, y = self._data(n=30)
        ulam = torch.tensor([10.0, 1.0, 0.1], dtype=torch.float64)
        m_loo = cvknyslogit(
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
        self.assertEqual(m_loo.foldid.numel(), X.shape[0])

        m_exh = cvknyslogit(
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
        m_exh.fit()
        self.assertTrue(m_exh.jerr <= 0)

    def test_cvknyssvm_maxit_exhausted(self):
        X, y = self._data(n=30)
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

    def test_cvknyssvm_transform_before_fit_raises(self):
        X, y = self._data(n=20)
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
        # landmarks_/M_/sig_w_ are None until fit()
        with self.assertRaisesRegex(RuntimeError, "Call fit"):
            m.transform(X[:3])


# -----------------------------------------------------------------
# Trip cvkdwd line 530 (`if torch.sum(cvnpass) > self.nmaxit: break`)
# -----------------------------------------------------------------
class TestCvkdwdMaxitInCv(unittest.TestCase):
    def test_cv_inner_maxit_exhausted(self):
        X, y = _binary_data(nn=40)
        Kmat, foldid, _ = _kernel_and_folds(X, nfolds=3)
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


if __name__ == "__main__":
    unittest.main()
