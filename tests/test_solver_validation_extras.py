# SPDX-License-Identifier: MIT
"""Constructor validation for the solver classes not already covered by
``test_solver_validation.py`` (which handles ``cvksvm`` / ``cvkqr``).

Each test exercises a documented API contract — invalid label conventions
are rejected, non-tensor inputs raise ``TypeError``, etc.
"""

import unittest

import numpy as np
import torch

from torchkm.cvkdwd import cvkdwd
from torchkm.cvknysdwd import cvknysdwd
from torchkm.cvknyslogit import cvknyslogit
from torchkm.cvknyssvm import cvknyssvm
from torchkm.functions import rbf_kernel, sigest, standardize


def _binary_data(nn=40, pp=4, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(nn, pp, dtype=torch.float64)
    w = torch.randn(pp, dtype=torch.float64)
    y = torch.where(X @ w > 0, torch.tensor(1.0), torch.tensor(-1.0))
    return standardize(X), y


def _kernel_and_folds(X, nfolds=3, seed=0):
    sig = sigest(X)
    K = rbf_kernel(X, sig)
    torch.manual_seed(seed)
    foldid = torch.randperm(X.shape[0]) % nfolds + 1
    return K, foldid


class TestCvkdwdConstructorValidation(unittest.TestCase):
    """Input-validation contract for the dense DWD solver."""

    def setUp(self):
        self.X, self.y = _binary_data(nn=40)
        self.Kmat, self.foldid = _kernel_and_folds(self.X, nfolds=3)
        self.ulam = torch.logspace(0, -2, steps=3, dtype=torch.float64)

    def _kwargs(self, **overrides):
        base = dict(
            Kmat=self.Kmat,
            y=self.y,
            nlam=3,
            ulam=self.ulam,
            foldid=self.foldid,
            nfolds=3,
            device="cpu",
        )
        base.update(overrides)
        return base

    def test_non_tensor_kmat_raises(self):
        # Numpy kernel should be rejected before any solver setup runs.
        with self.assertRaisesRegex(TypeError, "Kmat must be a torch.Tensor"):
            cvkdwd(**self._kwargs(Kmat=self.Kmat.numpy()))

    def test_non_tensor_y_raises(self):
        with self.assertRaisesRegex(TypeError, "y must be a torch.Tensor"):
            cvkdwd(**self._kwargs(y=self.y.numpy()))

    def test_rejects_multiclass_labels(self):
        # A third label value should be flagged as multi-class.
        y_bad = self.y.clone()
        y_bad[:5] = 2.0
        with self.assertRaisesRegex(ValueError, "Multi-class"):
            cvkdwd(**self._kwargs(y=y_bad))

    def test_rejects_non_pm1_labels(self):
        # Labels outside {-1, +1} should raise a clear error.
        y_bad = self.y.clone()
        y_bad[self.y == -1] = 0.0
        with self.assertRaisesRegex(ValueError, "Invalid labels"):
            cvkdwd(**self._kwargs(y=y_bad))

    def test_non_tensor_ulam_raises(self):
        with self.assertRaisesRegex(TypeError, "ulam must be a torch.Tensor"):
            cvkdwd(**self._kwargs(ulam=self.ulam.numpy()))

    def test_non_tensor_foldid_raises(self):
        with self.assertRaisesRegex(TypeError, "foldid must be a torch.Tensor"):
            cvkdwd(**self._kwargs(foldid=self.foldid.numpy()))

    def test_non_square_kmat_raises(self):
        with self.assertRaisesRegex(ValueError, "square"):
            cvkdwd(**self._kwargs(Kmat=self.Kmat[:, :10]))

    def test_kmat_y_size_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            cvkdwd(**self._kwargs(y=self.y[:10]))

    def test_leave_one_out_default_foldid(self):
        # When ``nfolds == nobs`` the solver should assign one fold per sample.
        m = cvkdwd(**self._kwargs(foldid=None, nfolds=self.Kmat.shape[0]))
        self.assertEqual(m.foldid.numel(), self.Kmat.shape[0])

    def test_random_default_foldid(self):
        # Without an explicit foldid the solver generates a balanced random one.
        m = cvkdwd(**self._kwargs(foldid=None, nfolds=4))
        self.assertEqual(m.foldid.numel(), self.Kmat.shape[0])


class TestNystromConstructorValidation(unittest.TestCase):
    """Input-validation contract for the Nyström-based solvers."""

    def setUp(self):
        self.X, self.y = _binary_data(nn=40)
        self.ulam = torch.logspace(0, -2, steps=3, dtype=torch.float64)

    def _ny_kwargs(self, **overrides):
        base = dict(
            Xmat=self.X,
            X_test=self.X,
            y=self.y,
            nlam=3,
            ulam=self.ulam,
            foldid=None,
            nfolds=3,
            num_landmarks=10,
            k=5,
            device="cpu",
        )
        base.update(overrides)
        return base

    def test_cvknyssvm_non_tensor_xmat_raises(self):
        with self.assertRaisesRegex(TypeError, "Xmat must be a torch.Tensor"):
            cvknyssvm(**self._ny_kwargs(Xmat=self.X.numpy()))

    def test_cvknyssvm_non_tensor_y_raises(self):
        with self.assertRaisesRegex(TypeError, "y must be a torch.Tensor"):
            cvknyssvm(**self._ny_kwargs(y=self.y.numpy()))

    def test_cvknyssvm_rejects_multiclass(self):
        y_bad = self.y.clone()
        y_bad[:5] = 2.0
        with self.assertRaisesRegex(ValueError, "Multi-class"):
            cvknyssvm(**self._ny_kwargs(y=y_bad))

    def test_cvknyssvm_rejects_non_pm1_labels(self):
        y_bad = self.y.clone()
        y_bad[self.y == -1] = 0.0
        with self.assertRaisesRegex(ValueError, "Invalid labels"):
            cvknyssvm(**self._ny_kwargs(y=y_bad))

    def test_cvknyssvm_non_tensor_ulam_raises(self):
        with self.assertRaisesRegex(TypeError, "ulam must be a torch.Tensor"):
            cvknyssvm(**self._ny_kwargs(ulam=self.ulam.numpy()))

    def test_cvknyssvm_non_tensor_foldid_raises(self):
        with self.assertRaisesRegex(TypeError, "foldid must be a torch.Tensor"):
            cvknyssvm(**self._ny_kwargs(foldid=np.zeros(40)))

    def test_cvknyssvm_size_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            cvknyssvm(**self._ny_kwargs(y=self.y[:10]))

    def test_cvknyssvm_leave_one_out_default_foldid(self):
        X, y = _binary_data(nn=30)
        m = cvknyssvm(
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
        self.assertEqual(m.foldid.numel(), X.shape[0])

    def test_cvknyssvm_random_default_foldid(self):
        X, y = _binary_data(nn=30)
        m = cvknyssvm(
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
        self.assertEqual(m.foldid.numel(), X.shape[0])

    def test_cvknysdwd_non_tensor_xmat_raises(self):
        with self.assertRaisesRegex(TypeError, "Xmat must be a torch.Tensor"):
            cvknysdwd(
                Xmat=self.X.numpy(),
                X_test=self.X,
                y=self.y,
                nlam=3,
                ulam=self.ulam,
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )

    def test_cvknysdwd_rejects_multiclass(self):
        y_bad = self.y.clone()
        y_bad[:5] = 2.0
        with self.assertRaisesRegex(ValueError, "Multi-class"):
            cvknysdwd(
                Xmat=self.X,
                X_test=self.X,
                y=y_bad,
                nlam=3,
                ulam=self.ulam,
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )

    def test_cvknysdwd_rejects_non_pm1_labels(self):
        y_bad = self.y.clone()
        y_bad[self.y == -1] = 0.0
        with self.assertRaisesRegex(ValueError, "Invalid labels"):
            cvknysdwd(
                Xmat=self.X,
                X_test=self.X,
                y=y_bad,
                nlam=3,
                ulam=self.ulam,
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )

    def test_cvknyslogit_non_tensor_xmat_raises(self):
        with self.assertRaisesRegex(TypeError, "Xmat must be a torch.Tensor"):
            cvknyslogit(
                Xmat=self.X.numpy(),
                X_test=self.X,
                y=self.y,
                nlam=3,
                ulam=self.ulam,
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )

    def test_cvknyslogit_rejects_multiclass(self):
        y_bad = self.y.clone()
        y_bad[:5] = 2.0
        with self.assertRaisesRegex(ValueError, "Multi-class"):
            cvknyslogit(
                Xmat=self.X,
                X_test=self.X,
                y=y_bad,
                nlam=3,
                ulam=self.ulam,
                foldid=None,
                nfolds=3,
                num_landmarks=10,
                k=5,
                device="cpu",
            )

    def test_cvknyqr_default_device(self):
        # When ``device=None`` the solver should pick CUDA if available
        # and otherwise fall back to CPU.
        from torchkm.cvknyqr import cvknyqr

        torch.manual_seed(0)
        X = torch.randn(30, 4, dtype=torch.float64)
        y = X[:, 0] + 0.1 * torch.randn(30, dtype=torch.float64)
        X = standardize(X)
        m = cvknyqr(
            Xmat=X,
            y=y,
            ulam=torch.tensor([1.0, 0.1], dtype=torch.float64),
            nlam=2,
            tau=0.5,
            nfolds=3,
            num_landmarks=8,
            k=4,
            device=None,
        )
        self.assertIn(m.device.type, ("cpu", "cuda"))


class TestCvkqrConstructorDefaults(unittest.TestCase):
    """Default-argument behaviour for ``cvkqr`` constructor."""

    def test_default_device_resolves_to_cpu_or_cuda(self):
        # ``device=None`` should auto-pick CUDA where available and
        # otherwise fall back to CPU — the same convention as the other
        # solvers.
        from torchkm.cvkqr import cvkqr

        torch.manual_seed(0)
        X = torch.randn(30, 4, dtype=torch.float64)
        y = X[:, 0] + 0.1 * torch.randn(30, dtype=torch.float64)
        X = standardize(X)
        Kmat, foldid = _kernel_and_folds(X, nfolds=3)
        m = cvkqr(
            Kmat=Kmat,
            y=y,
            nlam=2,
            ulam=torch.tensor([1.0, 0.1], dtype=torch.float64),
            tau=0.5,
            foldid=foldid,
            nfolds=3,
        )
        self.assertIn(m.device.type, ("cpu", "cuda"))


if __name__ == "__main__":
    unittest.main()
