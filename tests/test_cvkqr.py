# SPDX-License-Identifier: MIT
import unittest
import torch
import numpy
from torchkm.cvkqr import cvkqr
from torchkm.functions import sigest, rbf_kernel, kernelMult, standardize


def _make_reg_data(nn=200, pp=5, sdn=42):
    torch.manual_seed(sdn)
    X = torch.randn(nn, pp, dtype=torch.float64)
    y = X[:, 0] + 0.3 * torch.randn(nn, dtype=torch.float64)
    X = standardize(X)
    return X, y


def _make_model(X, y, nlam=5, nfolds=3, tau=0.5, sdn=42, **kwargs):
    sig = sigest(X)
    Kmat = rbf_kernel(X, sig)
    ulam = torch.logspace(0, -3, steps=nlam, dtype=torch.float64)

    torch.manual_seed(sdn)
    foldid = torch.randperm(X.shape[0]) % nfolds + 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cvkqr(
        Kmat=Kmat,
        y=y,
        nlam=nlam,
        ulam=ulam,
        tau=tau,
        foldid=foldid,
        nfolds=nfolds,
        eps=1e-4,
        maxit=500,
        gamma=1e-7,
        is_exact=0,
        device=device,
        **kwargs,
    )
    return model, Kmat, sig, ulam


class TestCvkqr(unittest.TestCase):
    def test_fit_basic(self):
        nn = 200
        X, y = _make_reg_data(nn=nn)
        model, Kmat, sig, ulam = _make_model(X, y)
        model.fit()

        # alpmat shape: (n+1) x nlam
        self.assertEqual(model.alpmat.shape, (nn + 1, model.nlam))
        # pred populated by integrated CV — must not be all zeros
        self.assertFalse(
            torch.all(model.pred == 0).item(),
            "pred should be populated after fit()",
        )
        self.assertEqual(model.jerr, 0)

    def test_cv_selects_best_lambda(self):
        X, y = _make_reg_data(nn=200)
        model, Kmat, sig, ulam = _make_model(X, y, nlam=5, nfolds=3)
        model.fit()

        y_cpu = y.to("cpu")
        pred_cpu = model.pred.to("cpu")
        cv_loss = model.cv(pred_cpu, y_cpu)

        self.assertEqual(cv_loss.shape, (model.nlam,))
        # at least one lambda must yield a finite CV loss
        self.assertTrue(torch.any(torch.isfinite(cv_loss)).item())

        best_ind = int(numpy.nanargmin(cv_loss.numpy()))
        self.assertGreaterEqual(best_ind, 0)
        self.assertLess(best_ind, model.nlam)

    def test_predict(self):
        nn = 200
        X, y = _make_reg_data(nn=nn)
        X_test, y_test = _make_reg_data(nn=40, sdn=99)
        model, Kmat, sig, ulam = _make_model(X, y, nlam=5, nfolds=3)
        model.fit()

        y_cpu = y.to("cpu")
        cv_loss = model.cv(model.pred.to("cpu"), y_cpu)
        best_ind = int(numpy.nanargmin(cv_loss.numpy()))

        alpmat = model.alpmat.to("cpu")
        Kmat_new = kernelMult(X_test, X, sig).double()
        result = model.predict(Kmat_new, y_test, alpmat[:, best_ind])

        self.assertEqual(result.shape, (40,))
        self.assertTrue(torch.all(torch.isfinite(result)).item())

    def test_obj_value(self):
        X, y = _make_reg_data(nn=100)
        model, Kmat, sig, ulam = _make_model(X, y, nlam=3, nfolds=3)
        model.fit()

        alpmat = model.alpmat.to("cpu")
        obj = model.obj_value(alpmat[:, 0], ulam[0].item())
        self.assertTrue(torch.isfinite(obj).item())

    def test_check_loss(self):
        # static method: check-loss values for positives and negatives
        u = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        tau = 0.5
        loss = cvkqr.check_loss(u, tau)
        # u=-1: (tau-1)*(-1) = 0.5, u=0: 0.0, u=1: tau*1 = 0.5
        expected = torch.tensor([0.5, 0.0, 0.5], dtype=torch.float64)
        self.assertTrue(torch.allclose(loss, expected))

    def test_npass_cvnpass_populated(self):
        X, y = _make_reg_data(nn=100)
        model, Kmat, sig, ulam = _make_model(X, y, nlam=3, nfolds=3)
        model.fit()

        self.assertEqual(model.npass.shape, (model.nlam,))
        self.assertTrue(torch.any(model.npass > 0).item())
        self.assertEqual(model.cvnpass.shape, (model.nlam,))

    def test_foldid_respected(self):
        X, y = _make_reg_data(nn=120)
        sig = sigest(X)
        Kmat = rbf_kernel(X, sig)
        ulam = torch.logspace(0, -2, steps=3, dtype=torch.float64)
        nfolds = 4
        device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(7)
        foldid = torch.randperm(120) % nfolds + 1
        m1 = cvkqr(
            Kmat=Kmat,
            y=y,
            nlam=3,
            ulam=ulam,
            tau=0.5,
            foldid=foldid,
            nfolds=nfolds,
            eps=1e-4,
            maxit=300,
            gamma=1e-7,
            device=device,
        )
        m1.fit()

        torch.manual_seed(7)
        foldid2 = torch.randperm(120) % nfolds + 1
        m2 = cvkqr(
            Kmat=Kmat,
            y=y,
            nlam=3,
            ulam=ulam,
            tau=0.5,
            foldid=foldid2,
            nfolds=nfolds,
            eps=1e-4,
            maxit=300,
            gamma=1e-7,
            device=device,
        )
        m2.fit()

        # same foldid → identical pred
        self.assertTrue(torch.allclose(m1.pred, m2.pred))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_device_cuda(self):
        X, y = _make_reg_data(nn=100)
        model, Kmat, sig, ulam = _make_model(X, y, nlam=3, nfolds=3)
        model.fit()

        self.assertEqual(model.alpmat.device.type, "cuda")
        self.assertEqual(model.pred.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
