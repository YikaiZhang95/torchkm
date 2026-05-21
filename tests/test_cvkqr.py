import unittest
import torch
import numpy
from torchkm.cvkqr import cvkqr
from torchkm.functions import data_gen, sigest, rbf_kernel, kernelMult, standardize


class Testcvkqr(unittest.TestCase):
    def test_fit_predict(self):
        nn = 200
        nm = 5
        pp = 3
        p1 = p2 = pp // 2
        mu = 2.0
        ro = 3
        sdn = 315

        X_train, y_train, _ = data_gen(nn, nm, pp, p1, p2, mu, ro, sdn)
        X_train = standardize(X_train)

        nlam = 5
        torch.manual_seed(sdn)
        ulam = torch.logspace(-3, 3, steps=nlam)

        sig = sigest(X_train)
        Kmat = rbf_kernel(X_train, sig)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model1 = cvkqr(
            Kmat=Kmat,
            y=y_train,
            nlam=nlam,
            ulam=ulam,
            tau=0.5,
            eps=1e-2,
            maxit=1000000,
            gamma=1e-7,
            is_exact=0,
            device=device,
        )
        model1.fit()

        cv = model1.cross_validate()
        best_ind = cv[1]

        alpmat = model1.alpmat.to("cpu")

        Kmat_new = kernelMult(X_train, X_train, sig)
        Kmat_new = Kmat_new.double()

        result = torch.mv(Kmat_new, alpmat[1:, best_ind]) + alpmat[0, best_ind]
        ypred = model1.predict(Kmat_new, y_train, alpmat[0:, best_ind])
        cvkqr.check_loss(result, 0.5)


if __name__ == "__main__":
    unittest.main()
