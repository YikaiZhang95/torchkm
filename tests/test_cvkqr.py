import unittest
import torch
import os
import numpy
import time
from torchkm.cvkqr import cvkqr
from torchkm.functions import *

class Testcvkqr(unittest.TestCase):
    def test_fit_predict(self):
        sim1 = Data2(i = 315, num = 200, dim = 3)
        X_train = sim1['x']
        y_train = sim1['y']

        nlam = 5
        torch.manual_seed(52)
        ulam = torch.logspace(-3, 3, steps=nlam)

        sig = sigest(X_train)
        Kmat = rbf_kernel(X_train, sig)

        model1 = cvkqr(Kmat=Kmat, y=y_train, nlam=nlam, ulam=ulam, tau=0.5, eps=1e-2, maxit=1000000, gamma=1e-7, is_exact=0, device='cuda')
        model1.fit()

        cv = model1.cross_validate()

        best_ind = cv[1]

        Kmat = Kmat.double()
        alpmat = model1.alpmat.to('cpu')

        Kmat_new = kernelMult(X_train, X_train, sig)
        Kmat_new = Kmat_new.double()

        result = torch.mv(Kmat_new, alpmat[1:,best_ind]) + alpmat[0, best_ind]

        ypred = model1.predict(Kmat_new, y_train, alpmat[0:, best_ind])

        cvkqr.check_loss(result, 0.5)
        # # Check that alpha is set
        # self.assertIsNotNone(model.alpha, "Alpha should not be None after fit")

        # X_test = np.array([[0.5, 0.5]])
        # y_pred = model.predict(X_test)
        
        # # This is a naive test checking the shape of predictions
        # self.assertEqual(len(y_pred), 1, "Prediction shape mismatch")

        # # Optionally, check sign or magnitude
        # # For a real test, you'd compare against known correct labels
        # self.assertIn(y_pred[0], [-1, 1], "Predicted label should be -1 or 1")

if __name__ == '__main__':
    unittest.main()
