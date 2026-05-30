# SPDX-License-Identifier: MIT
def test_import_torchkmkqr():
    from torchkm.estimators import TorchKMKQR

    assert TorchKMKQR is not None


def test_import_cvknyqr_backend():
    from torchkm.cvknyqr import cvknyqr

    assert cvknyqr is not None
