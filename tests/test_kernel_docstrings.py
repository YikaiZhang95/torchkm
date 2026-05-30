# SPDX-License-Identifier: MIT
import inspect

import numpy as np

from torchkm.kernels import linear_kernel, polynomial_kernel, rbf_kernel


def test_public_kernel_docstrings_are_present():
    for func in (linear_kernel, polynomial_kernel, rbf_kernel):
        doc = inspect.getdoc(func)
        assert doc is not None
        assert "Parameters" in doc
        assert "Returns" in doc


def test_kernel_functions_smoke():
    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([0.5, -1.0, 2.0])

    assert np.isclose(linear_kernel(x1, x2), np.dot(x1, x2))
    assert np.isclose(
        polynomial_kernel(x1, x2, degree=2, coef0=1.0, gamma=0.5),
        (0.5 * np.dot(x1, x2) + 1.0) ** 2,
    )
    assert np.isclose(
        rbf_kernel(x1, x2, gamma=0.2),
        np.exp(-0.2 * np.linalg.norm(x1 - x2) ** 2),
    )
