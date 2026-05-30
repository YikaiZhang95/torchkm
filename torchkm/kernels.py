# SPDX-License-Identifier: MIT
import numpy as np


def linear_kernel(x1, x2, **kwargs):
    """Compute the linear kernel between two input vectors.

    Parameters
    ----------
    x1 : numpy.ndarray
        First input vector.
    x2 : numpy.ndarray
        Second input vector.
    **kwargs
        Ignored keyword arguments accepted for compatibility with other kernel
        functions.

    Returns
    -------
    float or numpy.ndarray
        The dot product ``np.dot(x1, x2)``.
    """
    return np.dot(x1, x2)


def polynomial_kernel(x1, x2, degree=3, coef0=1, gamma=1, **kwargs):
    """Compute the polynomial kernel between two input vectors.

    The kernel is defined as ``(gamma * np.dot(x1, x2) + coef0) ** degree``.

    Parameters
    ----------
    x1 : numpy.ndarray
        First input vector.
    x2 : numpy.ndarray
        Second input vector.
    degree : int, default=3
        Degree of the polynomial kernel.
    coef0 : float, default=1
        Additive constant in the polynomial kernel.
    gamma : float, default=1
        Multiplicative scale applied to the dot product.
    **kwargs
        Ignored keyword arguments accepted for compatibility with other kernel
        functions.

    Returns
    -------
    float or numpy.ndarray
        Polynomial kernel value.
    """
    return (gamma * np.dot(x1, x2) + coef0) ** degree


def rbf_kernel(x1, x2, gamma=0.1, **kwargs):
    """Compute the radial basis function kernel between two input vectors.

    The kernel is defined as ``exp(-gamma * ||x1 - x2||^2)``.

    Parameters
    ----------
    x1 : numpy.ndarray
        First input vector.
    x2 : numpy.ndarray
        Second input vector.
    gamma : float, default=0.1
        Positive scale parameter controlling the width of the RBF kernel.
    **kwargs
        Ignored keyword arguments accepted for compatibility with other kernel
        functions.

    Returns
    -------
    float
        RBF kernel value.
    """
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
