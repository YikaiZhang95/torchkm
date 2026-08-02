# SPDX-License-Identifier: MIT
"""Warning classes used by the torchkm solvers."""

try:
    # Reuse sklearn's class so callers can filter with the familiar
    # sklearn.exceptions.ConvergenceWarning.
    from sklearn.exceptions import ConvergenceWarning
except ImportError:  # pragma: no cover - solvers stay usable without sklearn

    class ConvergenceWarning(UserWarning):
        """Warning raised when a solver stops before reaching convergence."""


__all__ = ["ConvergenceWarning"]
