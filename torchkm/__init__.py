# SPDX-License-Identifier: MIT
from importlib.metadata import PackageNotFoundError, version

from .estimators import (
    TorchKMDWD,
    TorchKMKQR,
    TorchKMLogit,
    TorchKMSVC,
)
from .functions import data_gen, sigest, rbf_kernel, kernelMult, standardize

# Resolve the version from the installed distribution metadata rather than
# duplicating the literal in setup.cfg. This keeps ``torchkm.__version__``,
# ``pip freeze``, and the packaged metadata in agreement, which matters for
# reproducible GPU/MLOSS snapshots.
try:
    __version__ = version("torchkm")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    # Running straight from a checkout that was never pip-installed.
    __version__ = "0.0.0+unknown"

__all__ = [
    "TorchKMSVC",
    "TorchKMDWD",
    "TorchKMLogit",
    "TorchKMKQR",
    "data_gen",
    "sigest",
    "rbf_kernel",
    "kernelMult",
    "standardize",
]
