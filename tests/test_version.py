# SPDX-License-Identifier: MIT
"""``torchkm`` exposes a ``__version__`` resolved from installed metadata."""

from importlib.metadata import PackageNotFoundError, version

import pytest

import torchkm


def test_version_attribute_is_nonempty_string():
    assert isinstance(torchkm.__version__, str)
    assert torchkm.__version__


def test_version_matches_installed_distribution_metadata():
    # ``__version__`` must reflect the *installed* distribution rather than a
    # hard-coded literal, so that a snapshot's ``pip freeze`` and
    # ``torchkm.__version__`` can never disagree. When the package is run from a
    # source tree that was never pip-installed the metadata is absent and the
    # module falls back to a sentinel, so skip the comparison in that case.
    try:
        expected = version("torchkm")
    except PackageNotFoundError:
        pytest.skip("torchkm distribution metadata is not installed")
    assert torchkm.__version__ == expected
