# SPDX-License-Identifier: MIT
"""``torchkm`` exposes a ``__version__`` resolved from installed metadata."""

from importlib.metadata import PackageNotFoundError, version

import pytest

import torchkm


def test_version_attribute_is_nonempty_string():
    # A missing attribute here almost always means a stale install is shadowing
    # the source tree (a non-editable ``pip install`` that predates this change,
    # or the published wheel), so surface the imported path to make that obvious.
    assert hasattr(torchkm, "__version__"), (
        f"torchkm imported from {torchkm.__file__!r} exposes no __version__ "
        "- reinstall the checkout with `pip install -e .`"
    )
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
