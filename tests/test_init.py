"""Tests for the top-level fuellib package init."""

import importlib
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import fuellib


class TestVersion:
    def test_version_falls_back_when_package_not_found(self):
        with patch(
            "importlib.metadata.version",
            side_effect=PackageNotFoundError("fuellib"),
        ):
            importlib.reload(fuellib)
        try:
            assert fuellib.__version__ == "unknown"
        finally:
            # Restore the normal module state for any subsequent tests.
            importlib.reload(fuellib)
