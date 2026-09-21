"""Group Contribution Methods for predicting thermophysical properties of chemical compounds."""

from .core import GCMRegistry
from .gani import gani_gcm

__all__ = ["GCMRegistry", "gani_gcm"]
