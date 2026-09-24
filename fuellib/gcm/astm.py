"""ASTM D86 correlation methods for GCM."""

from typing import TYPE_CHECKING

from ..utils.logger import logger
from ..utils.types import PintVector
from . import gani
from .core import GCMRegistry

if TYPE_CHECKING:
    from ..fuel import Fuel

astm_gcm = GCMRegistry.register("astm", property_fns=[])


@astm_gcm.register_property
def Tm(fuel: "Fuel") -> PintVector:
    """Predict the melting temperature (Tm) for the given fuel's components using ASTM D86 correlation."""
    logger.info("ASTM Tm not implemented; using Gani Tm.")
    return gani.Tm(fuel)


@astm_gcm.register_property
def Tb(fuel: "Fuel") -> PintVector:
    """Predict the boiling temperature (Tb) for the given fuel's components using ASTM D86 correlation."""
    logger.info("ASTM Tb not implemented; using Gani Tb.")
    return gani.Tb(fuel)
