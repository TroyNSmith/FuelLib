"""
FuelLib: Fuel Library for Group Contribution Method calculations.

FuelLib utilizes the Group Contribution Method (GCM) as proposed by Constantinou
and Gani (1994, 1995) to calculate thermodynamic and mixture properties of fuels.

See :class:`Fuel` for the main class and complete API documentation.
"""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("fuellib")
except PackageNotFoundError:
    __version__ = "unknown"

# Import submodules for namespacing
from . import comp, fuel, gcm

# Import data locator functions
from ._data_locator import *
from .comp import Component
from .fuel import Fuel
from .utils import constants, convert, helpers

__all__ = [
    "Component",
    "Fuel",
    "comp",
    "constants",
    "convert",
    "fuel",
    "gcm",
    "get_data_dir",
    "get_fueldata_dir",
    "get_fueldata_props_dir",
    "get_gcmtable_dir",
    "helpers",
    "list_fuel_names",
]
