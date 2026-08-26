"""
FuelLib: Fuel Library for Group Contribution Method calculations.

FuelLib utilizes the Group Contribution Method (GCM) as proposed by Constantinou
and Gani (1994, 1995) to calculate thermodynamic and mixture properties of fuels.

See :class:`Fuel` for the main class and complete API documentation.
"""

try:
    from importlib.metadata import version

    __version__ = version("fuellib")
except Exception:
    __version__ = "unknown"

# Import submodules for namespacing
from . import comp, constants, convert, fuel, gcm, utility

# Import data locator functions
from ._data_locator import *
from .comp import Component
from .fuel import Fuel

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
    "list_fuel_names",
    "utility",
]
