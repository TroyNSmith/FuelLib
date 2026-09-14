"""Units handling and conversions.

This module re-exports all of ``astropy.units`` (``Quantity``, ``Unit``, and every
unit symbol such as ``K``, ``Pa``, ``m``, ``kg``, etc.) alongside FuelLib's custom
units and helper functions. This allows the entire units API to be accessed from
a single namespace, e.g. ``fl.units.Quantity`` or ``fl.units.K``, without needing
a separate ``import astropy.units`` statement.
"""

import astropy.units as _astropy_units
import numpy as np
from astropy.units import *  # noqa: F401,F403

# astropy does not have some common unit strings; we need to define and register them
## Pressure (defined in terms of Pa)
atm = _astropy_units.def_unit(
    "atm", 101325 * _astropy_units.Pa, doc="Standard atmosphere"
)
dyne_cm2 = _astropy_units.def_unit(
    "dyne/cm^2", 0.1 * _astropy_units.Pa, doc="Dyne per square centimeter"
)
cgs = _astropy_units.def_unit(
    "cgs", 0.1 * _astropy_units.Pa, doc="CGS unit of pressure"
)
mks = _astropy_units.def_unit("mks", 1 * _astropy_units.Pa, doc="MKS unit of pressure")
## Temperature
fahrenheit = _astropy_units.def_unit(
    "Fahrenheit",
    1 * _astropy_units.imperial.deg_F,
    doc="Fahrenheit temperature unit",
)
## Undefined
dimensionless = _astropy_units.def_unit(
    "dimensionless", 1 * _astropy_units.dimensionless_unscaled, doc="Dimensionless unit"
)

## Register the new units with astropy so they can be used in unxt.Quantity objects.
_astropy_units.add_enabled_units([atm, mks, dyne_cm2, cgs, fahrenheit, dimensionless])

## Enable temperature equivalencies globally so that ``.to()`` can convert between
## temperature scales (K, Celsius, Fahrenheit, ...) just like any other astropy
## unit conversion, without callers needing to enable the equivalency themselves.
_astropy_units.add_enabled_equivalencies(_astropy_units.temperature())


def ustrip(quant: _astropy_units.Quantity) -> np.ndarray:
    """
    Strip the unit from an astropy Quantity, returning the raw value as a numpy array.

    :param quant: Quantity to strip the unit from.
    :type quant: u.Quantity
    :return: Raw value without units.
    :rtype: np.ndarray
    """
    return np.array(quant.value)


__all__ = [
    *_astropy_units.__all__,
    "atm",
    "cgs",
    "dimensionless",
    "dyne_cm2",
    "fahrenheit",
    "mks",
    "ustrip",
]
