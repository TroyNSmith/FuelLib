"""Shared unxt Quantity helpers used across FuelLib."""

from typing import Literal, Union, cast

import astropy.units as apyu
import numpy as np
import unxt as u
from jax import Array
from unxt import AbstractQuantity, Quantity

# astropy has no built-in "atm" unit string; register one so it can be used like any other unit.
apyu.add_enabled_units(
    [apyu.def_unit("atm", 101325 * apyu.Pa, doc="Standard atmosphere")]
)


def load_quantity(
    arr: Array | np.ndarray | float, native_unit: str, target_unit: str | None = None
) -> u.Quantity:
    """
    Load a quantity with optional unit conversion.

    :param arr: Raw (unitless) array to attach a unit to.
    :type arr: jax.Array, numpy.ndarray, or float
    :param native_unit: Unit that `arr` is expressed in.
    :type native_unit: str
    :param target_unit: Unit to convert to, if different from `native_unit`.
    :type target_unit: str, optional
    :return: Quantity wrapping `arr`.
    :rtype: unxt.Quantity
    """
    q = u.Q(arr, native_unit)
    if target_unit is not None:
        q = q.uconvert(target_unit)
    return cast(u.Quantity, q)


def convert_temperature(temp: AbstractQuantity, target_unit: str) -> AbstractQuantity:
    """
    Convert a temperature quantity to a different unit.

    :param temp: Temperature quantity to convert.
    :type temp: AbstractQuantity
    :param target_unit: Unit to convert to.
    :type target_unit: str
    :return: Converted temperature quantity.
    :rtype: unxt.AbstractQuantity
    """
    with apyu.add_enabled_equivalencies(apyu.temperature()):
        return temp.uconvert(target_unit)


def convert_pressure_to_atm(pressure: AbstractQuantity) -> AbstractQuantity:
    """
    Convert a pressure quantity to atmospheres.

    :param pressure: Pressure quantity to convert.
    :type pressure: AbstractQuantity
    :return: Converted pressure quantity in atmospheres.
    :rtype: unxt.AbstractQuantity
    """
    return pressure.to("atm")


__all__ = ["convert_pressure_to_atm", "convert_temperature", "load_quantity"]
