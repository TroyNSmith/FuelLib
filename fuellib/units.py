"""Shared unxt Quantity helpers used across FuelLib."""

from typing import cast

import numpy as np
import unxt as u
from jax import Array


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


__all__ = ["load_quantity"]
