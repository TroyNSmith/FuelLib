"""Component-wise correlation methods."""

from typing import TYPE_CHECKING, overload

import numpy as np

from ..utils.constants import PintUnits
from ..utils.types import PintMatrix, PintScalar, PintVector

if TYPE_CHECKING:
    from ..fuel import Fuel


def atleast_1d(quantity: PintScalar | PintVector) -> PintVector:
    """
    Ensure a pint Quantity has at least one dimension, preserving its units.

    :param quantity: The quantity to convert.
    :type quantity: PintScalar | PintVector
    :return: The quantity with its magnitude passed through ``np.atleast_1d``.
    :rtype: PintVector
    """
    return PintUnits.Quantity(np.atleast_1d(quantity.magnitude), quantity.units)


@overload
def psat_lee_kesler(fuel: "Fuel", T: PintScalar) -> PintVector: ...
@overload
def psat_lee_kesler(fuel: "Fuel", T: PintVector) -> PintMatrix: ...
def psat_lee_kesler(
    fuel: "Fuel", T: PintScalar | PintVector
) -> PintVector | PintMatrix:
    """
    Calculate the Lee-Kesler pure-component saturation pressures for the given fuel.

    :param fuel: The fuel object containing component properties.
    :type fuel: Fuel
    :param T: Temperature at which to calculate the saturation pressures.
    :type T: PintScalar | PintVector
    :return: Saturation pressures of each component in Pascal. Returns a PintVector if
    T is a PintScalar, or a PintMatrix if T is a PintVector.
    :rtype: PintVector | PintMatrix
    """
    is_scalar_T = np.ndim(T.magnitude) == 0
    T = atleast_1d(T)

    Tc = fuel.get_gcm_property("gani", "Tc")[:, np.newaxis]
    Pc = fuel.get_gcm_property("gani", "Pc").to("Pa")[:, np.newaxis]
    w = fuel.get_gcm_property("gani", "omega")[:, np.newaxis]

    Tr = T / Tc
    f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * (Tr**6)
    f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * (Tr**6)
    rhs = np.exp(f0 + w * f1)

    psat = (Pc * rhs).to("Pa")
    return psat[:, 0] if is_scalar_T else psat


def flash_point_alqaheem(fuel: "Fuel") -> PintVector:
    """
    Calculate the Alqaheem-Riazi pure-component flash points for the given fuel.

    :param fuel: The fuel object containing component properties.
    :type fuel: Fuel
    :return: Flash points of each component in Kelvin.
    :rtype: PintVector
    """
    return 0.70 * fuel.get_gcm_property("astm", "Tb").to("K")


def flash_point_alibakhshi(fuel: "Fuel") -> PintVector:
    """
    Calculate the Alibakhshi et al. (2015) pure-component flash points for the given fuel.

    :param fuel: The fuel object containing component properties.
    :type fuel: Fuel
    :return: Flash points of each component in Kelvin.
    :rtype: PintVector
    """
    Tb = fuel.get_gcm_property("astm", "Tb").to("K")
    phi_sum = fuel.get_gcm_property("gani", "alibakhshi_phi").to("K")
    return PintUnits.Quantity(12.14, "K") + 0.73 * Tb + phi_sum


__all__ = [
    "flash_point_alibakhshi",
    "flash_point_alqaheem",
    "psat_lee_kesler",
]
