"""Module for calculating correlation properties of fuel compounds."""

from typing import TYPE_CHECKING

import quaxed.numpy as qnp
from unxt import AbstractQuantity, Quantity

from ..units import convert_pressure_to_atm, convert_temperature

if TYPE_CHECKING:
    from .core import Fuel


def tee_sigma(fuel: "Fuel", *, unit: str = "Angstrom") -> AbstractQuantity:
    """
    Calculate the diffusion coefficient of fuel compounds in a gas using the Tee correlation.

    :param fuel: Fuel object containing properties of the fuel compounds.
    :type fuel: Fuel
    :param unit: Unit for the output diffusion coefficient. Defaults to "Angstrom".
    :type unit: str
    :return: Diffusion coefficient of fuel compounds in the specified unit.
    :rtype: AbstractQuantity
    """
    Tc = convert_temperature(fuel.Tc, "K")
    Pc = convert_pressure_to_atm(fuel.Pc)
    C = Quantity(1.0, "Angstrom*atm^(1/3)/K^(1/3)")
    return (C * (2.3551 - 0.0874 * fuel._omega) * qnp.power(Tc / Pc, 1 / 3)).to(unit)


def tee_epsilon(fuel: "Fuel", *, unit: str = "K") -> AbstractQuantity:
    """
    Calculate the Lennard-Jones potential well depth (epsilon/k) for fuel compounds using the Tee correlation.

    :param fuel: Fuel object containing properties of the fuel compounds.
    :type fuel: Fuel
    :param unit: Desired unit for the output. Defaults to "K".
    :type unit: str
    :return: Lennard-Jones potential well depth (epsilon/k) in the specified unit.
    :rtype: AbstractQuantity
    """
    Tc = convert_temperature(fuel.Tc, "K")
    return convert_temperature((0.7915 + 0.1693 * fuel._omega) * Tc, unit)
