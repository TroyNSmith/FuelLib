"""Module for calculating correlation properties of fuel compounds."""

from typing import TYPE_CHECKING, Literal

import quaxed.numpy as qnp
from jax import Array
from unxt import AbstractQuantity, Quantity

from ..units import convert_temperature

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
    Pc = fuel.Pc.to("atm")
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


def mixing_rule(
    var_n: AbstractQuantity,
    X: Array,
    *,
    pseudo_prop: Literal["arithmetic", "geometric"] = "arithmetic",
) -> AbstractQuantity:
    """
    Mixing rules for computing mixture properties.

    :param var_n: Individual compound properties (plain array or unxt.Quantity).
    :type var_n: AbstractQuantity
    :param X: Mole fractions of the compounds.
    :type X: jax.Array
    :param pseudo_prop: Type of mean ("arithmetic" or "geometric"). Defaults to "arithmetic".
    :type pseudo_prop: str, optional
    :return: Mixture property value.
    :rtype: AbstractQuantity
    """
    # Leading "..." batch axis lets var_n be (num_compounds,) for a scalar T
    # or (num_times, num_compounds) for an array of temperatures; X stays (num_compounds,).
    arr = var_n.value
    unit = var_n.unit
    if pseudo_prop == "geometric":
        # Use geometric mean definition for the pseudo property
        var_ij = qnp.sqrt(arr[..., :, None] * arr[..., None, :])
    elif pseudo_prop == "arithmetic":
        # Use arithmetic mean definition for the pseudo property
        var_ij = (arr[..., :, None] + arr[..., None, :]) / 2
    else:
        raise ValueError(f"Invalid pseudo_prop value: {pseudo_prop}")

    return Quantity(
        qnp.sum(X[..., :, None] * X[..., None, :] * var_ij, axis=(-2, -1)), unit
    )
