"""Utility functions for mixture calculations and droplet properties."""

import numpy as np

from .utils.units import ureg


def mixing_rule(component_values, mole_fractions, pseudo_prop="arithmetic"):
    """
    Mixing rules for computing mixture properties.

    :param component_values: Individual compound properties.
    :type component_values: np.ndarray
    :param mole_fractions: Mole fractions of the compounds.
    :type mole_fractions: np.ndarray
    :param pseudo_prop: Type of mean ("arithmetic" or "geometric").
    :type pseudo_prop: str, optional
    :return: Mixture property value.
    :rtype: float
    """
    num_comps = len(component_values)
    var_mix = 0.0
    for i in range(num_comps):
        for j in range(num_comps):
            if pseudo_prop.casefold() == "geometric":
                # Use geometric mean definition for the pseudo property
                var_ij = (component_values[i] * component_values[j]) ** (0.5)
            else:
                # Use arithmetic definition for the pseudo property
                var_ij = (component_values[i] + component_values[j]) / 2
            var_mix += mole_fractions[i] * mole_fractions[j] * var_ij
    return var_mix


def droplet_volume(radius):
    """
    Calculate spherical volume of a droplet given the radius.

    :param radius: Radius of the droplet in meters.
    :type radius: float
    :return: Spherical volume of droplet in cubic meters.
    :rtype: float
    """
    return 4.0 / 3.0 * np.pi * radius**3


def droplet_mass(fuel, radius, mass_fractions, temperature):
    """
    Calculate the mass of each compound in the fuel provided the radius of the droplet.

    :param fuel: An instance of the Fuel class.
    :type fuel: Fuel object
    :param radius: Radius of the droplet in meters.
    :type radius: float
    :param mass_fractions: Mass fractions of each compound.
    :type mass_fractions: np.ndarray
    :param temperature: Droplet temperature.
    :type temperature: pint.Quantity
    :return: Mass of each compound in droplet, in kg.
    :rtype: pint.Quantity
    """
    from .fuel import (
        molar_liquid_vol,
    )  # local import avoids circular import with fuel.py

    volume = droplet_volume(radius) * ureg("m**3")
    if volume.magnitude > 0:
        return (
            volume
            / (molar_liquid_vol(fuel, temperature) @ mass_fractions)
            * mass_fractions
            * fuel.molecular_weight
        )
    else:
        return np.zeros_like(fuel.molecular_weight.magnitude) * ureg.kg


__all__ = ["droplet_mass", "droplet_volume", "mixing_rule"]
