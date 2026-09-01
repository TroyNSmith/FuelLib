"""Utility functions for mixture calculations and droplet properties."""

import quaxed.numpy as jnp
import unxt as u


def mixing_rule(var_n, X, pseudo_prop="arithmetic"):
    """
    Mixing rules for computing mixture properties.

    :param var_n: Individual compound properties (plain array or unxt.Quantity).
    :type var_n: jax.Array or unxt.Quantity
    :param X: Mole fractions of the compounds.
    :type X: jax.Array
    :param pseudo_prop: Type of mean ("arithmetic" or "geometric").
    :type pseudo_prop: str, optional
    :return: Mixture property value.
    :rtype: float or unxt.Quantity
    """
    # Leading "..." batch axis lets var_n be (num_compounds,) for a scalar T
    # or (num_times, num_compounds) for an array of temperatures; X stays (num_compounds,).
    if pseudo_prop.casefold() == "geometric":
        # Use geometric mean definition for the pseudo property
        var_ij = jnp.sqrt(var_n[..., :, None] * var_n[..., None, :])
    else:
        # Use arithmetic definition for the pseudo property
        var_ij = (var_n[..., :, None] + var_n[..., None, :]) / 2
    return jnp.sum(X[..., :, None] * X[..., None, :] * var_ij, axis=(-2, -1))


def droplet_volume(r):
    """
    Calculate spherical volume of a droplet given the radius.

    :param r: Radius of the droplet.
    :type r: unxt.Quantity
    :return: Spherical volume of droplet.
    :rtype: unxt.Quantity
    """
    return 4.0 / 3.0 * jnp.pi * r**3


def droplet_mass(fuel, r, Yi, T):
    """
    Calculate the mass of each compound in the fuel provided the radius of the droplet.

    :param fuel: An instance of the fuel class.
    :type fuel: fuel object
    :param r: Radius of the droplet.
    :type r: unxt.Quantity
    :param Yi: Mass fractions of each compound.
    :type Yi: jax.Array
    :param T: Droplet temperature.
    :type T: unxt.Quantity
    :return: Mass of each compound in droplet.
    :rtype: unxt.Quantity
    """
    volume = droplet_volume(r)
    if volume > 0:
        return volume / (fuel.molar_liquid_vol(T) @ Yi) * Yi * fuel.MW
    else:
        # zero mass, not zero moles - can't reuse fuel.MW's kg/mol unit here
        return u.Q(jnp.zeros_like(fuel.MW.value), "kg")


__all__ = ["droplet_mass", "droplet_volume", "mixing_rule"]
