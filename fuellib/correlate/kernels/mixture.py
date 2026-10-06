"""Unitless mixture kernels (NumPy or JAX, dispatched on inputs)."""

from ...utils.backend import get_namespace
from ...utils.types import ArrayLike


def density(Y: ArrayLike, rho_i: ArrayLike) -> ArrayLike:
    """Calculate the density of a mixture.

    Args:
        Y: Mass fractions of the components in the mixture.
        rho_i: Densities of the pure components.

    Returns:
        Density of the mixture, in the same units as `rho_i`.
    """
    xp = get_namespace(Y, rho_i)
    return xp.asarray(Y) @ xp.asarray(rho_i)


def kinematic_viscosity_arrhenius(
    Y: ArrayLike, MW: ArrayLike, nu_i: ArrayLike
) -> ArrayLike:
    """Calculate the kinematic viscosity of a mixture using Arrhenius mixing.

    Args:
        Y: Mass fractions of the components in the mixture.
        MW: Molecular weights of the components (any consistent units).
        nu_i: Kinematic viscosities of the pure components.

    Returns:
        Kinematic viscosity of the mixture, in the same units as `nu_i`.
    """
    xp = get_namespace(Y, MW, nu_i)
    N = xp.asarray(Y) / xp.asarray(MW)
    X = N / xp.sum(N)
    nu_i = xp.asarray(nu_i)
    return xp.exp(xp.sum(X * xp.log(nu_i)))


def kinematic_viscosity_kendall_monroe(
    Y: ArrayLike, MW: ArrayLike, nu_i: ArrayLike
) -> ArrayLike:
    """Calculate the kinematic viscosity of a mixture using Kendall-Monroe mixing.

    Args:
        Y: Mass fractions of the components in the mixture.
        MW: Molecular weights of the components (any consistent units).
        nu_i: Kinematic viscosities of the pure components.

    Returns:
        Kinematic viscosity of the mixture, in the same units as `nu_i`.
    """
    xp = get_namespace(Y, MW, nu_i)
    N = xp.asarray(Y) / xp.asarray(MW)
    X = N / xp.sum(N)
    nu_i = xp.asarray(nu_i)
    return xp.power(xp.sum(X * xp.power(nu_i, 1.0 / 3.0)), 3.0)


__all__ = [
    "density",
    "kinematic_viscosity_arrhenius",
    "kinematic_viscosity_kendall_monroe",
]
