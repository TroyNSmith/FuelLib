"""Helper functions for correlations."""

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import root

from ..utils import Units, types

if TYPE_CHECKING:
    from ..fuel import Fuel


def arithmetic_mixing_rule(
    X: types.Quantity1D,
    var_n: types.Quantity1D,
) -> types.Quantity0D:
    """Arithmetic mixing rule for computing mixture properties.

    Args:
        X: Mole fractions of the compounds.
        var_n: Individual compound properties.

    Returns:
        Mixture property value.
    """
    values = var_n.magnitude
    mag_X = X.to("dimensionless").magnitude

    num_comps = len(values)
    var_mix = 0.0
    for i in range(num_comps):
        for j in range(num_comps):
            # Use arithmetic definition for the pseudo property
            var_ij = (values[i] + values[j]) / 2
            var_mix += mag_X[i] * mag_X[j] * var_ij

    return Units.Quantity(var_mix, var_n.units)  # Attach original units from var_n


def geometric_mixing_rule(
    X: types.Quantity1D,
    var_n: types.Quantity1D,
) -> types.Quantity0D:
    """Geometric mixing rule for computing mixture properties.

    Args:
        X: Mole fractions of the compounds.
        var_n: Individual compound properties.

    Returns:
        Mixture property value.
    """
    values = var_n.magnitude
    mag_X = X.to("dimensionless").magnitude

    num_comps = len(values)
    var_mix = 0.0
    for i in range(num_comps):
        for j in range(num_comps):
            # Use geometric definition for the pseudo property
            var_ij = (values[i] * values[j]) ** 0.5
            var_mix += mag_X[i] * mag_X[j] * var_ij

    return Units.Quantity(var_mix, var_n.units)  # Attach original units from var_n


def mass_to_mass_fractions(fuel: "Fuel", mass: types.Quantity1D) -> types.Quantity1D:
    """Convert mass of each component to mass fractions (Yi).

    Args:
        fuel: Fuel object.
        mass: Mass of each compound.

    Returns:
        Mass fractions of the compounds (shape: num_compounds,).
    """
    # Normalize to get group mole fractions
    mass = mass.to("kg")
    total_mass = mass.magnitude.sum()
    if total_mass != 0:
        Yi = (mass / total_mass).magnitude
    else:
        Yi = np.zeros_like(fuel.MW.magnitude)

    return Units.Quantity(Yi, "dimensionless")


def mass_to_mole_fractions(fuel: "Fuel", mass: types.Quantity1D) -> types.Quantity1D:
    """Convert mass of each component to mole fractions (Xi).

    Args:
        fuel: Fuel object.
        mass: Mass of each compound in the mixture.

    Returns:
        Mole fractions of the compounds (shape: num_compounds,).
    """
    mass = mass.to("kg")

    # Calculate the number of moles for each compound
    num_mole = mass / fuel.MW

    # Normalize to get group mole fractions
    total_moles = np.sum(num_mole)
    if total_moles != 0:
        Xi = (num_mole / total_moles).magnitude
    else:
        Xi = np.zeros_like(fuel.MW.magnitude)

    return Units.Quantity(Xi, "dimensionless")


def mole_fractions_to_mass_fractions(
    fuel: "Fuel", Xi: types.Quantity1D
) -> types.Quantity1D:
    """Convert mole fractions (Xi) to mass fractions (Yi).

    Args:
        fuel: Fuel object.
        Xi: Mole fractions of each compound in the mixture.

    Returns:
        Mass fractions of the compounds (shape: num_compounds,).
    """
    # Calculate the mass for each compound
    mass = fuel.MW * Xi
    # Normalize to get group mass fractions
    total_mass = np.sum(mass)
    Yi: types.Array1D = (
        (mass / total_mass).magnitude
        if total_mass != 0
        else np.zeros_like(fuel.MW.magnitude)
    )

    return Units.Quantity(Yi, "dimensionless")


def mass_fractions_to_mole_fractions(
    fuel: "Fuel", Yi: types.Quantity1D
) -> types.Quantity1D:
    """Convert mass fractions (Yi) to mole fractions (Xi).

    Args:
        fuel: Fuel object.
        Yi: Mass fractions of each compound in the mixture.

    Returns:
        Mole fractions of the compounds (shape: num_compounds,).
    """
    Mbar = fuel.mean_molecular_weight(Yi)
    if np.sum(Yi) != 0:
        Xi = (Mbar * Yi / fuel.MW).magnitude
    else:
        Xi = np.zeros_like(fuel.MW.magnitude)

    return Units.Quantity(Xi, "dimensionless")


def _psat_lee_kesler(
    T: types.Array1D | float,
    Tc: types.Array1D,
    Pc: types.Array1D,
    omega: types.Array1D,
) -> types.Array1D:
    """Lee-Kesler saturated vapor pressure for each compound.

    Args:
        T: Temperature in K.
        Tc: Critical temperature of each compound in K.
        Pc: Critical pressure of each compound.
        omega: Acentric factor of each compound.

    Returns:
        Saturated vapor pressure (units follow whatever `Pc` is given in).
    """
    Tr = T / Tc
    f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * (Tr**6)
    f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * (Tr**6)
    return Pc * np.exp(f0 + omega * f1)


def liaw_chiu_flash_point(
    Xi: types.Array1D,
    Tf_i: types.Array1D,
    Tc: types.Array1D,
    Pc: types.Array1D,
    omega: types.Array1D,
) -> float:
    """Solve the ideal Liaw-Chiu mixture flash-point criterion.

    Solves for the mixture flash point T such that
    sum(Xi * psat(T) / psat(Tf_i)) = 1, where psat is evaluated with the
    Lee-Kesler correlation for each compound at its own critical properties.
    A scipy root finder is used in place of the fixed-point iteration used
    in the original reference implementation.

    Args:
        Xi: Mole fractions of each compound in the mixture.
        Tf_i: Flash point of each compound in K.
        Tc: Critical temperature of each compound in K.
        Pc: Critical pressure of each compound.
        omega: Acentric factor of each compound.

    Returns:
        Mixture flash point in K.

    Raises:
        RuntimeError: If the root finder fails to converge.
    """
    psat_ref = _psat_lee_kesler(Tf_i, Tc, Pc, omega)

    def residual(T: types.Array1D) -> types.Array1D:
        """Residual of the ideal Liaw-Chiu mixture flash-point criterion.

        Args:
            T: Candidate mixture flash point in K.

        Returns:
            Residual of the flash-point criterion.
        """
        psat_T = _psat_lee_kesler(T[0], Tc, Pc, omega)
        return np.array([np.sum(Xi * psat_T / psat_ref) - 1.0])

    T0 = np.sum(Xi * Tf_i)
    sol = root(residual, [T0])
    if not sol.success:
        raise RuntimeError(
            f"Flash point root-finding failed to converge: {sol.message}"
        )

    return sol.x[0]


__all__ = [
    "arithmetic_mixing_rule",
    "geometric_mixing_rule",
    "liaw_chiu_flash_point",
    "mass_fractions_to_mole_fractions",
    "mass_to_mass_fractions",
    "mass_to_mole_fractions",
    "mole_fractions_to_mass_fractions",
]
