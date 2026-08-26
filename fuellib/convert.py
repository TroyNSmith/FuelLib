"""Unit conversion functions.

Temperature conversions are handled by pint (see ``fuellib.utils.units``);
this module only retains conversions outside pint's built-in unit system.
"""

from .constants import N_A, k_B


def epsilon_to_characteristic_temperature(epsilon_j_per_mol):
    """
    Convert Lennard-Jones epsilon from J/mol to characteristic temperature in Kelvin.

    The characteristic temperature (epsilon/k_B) is used in transport property
    correlations and is required by combustion codes like CHEMKIN.

    Uses the relation: T* = (epsilon_J/mol) / (N_A * k_B)

    :param epsilon_j_per_mol: Lennard-Jones well depth epsilon in J/mol.
    :type epsilon_j_per_mol: float
    :return: Characteristic temperature (epsilon/k_B) in Kelvin.
    :rtype: float
    """
    epsilon_per_molecule = epsilon_j_per_mol / N_A
    lj_welldepth_K = epsilon_per_molecule / k_B
    return lj_welldepth_K


__all__ = [
    "epsilon_to_characteristic_temperature",
]
