"""Physical constants used in FuelLib calculations."""

import unxt as u
from unxt import Quantity

# Physical constants
#: Boltzmann's constant in J/K.
k_B = u.Q(1.380649e-23, "J/K")

#: Avogadro's number in 1/mol.
N_A = u.Q(6.02214076e23, "1/mol")

# Lennard-Jones parameters for air (used in diffusion coefficient calculations)
## Lennard-Jones collision diameter of the gas in meters.
SIGMA_GAS = Quantity(3.62e-10, "m")
## Well depth of the Lennard-Jones potential divided by Boltzmann's constant in Kelvin.
EPSILON_BY_KB_GAS = Quantity(97.0, "K")
## Mean molecular weight of ambient gas in kg/mol.
MW_GAS = Quantity(28.97e-3, "kg/mol")

__all__ = ["EPSILON_BY_KB_GAS", "MW_GAS", "N_A", "SIGMA_GAS", "k_B"]
