"""Physical constants used in FuelLib calculations."""

import unxt as u

# Physical constants
#: Boltzmann's constant in J/K.
k_B = u.Q(1.380649e-23, "J/K")

#: Avogadro's number in 1/mol.
N_A = u.Q(6.02214076e23, "1/mol")

__all__ = ["N_A", "k_B"]
