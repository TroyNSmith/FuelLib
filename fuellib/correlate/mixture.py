"""Mixture-wise correlation methods."""

from typing import TYPE_CHECKING

import numpy as np

from ..utils.constants import T_STP
from ..utils.types import PintScalar, PintVector

if TYPE_CHECKING:
    from ..fuel import Fuel


def molar_liquid_vol(
    fuel: "Fuel",
    T: PintScalar | PintVector,
) -> PintVector:
    """
    Compute molar liquid volume with temperature correction.

    :param fuel: The fuel object containing component properties.
    :type fuel: Fuel
    :param T: Temperature in Kelvin. May be a scalar or an array (e.g. a
        temperature sweep); an array ``T`` is broadcast against every
        compound, producing an output of shape ``(*T.shape, num_compounds)``.
    :type T: PintScalar | PintVector
    :return: Molar liquid volume in m^3/mol.
    :rtype: PintVector
    """
    Tc = fuel.Tc.to("K")
    omega = fuel.omega
    Vm_stp = fuel.Vm_stp.to("m^3/mol")

    if T.ndim > 0:
        # Add a trailing axis so an array of temperatures broadcasts
        # against the per-compound properties instead of colliding with
        # the compound axis.
        T = T[..., np.newaxis]

    x = -np.power((1 - (T_STP / Tc)), 2.0 / 7.0)
    y = np.power((1 - (T / Tc)), 2.0 / 7.0) + x
    phi = np.where(T > Tc, x, y)

    z = 0.29056 - 0.08775 * omega.magnitude
    return Vm_stp * np.power(z, phi)
