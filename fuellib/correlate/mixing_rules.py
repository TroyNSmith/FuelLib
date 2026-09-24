"""Mixing rules for fuel properties."""

from typing import TYPE_CHECKING

import numpy as np

from ..utils.types import FloatVector, PintScalar, PintVector
from ..utils.units import PintUnits

if TYPE_CHECKING:
    from ..fuel import Fuel


def linear(Xi: FloatVector, property_values: PintVector) -> PintScalar:
    """
    Calculate the linear mixing rule for a given property.

    :param Xi: Mole fractions of the components.
    :type Xi: FloatVector
    :param property_values: Property values of the components.
    :type property_values: PintVector
    :return: Mixture property calculated using the linear mixing rule.
    :rtype: PintScalar
    """
    return PintUnits.Quantity(np.sum(Xi * property_values), property_values.units)
