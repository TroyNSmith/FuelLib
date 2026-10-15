"""Fuel class for Group Contribution Method calculations.

`Fuel` is composed of a core class (data loading) and three mixins:

* `ComponentPropertiesMixin`: properties of each compound.
* `ComponentCorrelationsMixin`: temperature-dependent correlations for each compound.
* `MixtureCorrelationsMixin`: composition conversions and mixture correlations.
"""

from .._reference_data import DEFAULT_DATA_DIR
from .component_correlations import ComponentCorrelationsMixin
from .component_properties import ComponentPropertiesMixin
from .core import Fuel
from .mixture_correlations import MixtureCorrelationsMixin

__all__ = [
    "DEFAULT_DATA_DIR",
    "ComponentCorrelationsMixin",
    "ComponentPropertiesMixin",
    "Fuel",
    "MixtureCorrelationsMixin",
]
