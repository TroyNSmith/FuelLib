"""Fuel class for Group Contribution Method calculations.

`Fuel` is composed of a core class (data loading) and three classes:

* `ComponentProperties`: properties of each compound.
* `ComponentCorrelations`: temperature-dependent correlations for each compound.
* `MixtureCorrelations`: composition conversions and mixture correlations.
"""

from .._reference_data import DEFAULT_DATA_DIR
from .component_correlations import ComponentCorrelations
from .component_properties import ComponentProperties
from .core import Fuel
from .mixture_correlations import MixtureCorrelations

__all__ = [
    "DEFAULT_DATA_DIR",
    "ComponentCorrelations",
    "ComponentProperties",
    "Fuel",
    "MixtureCorrelations",
]
