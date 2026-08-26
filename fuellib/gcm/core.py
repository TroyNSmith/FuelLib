"""BaseModels for the group contribution method (GCM) implementations."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import pandas as pd
from pint import Quantity
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ..comp import Component


class BaseMethod(BaseModel):
    """Base class for group contribution method (GCM) **method** models."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    groups: pd.DataFrame = Field(default_factory=pd.DataFrame, repr=False)

    def model_post_init(self, context: Any) -> None:
        """Load the groups for the method."""
        self.load_groups()

    @abstractmethod
    def load_groups(self) -> None:
        """Load the groups for the method."""

    # --- Abstract property calculation methods ---
    # Each takes a Component and returns a pint.Quantity with appropriate units.

    @abstractmethod
    def calc_molecular_weight(self, comp: "Component") -> Quantity:
        """Calculate molecular weight in kg/mol."""

    @abstractmethod
    def calc_critical_temperature(self, comp: "Component") -> Quantity:
        """Calculate critical temperature in K."""

    @abstractmethod
    def calc_critical_pressure(self, comp: "Component") -> Quantity:
        """Calculate critical pressure in Pa."""

    @abstractmethod
    def calc_critical_volume(self, comp: "Component") -> Quantity:
        """Calculate critical volume in m³/mol."""

    @abstractmethod
    def calc_boiling_temperature(self, comp: "Component") -> Quantity:
        """Calculate boiling temperature in K."""

    @abstractmethod
    def calc_melting_temperature(self, comp: "Component") -> Quantity:
        """Calculate melting temperature in K."""

    @abstractmethod
    def calc_enthalpy_of_formation(self, comp: "Component") -> Quantity:
        """Calculate enthalpy of formation in J/mol."""

    @abstractmethod
    def calc_gibbs_free_energy(self, comp: "Component") -> Quantity:
        """Calculate Gibbs free energy in J/mol."""

    @abstractmethod
    def calc_enthalpy_of_vaporization_stp(self, comp: "Component") -> Quantity:
        """Calculate enthalpy of vaporization at 298 K in J/mol."""

    @abstractmethod
    def calc_acentric_factor(self, comp: "Component") -> Quantity:
        """Calculate acentric factor (dimensionless)."""

    @abstractmethod
    def calc_molar_liquid_volume_stp(self, comp: "Component") -> Quantity:
        """Calculate molar liquid volume at 298 K in m³/mol."""

    @abstractmethod
    def calc_heat_capacity_coeffs(
        self, comp: "Component"
    ) -> tuple[Quantity, Quantity, Quantity]:
        """Calculate heat capacity coefficients (A, B, C) in J/mol/K."""

    @abstractmethod
    def calc_carbon_number(self, comp: "Component") -> Quantity:
        """Calculate carbon number (dimensionless)."""
