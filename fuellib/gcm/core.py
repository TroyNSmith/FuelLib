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
    def calc_MW(self, comp: "Component") -> Quantity:
        """Calculate molecular weight in kg/mol."""

    @abstractmethod
    def calc_Tc(self, comp: "Component") -> Quantity:
        """Calculate critical temperature in K."""

    @abstractmethod
    def calc_Pc(self, comp: "Component") -> Quantity:
        """Calculate critical pressure in Pa."""

    @abstractmethod
    def calc_Vc(self, comp: "Component") -> Quantity:
        """Calculate critical volume in m³/mol."""

    @abstractmethod
    def calc_Tb(self, comp: "Component") -> Quantity:
        """Calculate boiling temperature in K."""

    @abstractmethod
    def calc_Tm(self, comp: "Component") -> Quantity:
        """Calculate melting temperature in K."""

    @abstractmethod
    def calc_Hf(self, comp: "Component") -> Quantity:
        """Calculate enthalpy of formation in J/mol."""

    @abstractmethod
    def calc_Gf(self, comp: "Component") -> Quantity:
        """Calculate Gibbs free energy in J/mol."""

    @abstractmethod
    def calc_Hv_stp(self, comp: "Component") -> Quantity:
        """Calculate enthalpy of vaporization at 298 K in J/mol."""

    @abstractmethod
    def calc_omega(self, comp: "Component") -> Quantity:
        """Calculate acentric factor (dimensionless)."""

    @abstractmethod
    def calc_Vm_stp(self, comp: "Component") -> Quantity:
        """Calculate molar liquid volume at 298 K in m³/mol."""

    @abstractmethod
    def calc_Cp_coeffs(self, comp: "Component") -> tuple[Quantity, Quantity, Quantity]:
        """Calculate specific heat coefficients (Cp_A, Cp_B, Cp_C) in J/mol/K."""
