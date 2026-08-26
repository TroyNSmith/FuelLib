"""Fuel components."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pint import Quantity
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .utils.units import ureg

if TYPE_CHECKING:
    from .gcm.core import BaseMethod


class Component(BaseModel):
    """A fuel component with GCM property calculations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    smiles: str | None = None
    decomposition: set[tuple[str, int]] = Field(repr=False)
    method: Any = Field(
        default=None, repr=False
    )  # BaseMethod | None (avoid forward ref)

    #: Hydrocarbon type (set externally or None)
    hc_type: str | None = None

    #: Family code for thermal conductivity (set externally or None)
    fam: int | None = None

    # Private cache for computed properties
    _cache: dict = PrivateAttr(default_factory=dict)

    def _require_method(self) -> "BaseMethod":
        """Ensure a method is set, raising if not."""
        if self.method is None:
            raise ValueError(
                f"Component '{self.name}' has no GCM method set. "
                "Assign a method (e.g., ConstantinouMethod()) to compute properties."
            )
        return self.method

    def _get_cached(self, key: str, calc_func) -> Quantity:
        """Get a cached property value, computing if necessary."""
        if key not in self._cache:
            self._cache[key] = calc_func()
        return self._cache[key]

    # --- GCM Properties (computed via method) ---

    @property
    def MW(self) -> Quantity:
        """Molecular weight in kg/mol."""
        return self._get_cached("MW", lambda: self._require_method().calc_MW(self))

    @property
    def Tc(self) -> Quantity:
        """Critical temperature in K."""
        return self._get_cached("Tc", lambda: self._require_method().calc_Tc(self))

    @property
    def Pc(self) -> Quantity:
        """Critical pressure in Pa."""
        return self._get_cached("Pc", lambda: self._require_method().calc_Pc(self))

    @property
    def Vc(self) -> Quantity:
        """Critical volume in m³/mol."""
        return self._get_cached("Vc", lambda: self._require_method().calc_Vc(self))

    @property
    def Tb(self) -> Quantity:
        """Boiling temperature in K."""
        return self._get_cached("Tb", lambda: self._require_method().calc_Tb(self))

    @property
    def Tm(self) -> Quantity:
        """Melting temperature in K."""
        return self._get_cached("Tm", lambda: self._require_method().calc_Tm(self))

    @property
    def Hf(self) -> Quantity:
        """Enthalpy of formation in J/mol."""
        return self._get_cached("Hf", lambda: self._require_method().calc_Hf(self))

    @property
    def Gf(self) -> Quantity:
        """Gibbs free energy in J/mol."""
        return self._get_cached("Gf", lambda: self._require_method().calc_Gf(self))

    @property
    def Hv_stp(self) -> Quantity:
        """Enthalpy of vaporization at 298 K in J/mol."""
        return self._get_cached(
            "Hv_stp", lambda: self._require_method().calc_Hv_stp(self)
        )

    @property
    def omega(self) -> Quantity:
        """Acentric factor (dimensionless)."""
        return self._get_cached(
            "omega", lambda: self._require_method().calc_omega(self)
        )

    @property
    def Vm_stp(self) -> Quantity:
        """Molar liquid volume at 298 K in m³/mol."""
        return self._get_cached(
            "Vm_stp", lambda: self._require_method().calc_Vm_stp(self)
        )

    @property
    def Cp_coeffs(self) -> tuple[Quantity, Quantity, Quantity]:
        """Specific heat coefficients (Cp_A, Cp_B, Cp_C) in J/mol/K."""
        return self._get_cached(
            "Cp_coeffs", lambda: self._require_method().calc_Cp_coeffs(self)
        )

    @property
    def Cp_stp(self) -> Quantity:
        """Molar specific heat at 298 K in J/mol/K."""
        return self.Cp_coeffs[0]

    @property
    def Lv_stp(self) -> Quantity:
        """Latent heat of vaporization at 298 K in J/kg."""
        return (self.Hv_stp / self.MW).to("J/kg")

    @property
    def sigma(self) -> Quantity:
        """Lennard-Jones collision diameter in m (Tee et al. 1966)."""
        Pc_atm = self.Pc.to("atm").magnitude
        Tc_val = self.Tc.magnitude
        omega_val = self.omega.magnitude
        sigma_angstrom = (2.3551 - 0.0874 * omega_val) * (Tc_val / Pc_atm) ** (1.0 / 3)
        return (sigma_angstrom * ureg.angstrom).to("m")

    @property
    def epsilonByKB(self) -> Quantity:
        """Lennard-Jones well depth (epsilon/k_B) in K (Tee et al. 1966)."""
        return (0.7915 + 0.1693 * self.omega.magnitude) * self.Tc

    # --- Temperature-dependent methods ---

    def molar_liquid_vol(self, T: Quantity) -> Quantity:
        """Compute molar liquid volume with temperature correction.

        Args:
            T: Temperature in Kelvin.

        Returns:
            Molar liquid volume in m³/mol.
        """
        T_val = T.to("K").magnitude
        Tstp = 298.0
        Tc_val = self.Tc.magnitude
        omega_val = self.omega.magnitude
        Vm_stp_val = self.Vm_stp.magnitude

        if T_val > Tc_val:
            phi = -((1 - (Tstp / Tc_val)) ** (2.0 / 7.0))
        else:
            phi = ((1 - (T_val / Tc_val)) ** (2.0 / 7.0)) - (
                (1 - (Tstp / Tc_val)) ** (2.0 / 7.0)
            )

        z = 0.29056 - 0.08775 * omega_val
        Vm = Vm_stp_val * (z**phi)
        return Vm * ureg("m**3/mol")

    def density(self, T: Quantity) -> Quantity:
        """Calculate density at temperature T.

        Args:
            T: Temperature in Kelvin.

        Returns:
            Density in kg/m³.
        """
        Vm = self.molar_liquid_vol(T)
        return (self.MW / Vm).to("kg/m**3")

    def viscosity_kinematic(self, T: Quantity) -> Quantity:
        """Calculate kinematic viscosity using Dutt's equation.

        Args:
            T: Temperature in Kelvin.

        Returns:
            Kinematic viscosity in m²/s.
        """
        T_cels = T.to("degC").magnitude
        Tb_cels = self.Tb.to("K").magnitude - 273.15  # Convert K to C

        rhs = -3.0171 + (442.78 + 1.6452 * Tb_cels) / (T_cels + 239 - 0.19 * Tb_cels)
        nu = np.exp(rhs)  # mm²/s
        return (nu * ureg("mm**2/s")).to("m**2/s")

    def viscosity_dynamic(self, T: Quantity) -> Quantity:
        """Calculate dynamic viscosity.

        Args:
            T: Temperature in Kelvin.

        Returns:
            Dynamic viscosity in Pa·s.
        """
        nu = self.viscosity_kinematic(T)
        rho = self.density(T)
        return (nu * rho).to("Pa*s")

    def Cp(self, T: Quantity) -> Quantity:
        """Compute molar specific heat capacity at a given temperature.

        Args:
            T: Temperature in Kelvin.

        Returns:
            Molar specific heat capacity in J/mol/K.
        """
        T_val = T.to("K").magnitude
        theta = (T_val - 298) / 700
        Cp_A, Cp_B, Cp_C = self.Cp_coeffs
        return Cp_A + Cp_B * theta + Cp_C * theta**2

    def Cl(self, T: Quantity) -> Quantity:
        """Compute liquid mass specific heat capacity.

        Args:
            T: Temperature in Kelvin.

        Returns:
            Mass specific heat capacity in J/kg/K.
        """
        return (self.Cp(T) / self.MW).to("J/kg/K")

    def psat(
        self,
        T: Quantity,
        correlation: Literal["Lee-Kesler", "Ambrose-Walton"] = "Lee-Kesler",
    ) -> Quantity:
        """Compute saturated vapor pressure.

        Args:
            T: Temperature in Kelvin.
            correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").

        Returns:
            Saturated vapor pressure in Pa.
        """
        T_val = T.to("K").magnitude
        Tr = T_val / self.Tc.magnitude
        Pc_val = self.Pc.magnitude
        omega_val = self.omega.magnitude

        if correlation.casefold() == "Ambrose-Walton".casefold():
            tau = 1 - Tr
            f0 = (
                -5.97616 * tau
                + 1.29874 * tau**1.5
                - 0.60394 * tau**2.5
                - 1.06841 * tau**5.0
            ) / Tr
            f1 = (
                -5.03365 * tau
                + 1.11505 * tau**1.5
                - 5.41217 * tau**2.5
                - 7.46628 * tau**5.0
            ) / Tr
            f2 = (
                -0.64771 * tau
                + 2.41539 * tau**1.5
                - 4.26979 * tau**2.5
                - 3.25259 * tau**5.0
            ) / Tr
            rhs = np.exp(f0 + omega_val * f1 + omega_val**2 * f2)
        else:  # Lee-Kesler
            f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * (Tr**6)
            f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * (Tr**6)
            rhs = np.exp(f0 + omega_val * f1)

        return (Pc_val * rhs) * ureg.Pa

    def latent_heat_vaporization(self, T: Quantity) -> Quantity:
        """Calculate latent heat of vaporization adjusted for temperature.

        Args:
            T: Temperature in Kelvin.

        Returns:
            Latent heat of vaporization in J/kg.
        """
        T_val = T.to("K").magnitude
        Tc_val = self.Tc.magnitude
        Tb_val = self.Tb.magnitude
        Lv_stp_val = self.Lv_stp.magnitude

        Tr = T_val / Tc_val
        Trb = Tb_val / Tc_val

        if T_val > Tc_val:
            return 0.0 * ureg("J/kg")

        Lv = Lv_stp_val * (((1.0 - Tr) / (1.0 - Trb)) ** 0.38)
        return Lv * ureg("J/kg")

    def surface_tension(
        self, T: Quantity, correlation: Literal["Brock-Bird", "Pitzer"] = "Brock-Bird"
    ) -> Quantity:
        """Calculate surface tension at a given temperature.

        Args:
            T: Temperature in Kelvin.
            correlation: Correlation method ("Brock-Bird" or "Pitzer").

        Returns:
            Surface tension in N/m.
        """
        T_val = T.to("K").magnitude
        Tc_val = self.Tc.magnitude
        Pc_bar = self.Pc.to("bar").magnitude
        Tb_val = self.Tb.magnitude
        omega_val = self.omega.magnitude

        Tr = T_val / Tc_val

        if correlation.casefold() == "Brock-Bird".casefold():
            Tbr = Tb_val / Tc_val
            Q = 0.1196 * (1.0 + (Tbr * np.log(Pc_bar / 1.01325)) / (1.0 - Tbr)) - 0.279
        else:  # Pitzer
            w = omega_val
            Q = (
                (1.86 + 1.18 * w)
                / 19.05
                * (((3.75 + 0.91 * w) / (0.291 - 0.08 * w)) ** (2.0 / 3.0))
            )

        st = (
            Pc_bar ** (2.0 / 3.0) * Tc_val ** (1.0 / 3.0) * Q * (1 - Tr) ** (11.0 / 9.0)
        )
        return (st * ureg("dyn/cm")).to("N/m")

    def thermal_conductivity(self, T: Quantity) -> Quantity:
        """Calculate thermal conductivity at a given temperature (Latini et al.).

        Args:
            T: Temperature in Kelvin.

        Returns:
            Thermal conductivity in W/m/K.
        """
        T_val = T.to("K").magnitude
        MW_gmol = self.MW.to("g/mol").magnitude
        Tc_val = self.Tc.magnitude
        Tb_val = self.Tb.magnitude
        fam = self.fam if self.fam is not None else 0  # Default to saturated HC

        alpha = 1.2
        gamma = 0.167
        Tr = T_val / Tc_val

        # Family-specific parameters
        if fam == 1:  # Aromatics
            Astar = 0.0346
            beta = 1.0
        elif fam == 2:  # Cycloparaffins
            Astar = 0.0310
            beta = 1.0
        elif fam == 3:  # Olefins
            Astar = 0.0361
            beta = 1.0
        else:  # Saturated hydrocarbons (default)
            Astar = 0.00350
            beta = 0.5

        MW_beta = MW_gmol**beta
        A = Astar * Tb_val**alpha / (MW_beta * Tc_val**gamma)
        tc = A * (1 - Tr) ** 0.38 / (Tr ** (1 / 6))
        return tc * ureg("W/m/K")

    @classmethod
    def from_csv(cls, path: str | Path) -> list["Component"]:
        """Load components from a CSV file."""
        df = pd.read_csv(path, header=0, index_col=0)

        components = []
        for row in df.itertuples():
            name = str(row[0])
            smiles = str(row.smiles) if hasattr(row, "smiles") else None
            decomposition = {
                (group, int(count))
                for group, count in zip(df.columns, row[1:])
                if count != 0 and str(count).isdigit()
            }
            components.append(
                cls(name=name, smiles=smiles, decomposition=decomposition)
            )
        return components
