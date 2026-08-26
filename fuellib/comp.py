"""Fuel components."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
import pandas as pd
from pint import Quantity
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .utils.units import ureg

if TYPE_CHECKING:
    from .gcm.core import BaseMethod

_T = TypeVar("_T")


class Component(BaseModel):
    """A fuel component with GCM property calculations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    smiles: str | None = None
    pelephysics_key: str | None = None
    decomposition: set[tuple[str, int]] = Field(repr=False)
    method: Any = Field(
        default=None, repr=False
    )  # BaseMethod | None (avoid forward ref)

    # Private cache for computed properties
    _cache: dict = PrivateAttr(default_factory=dict)

    def _require_method(self) -> BaseMethod:
        """Ensure a method is set, raising if not."""
        if self.method is None:
            raise ValueError(
                f"Component '{self.name}' has no GCM method set. "
                "Assign a method (e.g., ConstantinouMethod()) to compute properties."
            )
        return self.method

    def _get_cached(self, key: str, calc_func: Callable[[], _T]) -> _T:
        """Get a cached property value, computing if necessary."""
        if key not in self._cache:
            self._cache[key] = calc_func()
        return self._cache[key]

    # --- GCM Properties (computed via method) ---

    @property
    def molecular_weight(self) -> Quantity:
        """Molecular weight in kg/mol."""
        return self._get_cached(
            "molecular_weight",
            lambda: self._require_method().calc_molecular_weight(self),
        )

    @property
    def critical_temperature(self) -> Quantity:
        """Critical temperature in K."""
        return self._get_cached(
            "critical_temperature",
            lambda: self._require_method().calc_critical_temperature(self),
        )

    @property
    def critical_pressure(self) -> Quantity:
        """Critical pressure in Pa."""
        return self._get_cached(
            "critical_pressure",
            lambda: self._require_method().calc_critical_pressure(self),
        )

    @property
    def critical_volume(self) -> Quantity:
        """Critical volume in m³/mol."""
        return self._get_cached(
            "critical_volume", lambda: self._require_method().calc_critical_volume(self)
        )

    @property
    def boiling_temperature(self) -> Quantity:
        """Boiling temperature in K."""
        return self._get_cached(
            "boiling_temperature",
            lambda: self._require_method().calc_boiling_temperature(self),
        )

    @property
    def melting_temperature(self) -> Quantity:
        """Melting temperature in K."""
        return self._get_cached(
            "melting_temperature",
            lambda: self._require_method().calc_melting_temperature(self),
        )

    @property
    def enthalpy_of_formation(self) -> Quantity:
        """Enthalpy of formation in J/mol."""
        return self._get_cached(
            "enthalpy_of_formation",
            lambda: self._require_method().calc_enthalpy_of_formation(self),
        )

    @property
    def gibbs_free_energy(self) -> Quantity:
        """Gibbs free energy in J/mol."""
        return self._get_cached(
            "gibbs_free_energy",
            lambda: self._require_method().calc_gibbs_free_energy(self),
        )

    @property
    def enthalpy_of_vaporization_stp(self) -> Quantity:
        """Enthalpy of vaporization at 298 K in J/mol."""
        return self._get_cached(
            "enthalpy_of_vaporization_stp",
            lambda: self._require_method().calc_enthalpy_of_vaporization_stp(self),
        )

    @property
    def acentric_factor(self) -> Quantity:
        """Acentric factor (dimensionless)."""
        return self._get_cached(
            "acentric_factor", lambda: self._require_method().calc_acentric_factor(self)
        )

    @property
    def molar_liquid_volume_stp(self) -> Quantity:
        """Molar liquid volume at 298 K in m³/mol."""
        return self._get_cached(
            "molar_liquid_volume_stp",
            lambda: self._require_method().calc_molar_liquid_volume_stp(self),
        )

    @property
    def heat_capacity_coeffs(self) -> tuple[Quantity, Quantity, Quantity]:
        """Heat capacity coefficients (A, B, C) in J/mol/K."""
        return self._get_cached(
            "heat_capacity_coeffs",
            lambda: self._require_method().calc_heat_capacity_coeffs(self),
        )

    @property
    def heat_capacity_stp(self) -> Quantity:
        """Molar heat capacity at 298 K in J/mol/K."""
        return self.heat_capacity_coeffs[0]

    @property
    def heat_capacity_coeff_b(self) -> Quantity:
        """Temperature-correction coefficient B for molar heat capacity in J/mol/K."""
        return self.heat_capacity_coeffs[1]

    @property
    def heat_capacity_coeff_c(self) -> Quantity:
        """Temperature-correction coefficient C for molar heat capacity in J/mol/K."""
        return self.heat_capacity_coeffs[2]

    @property
    def carbon_number(self) -> Quantity:
        """Carbon number, from alkyl/olefinic/aromatic group contributions."""
        return self._get_cached(
            "carbon_number", lambda: self._require_method().calc_carbon_number(self)
        )

    @property
    def latent_heat_vaporization_stp(self) -> Quantity:
        """Latent heat of vaporization at 298 K in J/kg."""
        return (self.enthalpy_of_vaporization_stp / self.molecular_weight).to("J/kg")

    @property
    def lennard_jones_diameter(self) -> Quantity:
        """Lennard-Jones collision diameter in m (Tee et al. 1966)."""
        critical_pressure_atm = self.critical_pressure.to("atm").magnitude
        critical_temperature_val = self.critical_temperature.magnitude
        acentric_factor_val = self.acentric_factor.magnitude
        diameter_angstrom = (2.3551 - 0.0874 * acentric_factor_val) * (
            critical_temperature_val / critical_pressure_atm
        ) ** (1.0 / 3)
        return (diameter_angstrom * ureg.angstrom).to("m")

    @property
    def epsilon_over_kb(self) -> Quantity:
        """Lennard-Jones well depth (epsilon/k_B) in K (Tee et al. 1966)."""
        return (
            0.7915 + 0.1693 * self.acentric_factor.magnitude
        ) * self.critical_temperature

    @property
    def hc_type(self) -> str:
        """Hydrocarbon type, by priority: aromatic > cyclo-alkane > alkene > iso-alkane > n-alkane."""
        return self._get_cached("hc_type", self._calc_hc_type)

    def _calc_hc_type(self) -> str:
        groups = self._require_method().groups
        types = [groups.loc["type", g] for g, _ in self.decomposition]
        if "aromatic" in types:
            return "aromatic"
        if "cycloalkane" in types:
            return "cyclo-alkane"
        if "alkene" in types:
            return "alkene"
        if "branched" in types:
            return "iso-alkane"
        return "n-alkane"

    @property
    def family_code(self) -> int:
        """Family code for thermal conductivity (0: saturated, 1: aromatic, 2: cycloparaffin, 3: olefin)."""
        return {"aromatic": 1, "cyclo-alkane": 2, "alkene": 3}.get(self.hc_type, 0)

    @classmethod
    def from_csv(cls, path: str | Path) -> list[Component]:
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


# --- Temperature-dependent functions ---


def molar_liquid_vol(comp: Component, temperature: Quantity) -> Quantity:
    """Compute molar liquid volume with temperature correction.

    Args:
        comp: Component to evaluate.
        temperature: Temperature in Kelvin.

    Returns:
        Molar liquid volume in m³/mol.
    """
    temperature_val = temperature.to("K").magnitude
    temperature_stp = 298.0
    critical_temperature_val = comp.critical_temperature.magnitude
    acentric_factor_val = comp.acentric_factor.magnitude
    molar_liquid_volume_stp_val = comp.molar_liquid_volume_stp.magnitude

    if temperature_val > critical_temperature_val:
        phi = -((1 - (temperature_stp / critical_temperature_val)) ** (2.0 / 7.0))
    else:
        phi = ((1 - (temperature_val / critical_temperature_val)) ** (2.0 / 7.0)) - (
            (1 - (temperature_stp / critical_temperature_val)) ** (2.0 / 7.0)
        )

    z = 0.29056 - 0.08775 * acentric_factor_val
    molar_volume = molar_liquid_volume_stp_val * (z**phi)
    return molar_volume * ureg("m**3/mol")


def density(comp: Component, temperature: Quantity) -> Quantity:
    """Calculate density at a given temperature.

    Args:
        comp: Component to evaluate.
        temperature: Temperature in Kelvin.

    Returns:
        Density in kg/m³.
    """
    molar_volume = molar_liquid_vol(comp, temperature)
    return (comp.molecular_weight / molar_volume).to("kg/m**3")


def viscosity_kinematic(comp: Component, temperature: Quantity) -> Quantity:
    """Calculate kinematic viscosity using Dutt's equation.

    Args:
        comp: Component to evaluate.
        temperature: Temperature in Kelvin.

    Returns:
        Kinematic viscosity in m²/s.
    """
    temperature_cels = temperature.to("degC").magnitude
    boiling_temperature_cels = (
        comp.boiling_temperature.to("K").magnitude - 273.15
    )  # Convert K to C

    rhs = -3.0171 + (442.78 + 1.6452 * boiling_temperature_cels) / (
        temperature_cels + 239 - 0.19 * boiling_temperature_cels
    )
    nu = np.exp(rhs)  # mm²/s
    return (nu * ureg("mm**2/s")).to("m**2/s")


def viscosity_dynamic(comp: Component, temperature: Quantity) -> Quantity:
    """Calculate dynamic viscosity.

    Args:
        comp: Component to evaluate.
        temperature: Temperature in Kelvin.

    Returns:
        Dynamic viscosity in Pa·s.
    """
    nu = viscosity_kinematic(comp, temperature)
    rho = density(comp, temperature)
    return (nu * rho).to("Pa*s")


def molar_heat_capacity(comp: Component, temperature: Quantity) -> Quantity:
    """Compute molar heat capacity at a given temperature.

    Args:
        comp: Component to evaluate.
        temperature: Temperature in Kelvin.

    Returns:
        Molar heat capacity in J/mol/K.
    """
    temperature_val = temperature.to("K").magnitude
    theta = (temperature_val - 298) / 700
    coeff_a, coeff_b, coeff_c = comp.heat_capacity_coeffs
    return coeff_a + coeff_b * theta + coeff_c * theta**2


def mass_heat_capacity(comp: Component, temperature: Quantity) -> Quantity:
    """Compute liquid mass heat capacity.

    Args:
        comp: Component to evaluate.
        temperature: Temperature in Kelvin.

    Returns:
        Mass heat capacity in J/kg/K.
    """
    return (molar_heat_capacity(comp, temperature) / comp.molecular_weight).to("J/kg/K")


def saturation_pressure(
    comp: Component,
    temperature: Quantity,
    correlation: Literal["Lee-Kesler", "Ambrose-Walton"] = "Lee-Kesler",
) -> Quantity:
    """Compute saturated vapor pressure.

    Args:
        comp: Component to evaluate.
        temperature: Temperature in Kelvin.
        correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").

    Returns:
        Saturated vapor pressure in Pa.
    """
    temperature_val = temperature.to("K").magnitude
    Tr = temperature_val / comp.critical_temperature.magnitude
    critical_pressure_val = comp.critical_pressure.magnitude
    acentric_factor_val = comp.acentric_factor.magnitude

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
        rhs = np.exp(f0 + acentric_factor_val * f1 + acentric_factor_val**2 * f2)
    else:  # Lee-Kesler
        f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * (Tr**6)
        f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * (Tr**6)
        rhs = np.exp(f0 + acentric_factor_val * f1)

    return (critical_pressure_val * rhs) * ureg.Pa


def latent_heat_vaporization(comp: Component, temperature: Quantity) -> Quantity:
    """Calculate latent heat of vaporization adjusted for temperature.

    Args:
        comp: Component to evaluate.
        temperature: Temperature in Kelvin.

    Returns:
        Latent heat of vaporization in J/kg.
    """
    temperature_val = temperature.to("K").magnitude
    critical_temperature_val = comp.critical_temperature.magnitude
    boiling_temperature_val = comp.boiling_temperature.magnitude
    latent_heat_vaporization_stp_val = comp.latent_heat_vaporization_stp.magnitude

    Tr = temperature_val / critical_temperature_val
    Trb = boiling_temperature_val / critical_temperature_val

    if temperature_val > critical_temperature_val:
        return 0.0 * ureg("J/kg")

    latent_heat = latent_heat_vaporization_stp_val * (
        ((1.0 - Tr) / (1.0 - Trb)) ** 0.38
    )
    return latent_heat * ureg("J/kg")


def surface_tension(
    comp: Component,
    temperature: Quantity,
    correlation: Literal["Brock-Bird", "Pitzer"] = "Brock-Bird",
) -> Quantity:
    """Calculate surface tension at a given temperature.

    Args:
        comp: Component to evaluate.
        temperature: Temperature in Kelvin.
        correlation: Correlation method ("Brock-Bird" or "Pitzer").

    Returns:
        Surface tension in N/m.
    """
    temperature_val = temperature.to("K").magnitude
    critical_temperature_val = comp.critical_temperature.magnitude
    critical_pressure_bar = comp.critical_pressure.to("bar").magnitude
    boiling_temperature_val = comp.boiling_temperature.magnitude
    acentric_factor_val = comp.acentric_factor.magnitude

    Tr = temperature_val / critical_temperature_val

    if correlation.casefold() == "Brock-Bird".casefold():
        Tbr = boiling_temperature_val / critical_temperature_val
        Q = (
            0.1196
            * (1.0 + (Tbr * np.log(critical_pressure_bar / 1.01325)) / (1.0 - Tbr))
            - 0.279
        )
    else:  # Pitzer
        w = acentric_factor_val
        Q = (
            (1.86 + 1.18 * w)
            / 19.05
            * (((3.75 + 0.91 * w) / (0.291 - 0.08 * w)) ** (2.0 / 3.0))
        )

    st = (
        critical_pressure_bar ** (2.0 / 3.0)
        * critical_temperature_val ** (1.0 / 3.0)
        * Q
        * (1 - Tr) ** (11.0 / 9.0)
    )
    return (st * ureg("dyn/cm")).to("N/m")


def thermal_conductivity(comp: Component, temperature: Quantity) -> Quantity:
    """Calculate thermal conductivity at a given temperature (Latini et al.).

    Args:
        comp: Component to evaluate.
        temperature: Temperature in Kelvin.

    Returns:
        Thermal conductivity in W/m/K.
    """
    temperature_val = temperature.to("K").magnitude
    molecular_weight_gmol = comp.molecular_weight.to("g/mol").magnitude
    critical_temperature_val = comp.critical_temperature.magnitude
    boiling_temperature_val = comp.boiling_temperature.magnitude
    family_code = comp.family_code

    alpha = 1.2
    gamma = 0.167
    Tr = temperature_val / critical_temperature_val

    # Family-specific parameters
    if family_code == 1:  # Aromatics
        coeff_star = 0.0346
        beta = 1.0
    elif family_code == 2:  # Cycloparaffins
        coeff_star = 0.0310
        beta = 1.0
    elif family_code == 3:  # Olefins
        coeff_star = 0.0361
        beta = 1.0
    else:  # Saturated hydrocarbons (default)
        coeff_star = 0.00350
        beta = 0.5

    molecular_weight_beta = molecular_weight_gmol**beta
    coeff = (
        coeff_star
        * boiling_temperature_val**alpha
        / (molecular_weight_beta * critical_temperature_val**gamma)
    )
    conductivity = coeff * (1 - Tr) ** 0.38 / (Tr ** (1 / 6))
    return conductivity * ureg("W/m/K")


__all__ = [
    "Component",
    "density",
    "latent_heat_vaporization",
    "mass_heat_capacity",
    "molar_heat_capacity",
    "molar_liquid_vol",
    "saturation_pressure",
    "surface_tension",
    "thermal_conductivity",
    "viscosity_dynamic",
    "viscosity_kinematic",
]
