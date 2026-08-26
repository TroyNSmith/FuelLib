"""Fuel class for Group Contribution Method calculations."""

import json
from pathlib import Path
from typing import Literal, cast

import numpy as np
from pint import Quantity
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import curve_fit

from . import comp as comp_module
from ._data_locator import get_fueldata_dir
from .comp import Component
from .gcm import ConstantinouMethod
from .gcm.core import BaseMethod
from .utils.helpers import mixing_rule
from .utils.units import ureg


class Fuel(BaseModel):
    """A fuel class for Group Contribution Method calculations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    components: list[tuple[Component, float]] = Field(
        description="List of (Component, mass_fraction) tuples"
    )
    method: BaseMethod = Field(default_factory=ConstantinouMethod, repr=False)
    properties: dict = Field(
        default_factory=dict,
        repr=False,
        description="Raw mixture-level validation data from the 'properties' JSON block",
    )
    metadata: dict = Field(
        default_factory=dict,
        repr=False,
        description="Raw fuel-level metadata from the 'metadata' JSON block",
    )

    def model_post_init(self, __context, /) -> None:
        """Assign method to all components after initialization."""
        for comp, _ in self.components:
            if comp.method is None:
                comp.method = self.method

    # --- Basic properties ---

    @property
    def compounds(self) -> list[str]:
        """List of compound names in the mixture."""
        return [comp.name for comp, _ in self.components]

    @property
    def num_compounds(self) -> int:
        """Number of compounds in the mixture."""
        return len(self.components)

    @property
    def initial_mass_fractions(self) -> np.ndarray:
        """Initial mass fractions of each compound."""
        fracs = np.array([frac for _, frac in self.components])
        return fracs / fracs.sum()  # Normalize

    # --- Aggregate GCM properties (as arrays) ---

    @property
    def molecular_weight(self) -> Quantity:
        """Molecular weights in kg/mol. Shape: (num_compounds,)"""
        return ureg("kg/mol") * np.array(
            [comp.molecular_weight.magnitude for comp, _ in self.components]
        )

    @property
    def critical_temperature(self) -> Quantity:
        """Critical temperatures in K. Shape: (num_compounds,)"""
        return (
            np.array(
                [comp.critical_temperature.magnitude for comp, _ in self.components]
            )
            * ureg.kelvin
        )

    @property
    def critical_pressure(self) -> Quantity:
        """Critical pressures in Pa. Shape: (num_compounds,)"""
        return (
            np.array([comp.critical_pressure.magnitude for comp, _ in self.components])
            * ureg.Pa
        )

    @property
    def critical_volume(self) -> Quantity:
        """Critical volumes in m³/mol. Shape: (num_compounds,)"""
        return ureg("m**3/mol") * np.array(
            [comp.critical_volume.magnitude for comp, _ in self.components]
        )

    @property
    def boiling_temperature(self) -> Quantity:
        """Boiling temperatures in K. Shape: (num_compounds,)"""
        return (
            np.array(
                [comp.boiling_temperature.magnitude for comp, _ in self.components]
            )
            * ureg.kelvin
        )

    @property
    def melting_temperature(self) -> Quantity:
        """Melting temperatures in K. Shape: (num_compounds,)"""
        return (
            np.array(
                [comp.melting_temperature.magnitude for comp, _ in self.components]
            )
            * ureg.kelvin
        )

    @property
    def enthalpy_of_formation(self) -> Quantity:
        """Enthalpy of formation in J/mol. Shape: (num_compounds,)"""
        return ureg("J/mol") * np.array(
            [comp.enthalpy_of_formation.magnitude for comp, _ in self.components]
        )

    @property
    def gibbs_free_energy(self) -> Quantity:
        """Gibbs free energy in J/mol. Shape: (num_compounds,)"""
        return ureg("J/mol") * np.array(
            [comp.gibbs_free_energy.magnitude for comp, _ in self.components]
        )

    @property
    def enthalpy_of_vaporization_stp(self) -> Quantity:
        """Enthalpy of vaporization at 298 K in J/mol. Shape: (num_compounds,)"""
        return ureg("J/mol") * np.array(
            [comp.enthalpy_of_vaporization_stp.magnitude for comp, _ in self.components]
        )

    @property
    def latent_heat_vaporization_stp(self) -> Quantity:
        """Latent heat of vaporization at 298 K in J/kg. Shape: (num_compounds,)"""
        return ureg("J/kg") * np.array(
            [comp.latent_heat_vaporization_stp.magnitude for comp, _ in self.components]
        )

    @property
    def heat_capacity_stp(self) -> Quantity:
        """Molar heat capacity at 298 K in J/mol/K. Shape: (num_compounds,)"""
        return ureg("J/mol/K") * np.array(
            [comp.heat_capacity_stp.magnitude for comp, _ in self.components]
        )

    @property
    def heat_capacity_coeff_b(self) -> Quantity:
        """Temperature-correction coefficient B for molar heat capacity in J/mol/K. Shape: (num_compounds,)"""
        return ureg("J/mol/K") * np.array(
            [comp.heat_capacity_coeff_b.magnitude for comp, _ in self.components]
        )

    @property
    def heat_capacity_coeff_c(self) -> Quantity:
        """Temperature-correction coefficient C for molar heat capacity in J/mol/K. Shape: (num_compounds,)"""
        return ureg("J/mol/K") * np.array(
            [comp.heat_capacity_coeff_c.magnitude for comp, _ in self.components]
        )

    @property
    def carbon_number(self) -> Quantity:
        """Carbon number of each compound. Shape: (num_compounds,)"""
        return (
            np.array([comp.carbon_number.magnitude for comp, _ in self.components])
            * ureg.dimensionless
        )

    @property
    def molar_liquid_volume_stp(self) -> Quantity:
        """Molar liquid volume at 298 K in m³/mol. Shape: (num_compounds,)"""
        return ureg("m**3/mol") * np.array(
            [comp.molar_liquid_volume_stp.magnitude for comp, _ in self.components]
        )

    @property
    def acentric_factor(self) -> Quantity:
        """Acentric factors. Shape: (num_compounds,)"""
        return (
            np.array([comp.acentric_factor.magnitude for comp, _ in self.components])
            * ureg.dimensionless
        )

    @property
    def lennard_jones_diameter(self) -> Quantity:
        """Lennard-Jones collision diameters in m. Shape: (num_compounds,)"""
        return (
            np.array(
                [comp.lennard_jones_diameter.magnitude for comp, _ in self.components]
            )
            * ureg.m
        )

    @property
    def epsilon_over_kb(self) -> Quantity:
        """Lennard-Jones well depths in K. Shape: (num_compounds,)"""
        return (
            np.array([comp.epsilon_over_kb.magnitude for comp, _ in self.components])
            * ureg.kelvin
        )

    @property
    def hc_type(self) -> np.ndarray:
        """Hydrocarbon types. Shape: (num_compounds,)"""
        return np.array([comp.hc_type for comp, _ in self.components], dtype=object)

    @property
    def family_code(self) -> np.ndarray:
        """Family codes for thermal conductivity. Shape: (num_compounds,)"""
        return np.array([comp.family_code for comp, _ in self.components], dtype=int)

    @property
    def pelephysics_keys(self) -> np.ndarray | None:
        """PelePhysics keys for each compound, or None if none are set."""
        keys = [comp.pelephysics_key for comp, _ in self.components]
        if all(key is None for key in keys):
            return None
        return np.array(keys, dtype=object)

    # --- Factory methods ---

    @classmethod
    def from_json(cls, path: str | Path, method: BaseMethod | None = None) -> "Fuel":
        """Load a fuel from a JSON file.

        Args:
            path: Path to JSON file.
            method: GCM method to use (default: ConstantinouMethod).

        Returns:
            Fuel instance.
        """
        if method is None:
            method = ConstantinouMethod()

        data = json.loads(Path(path).read_text())

        components = []
        for comp_name, comp_data in data["components"].items():
            comp = Component(
                name=comp_name,
                smiles=comp_data.get("smiles", None),
                pelephysics_key=comp_data.get("pelephysics_key", None),
                decomposition={
                    (group, count)
                    for group, count in comp_data.get("decomposition", {}).items()
                },
                method=method,
            )
            components.append((comp, comp_data["weight_percent"] / 100.0))

        return cls(
            name=Path(path).stem,
            components=components,
            method=method,
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_name(
        cls,
        name: str,
        fuel_data_dir: str | Path | None = None,
        method: BaseMethod | None = None,
    ) -> "Fuel":
        """Load a fuel by name from the embedded or a custom fuel data directory.

        Args:
            name: Fuel name, matching a "<name>.json" file in the fuel data directory.
            fuel_data_dir: Directory containing "<name>.json" files (default: embedded data).
            method: GCM method to use (default: ConstantinouMethod).

        Returns:
            Fuel instance.
        """
        if fuel_data_dir is None:
            fuel_data_dir = get_fueldata_dir()
        return cls.from_json(Path(fuel_data_dir) / f"{name}.json", method=method)


# --- Fraction conversion functions ---


def mean_molecular_weight(fuel: Fuel, mass_fractions: np.ndarray) -> Quantity:
    """Calculate the mean molecular weight of the mixture.

    Args:
        fuel: Fuel to evaluate.
        mass_fractions: Mass fractions of each compound.

    Returns:
        Mean molecular weight in kg/mol.
    """
    molecular_weights = fuel.molecular_weight.magnitude
    if np.sum(mass_fractions) != 0:
        mean_mw = 1 / np.sum(mass_fractions / molecular_weights)
    else:
        mean_mw = 0.0
    return mean_mw * ureg("kg/mol")


def mass_to_mass_fraction(fuel: Fuel, mass: np.ndarray) -> np.ndarray:
    """Calculate mass fractions from the mass of each component.

    Args:
        fuel: Fuel to evaluate.
        mass: Mass of each compound.

    Returns:
        Mass fractions of the compounds (shape: num_compounds,).
    """
    total_mass = np.sum(mass)
    if total_mass != 0:
        return mass / total_mass
    return np.zeros(fuel.num_compounds)


def mass_to_mole_fraction(fuel: Fuel, mass: np.ndarray) -> np.ndarray:
    """Calculate mole fractions from the mass of each component.

    Args:
        fuel: Fuel to evaluate.
        mass: Mass of each compound.

    Returns:
        Mole fractions of the compounds (shape: num_compounds,).
    """
    molecular_weights = fuel.molecular_weight.magnitude
    num_mole = mass / molecular_weights
    total_moles = np.sum(num_mole)
    if total_moles != 0:
        return num_mole / total_moles
    return np.zeros(fuel.num_compounds)


def mole_fraction_to_mass_fraction(
    fuel: Fuel, mole_fractions: np.ndarray
) -> np.ndarray:
    """Calculate mass fractions from mole fractions.

    Args:
        fuel: Fuel to evaluate.
        mole_fractions: Mole fractions of each compound.

    Returns:
        Mass fractions of the compounds (shape: num_compounds,).
    """
    molecular_weights = fuel.molecular_weight.magnitude
    mass = mole_fractions * molecular_weights
    total_mass = np.sum(mass)
    if total_mass != 0:
        return mass / total_mass
    return np.zeros(fuel.num_compounds)


def mass_fraction_to_mole_fraction(
    fuel: Fuel, mass_fractions: np.ndarray
) -> np.ndarray:
    """Calculate mole fractions from mass fractions.

    Args:
        fuel: Fuel to evaluate.
        mass_fractions: Mass fractions of each compound.

    Returns:
        Mole fractions of the compounds (shape: num_compounds,).
    """
    mean_mw = mean_molecular_weight(fuel, mass_fractions).magnitude
    molecular_weights = fuel.molecular_weight.magnitude
    if np.sum(mass_fractions) != 0:
        return mean_mw * mass_fractions / molecular_weights
    return np.zeros(fuel.num_compounds)


# --- Temperature-dependent component functions ---


def density(fuel: Fuel, temperature: Quantity, comp_idx: int | None = None) -> Quantity:
    """Calculate density at a given temperature.

    Args:
        fuel: Fuel to evaluate.
        temperature: Temperature in Kelvin.
        comp_idx: Index of compound (None for all).

    Returns:
        Density in kg/m³.
    """
    if comp_idx is not None:
        comp, _ = fuel.components[comp_idx]
        return comp_module.density(comp, temperature)
    return ureg("kg/m**3") * np.array(
        [
            comp_module.density(comp, temperature).magnitude
            for comp, _ in fuel.components
        ]
    )


def viscosity_kinematic(
    fuel: Fuel, temperature: Quantity, comp_idx: int | None = None
) -> Quantity:
    """Calculate kinematic viscosity at a given temperature.

    Args:
        fuel: Fuel to evaluate.
        temperature: Temperature in Kelvin.
        comp_idx: Index of compound (None for all).

    Returns:
        Kinematic viscosity in m²/s.
    """
    if comp_idx is not None:
        comp, _ = fuel.components[comp_idx]
        return comp_module.viscosity_kinematic(comp, temperature)
    return ureg("m**2/s") * np.array(
        [
            comp_module.viscosity_kinematic(comp, temperature).magnitude
            for comp, _ in fuel.components
        ]
    )


def viscosity_dynamic(
    fuel: Fuel, temperature: Quantity, comp_idx: int | None = None
) -> Quantity:
    """Calculate dynamic viscosity at a given temperature.

    Args:
        fuel: Fuel to evaluate.
        temperature: Temperature in Kelvin.
        comp_idx: Index of compound (None for all).

    Returns:
        Dynamic viscosity in Pa·s.
    """
    if comp_idx is not None:
        comp, _ = fuel.components[comp_idx]
        return comp_module.viscosity_dynamic(comp, temperature)
    return ureg("Pa*s") * np.array(
        [
            comp_module.viscosity_dynamic(comp, temperature).magnitude
            for comp, _ in fuel.components
        ]
    )


def molar_heat_capacity(
    fuel: Fuel, temperature: Quantity, comp_idx: int | None = None
) -> Quantity:
    """Compute molar heat capacity at a given temperature.

    Args:
        fuel: Fuel to evaluate.
        temperature: Temperature in Kelvin.
        comp_idx: Index of compound (None for all).

    Returns:
        Molar heat capacity in J/mol/K.
    """
    if comp_idx is not None:
        comp, _ = fuel.components[comp_idx]
        return comp_module.molar_heat_capacity(comp, temperature)
    return ureg("J/mol/K") * np.array(
        [
            comp_module.molar_heat_capacity(comp, temperature).magnitude
            for comp, _ in fuel.components
        ]
    )


def mass_heat_capacity(
    fuel: Fuel, temperature: Quantity, comp_idx: int | None = None
) -> Quantity:
    """Compute mass heat capacity at a given temperature.

    Args:
        fuel: Fuel to evaluate.
        temperature: Temperature in Kelvin.
        comp_idx: Index of compound (None for all).

    Returns:
        Mass heat capacity in J/kg/K.
    """
    if comp_idx is not None:
        comp, _ = fuel.components[comp_idx]
        return comp_module.mass_heat_capacity(comp, temperature)
    return ureg("J/kg/K") * np.array(
        [
            comp_module.mass_heat_capacity(comp, temperature).magnitude
            for comp, _ in fuel.components
        ]
    )


def saturation_pressure(
    fuel: Fuel,
    temperature: Quantity,
    comp_idx: int | None = None,
    correlation: Literal["Lee-Kesler", "Ambrose-Walton"] = "Lee-Kesler",
) -> Quantity:
    """Compute saturated vapor pressure.

    Args:
        fuel: Fuel to evaluate.
        temperature: Temperature in Kelvin.
        comp_idx: Index of compound (None for all).
        correlation: Correlation method.

    Returns:
        Saturated vapor pressure in Pa.
    """
    if comp_idx is not None:
        comp, _ = fuel.components[comp_idx]
        return comp_module.saturation_pressure(
            comp, temperature, correlation=correlation
        )
    return (
        np.array(
            [
                comp_module.saturation_pressure(
                    comp, temperature, correlation=correlation
                ).magnitude
                for comp, _ in fuel.components
            ]
        )
        * ureg.Pa
    )


def molar_liquid_vol(
    fuel: Fuel, temperature: Quantity, comp_idx: int | None = None
) -> Quantity:
    """Compute molar liquid volume with temperature correction.

    Args:
        fuel: Fuel to evaluate.
        temperature: Temperature in Kelvin.
        comp_idx: Index of compound (None for all).

    Returns:
        Molar liquid volume in m³/mol.
    """
    if comp_idx is not None:
        comp, _ = fuel.components[comp_idx]
        return comp_module.molar_liquid_vol(comp, temperature)
    return ureg("m**3/mol") * np.array(
        [
            comp_module.molar_liquid_vol(comp, temperature).magnitude
            for comp, _ in fuel.components
        ]
    )


def latent_heat_vaporization(
    fuel: Fuel, temperature: Quantity, comp_idx: int | None = None
) -> Quantity:
    """Calculate latent heat of vaporization adjusted for temperature.

    Args:
        fuel: Fuel to evaluate.
        temperature: Temperature in Kelvin.
        comp_idx: Index of compound (None for all).

    Returns:
        Latent heat of vaporization in J/kg.
    """
    if comp_idx is not None:
        comp, _ = fuel.components[comp_idx]
        return comp_module.latent_heat_vaporization(comp, temperature)
    return ureg("J/kg") * np.array(
        [
            comp_module.latent_heat_vaporization(comp, temperature).magnitude
            for comp, _ in fuel.components
        ]
    )


def surface_tension(
    fuel: Fuel,
    temperature: Quantity,
    comp_idx: int | None = None,
    correlation: Literal["Brock-Bird", "Pitzer"] = "Brock-Bird",
) -> Quantity:
    """Calculate surface tension at a given temperature.

    Args:
        fuel: Fuel to evaluate.
        temperature: Temperature in Kelvin.
        comp_idx: Index of compound (None for all).
        correlation: Correlation method.

    Returns:
        Surface tension in N/m.
    """
    if comp_idx is not None:
        comp, _ = fuel.components[comp_idx]
        return comp_module.surface_tension(comp, temperature, correlation=correlation)
    return ureg("N/m") * np.array(
        [
            comp_module.surface_tension(
                comp, temperature, correlation=correlation
            ).magnitude
            for comp, _ in fuel.components
        ]
    )


def thermal_conductivity(
    fuel: Fuel, temperature: Quantity, comp_idx: int | None = None
) -> Quantity:
    """Calculate thermal conductivity at a given temperature.

    Args:
        fuel: Fuel to evaluate.
        temperature: Temperature in Kelvin.
        comp_idx: Index of compound (None for all).

    Returns:
        Thermal conductivity in W/m/K.
    """
    if comp_idx is not None:
        comp, _ = fuel.components[comp_idx]
        return comp_module.thermal_conductivity(comp, temperature)
    return ureg("W/m/K") * np.array(
        [
            comp_module.thermal_conductivity(comp, temperature).magnitude
            for comp, _ in fuel.components
        ]
    )


# --- Mixture functions ---


def mixture_density(
    fuel: Fuel, mass_fractions: np.ndarray, temperature: Quantity
) -> Quantity:
    """Calculate mixture density at a given temperature.

    Args:
        fuel: Fuel to evaluate.
        mass_fractions: Mass fractions of each compound.
        temperature: Temperature in Kelvin.

    Returns:
        Mixture density in kg/m³.
    """
    molecular_weights = fuel.molecular_weight.magnitude
    molar_volumes = molar_liquid_vol(fuel, temperature).magnitude
    rho = mass_fractions @ (molecular_weights / molar_volumes)
    return rho * ureg("kg/m**3")


def mixture_kinematic_viscosity(
    fuel: Fuel,
    mass_fractions: np.ndarray,
    temperature: Quantity,
    correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
) -> Quantity:
    """Calculate kinematic viscosity of the mixture.

    Args:
        fuel: Fuel to evaluate.
        mass_fractions: Mass fractions of each compound.
        temperature: Temperature in Kelvin.
        correlation: Mixing model.

    Returns:
        Mixture kinematic viscosity in m²/s.
    """
    nu_arr = viscosity_kinematic(fuel, temperature).magnitude
    mole_fractions = mass_fraction_to_mole_fraction(fuel, mass_fractions)

    if correlation.casefold() == "Arrhenius".casefold():
        nu = np.exp(np.sum(mole_fractions * np.log(nu_arr)))
    else:  # Kendall-Monroe
        nu = np.sum(mole_fractions * (nu_arr ** (1.0 / 3.0))) ** 3.0

    return nu * ureg("m**2/s")


def mixture_dynamic_viscosity(
    fuel: Fuel,
    mass_fractions: np.ndarray,
    temperature: Quantity,
    correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
) -> Quantity:
    """Calculate dynamic viscosity of the mixture.

    Args:
        fuel: Fuel to evaluate.
        mass_fractions: Mass fractions of each compound.
        temperature: Temperature in Kelvin.
        correlation: Mixing model.

    Returns:
        Mixture dynamic viscosity in Pa·s.
    """
    nu = mixture_kinematic_viscosity(
        fuel, mass_fractions, temperature, correlation=correlation
    )
    rho = mixture_density(fuel, mass_fractions, temperature)
    return (rho * nu).to("Pa*s")


def mixture_vapor_pressure(
    fuel: Fuel,
    mass_fractions: np.ndarray,
    temperature: Quantity,
    correlation: Literal["Lee-Kesler", "Ambrose-Walton"] = "Lee-Kesler",
) -> Quantity:
    """Calculate vapor pressure of the mixture via Raoult's law.

    Args:
        fuel: Fuel to evaluate.
        mass_fractions: Mass fractions of each compound.
        temperature: Temperature in Kelvin.
        correlation: Correlation method.

    Returns:
        Mixture vapor pressure in Pa.
    """
    mole_fractions = mass_fraction_to_mole_fraction(fuel, mass_fractions)
    saturation_pressures = saturation_pressure(
        fuel, temperature, correlation=correlation
    ).magnitude
    vapor_pressure = saturation_pressures @ mole_fractions
    return vapor_pressure * ureg.Pa


def mixture_vapor_pressure_antoine_coeffs(
    fuel: Fuel,
    mass_fractions: np.ndarray,
    temperature_values: np.ndarray | None = None,
    units: Literal["mks", "cgs", "bar", "atm"] = "mks",
    correlation: Literal["Lee-Kesler", "Ambrose-Walton"] = "Lee-Kesler",
) -> tuple[float, float, float, float]:
    """Estimate Antoine coefficients for vapor pressure of the mixture.

    Args:
        fuel: Fuel to evaluate.
        mass_fractions: Mass fractions of each compound.
        temperature_values: Temperature range or nodes for fit in K.
        units: Units for pressure in fit.
        correlation: Correlation method.

    Returns:
        Coefficients A, B, C, D.
    """

    def antoine_eq(temperature_k, coeff_a, coeff_b, coeff_c):
        return coeff_a - coeff_b / (temperature_k + coeff_c)

    unit_factor = {"mks": 1, "bar": 1e5, "atm": 1.01325e5, "cgs": 0.1}.get(
        units.lower(), 1
    )

    if temperature_values is None:
        mole_fractions = mass_fraction_to_mole_fraction(fuel, mass_fractions)
        boiling_temperature_mix = mixing_rule(
            fuel.boiling_temperature.magnitude, mole_fractions
        )
        temperature_k = np.linspace(273.15, np.min(boiling_temperature_mix), 20)
    elif len(temperature_values) == 2:
        temperature_k = np.linspace(temperature_values[0], temperature_values[1], 20)
    else:
        temperature_k = temperature_values

    pressure_values = np.array(
        [
            mixture_vapor_pressure(
                fuel, mass_fractions, t * ureg.K, correlation=correlation
            ).magnitude
            / unit_factor
            for t in temperature_k
        ]
    )
    log_pressure = np.log10(pressure_values)
    popt, _ = curve_fit(antoine_eq, temperature_k, log_pressure, p0=[1, 1e3, -1])
    coeff_a, coeff_b, coeff_c = popt

    return float(coeff_a), float(coeff_b), float(coeff_c), float(unit_factor)


def mixture_surface_tension(
    fuel: Fuel,
    mass_fractions: np.ndarray,
    temperature: Quantity,
    correlation: Literal["Brock-Bird", "Pitzer"] = "Brock-Bird",
) -> Quantity:
    """Calculate surface tension of the mixture.

    Args:
        fuel: Fuel to evaluate.
        mass_fractions: Mass fractions of each compound.
        temperature: Temperature in Kelvin.
        correlation: Correlation method.

    Returns:
        Mixture surface tension in N/m.
    """
    mole_fractions = mass_fraction_to_mole_fraction(fuel, mass_fractions)
    surface_tension_arr = surface_tension(
        fuel, temperature, correlation=correlation
    ).magnitude
    st = mixing_rule(surface_tension_arr, mole_fractions, "arithmetic")
    return st * ureg("N/m")


def mixture_thermal_conductivity(
    fuel: Fuel, mass_fractions: np.ndarray, temperature: Quantity
) -> Quantity:
    """Calculate thermal conductivity of the mixture.

    Args:
        fuel: Fuel to evaluate.
        mass_fractions: Mass fractions of each compound.
        temperature: Temperature in Kelvin.

    Returns:
        Thermal conductivity in W/m/K.
    """
    thermal_conductivity_arr = thermal_conductivity(fuel, temperature).magnitude
    conductivity = np.sum(mass_fractions * thermal_conductivity_arr ** (-2)) ** (-0.5)
    return conductivity * ureg("W/m/K")


# --- Special functions ---


def diffusion_coeff(
    fuel: Fuel,
    pressure: Quantity,
    temperature: Quantity,
    collision_diameter_gas: Quantity = 3.62e-10 * ureg.m,
    epsilon_over_kb_gas: Quantity = 97.0 * ureg.kelvin,
    molecular_weight_gas: Quantity = 28.97e-3 * ureg.kg / ureg.mol,
    correlation: Literal["Tee", "Wilke"] = "Tee",
) -> Quantity:
    """Compute diffusion coefficients using Lennard-Jones parameters.

    Args:
        fuel: Fuel to evaluate.
        pressure: Pressure in Pa.
        temperature: Temperature in Kelvin.
        collision_diameter_gas: Collision diameter of ambient gas in m.
        epsilon_over_kb_gas: Well depth over Boltzmann constant in K.
        molecular_weight_gas: Mean molecular weight of ambient gas in kg/mol.
        correlation: Method for collision diameter/well depth ("Tee" or "Wilke").

    Returns:
        Diffusion coefficient in m²/s.
    """
    temperature_val = temperature.to("K").magnitude
    pressure_bar = pressure.to("bar").magnitude

    if correlation.casefold() == "Tee".casefold():
        diameter_i = fuel.lennard_jones_diameter.to("angstrom").magnitude
        epsilon_over_kb_i = fuel.epsilon_over_kb.magnitude
    else:  # Wilke
        molar_volume_boiling_i = np.array(
            [
                comp_module.molar_liquid_vol(comp, comp.boiling_temperature)
                .to("cm**3/mol")
                .magnitude
                for comp, _ in fuel.components
            ]
        )
        diameter_i = 1.18 * molar_volume_boiling_i ** (1 / 3)  # Angstroms
        epsilon_over_kb_i = 1.15 * fuel.boiling_temperature.magnitude

    diameter_gas_angstrom = collision_diameter_gas.to("angstrom").magnitude
    diameter_ab_i = (diameter_gas_angstrom + diameter_i) / 2
    epsilon_ab_over_kb_i = (epsilon_over_kb_gas.magnitude * epsilon_over_kb_i) ** 0.5

    # Dimensionless collision integral
    reduced_temperature_i = temperature_val / epsilon_ab_over_kb_i
    coeff_a, coeff_b, coeff_c, coeff_d, coeff_e, coeff_f, coeff_g, coeff_h = (
        1.06036,
        0.15610,
        0.193,
        0.47635,
        1.03587,
        1.52996,
        1.76474,
        3.89411,
    )
    collision_integral_i = (
        coeff_a / (reduced_temperature_i**coeff_b)
        + coeff_c / np.exp(coeff_d * reduced_temperature_i)
        + coeff_e / np.exp(coeff_f * reduced_temperature_i)
        + coeff_g / np.exp(coeff_h * reduced_temperature_i)
    )

    molecular_weight_gas_gmol = molecular_weight_gas.to("g/mol").magnitude
    molecular_weight_i_gmol = fuel.molecular_weight.to("g/mol").magnitude
    mean_molecular_weight_ab_i = (
        2
        * (molecular_weight_i_gmol * molecular_weight_gas_gmol)
        / (molecular_weight_i_gmol + molecular_weight_gas_gmol)
    )

    diffusion_coeff_i = (
        1e-3
        * (3.03 - 0.98 / (mean_molecular_weight_ab_i**0.5))
        * (temperature_val**1.5)
        / (
            pressure_bar
            * mean_molecular_weight_ab_i**0.5
            * diameter_ab_i**2
            * collision_integral_i
        )
    )  # cm²/s
    return (diffusion_coeff_i * ureg("cm**2/s")).to("m**2/s")


def saturation_pressure_antoine_coeffs(
    fuel: Fuel,
    temperature_values: np.ndarray | None = None,
    units: Literal["mks", "cgs", "bar", "atm"] = "mks",
    correlation: Literal["Lee-Kesler", "Ambrose-Walton"] = "Lee-Kesler",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate Antoine coefficients for vapor pressure of individual compounds.

    Args:
        fuel: Fuel to evaluate.
        temperature_values: Temperature range or nodes for fit in K (default [273.15, boiling_temperature_i]).
        units: Units for pressure in fit.
        correlation: Correlation method.

    Returns:
        Coefficients A, B, C, D as arrays.
    """

    def antoine_eq(temperature_k, coeff_a, coeff_b, coeff_c):
        return coeff_a - coeff_b / (temperature_k + coeff_c)

    unit_factor = {"mks": 1, "bar": 1e5, "atm": 1.01325e5, "cgs": 0.1}.get(
        units.lower(), 1
    )

    coeff_a_arr = np.zeros(fuel.num_compounds)
    coeff_b_arr = np.zeros(fuel.num_compounds)
    coeff_c_arr = np.zeros(fuel.num_compounds)

    for i, (comp, _) in enumerate(fuel.components):
        if temperature_values is None:
            temperature_k = np.linspace(273.15, comp.boiling_temperature.magnitude, 20)
        elif len(temperature_values) == 2:
            temperature_k = np.linspace(
                temperature_values[0], temperature_values[1], 20
            )
        else:
            temperature_k = temperature_values

        pressure_values = np.array(
            [
                comp_module.saturation_pressure(
                    comp, t * ureg.K, correlation=correlation
                ).magnitude
                / unit_factor
                for t in temperature_k
            ]
        )
        log_pressure = np.log10(pressure_values)
        popt, _ = curve_fit(antoine_eq, temperature_k, log_pressure, p0=[1, 1e3, -1])
        coeff_a_arr[i], coeff_b_arr[i], coeff_c_arr[i] = popt

    unit_factor_arr = np.full(fuel.num_compounds, unit_factor)
    return coeff_a_arr, coeff_b_arr, coeff_c_arr, unit_factor_arr


# --- Validation data ---


def experimental_property(fuel: Fuel, prop_name: str) -> tuple[Quantity, Quantity]:
    """Fetch experimental validation data for the mixture from the loaded JSON.

    Args:
        fuel: Fuel to evaluate.
        prop_name: Name of the property block (e.g. "Density", "Viscosity").

    Returns:
        Tuple of (temperature in K, property values with their JSON-declared unit).
    """
    if prop_name not in fuel.properties:
        raise KeyError(f"No experimental data for '{prop_name}' in fuel '{fuel.name}'")

    # Map JSON temperature unit strings to pint-recognized unit names
    temp_unit_map = {"C": "degC", "F": "degF", "K": "kelvin"}
    temp_unit = temp_unit_map[fuel.properties.get("temperature_unit", "K")]

    entry = fuel.properties[prop_name]
    temperature_arr, val_arr = np.array(entry["data"]).T

    # degC/degF are offset units; must build the Quantity directly rather than
    # multiplying, which pint disallows for offset units.
    temperature = cast(Quantity, Quantity(temperature_arr, temp_unit).to("K"))
    values = ureg(entry["unit"]) * val_arr
    return temperature, values


__all__ = [
    "Fuel",
    "density",
    "diffusion_coeff",
    "experimental_property",
    "latent_heat_vaporization",
    "mass_fraction_to_mole_fraction",
    "mass_heat_capacity",
    "mass_to_mass_fraction",
    "mass_to_mole_fraction",
    "mean_molecular_weight",
    "mixture_density",
    "mixture_dynamic_viscosity",
    "mixture_kinematic_viscosity",
    "mixture_surface_tension",
    "mixture_thermal_conductivity",
    "mixture_vapor_pressure",
    "mixture_vapor_pressure_antoine_coeffs",
    "molar_heat_capacity",
    "molar_liquid_vol",
    "mole_fraction_to_mass_fraction",
    "saturation_pressure",
    "saturation_pressure_antoine_coeffs",
    "surface_tension",
    "thermal_conductivity",
    "viscosity_dynamic",
    "viscosity_kinematic",
]
