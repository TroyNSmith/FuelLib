"""Fuel class for Group Contribution Method calculations."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pint import Quantity
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import curve_fit

from .comp import Component
from .gcm import ConstantinouMethod
from .gcm.core import BaseMethod
from .utility import mixing_rule
from .utils.units import ureg


class FuelNew(BaseModel):
    """A fuel class for Group Contribution Method calculations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    components: list[tuple[Component, float]] = Field(
        description="List of (Component, mass_fraction) tuples"
    )
    method: BaseMethod = Field(default_factory=ConstantinouMethod, repr=False)

    def model_post_init(self, __context) -> None:
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
    def Y_0(self) -> np.ndarray:
        """Initial mass fractions of each compound."""
        fracs = np.array([frac for _, frac in self.components])
        return fracs / fracs.sum()  # Normalize

    # --- Aggregate GCM properties (as arrays) ---

    @property
    def MW(self) -> Quantity:
        """Molecular weights in kg/mol. Shape: (num_compounds,)"""
        return np.array([comp.MW.magnitude for comp, _ in self.components]) * ureg(
            "kg/mol"
        )

    @property
    def Tc(self) -> Quantity:
        """Critical temperatures in K. Shape: (num_compounds,)"""
        return (
            np.array([comp.Tc.magnitude for comp, _ in self.components]) * ureg.kelvin
        )

    @property
    def Pc(self) -> Quantity:
        """Critical pressures in Pa. Shape: (num_compounds,)"""
        return np.array([comp.Pc.magnitude for comp, _ in self.components]) * ureg.Pa

    @property
    def Vc(self) -> Quantity:
        """Critical volumes in m³/mol. Shape: (num_compounds,)"""
        return np.array([comp.Vc.magnitude for comp, _ in self.components]) * ureg(
            "m**3/mol"
        )

    @property
    def Tb(self) -> Quantity:
        """Boiling temperatures in K. Shape: (num_compounds,)"""
        return (
            np.array([comp.Tb.magnitude for comp, _ in self.components]) * ureg.kelvin
        )

    @property
    def Tm(self) -> Quantity:
        """Melting temperatures in K. Shape: (num_compounds,)"""
        return (
            np.array([comp.Tm.magnitude for comp, _ in self.components]) * ureg.kelvin
        )

    @property
    def Hf(self) -> Quantity:
        """Enthalpy of formation in J/mol. Shape: (num_compounds,)"""
        return np.array([comp.Hf.magnitude for comp, _ in self.components]) * ureg(
            "J/mol"
        )

    @property
    def Gf(self) -> Quantity:
        """Gibbs free energy in J/mol. Shape: (num_compounds,)"""
        return np.array([comp.Gf.magnitude for comp, _ in self.components]) * ureg(
            "J/mol"
        )

    @property
    def Hv_stp(self) -> Quantity:
        """Enthalpy of vaporization at 298 K in J/mol. Shape: (num_compounds,)"""
        return np.array([comp.Hv_stp.magnitude for comp, _ in self.components]) * ureg(
            "J/mol"
        )

    @property
    def Lv_stp(self) -> Quantity:
        """Latent heat of vaporization at 298 K in J/kg. Shape: (num_compounds,)"""
        return np.array([comp.Lv_stp.magnitude for comp, _ in self.components]) * ureg(
            "J/kg"
        )

    @property
    def Cp_stp(self) -> Quantity:
        """Molar specific heat at 298 K in J/mol/K. Shape: (num_compounds,)"""
        return np.array([comp.Cp_stp.magnitude for comp, _ in self.components]) * ureg(
            "J/mol/K"
        )

    @property
    def Vm_stp(self) -> Quantity:
        """Molar liquid volume at 298 K in m³/mol. Shape: (num_compounds,)"""
        return np.array([comp.Vm_stp.magnitude for comp, _ in self.components]) * ureg(
            "m**3/mol"
        )

    @property
    def omega(self) -> Quantity:
        """Acentric factors. Shape: (num_compounds,)"""
        return (
            np.array([comp.omega.magnitude for comp, _ in self.components])
            * ureg.dimensionless
        )

    @property
    def sigma(self) -> Quantity:
        """Lennard-Jones collision diameters in m. Shape: (num_compounds,)"""
        return np.array([comp.sigma.magnitude for comp, _ in self.components]) * ureg.m

    @property
    def epsilonByKB(self) -> Quantity:
        """Lennard-Jones well depths in K. Shape: (num_compounds,)"""
        return (
            np.array([comp.epsilonByKB.magnitude for comp, _ in self.components])
            * ureg.kelvin
        )

    @property
    def hc_type(self) -> np.ndarray:
        """Hydrocarbon types (may be None for components)."""
        return np.array([comp.hc_type for comp, _ in self.components], dtype=object)

    @property
    def fam(self) -> np.ndarray:
        """Family codes for thermal conductivity (may be None for components)."""
        return np.array([comp.fam for comp, _ in self.components], dtype=object)

    # --- Fraction conversion methods ---

    def mean_molecular_weight(self, Yi: np.ndarray) -> Quantity:
        """Calculate the mean molecular weight of the mixture.

        Args:
            Yi: Mass fractions of each compound.

        Returns:
            Mean molecular weight in kg/mol.
        """
        MW_arr = self.MW.magnitude
        if np.sum(Yi) != 0:
            Mbar = 1 / np.sum(Yi / MW_arr)
        else:
            Mbar = 0.0
        return Mbar * ureg("kg/mol")

    def mass2Y(self, mass: np.ndarray) -> np.ndarray:
        """Calculate mass fractions from the mass of each component.

        Args:
            mass: Mass of each compound.

        Returns:
            Mass fractions of the compounds (shape: num_compounds,).
        """
        total_mass = np.sum(mass)
        if total_mass != 0:
            return mass / total_mass
        return np.zeros(self.num_compounds)

    def mass2X(self, mass: np.ndarray) -> np.ndarray:
        """Calculate mole fractions from the mass of each component.

        Args:
            mass: Mass of each compound.

        Returns:
            Mole fractions of the compounds (shape: num_compounds,).
        """
        MW_arr = self.MW.magnitude
        num_mole = mass / MW_arr
        total_moles = np.sum(num_mole)
        if total_moles != 0:
            return num_mole / total_moles
        return np.zeros(self.num_compounds)

    def X2Y(self, Xi: np.ndarray) -> np.ndarray:
        """Calculate mass fractions from mole fractions.

        Args:
            Xi: Mole fractions of each compound.

        Returns:
            Mass fractions of the compounds (shape: num_compounds,).
        """
        MW_arr = self.MW.magnitude
        mass = Xi * MW_arr
        total_mass = np.sum(mass)
        if total_mass != 0:
            return mass / total_mass
        return np.zeros(self.num_compounds)

    def Y2X(self, Yi: np.ndarray) -> np.ndarray:
        """Calculate mole fractions from mass fractions.

        Args:
            Yi: Mass fractions of each compound.

        Returns:
            Mole fractions of the compounds (shape: num_compounds,).
        """
        Mbar = self.mean_molecular_weight(Yi).magnitude
        MW_arr = self.MW.magnitude
        if np.sum(Yi) != 0:
            return Mbar * Yi / MW_arr
        return np.zeros(self.num_compounds)

    # --- T-dependent component methods ---

    def density(self, T: Quantity, comp_idx: int | None = None) -> Quantity:
        """Calculate density at temperature T.

        Args:
            T: Temperature in Kelvin.
            comp_idx: Index of compound (None for all).

        Returns:
            Density in kg/m³.
        """
        if comp_idx is not None:
            comp, _ = self.components[comp_idx]
            return comp.density(T)
        return np.array(
            [comp.density(T).magnitude for comp, _ in self.components]
        ) * ureg("kg/m**3")

    def viscosity_kinematic(self, T: Quantity, comp_idx: int | None = None) -> Quantity:
        """Calculate kinematic viscosity at temperature T.

        Args:
            T: Temperature in Kelvin.
            comp_idx: Index of compound (None for all).

        Returns:
            Kinematic viscosity in m²/s.
        """
        if comp_idx is not None:
            comp, _ = self.components[comp_idx]
            return comp.viscosity_kinematic(T)
        return np.array(
            [comp.viscosity_kinematic(T).magnitude for comp, _ in self.components]
        ) * ureg("m**2/s")

    def viscosity_dynamic(self, T: Quantity, comp_idx: int | None = None) -> Quantity:
        """Calculate dynamic viscosity at temperature T.

        Args:
            T: Temperature in Kelvin.
            comp_idx: Index of compound (None for all).

        Returns:
            Dynamic viscosity in Pa·s.
        """
        if comp_idx is not None:
            comp, _ = self.components[comp_idx]
            return comp.viscosity_dynamic(T)
        return np.array(
            [comp.viscosity_dynamic(T).magnitude for comp, _ in self.components]
        ) * ureg("Pa*s")

    def Cp(self, T: Quantity, comp_idx: int | None = None) -> Quantity:
        """Compute molar specific heat capacity at temperature T.

        Args:
            T: Temperature in Kelvin.
            comp_idx: Index of compound (None for all).

        Returns:
            Molar specific heat capacity in J/mol/K.
        """
        if comp_idx is not None:
            comp, _ = self.components[comp_idx]
            return comp.Cp(T)
        return np.array([comp.Cp(T).magnitude for comp, _ in self.components]) * ureg(
            "J/mol/K"
        )

    def Cl(self, T: Quantity, comp_idx: int | None = None) -> Quantity:
        """Compute mass specific heat capacity at temperature T.

        Args:
            T: Temperature in Kelvin.
            comp_idx: Index of compound (None for all).

        Returns:
            Mass specific heat capacity in J/kg/K.
        """
        if comp_idx is not None:
            comp, _ = self.components[comp_idx]
            return comp.Cl(T)
        return np.array([comp.Cl(T).magnitude for comp, _ in self.components]) * ureg(
            "J/kg/K"
        )

    def psat(
        self,
        T: Quantity,
        comp_idx: int | None = None,
        correlation: Literal["Lee-Kesler", "Ambrose-Walton"] = "Lee-Kesler",
    ) -> Quantity:
        """Compute saturated vapor pressure.

        Args:
            T: Temperature in Kelvin.
            comp_idx: Index of compound (None for all).
            correlation: Correlation method.

        Returns:
            Saturated vapor pressure in Pa.
        """
        if comp_idx is not None:
            comp, _ = self.components[comp_idx]
            return comp.psat(T, correlation=correlation)
        return (
            np.array(
                [
                    comp.psat(T, correlation=correlation).magnitude
                    for comp, _ in self.components
                ]
            )
            * ureg.Pa
        )

    def molar_liquid_vol(self, T: Quantity, comp_idx: int | None = None) -> Quantity:
        """Compute molar liquid volume with temperature correction.

        Args:
            T: Temperature in Kelvin.
            comp_idx: Index of compound (None for all).

        Returns:
            Molar liquid volume in m³/mol.
        """
        if comp_idx is not None:
            comp, _ = self.components[comp_idx]
            return comp.molar_liquid_vol(T)
        return np.array(
            [comp.molar_liquid_vol(T).magnitude for comp, _ in self.components]
        ) * ureg("m**3/mol")

    def latent_heat_vaporization(
        self, T: Quantity, comp_idx: int | None = None
    ) -> Quantity:
        """Calculate latent heat of vaporization adjusted for temperature.

        Args:
            T: Temperature in Kelvin.
            comp_idx: Index of compound (None for all).

        Returns:
            Latent heat of vaporization in J/kg.
        """
        if comp_idx is not None:
            comp, _ = self.components[comp_idx]
            return comp.latent_heat_vaporization(T)
        return np.array(
            [comp.latent_heat_vaporization(T).magnitude for comp, _ in self.components]
        ) * ureg("J/kg")

    def surface_tension(
        self,
        T: Quantity,
        comp_idx: int | None = None,
        correlation: Literal["Brock-Bird", "Pitzer"] = "Brock-Bird",
    ) -> Quantity:
        """Calculate surface tension at temperature T.

        Args:
            T: Temperature in Kelvin.
            comp_idx: Index of compound (None for all).
            correlation: Correlation method.

        Returns:
            Surface tension in N/m.
        """
        if comp_idx is not None:
            comp, _ = self.components[comp_idx]
            return comp.surface_tension(T, correlation=correlation)
        return np.array(
            [
                comp.surface_tension(T, correlation=correlation).magnitude
                for comp, _ in self.components
            ]
        ) * ureg("N/m")

    def thermal_conductivity(
        self, T: Quantity, comp_idx: int | None = None
    ) -> Quantity:
        """Calculate thermal conductivity at temperature T.

        Args:
            T: Temperature in Kelvin.
            comp_idx: Index of compound (None for all).

        Returns:
            Thermal conductivity in W/m/K.
        """
        if comp_idx is not None:
            comp, _ = self.components[comp_idx]
            return comp.thermal_conductivity(T)
        return np.array(
            [comp.thermal_conductivity(T).magnitude for comp, _ in self.components]
        ) * ureg("W/m/K")

    # --- Mixture methods ---

    def mixture_density(self, Yi: np.ndarray, T: Quantity) -> Quantity:
        """Calculate mixture density at a given temperature.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature in Kelvin.

        Returns:
            Mixture density in kg/m³.
        """
        MW_arr = self.MW.magnitude
        Vm_arr = self.molar_liquid_vol(T).magnitude
        rho = Yi @ (MW_arr / Vm_arr)
        return rho * ureg("kg/m**3")

    def mixture_kinematic_viscosity(
        self,
        Yi: np.ndarray,
        T: Quantity,
        correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> Quantity:
        """Calculate kinematic viscosity of the mixture.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature in Kelvin.
            correlation: Mixing model.

        Returns:
            Mixture kinematic viscosity in m²/s.
        """
        nu_arr = self.viscosity_kinematic(T).magnitude
        Xi = self.Y2X(Yi)

        if correlation.casefold() == "Arrhenius".casefold():
            nu = np.exp(np.sum(Xi * np.log(nu_arr)))
        else:  # Kendall-Monroe
            nu = np.sum(Xi * (nu_arr ** (1.0 / 3.0))) ** 3.0

        return nu * ureg("m**2/s")

    def mixture_dynamic_viscosity(
        self,
        Yi: np.ndarray,
        T: Quantity,
        correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> Quantity:
        """Calculate dynamic viscosity of the mixture.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature in Kelvin.
            correlation: Mixing model.

        Returns:
            Mixture dynamic viscosity in Pa·s.
        """
        nu = self.mixture_kinematic_viscosity(Yi, T, correlation=correlation)
        rho = self.mixture_density(Yi, T)
        return (rho * nu).to("Pa*s")

    def mixture_vapor_pressure(
        self,
        Yi: np.ndarray,
        T: Quantity,
        correlation: Literal["Lee-Kesler", "Ambrose-Walton"] = "Lee-Kesler",
    ) -> Quantity:
        """Calculate vapor pressure of the mixture via Raoult's law.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature in Kelvin.
            correlation: Correlation method.

        Returns:
            Mixture vapor pressure in Pa.
        """
        Xi = self.Y2X(Yi)
        p_sat_arr = self.psat(T, correlation=correlation).magnitude
        p_v = p_sat_arr @ Xi
        return p_v * ureg.Pa

    def mixture_surface_tension(
        self,
        Yi: np.ndarray,
        T: Quantity,
        correlation: Literal["Brock-Bird", "Pitzer"] = "Brock-Bird",
    ) -> Quantity:
        """Calculate surface tension of the mixture.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature in Kelvin.
            correlation: Correlation method.

        Returns:
            Mixture surface tension in N/m.
        """
        Xi = self.Y2X(Yi)
        st_arr = self.surface_tension(T, correlation=correlation).magnitude
        st = mixing_rule(st_arr, Xi, "arithmetic")
        return st * ureg("N/m")

    def mixture_thermal_conductivity(self, Yi: np.ndarray, T: Quantity) -> Quantity:
        """Calculate thermal conductivity of the mixture.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature in Kelvin.

        Returns:
            Thermal conductivity in W/m/K.
        """
        tc_arr = self.thermal_conductivity(T).magnitude
        tc = np.sum(Yi * tc_arr ** (-2)) ** (-0.5)
        return tc * ureg("W/m/K")

    # --- Special methods ---

    def diffusion_coeff(
        self,
        p: Quantity,
        T: Quantity,
        sigma_gas: Quantity = 3.62e-10 * ureg.m,
        epsilonByKB_gas: Quantity = 97.0 * ureg.kelvin,
        MW_gas: Quantity = 28.97e-3 * ureg("kg/mol"),
        correlation: Literal["Tee", "Wilke"] = "Tee",
    ) -> Quantity:
        """Compute diffusion coefficients using Lennard-Jones parameters.

        Args:
            p: Pressure in Pa.
            T: Temperature in Kelvin.
            sigma_gas: Collision diameter of ambient gas in m.
            epsilonByKB_gas: Well depth over Boltzmann constant in K.
            MW_gas: Mean molecular weight of ambient gas in kg/mol.
            correlation: Method for sigma/epsilon ("Tee" or "Wilke").

        Returns:
            Diffusion coefficient in m²/s.
        """
        T_val = T.to("K").magnitude
        p_bar = p.to("bar").magnitude

        if correlation.casefold() == "Tee".casefold():
            sigma_i = self.sigma.to("angstrom").magnitude
            epsilonByKB_i = self.epsilonByKB.magnitude
        else:  # Wilke
            Vmb_i = np.array(
                [
                    comp.molar_liquid_vol(comp.Tb).to("cm**3/mol").magnitude
                    for comp, _ in self.components
                ]
            )
            sigma_i = 1.18 * Vmb_i ** (1 / 3)  # Angstroms
            epsilonByKB_i = 1.15 * self.Tb.magnitude

        sigma_gas_A = sigma_gas.to("angstrom").magnitude
        sigmaAB_i = (sigma_gas_A + sigma_i) / 2
        epsilonAB_byKB_i = (epsilonByKB_gas.magnitude * epsilonByKB_i) ** 0.5

        # Dimensionless collision integral
        Tstar_i = T_val / epsilonAB_byKB_i
        A, B, C, D, E, F, G, H = (
            1.06036,
            0.15610,
            0.193,
            0.47635,
            1.03587,
            1.52996,
            1.76474,
            3.89411,
        )
        omegaD_i = (
            A / (Tstar_i**B)
            + C / np.exp(D * Tstar_i)
            + E / np.exp(F * Tstar_i)
            + G / np.exp(H * Tstar_i)
        )

        MW_gas_gmol = MW_gas.to("g/mol").magnitude
        MW_i_gmol = self.MW.to("g/mol").magnitude
        M_AB_i = 2 * (MW_i_gmol * MW_gas_gmol) / (MW_i_gmol + MW_gas_gmol)

        D_AB_i = (
            1e-3
            * (3.03 - 0.98 / (M_AB_i**0.5))
            * (T_val**1.5)
            / (p_bar * M_AB_i**0.5 * sigmaAB_i**2 * omegaD_i)
        )  # cm²/s
        return (D_AB_i * ureg("cm**2/s")).to("m**2/s")

    def psat_antoine_coeffs(
        self,
        Tvals: np.ndarray | None = None,
        units: Literal["mks", "cgs", "bar", "atm"] = "mks",
        correlation: Literal["Lee-Kesler", "Ambrose-Walton"] = "Lee-Kesler",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Estimate Antoine coefficients for vapor pressure of individual compounds.

        Args:
            Tvals: Temperature range or nodes for fit in K (default [273.15, Tb_i]).
            units: Units for pressure in fit.
            correlation: Correlation method.

        Returns:
            Coefficients A, B, C, D as arrays.
        """

        def antoine_eq(T, A, B, C):
            return A - B / (T + C)

        D_factor = {"mks": 1, "bar": 1e5, "atm": 1.01325e5, "cgs": 0.1}.get(
            units.lower(), 1
        )

        A_arr = np.zeros(self.num_compounds)
        B_arr = np.zeros(self.num_compounds)
        C_arr = np.zeros(self.num_compounds)

        for i, (comp, _) in enumerate(self.components):
            if Tvals is None:
                T = np.linspace(273.15, comp.Tb.magnitude, 20)
            elif len(Tvals) == 2:
                T = np.linspace(Tvals[0], Tvals[1], 20)
            else:
                T = Tvals

            Pvals = np.array(
                [
                    comp.psat(t * ureg.K, correlation=correlation).magnitude / D_factor
                    for t in T
                ]
            )
            logP = np.log10(Pvals)
            popt, _ = curve_fit(antoine_eq, T, logP, p0=[1, 1e3, -1])
            A_arr[i], B_arr[i], C_arr[i] = popt

        D_arr = np.full(self.num_compounds, D_factor)
        return A_arr, B_arr, C_arr, D_arr

    def mixture_vapor_pressure_antoine_coeffs(
        self,
        Yi: np.ndarray,
        Tvals: np.ndarray | None = None,
        units: Literal["mks", "cgs", "bar", "atm"] = "mks",
        correlation: Literal["Lee-Kesler", "Ambrose-Walton"] = "Lee-Kesler",
    ) -> tuple[float, float, float, float]:
        """Estimate Antoine coefficients for vapor pressure of the mixture.

        Args:
            Yi: Mass fractions of each compound.
            Tvals: Temperature range or nodes for fit in K.
            units: Units for pressure in fit.
            correlation: Correlation method.

        Returns:
            Coefficients A, B, C, D.
        """

        def antoine_eq(T, A, B, C):
            return A - B / (T + C)

        D_factor = {"mks": 1, "bar": 1e5, "atm": 1.01325e5, "cgs": 0.1}.get(
            units.lower(), 1
        )

        if Tvals is None:
            X = self.Y2X(Yi)
            Tb_mix = mixing_rule(self.Tb.magnitude, X)
            T = np.linspace(273.15, np.min(Tb_mix), 20)
        elif len(Tvals) == 2:
            T = np.linspace(Tvals[0], Tvals[1], 20)
        else:
            T = Tvals

        Pvals = np.array(
            [
                self.mixture_vapor_pressure(
                    Yi, t * ureg.K, correlation=correlation
                ).magnitude
                / D_factor
                for t in T
            ]
        )
        logP = np.log10(Pvals)
        popt, _ = curve_fit(antoine_eq, T, logP, p0=[1, 1e3, -1])
        A, B, C = popt

        return float(A), float(B), float(C), float(D_factor)

    # --- Factory methods ---

    @classmethod
    def from_json(cls, path: str | Path, method: BaseMethod | None = None) -> "FuelNew":
        """Load a fuel from a JSON file.

        Args:
            path: Path to JSON file.
            method: GCM method to use (default: ConstantinouMethod).

        Returns:
            FuelNew instance.
        """
        if method is None:
            method = ConstantinouMethod()

        components = []
        for comp_name, comp_data in json.loads(Path(path).read_text()).items():
            comp = Component(
                name=comp_name,
                smiles=comp_data.get("smiles", None),
                decomposition={
                    (group, count)
                    for group, count in comp_data.get("decomposition", {}).items()
                },
                method=method,
            )
            components.append((comp, comp_data["weight_percent"] / 100.0))

        return cls(name=Path(path).stem, components=components, method=method)
