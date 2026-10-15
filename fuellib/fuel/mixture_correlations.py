"""Mixin providing mixture correlations and composition conversions for `Fuel`."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import curve_fit, root

from .. import constants
from ..utility import arithmetic
from ..utils import Units, types
from .component_correlations import ComponentCorrelationsMixin, _antoine_eq


def _liaw_chiu_flash_point(
    Xi: types.Array1D,
    Tf_i: types.Array1D,
    Tc: types.Array1D,
    Pc: types.Array1D,
    omega: types.Array1D,
) -> float:
    """Solve the ideal Liaw-Chiu mixture flash-point criterion.

    Solves for the mixture flash point T such that
    sum(Xi * psat(T) / psat(Tf_i)) = 1, where psat is evaluated with the
    Lee-Kesler correlation for each compound at its own critical properties.

    Args:
        Xi: Mole fractions of each compound in the mixture.
        Tf_i: Flash point of each compound in K.
        Tc: Critical temperature of each compound in K.
        Pc: Critical pressure of each compound.
        omega: Acentric factor of each compound.

    Returns:
        Mixture flash point in K.

    Raises:
        RuntimeError: If the root finder fails to converge.
    """

    def _psat_lee_kesler(T: types.Array1D | float) -> types.Array1D:
        """Lee-Kesler saturated vapor pressure for each compound (units of `Pc`).

        Args:
            T: Temperature in K.

        Returns:
            Saturated vapor pressure.
        """
        Tr = T / Tc
        f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * (Tr**6)
        f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * (Tr**6)
        return Pc * np.exp(f0 + omega * f1)

    psat_ref = _psat_lee_kesler(Tf_i)

    def residual(T: types.Array1D) -> types.Array1D:
        """Residual of the ideal Liaw-Chiu mixture flash-point criterion.

        Args:
            T: Candidate mixture flash point in K.

        Returns:
            Residual of the flash-point criterion.
        """
        psat_T = _psat_lee_kesler(T[0])
        return np.array([np.sum(Xi * psat_T / psat_ref) - 1.0])

    sol = root(residual, [np.sum(Xi * Tf_i)])
    if not sol.success:
        raise RuntimeError(
            f"Flash point root-finding failed to converge: {sol.message}"
        )

    return sol.x[0]


class MixtureCorrelationsMixin(ComponentCorrelationsMixin):
    """Mixin for composition conversions and correlations of the fuel mixture."""

    # Attributes provided by the core `Fuel` class.
    Y_0: types.Quantity1D

    # -------------------------------------------------------------------------
    # Composition conversions
    # -------------------------------------------------------------------------
    def mean_molecular_weight(
        self, Yi: types.Quantity1D | None = None
    ) -> types.Quantity0D:
        """Calculate the mean molecular weight of the mixture.

        Args:
            Yi: Mass fractions of each compound.
                Defaults to `self.Y_0` (initial mass fractions).

        Returns:
            Mean molecular weight of the mixture in kg/mol.
        """
        Yi = Yi if Yi is not None else self.Y_0
        return 1 / np.sum(Yi / self.MW.to("kg/mol"))

    def mass2Y(self, mass: types.Quantity1D) -> types.Quantity1D:
        """Convert mass of each component to mass fractions (Yi).

        Args:
            mass: Mass of each compound.

        Returns:
            Mass fractions of the compounds (shape: num_compounds,).
        """
        mass = mass.to("kg")
        total_mass = mass.magnitude.sum()
        if total_mass != 0:
            Yi = (mass / total_mass).magnitude
        else:
            Yi = np.zeros_like(self.MW.magnitude)
        return Units.Quantity(Yi, "dimensionless")

    def mass2X(self, mass: types.Quantity1D) -> types.Quantity1D:
        """Convert mass of each component to mole fractions (Xi).

        Args:
            mass: Mass of each compound.

        Returns:
            Mole fractions of the compounds (shape: num_compounds,).
        """
        num_mole = mass.to("kg") / self.MW
        total_moles = np.sum(num_mole)
        if total_moles != 0:
            Xi = (num_mole / total_moles).magnitude
        else:
            Xi = np.zeros_like(self.MW.magnitude)
        return Units.Quantity(Xi, "dimensionless")

    def X2Y(self, Xi: types.Quantity1D) -> types.Quantity1D:
        """Convert mole fractions (Xi) to mass fractions (Yi).

        Args:
            Xi: Mole fractions of each compound.

        Returns:
            Mass fractions of the compounds (shape: num_compounds,).
        """
        mass = self.MW * Xi
        total_mass = np.sum(mass)
        Yi: types.Array1D = (
            (mass / total_mass).magnitude
            if total_mass != 0
            else np.zeros_like(self.MW.magnitude)
        )
        return Units.Quantity(Yi, "dimensionless")

    def Y2X(self, Yi: types.Quantity1D) -> types.Quantity1D:
        """Convert mass fractions (Yi) to mole fractions (Xi).

        Args:
            Yi: Mass fractions of each compound.

        Returns:
            Mole fractions of the compounds (shape: num_compounds,).
        """
        Mbar = self.mean_molecular_weight(Yi)
        if np.sum(Yi) != 0:
            Xi = (Mbar * Yi / self.MW).magnitude
        else:
            Xi = np.zeros_like(self.MW.magnitude)
        return Units.Quantity(Xi, "dimensionless")

    # -------------------------------------------------------------------------
    # Temperature-dependent mixture properties
    # -------------------------------------------------------------------------
    # NOTE: Cannot make Yi optional in these functions because switching the order
    # of Yi and T would break the original function signatures.
    def mixture_density(
        self, Yi: types.Quantity1D, T: types.Quantity0D
    ) -> types.Quantity1D:
        """Calculate mixture density at a given temperature.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature to compute property.

        Returns:
            Mixture density in kg/m^3.
        """
        return Yi @ self.density(T)

    def mixture_kinematic_viscosity(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> types.Quantity0D:
        """Calculate kinematic viscosity of the mixture.

        Uses Kendall-Monroe (default) or Arrhenius mixing correlations.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature to compute property.
            correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
                Defaults to "Kendall-Monroe".

        Returns:
            Mixture kinematic viscosity in m^2/s.
        """
        nu_i = self.viscosity_kinematic(T).to("m^2/s").magnitude
        Xi = self.Y2X(Yi).magnitude
        if correlation.casefold() == "Arrhenius".casefold():
            nu = np.exp(np.sum(Xi * np.log(nu_i)))
        else:  # Default: Kendall-Monroe
            nu = np.sum(Xi * (nu_i ** (1.0 / 3.0))) ** 3.0
        return Units.Quantity(nu, "m^2/s")

    def mixture_dynamic_viscosity(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> types.Quantity0D:
        """Calculate dynamic viscosity of the mixture.

        Args:
            Yi: Mass fractions of each compound.
            T: Temperature to compute property.
            correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").

        Returns:
            Mixture dynamic viscosity in Pa*s.
        """
        nu = self.mixture_kinematic_viscosity(Yi, T, correlation=correlation)
        rho = self.mixture_density(Yi, T)
        return (rho * nu).to("Pa*s")

    def mixture_vapor_pressure(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> types.Quantity0D:
        """Calculate saturated vapor pressure of the mixture via Raoult's law.

        Args:
            Yi: Mass fractions of each compound in the mixture.
            T: Temperature to compute property.
            correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").

        Returns:
            Mixture saturated vapor pressure in Pa.
        """
        return self.psat(T, correlation=correlation).to("Pa") @ self.Y2X(Yi)

    def mixture_vapor_pressure_antoine_coeffs(
        self,
        Yi: types.Quantity1D,
        Tvals: types.Quantity1D | None = None,
        units: Literal["mks", "cgs", "dyne/cm^2", "Pa"] = "mks",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> tuple[float, float, float, float]:
        """Estimate Antoine coefficients for vapor pressure of the mixture.

        Args:
            Yi: Mass fractions of each compound in the mixture.
            Tvals: Temperature range or nodes for Antoine fit in Kelvin.
                Defaults to [273.15, min(Tb_mix)].
            units: Units for pressure in fit.
                Defaults to "mks" (Pa).
            correlation: Correlation method.
                Defaults to "Lee-Kesler".

        Returns:
            Coefficients A, B, C, D.

        Raises:
            ValueError: If units or Tvals are invalid.
        """
        if units == "cgs":
            units = "dyne/cm^2"
        elif units == "mks":
            units = "Pa"
        elif units not in ["dyne/cm^2", "Pa"]:
            raise ValueError("units must be either 'mks', 'cgs', 'dyne/cm^2', or 'Pa'.")

        if Tvals is not None:
            Tvals = Tvals.to("K")

        # Define or get temperature nodes for fit
        if Tvals is None:
            print("Tvals not specified, using [273.15, min(Tb_mix)] for mixture.")
            Tb = arithmetic(self.Y2X(Yi), self.Tb)
            T = Units.Quantity(
                np.linspace(273.15, np.min(Tb.to("K").magnitude), 20), "K"
            )
        elif len(Tvals) == 2:
            T = Units.Quantity(
                np.linspace(Tvals[0].magnitude, Tvals[1].magnitude, 20), "K"
            )
        elif len(Tvals) > 2:
            T = Tvals
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        # D is the Pa-to-target-unit conversion factor, applied only when
        # evaluating psat(T) = D * 10**(A - B/(T + C)) in Pele.
        D = Units.Quantity(1, "Pa").to(units)

        T_magnitude = T.to("K").magnitude
        Pvals = np.zeros_like(T_magnitude)
        for k in range(len(T)):
            Pvals[k] = (
                self
                .mixture_vapor_pressure(Yi, T[k], correlation=correlation)
                .to(units)
                .magnitude
            )

        logP = np.log10(Pvals)
        popt, _ = curve_fit(_antoine_eq, T_magnitude, logP, p0=[1, 1e3, -1])
        A, B, C = popt

        return A, B, C, D.magnitude

    def mixture_surface_tension(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
        correlation: Literal["Pitzer", "Brock-Bird"] = "Brock-Bird",
    ) -> types.Quantity0D:
        """Calculate surface tension of the mixture.

        Uses arithmetic pseudo-property method recommended by Hugill and van
        Welsenes (1986).

        Args:
            Yi: Mass fractions of each compound in the mixture.
            T: Temperature to compute property.
            correlation: Correlation method ("Pitzer" or "Brock-Bird").
                Defaults to "Brock-Bird".

        Returns:
            Mixture surface tension in N/m.
        """
        sti = self.surface_tension(T, correlation=correlation)
        # Mixture surface tension via arithmetic mean, Poling (12-5.2)
        return arithmetic(self.Y2X(Yi), sti).to("N/m")

    def mixture_thermal_conductivity(
        self,
        Yi: types.Quantity1D,
        T: types.Quantity0D,
    ) -> types.Quantity0D:
        """Calculate thermal conductivity of the mixture.

        Uses the power law method of Vredeveld (Poling).

        Args:
            Yi: Mass fractions of each compound in the mixture.
            T: Temperature to compute property.

        Returns:
            Thermal conductivity in W/m/K.
        """
        tci = self.thermal_conductivity(T).to("W/(m*K)").magnitude
        tc = np.sum(Yi.magnitude * tci ** (-2)) ** (-0.5)
        return Units.Quantity(tc, "W/(m*K)")

    # -------------------------------------------------------------------------
    # Temperature-independent mixture properties
    # -------------------------------------------------------------------------
    def mixture_freeze_point(
        self,
        Yi: types.Quantity1D | None = None,
        *,
        alpha: float = 1.0,
    ) -> types.Quantity0D:
        """Calculate the freeze point of the mixture using the Boehm method.

        Solves the solid-liquid equilibrium model of Boehm et al. (2022),
        equation 21, for every compound using its mole fraction in the mixture,
        and returns the highest candidate temperature, which marks the first
        crystal to form on cooling.

        The enthalpy of fusion is `dH_fus_i = Tm_i * dS_fus_i`, and the
        solid-minus-liquid heat-capacity approximation is
        `dCp_i = -0.35 * Cp_L_i(298.15 K)` (Ruzicka-Domalski) on a molar basis.
        Equation 21 is solved per compound `j` against the ideal binary mixing
        entropy of that compound relative to the rest of the mixture,

            dS_mix_j = -(R / x_j) * [(1 - x_j) * ln(1 - x_j) + x_j * ln(x_j)],

            T_j = (dH_fus_j + x_j * dCp_j * (Tm_j - T_j))
                  / (dS_fus_j + x_j * dCp_j * ln(T_j / Tm_j) + alpha * dS_mix_j).

        Components with mole fraction at or below 1e-6 are excluded from the
        result. This is an equilibrium screening model; it does not represent
        cooling rate, supercooling, crystal kinetics, or detailed solid-phase
        nonideality.

        Args:
            Yi: Mass fractions of each compound in the mixture.
                Defaults to `self.Y_0` (initial mass fractions).
            alpha: Scaling applied to the ideal mixing-entropy term.
                Defaults to 1.0 (classical ideal-solution entropy term).

        Returns:
            Mixture freeze point in K.

        Raises:
            RuntimeError: If the root finder fails to converge.
        """
        Yi = Yi if Yi is not None else self.Y_0
        Xi = self.Y2X(Yi).to("dimensionless").magnitude

        R = constants.gas_constant.to("J/(mol*K)")
        Tm = self.Tm.to("K")
        dS_fus = self.dS_fus.to("J/(mol*K)")
        dH_fus = Tm * dS_fus
        dCp = (
            -0.35 * self.Cp(Units.Quantity(298.15, "K"), correlation="Ruzicka-Domalski")
        ).to("J/(mol*K)")

        # Ideal mixing entropy for each compound at its mole fraction in the
        # mixture. This is the only composition-dependent term in the model.
        Xi_safe = np.clip(Xi, 1e-6, 1.0 - 1e-6)
        dS_mix = (
            -R
            / Xi_safe
            * ((1.0 - Xi_safe) * np.log(1.0 - Xi_safe) + Xi_safe * np.log(Xi_safe))
        )

        Tm_mag = Tm.magnitude
        dH_fus_mag = dH_fus.magnitude
        dS_fus_mag = dS_fus.magnitude
        dCp_mag = dCp.magnitude
        dS_mix_mag = dS_mix.magnitude

        def residual(T: types.Array1D) -> types.Array1D:
            """Residual of Boehm et al. (2022), equation 21, for each compound.

            Args:
                T: Candidate freeze temperature of each compound in K.

            Returns:
                Residual of equation 21 for each compound.
            """
            T_safe = np.maximum(T, 1.0)
            lhs = T_safe * (
                dS_fus_mag
                + Xi_safe * dCp_mag * np.log(T_safe / Tm_mag)
                + alpha * dS_mix_mag
            )
            rhs = dH_fus_mag + Xi_safe * dCp_mag * (T_safe - Tm_mag)
            return lhs - rhs

        sol = root(residual, Tm_mag)
        if not sol.success:
            raise RuntimeError(
                f"Freeze point root-finding failed to converge: {sol.message}"
            )

        # Mark non-physical (non-positive or non-finite) solutions as -inf, and
        # only let compounds actually present in the mixture (Xi > 1e-6) set the
        # freeze point.
        T_candidates = np.where(np.isfinite(sol.x) & (sol.x > 0), sol.x, -np.inf)
        T_freeze = np.max(np.where(Xi > 1e-6, T_candidates, -np.inf))

        return Units.Quantity(T_freeze, "K")

    def mixture_flash_point(
        self,
        Yi: types.Quantity1D | None = None,
        *,
        correlation: Literal["Alibakhshi", "Alqaheem"] = "Alibakhshi",
        mixing_rule: Literal["linear", "Liaw"] = "Liaw",
    ) -> types.Quantity0D:
        """Calculate the flash point of the mixture.

        Component flash points (see `flash_point`) are combined with either the
        Liaw-Chiu (default) or linear mixing rule.

        For `mixing_rule="linear"`, the mixture flash point is the mass-fraction
        weighted average, `Tfp_mix = sum(Yi * Tfp_i)`.

        For `mixing_rule="Liaw"`, mole fractions `Xi` are used with the
        ideal-activity Liaw-Chiu (2006) relation, which solves for `Tfp_mix`
        satisfying

            sum(Xi * Psat_i(Tfp_mix) / Psat_i(Tfp_i)) = 1,

        where `Psat_i` is evaluated with the Lee-Kesler correlation using each
        compound's own critical properties. This ideal mixing rule does not
        model nonideal liquid activity coefficients.

        Args:
            Yi: Mass fractions of each compound in the mixture.
                Defaults to `self.Y_0` (initial mass fractions).
            correlation: Component flash point correlation
                ("Alibakhshi" or "Alqaheem"). Defaults to "Alibakhshi".
            mixing_rule: Mixing rule to use ("linear" or "Liaw").
                Defaults to "Liaw".

        Returns:
            Mixture flash point in K.
        """
        Yi = Yi if Yi is not None else self.Y_0
        T_fpi = self.flash_point(correlation=correlation)
        if mixing_rule.casefold() == "linear".casefold():
            return np.sum(Yi * T_fpi)

        T_flash = _liaw_chiu_flash_point(
            self.Y2X(Yi).to("dimensionless").magnitude,
            T_fpi.to("K").magnitude,
            self.Tc.to("K").magnitude,
            self.Pc.magnitude,
            self.omega.magnitude,
        )
        return Units.Quantity(T_flash, "K")

    def mixture_heat_of_combustion(
        self, Yi: types.Quantity1D | None = None
    ) -> types.Quantity0D:
        """Calculate the heat of combustion of the mixture using a Hess cycle.

        Combines each component's lower heating value (see
        `lower_heating_value`) with a mass-fraction weighted mixing rule,
        `dHc_mix = sum(Yi * LHV_i)`, reported as the positive heat released per
        unit mass. This is a net (lower) basis, consistent with gaseous-water
        combustion products.

        Args:
            Yi: Mass fractions of each compound in the mixture.
                Defaults to `self.Y_0` (initial mass fractions).

        Returns:
            Heat of combustion of the mixture in J/kg.
        """
        Yi = Yi if Yi is not None else self.Y_0
        return np.sum(Yi * self.lower_heating_value()).to("J/kg")

    def mixture_yield_sooting_index(
        self, Yi: types.Quantity1D | None = None
    ) -> types.Quantity0D:
        """Calculate the yield sooting index (YSI) of the mixture.

        Combines each component's yield sooting index (`YSI`) with a
        mole-fraction weighted mixing rule, `YSI_mix = sum(Xi * YSI_i)`.

        Args:
            Yi: Mass fractions of each compound in the mixture.
                Defaults to `self.Y_0` (initial mass fractions).

        Returns:
            Yield sooting index of the mixture.
        """
        Yi = Yi if Yi is not None else self.Y_0
        Xi = self.Y2X(Yi).to("dimensionless").magnitude
        ysi_i = self.YSI.magnitude  # Raises if any component is missing YSI
        return Units.Quantity(np.sum(Xi * ysi_i) / np.sum(Xi), "")

    def mixture_derived_cetane_number(
        self,
        Yi: types.Quantity1D | None = None,
        *,
        T_ref: types.Quantity0D = constants.T_stp,
    ) -> types.Quantity0D:
        """Calculate the derived cetane number (DCN) of the mixture.

        Combines each component's derived cetane number (`DCN`) with a linear,
        liquid volume fraction mixing rule. Volume fractions are derived from
        mass fractions using component liquid densities at `T_ref`.

        Args:
            Yi: Mass fractions of each compound in the mixture.
                Defaults to `self.Y_0` (initial mass fractions).
            T_ref: Reference temperature for liquid densities.
                Defaults to standard temperature.

        Returns:
            Derived cetane number of the mixture.
        """
        Yi = Yi if Yi is not None else self.Y_0
        dcn_i = self.DCN.magnitude  # Raises if any component is missing DCN
        rho_i = self.density(T_ref).magnitude
        vol = Yi.to("dimensionless").magnitude / rho_i
        return Units.Quantity(np.sum(vol / np.sum(vol) * dcn_i), "")


__all__ = ["MixtureCorrelationsMixin"]
