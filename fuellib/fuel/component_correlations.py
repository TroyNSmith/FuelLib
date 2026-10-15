"""Mixin providing temperature-dependent correlations for each component of `Fuel`."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import curve_fit

from .. import constants
from ..utils import Units, types
from .component_properties import ComponentPropertiesMixin


def _antoine_eq(
    T: float | types.Array1D, A: float, B: float, C: float
) -> float | types.Array1D:
    """Antoine equation for vapor pressure, log10(p) = A - B/(T + C).

    Args:
        T: Temperature.
        A: Antoine coefficient A.
        B: Antoine coefficient B.
        C: Antoine coefficient C.

    Returns:
        log10(pressure).
    """
    return A - B / (T + C)


class ComponentCorrelationsMixin(ComponentPropertiesMixin):
    """Mixin for correlations evaluated for each compound in the fuel."""

    def molar_liquid_vol(
        self, T: types.Quantity0D, comp_idx: int | None = None
    ) -> types.Quantity1D:
        """Compute molar liquid volume with temperature correction.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Molar liquid volume in m^3/mol.
        """
        Tstp = Units.Quantity(298, "K")
        T = T.to("K")
        Tc = self.Tc.to("K")
        phi = np.zeros_like(Tc.magnitude)
        for i in range(len(Tc)):
            if T > Tc[i]:
                phi[i] = -((1 - (Tstp / Tc[i])) ** (2.0 / 7.0))
            else:
                phi[i] = (1 - (T / Tc[i])) ** (2.0 / 7.0) - (1 - (Tstp / Tc[i])) ** (
                    2.0 / 7.0
                )
        z = 0.29056 - 0.08775 * self.omega
        Vmi = self.Vm_stp * z**phi
        return Vmi[comp_idx] if comp_idx is not None else Vmi

    def density(
        self, T: types.Quantity0D, comp_idx: int | None = None
    ) -> types.Quantity1D:
        """Calculate the density of each component at temperature T.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Density of each compound in kg/m^3.
        """
        rho_i = (self.MW / self.molar_liquid_vol(T)).to("kg/m^3")
        return rho_i[comp_idx] if comp_idx is not None else rho_i

    def viscosity_kinematic(
        self, T: types.Quantity0D, comp_idx: int | None = None
    ) -> types.Quantity1D:
        """Calculate the viscosity using Dutt's equation.

        Uses Dutt's equation (4.23) from "Viscosity of Liquids". The equation
        predicts viscosity in mm^2/s and is converted to SI units.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Viscosity of each component in m^2/s.
        """
        T_C = T.to("celsius").magnitude
        Tb = self.Tb.to("celsius").magnitude
        # RHS of Dutt's equation (4.23) in Viscosity of Liquids
        rhs = -3.0171 + (442.78 + 1.6452 * Tb) / (T_C + 239 - 0.19 * Tb)
        nu_i = Units.Quantity(np.exp(rhs), "mm^2/s").to("m^2/s")
        return nu_i[comp_idx] if comp_idx is not None else nu_i

    def viscosity_dynamic(
        self, T: types.Quantity0D, comp_idx: int | None = None
    ) -> types.Quantity1D:
        """Calculate liquid dynamic viscosity based on droplet temperature and density.

        Uses Dutt's equation (4.23) for kinematic viscosity, combined with density.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Dynamic viscosity in Pa*s.
        """
        mu_i = (self.viscosity_kinematic(T) * self.density(T)).to("Pa*s")
        return mu_i[comp_idx] if comp_idx is not None else mu_i

    def Cp(
        self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        correlation: Literal[
            "Constantinou-Gani", "Ruzicka-Domalski"
        ] = "Constantinou-Gani",
    ) -> types.Quantity1D:
        """Compute molar specific heat capacity at a given temperature.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).
            correlation: Correlation method ("Constantinou-Gani" or
                "Ruzicka-Domalski"). Defaults to "Constantinou-Gani".

        Returns:
            Molar specific heat capacity in J/mol/K.
        """
        T = T.to("K")
        if correlation.casefold() == "Ruzicka-Domalski".casefold():
            Tr = T / 100.0
            rd_A, rd_B, rd_D = self.RD_coeffs
            cp = constants.gas_constant * (rd_A + rd_B * Tr + rd_D * Tr**2)
        else:
            theta = (T - Units.Quantity(298, "K")) / Units.Quantity(700, "K")
            cp = self.Cp_stp + self.Cp_B * theta + self.Cp_C * theta**2
        cp = cp.to("J/(mol*K)")
        return cp[comp_idx] if comp_idx is not None else cp

    def Cl(
        self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        correlation: Literal[
            "Constantinou-Gani", "Ruzicka-Domalski"
        ] = "Constantinou-Gani",
    ) -> types.Quantity1D:
        """Compute liquid mass specific heat capacity in J/kg/K at a given temperature.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).
            correlation: Correlation method for the molar heat capacity
                ("Constantinou-Gani" or "Ruzicka-Domalski").
                Defaults to "Constantinou-Gani".

        Returns:
            Mass specific heat capacity in J/kg/K.
        """
        cl = (self.Cp(T, correlation=correlation) / self.MW).to("J/(kg*K)")
        return cl[comp_idx] if comp_idx is not None else cl

    def psat(
        self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> types.Quantity1D:
        """Compute saturated vapor pressure.

        Can use Ambrose-Walton or Lee-Kesler correlations (default Lee-Kesler).

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).
            correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
                Defaults to "Lee-Kesler".

        Returns:
            Saturated vapor pressure in Pa.
        """
        T = T.to("K")
        Tr = T / self.Tc
        omega = self.omega

        if correlation.casefold() == "Ambrose-Walton".casefold():
            # May cause trouble at high temperatures
            tau = 1 - Tr
            f0 = (
                -5.97616 * tau
                + 1.29874 * tau**1.5
                - 0.60394 * tau**2.5
                - 1.06841 * tau**5.0
            )
            f0 /= Tr
            f1 = (
                -5.03365 * tau
                + 1.11505 * tau**1.5
                - 5.41217 * tau**2.5
                - 7.46628 * tau**5.0
            )
            f1 /= Tr
            f2 = (
                -0.64771 * tau
                + 2.41539 * tau**1.5
                - 4.26979 * tau**2.5
                - 3.25259 * tau**5.0
            )
            f2 /= Tr
            rhs = np.exp(f0 + omega * f1 + omega**2 * f2)
        else:  # Default correlation is Lee-Kesler
            f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * np.log(Tr) + 0.169347 * (Tr**6)
            f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * np.log(Tr) + 0.43577 * (Tr**6)
            rhs = np.exp(f0 + omega * f1)

        psat = (self.Pc * rhs).to("Pa")
        return psat[comp_idx] if comp_idx is not None else psat

    def psat_antoine_coeffs(
        self,
        Tvals: types.Quantity1D | None = None,
        units: Literal["mks", "cgs", "dyne/cm^2", "Pa"] = "mks",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> tuple[types.Array1D, types.Array1D, types.Array1D, types.Array1D]:
        """Estimate Antoine coefficients for vapor pressure of an individual compound.

        Args:
            Tvals: Temperature range or nodes for Antoine fit in Kelvin.
                Defaults to [273.15, Tb_i].
            units: Units for pressure in fit ("mks", "cgs", "dyne/cm^2", "Pa").
                Defaults to "mks".
            correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
                Defaults to "Lee-Kesler".

        Returns:
            Coefficients A, B, C, D for each compound.

        Raises:
            ValueError: If units or Tvals are invalid.
        """
        if units == "cgs" or units == "dyne/cm^2":
            units = "dyne/cm^2"
        elif units == "mks" or units == "Pa":
            units = "Pa"
        else:
            raise ValueError("units must be either 'mks', 'cgs', 'dyne/cm^2', or 'Pa'.")

        if Tvals is not None:
            Tvals = Tvals.to("K")

        # Define or get temperature nodes for fit
        if Tvals is None:
            print("Tvals not specified, using [273.15, Tb_i] for each compound.")
            # Initialize as zeros for now, calculated for each compound later
            T = Units.Quantity(np.zeros(20), "K")
        elif len(Tvals) == 2:
            T = Units.Quantity(
                np.linspace(Tvals[0].magnitude, Tvals[1].magnitude, 20), "K"
            )
        elif len(Tvals) > 2:
            T = Tvals
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        # Fit A, B, C against pressure in Pa (mks base) so the coefficients are
        # unit independent. D is the Pa-to-target-unit conversion factor, applied
        # only when evaluating psat(T) = D * 10**(A - B/(T + C)) in Pele.
        D = Units.Quantity(1, "Pa").to(units)

        A = np.zeros(self.num_compounds)
        B = np.zeros(self.num_compounds)
        C = np.zeros(self.num_compounds)
        for i in range(self.num_compounds):
            # Update T if not specified
            if Tvals is None:
                T = Units.Quantity(np.linspace(273.15, self.Tb[i].magnitude, 20), "K")
            T_magnitude = T.to("K").magnitude
            Pvals = np.zeros_like(T_magnitude)
            for k in range(len(T)):
                Pvals[k] = (
                    self
                    .psat(T[k], comp_idx=i, correlation=correlation)
                    .to("Pa")
                    .magnitude
                )

            logP = np.log10(Pvals)
            popt, _ = curve_fit(_antoine_eq, T_magnitude, logP, p0=[1, 1e3, -1])
            A[i], B[i], C[i] = popt
        D = D.magnitude + np.zeros(self.num_compounds)  # make D an array
        return A, B, C, D

    def latent_heat_vaporization(
        self, T: types.Quantity0D, comp_idx: int | None = None
    ) -> types.Quantity1D:
        """Calculate latent heat of vaporization adjusted for temperature.

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Latent heat of vaporization in J/kg.
        """
        T = T.to("K")
        Tc = self.Tc.to("K")
        Tb = self.Tb.to("K")
        Lv_stp = self.Lv_stp.to("J/kg")

        # Reduced temperatures
        Tr = T / Tc
        Trb = Tb / Tc

        Lvi = Units.Quantity(np.zeros_like(Tc.magnitude), "J/kg")
        for i in range(len(Tc)):
            if T > Tc[i]:
                Lvi.magnitude[i] = 0.0
            else:
                Lvi[i] = Lv_stp[i] * (((1.0 - Tr[i]) / (1.0 - Trb[i])) ** 0.38)

        return Lvi[comp_idx] if comp_idx is not None else Lvi

    def diffusion_coeff(
        self,
        p: types.Quantity0D,
        T: types.Quantity0D,
        sigma_gas: types.Quantity0D = constants.Sigma_gas,
        epsilonByKB_gas: types.Quantity0D = constants.EpsilonByKB_gas,
        MW_gas: types.Quantity0D = constants.MW_gas,
        correlation: Literal["Tee", "Wilke"] = "Tee",
    ) -> types.Quantity1D:
        """Compute diffusion coefficients using Lennard-Jones parameters.

        Uses Wilke and Lee method (Poling, equation 11-4.1). Ambient gas
        defaults to air parameters.

        Args:
            p: Pressure to compute property.
            T: Temperature to compute property.
            sigma_gas: Collision diameter.
                Default is 3.62 Angstroms.
            epsilonByKB_gas: Well depth over Boltzmann constant.
                Default is 97.0 K.
            MW_gas: Mean molecular weight of ambient gas.
                Default is 28.97 g/mol.
            correlation: Method to calculate sigma and epsilon ("Tee" or "Wilke").
                Default is "Tee".

        Returns:
            Diffusion coefficient in m^2/s.
        """
        p_bar = p.to("bar").magnitude
        T_K = T.to("K").magnitude
        sigma_gas_A = sigma_gas.to("angstrom").magnitude
        epsilonByKB_gas_K = epsilonByKB_gas.to("K").magnitude
        MW_gas_g = MW_gas.to("g/mol").magnitude

        if correlation.casefold() == "Tee".casefold():
            # Method of Tee for calculating liquid sigma and epsilon
            sigma_i = self.sigma.to("angstrom").magnitude
            epsilonByKB_i = self.epsilonByKB.to("K").magnitude
        else:
            # Method of Wilke & Lee calculating liquid sigma and epsilon
            Vmb_i = np.zeros_like(self.Tb.magnitude)
            for n in range(self.num_compounds):
                Vmb_i[n] = (
                    self
                    .molar_liquid_vol(self.Tb[n], comp_idx=n)
                    .to("cm^3/mol")
                    .magnitude
                )
            sigma_i = 1.18 * Vmb_i ** (1 / 3)  # Angstroms, Poling (11-4.2)
            epsilonByKB_i = 1.15 * self.Tb.to("K").magnitude  # K, Poling (11-4.3)

        # Compute binary sigma and epsilon
        sigmaAB_i = (sigma_gas_A + sigma_i) / 2  # Angstroms, Poling (11-3.5)
        epsilonAB_byKB_i = (epsilonByKB_gas_K * epsilonByKB_i) ** 0.5  # Poling (11-3.4)

        # Dimensionless collision integral for diffusion: Poling (11-3.6)
        Tstar_i = T_K / epsilonAB_byKB_i
        A = 1.06036
        B = 0.15610
        C = 0.193
        D = 0.47635
        E = 1.03587
        F = 1.52996
        G = 1.76474
        H = 3.89411
        omegaD_i = (
            A / (Tstar_i**B)
            + C / np.exp(D * Tstar_i)
            + E / np.exp(F * Tstar_i)
            + G / np.exp(H * Tstar_i)
        )

        MW_i = self.MW.to("g/mol").magnitude
        M_AB_i = 2 * (MW_i * MW_gas_g) / (MW_i + MW_gas_g)  # g/mol, Poling (11-3.1)

        # Binary diffusion coefficients, Poling (11-4.1)
        D_AB_i = (
            1e-3
            * (3.03 - 0.98 / (M_AB_i**0.5))
            * (T_K**1.5)
            / (p_bar * M_AB_i**0.5 * sigmaAB_i**2 * omegaD_i)
        )  # cm^2/s
        return Units.Quantity(D_AB_i, "cm^2/s").to("m^2/s")

    def surface_tension(
        self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
        correlation: Literal["Brock-Bird", "Pitzer"] = "Brock-Bird",
    ) -> types.Quantity1D:
        """Calculate surface tension of each compound at a given temperature.

        Uses Brock-Bird (default) or Pitzer correlations (Poling 12-3.5, 12-3.7).

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).
            correlation: Correlation method ("Brock-Bird" or "Pitzer").
                Defaults to "Brock-Bird".

        Returns:
            Surface tension in N/m.
        """
        Tc = self.Tc.to("K").magnitude
        Pc = self.Pc.to("bar").magnitude
        Tb = self.Tb.to("K").magnitude
        omega = self.omega.magnitude

        Tr = T.to("K").magnitude / Tc

        if correlation.casefold() == "Brock-Bird".casefold():
            Tbr = Tb / Tc
            Q = 0.1196 * (1.0 + (Tbr * np.log(Pc / 1.01325)) / (1.0 - Tbr)) - 0.279
        else:
            w = omega
            Q = (
                (1.86 + 1.18 * w)
                / 19.05
                * (((3.75 + 0.91 * w) / (0.291 - 0.08 * w)) ** (2.0 / 3.0))
            )

        st = Pc ** (2.0 / 3.0) * Tc ** (1.0 / 3.0) * Q * (1 - Tr) ** (11.0 / 9.0)
        st = Units.Quantity(st, "dyn/cm").to("N/m")
        return st[comp_idx] if comp_idx is not None else st

    def thermal_conductivity(
        self,
        T: types.Quantity0D,
        comp_idx: int | None = None,
    ) -> types.Quantity1D:
        """Calculate thermal conductivity at a given temperature.

        Uses Latini et al. method (Poling equation 10-9.1).

        Args:
            T: Temperature to compute property.
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Thermal conductivity in W/m/K.
        """
        MW = self.MW.to("g/mol").magnitude
        Tc = self.Tc.to("K").magnitude
        Tb = self.Tb.to("K").magnitude
        fam = self.fam

        Astar = 0.00350 + np.zeros_like(Tc)
        alpha = 1.2
        beta = 0.5 + np.zeros_like(Tc)
        gamma = 0.167
        Tr = T.to("K").magnitude / Tc

        for i in range(len(Tc)):
            if fam[i] == 1:
                # Aromatics
                Astar[i] = 0.0346
                beta[i] = 1.0
            elif fam[i] == 2:
                # Cycloparaffins
                Astar[i] = 0.0310
                beta[i] = 1.0
            elif fam[i] == 3:
                # Olefins
                Astar[i] = 0.0361
                beta[i] = 1.0

        A = Astar * Tb**alpha / (MW**beta * Tc**gamma)
        tc = Units.Quantity(A * (1 - Tr) ** (0.38) / (Tr ** (1 / 6)), "W/(m*K)")
        return tc[comp_idx] if comp_idx is not None else tc

    def flash_point(
        self,
        comp_idx: int | None = None,
        correlation: Literal["Alibakhshi", "Alqaheem"] = "Alibakhshi",
    ) -> types.Quantity1D:
        """Calculate the flash point of each compound.

        The Alibakhshi et al. (2015) modified group-contribution correlation
        (default) is

            Tfp_i = 12.14 + 0.73 * Tb_i + sum_k(N_ik * phi_k),

        where `phi_k` are the Alibakhshi group-contribution terms. The Alqaheem
        and Riazi (2017) correlation takes the flash point as a fixed fraction of
        the normal boiling point:

            Tfp_i = 0.70 * Tb_i.

        Args:
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).
            correlation: Correlation method ("Alibakhshi" or "Alqaheem").
                Defaults to "Alibakhshi".

        Returns:
            Component flash points in K.
        """
        if correlation.casefold() == "Alqaheem".casefold():
            Tfp = 0.70 * self.Tb
        else:
            Tfp = Units.Quantity(12.14, "K") + 0.73 * self.Tb + self.phi
        Tfp = Tfp.to("K")
        return Tfp[comp_idx] if comp_idx is not None else Tfp

    def lower_heating_value(self, comp_idx: int | None = None) -> types.Quantity1D:
        """Calculate the lower heating value of each compound from a Hess cycle.

        For component `i` with `nC` carbon and `nH` hydrogen atoms, the
        liquid-phase heat of formation is `Hf_liq_i = Hf_stp_i - Hv_stp_i`, the
        heat of combustion is

            dHc_i = nC_i * Hf(CO2, g) + (nH_i / 2) * Hf(H2O, g) - Hf_liq_i,

        using Hf(CO2, g) = -393.51 kJ/mol and Hf(H2O, g) = -241.83 kJ/mol at
        298.15 K, and the net (lower) heating value is `LHV_i = -dHc_i / MW_i`.

        This estimate corresponds to hydrocarbons only; it does not account for
        heteroatom-containing groups.

        Args:
            comp_idx: Index of compound to calculate property for.
                Defaults to None (all compounds).

        Returns:
            Lower heating values of the components in J/kg.
        """
        dH_form_co2 = Units.Quantity(-393.51, "kJ/mol")  # At 298.15 K
        dH_form_h2o = Units.Quantity(-241.83, "kJ/mol")  # At 298.15 K
        Hf_liq = (self.Hf_stp - self.Hv_stp).to("J/mol")
        nC = np.array(self.nC)
        nH = np.array(self.nH)
        lhv = (-(nC * dH_form_co2 + 0.5 * nH * dH_form_h2o - Hf_liq) / self.MW).to(
            "J/kg"
        )
        return lhv[comp_idx] if comp_idx is not None else lhv


__all__ = ["ComponentCorrelationsMixin"]
