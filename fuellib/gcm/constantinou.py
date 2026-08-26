"""Implementation of the Constantinou and Gani group contribution method."""

from pathlib import Path

import numpy as np
import pandas as pd
from pint import Quantity

from ..comp import Component
from ..utils.units import ureg
from .core import BaseMethod


class ConstantinouMethod(BaseMethod):
    """Method model for the Constantinou and Gani group contribution method."""

    @property
    def _contributions(self) -> dict[str, str]:
        """Field names and units for contributions in the Constantinou and Gani method."""
        return {
            "tck": "dimensionless",
            "pck": "bar**(-0.5)",
            "vck": "meter**3 / kilomole",
            "tbk": "dimensionless",
            "tmk": "dimensionless",
            "hfk": "kilojoule / mole",
            "gfk": "kilojoule / mole",
            "hvk": "kilojoule / mole",
            "wk": "dimensionless",
            "vmk": "meter**3 / kilomole",
            "CpAk": "joule / mole / kelvin",
            "CpBk": "joule / mole / kelvin",
            "CpCk": "joule / mole / kelvin",
            "MW": "gram / mole",
        }

    def load_groups(self) -> None:
        """Load the groups for the Constantinou and Gani method."""
        self.groups = pd.read_csv(
            Path(__file__).parent / "constantinou.csv", header=0, index_col=0
        )

        if not set(self._contributions.keys()).issubset(self.groups.index):
            raise ValueError(
                "The Constantinou and Gani groups file is missing required index columns."
            )

    def get_contributions(self, comp: Component, property_name: str) -> Quantity:
        """Fetch the contributions for a given property based on the decomposition of a component.

        Args:
            comp: The component for which to fetch contributions.
            property_name: The name of the property to calculate.

        Returns:
            Array of contributions with units attached.
        """
        if property_name not in self._contributions:
            msg = f"Property '{property_name}' is not a valid property.\nAvailable properties: {self._contributions.keys()}"
            raise ValueError(msg)

        groups, counts = zip(*comp.decomposition)

        return np.repeat(
            self.groups.loc[property_name, list(groups)].to_numpy(dtype=float),
            counts,
        ) * ureg(self._contributions[property_name])

    def _sum_contributions(self, comp: Component, property_name: str) -> Quantity:
        """Sum the contributions for a property (equivalent to Nij @ property_k)."""
        return self.get_contributions(comp, property_name).sum()

    # --- Property calculation implementations ---

    def calc_MW(self, comp: Component) -> Quantity:
        """Calculate molecular weight in kg/mol."""
        mw = self._sum_contributions(comp, "MW")  # g/mol
        return mw.to("kg/mol")

    def calc_Tc(self, comp: Component) -> Quantity:
        """Calculate critical temperature in K."""
        tck_sum = self._sum_contributions(comp, "tck").magnitude
        return (181.128 * np.log(tck_sum)) * ureg.kelvin

    def calc_Pc(self, comp: Component) -> Quantity:
        """Calculate critical pressure in Pa."""
        pck_sum = self._sum_contributions(comp, "pck").magnitude  # bar^(-0.5)
        pc_bar = 1.3705 + (pck_sum + 0.10022) ** (-2)  # bar
        return (pc_bar * ureg.bar).to("Pa")

    def calc_Vc(self, comp: Component) -> Quantity:
        """Calculate critical volume in m³/mol."""
        vck_sum = self._sum_contributions(comp, "vck")  # m^3/kmol
        vc = -0.00435 * ureg("m**3/kmol") + vck_sum
        return vc.to("m**3/mol")

    def calc_Tb(self, comp: Component) -> Quantity:
        """Calculate boiling temperature in K."""
        tbk_sum = self._sum_contributions(comp, "tbk").magnitude
        return (204.359 * np.log(tbk_sum)) * ureg.kelvin

    def calc_Tm(self, comp: Component) -> Quantity:
        """Calculate melting temperature in K."""
        tmk_sum = self._sum_contributions(comp, "tmk").magnitude
        return (102.425 * np.log(tmk_sum)) * ureg.kelvin

    def calc_Hf(self, comp: Component) -> Quantity:
        """Calculate enthalpy of formation in J/mol."""
        hfk_sum = self._sum_contributions(comp, "hfk")  # kJ/mol
        hf = 10.835 * ureg("kJ/mol") + hfk_sum
        return hf.to("J/mol")

    def calc_Gf(self, comp: Component) -> Quantity:
        """Calculate Gibbs free energy in J/mol."""
        gfk_sum = self._sum_contributions(comp, "gfk")  # kJ/mol
        gf = -14.828 * ureg("kJ/mol") + gfk_sum
        return gf.to("J/mol")

    def calc_Hv_stp(self, comp: Component) -> Quantity:
        """Calculate enthalpy of vaporization at 298 K in J/mol."""
        hvk_sum = self._sum_contributions(comp, "hvk")  # kJ/mol
        hv = 6.829 * ureg("kJ/mol") + hvk_sum
        return hv.to("J/mol")

    def calc_omega(self, comp: Component) -> Quantity:
        """Calculate acentric factor (dimensionless)."""
        wk_sum = self._sum_contributions(comp, "wk").magnitude
        omega = 0.4085 * np.log(wk_sum + 1.1507) ** (1.0 / 0.5050)
        return omega * ureg.dimensionless

    def calc_Vm_stp(self, comp: Component) -> Quantity:
        """Calculate molar liquid volume at 298 K in m³/mol."""
        vmk_sum = self._sum_contributions(comp, "vmk")  # m^3/kmol
        vm = 0.01211 * ureg("m**3/kmol") + vmk_sum
        return vm.to("m**3/mol")

    def calc_Cp_coeffs(self, comp: Component) -> tuple[Quantity, Quantity, Quantity]:
        """Calculate specific heat coefficients (Cp_A, Cp_B, Cp_C) in J/mol/K.

        Cp(T) = Cp_A + Cp_B * theta + Cp_C * theta^2
        where theta = (T - 298) / 700
        """
        cpak_sum = self._sum_contributions(comp, "CpAk")  # J/mol/K
        cpbk_sum = self._sum_contributions(comp, "CpBk")  # J/mol/K
        cpck_sum = self._sum_contributions(comp, "CpCk")  # J/mol/K

        cp_a = cpak_sum - 19.7779 * ureg("J/mol/K")
        cp_b = cpbk_sum
        cp_c = cpck_sum

        return (cp_a, cp_b, cp_c)
