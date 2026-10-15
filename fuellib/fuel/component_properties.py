"""Per-component (pure compound) properties for `Fuel`."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from pandas import DataFrame
from rdkit.Chem import Mol

from ..gcm import boehm_gcm, gani_gcm
from ..rdk import mol
from ..utils import Units, types

if TYPE_CHECKING:
    from ..gcm.core import GCM
    from .core import Fuel


class ComponentProperties:
    """Properties of each compound in the fuel.

    Properties are reference values (when `useRefProperties` is `True` and available),
    group contribution predictions, or values derived from them.
    """

    # Attributes provided by the core `Fuel` class.
    useRefProperties: bool
    compounds: list[str]
    smiles: list[str]
    compoundsData: DataFrame
    ganiDecomp: DataFrame

    def _predict(self, gcm: GCM, prop: str) -> types.Quantity1D:
        """Predict a property for each compound with a group contribution method.

        Args:
            gcm: Group contribution method used for the prediction.
            prop: Name of the property registered with `gcm`.

        Returns:
            Predicted property values for each compound.
        """
        return gcm.predict(prop, cast("Fuel", self))

    def _get_ref_values(self, prop: str, output_units: str) -> types.Quantity1D:
        """Get reference values for the given compounds in the specified output units.

        Args:
            prop: Property column prefix in the reference compounds file
                (e.g. "Tc" for the "Tc_Value" and "Tc_Units" columns).
            output_units: Desired output units for the reference values.

        Returns:
            Reference values for the given compounds in the specified output units.
        """
        out = np.full(len(self.compoundsData), np.nan, dtype=float)
        values = self.compoundsData[f"{prop}_Value"]
        units = self.compoundsData[f"{prop}_Units"]
        for i, (value, unit) in enumerate(zip(values, units)):
            if pd.isna(value):
                continue
            quantity = Units.Quantity(float(value), "" if pd.isna(unit) else unit)
            out[i] = quantity.to(output_units).magnitude
        return Units.Quantity(out, output_units)

    def _ref_or_predict(
        self, gani_prop: str, ref_prop: str, units: str
    ) -> types.Quantity1D:
        """Get reference values, falling back to Gani predictions where missing.

        Args:
            gani_prop: Name of the property registered with the Gani GCM.
            ref_prop: Property column prefix in the reference compounds file.
            units: Output units.

        Returns:
            Property values for each compound in `units`.
        """
        preds = self._predict(gani_gcm, gani_prop).to(units)
        if not self.useRefProperties:
            return preds
        refs = self._get_ref_values(ref_prop, units)
        return Units.Quantity(
            np.where(np.isnan(refs.magnitude), preds.magnitude, refs.magnitude), units
        )

    def _ref_only(self, prop: str, description: str) -> types.Quantity1D:
        """Get dimensionless reference values, requiring a value for every compound.

        Args:
            prop: Property column prefix in the reference compounds file.
            description: Human-readable property name used in error messages.

        Returns:
            Reference values for each compound.

        Raises:
            ValueError: If any reference values are missing.
        """
        refs = self._get_ref_values(prop, "")
        missing = np.isnan(refs.magnitude)
        if missing.any():
            names = ", ".join(
                self.compoundsData.loc[missing, "Reference Compound"].astype(str)
            )
            msg = f"Missing {description} reference values for: {names}."
            raise ValueError(msg)
        return refs

    # -------------------------------------------------------------------------
    # Critical properties
    # -------------------------------------------------------------------------
    @cached_property
    def Tc(self) -> types.Quantity1D:
        """Critical temperature for each compound in K."""
        return self._ref_or_predict("Tc", "Tc", "K")

    @cached_property
    def Pc(self) -> types.Quantity1D:
        """Critical pressure for each compound in Pa."""
        return self._ref_or_predict("Pc", "Pc", "Pa")

    @cached_property
    def Vc(self) -> types.Quantity1D:
        """Critical volume for each compound in m^3/mol."""
        return self._ref_or_predict("Vc", "Vc", "m^3/mol")

    @cached_property
    def Tm(self) -> types.Quantity1D:
        """Melting point for each compound in K."""
        return self._ref_or_predict("Tm", "Tm", "K")

    @cached_property
    def Tb(self) -> types.Quantity1D:
        """Boiling point for each compound in K."""
        return self._ref_or_predict("Tb", "Tb", "K")

    @cached_property
    def Hf_stp(self) -> types.Quantity1D:
        """Enthalpy of formation (at STP) for each compound in J/mol."""
        return self._ref_or_predict("Hf", "Hf_STP", "J/mol")

    @cached_property
    def Hv_stp(self) -> types.Quantity1D:
        """Enthalpy of vaporization (at STP) for each compound in J/mol."""
        return self._ref_or_predict("Hv_stp", "Hv_STP", "J/mol")

    @cached_property
    def Vm_stp(self) -> types.Quantity1D:
        """Molar volume (at STP) for each compound in m^3/mol."""
        return self._ref_or_predict("Vm_stp", "Vm_STP", "m^3/mol")

    @cached_property
    def omega(self) -> types.Quantity1D:
        """Acentric factor for each compound (dimensionless)."""
        return self._ref_or_predict("omega", "Omega", "")

    # -------------------------------------------------------------------------
    # Reference-only properties
    # -------------------------------------------------------------------------
    @cached_property
    def YSI(self) -> types.Quantity1D:
        """Yield Sooting Index (YSI) for each compound (dimensionless).

        A `ValueError` is raised if any reference values for YSI are missing.
        """  # ruff: ignore[property-docstring-starts-with-verb]
        return self._ref_only("YSI", "Yield Sooting Index (YSI)")

    @cached_property
    def DCN(self) -> types.Quantity1D:
        """Derived cetane number (DCN) for each compound (dimensionless).

        A `ValueError` is raised if any reference values for DCN are missing.
        """
        return self._ref_only("DCN", "Derived Cetane Number (DCN)")

    # -------------------------------------------------------------------------
    # Prediction-only properties
    # -------------------------------------------------------------------------
    @cached_property
    def Gf_stp(self) -> types.Quantity1D:
        """Gibbs free energy of formation (at STP) for each compound in J/mol."""
        return self._predict(gani_gcm, "Gf").to("J/mol")

    @cached_property
    def Cp_stp(self) -> types.Quantity1D:
        """Heat capacity at constant pressure for each compound in J/(mol*K)."""
        return self._predict(gani_gcm, "Cp_stp").to("J/(mol*K)")

    @cached_property
    def Cp_B(self) -> types.Quantity1D:
        """Temperature correction term B for heat capacity in J/(mol*K)."""
        return self._predict(gani_gcm, "Cp_B").to("J/(mol*K)")

    @cached_property
    def Cp_C(self) -> types.Quantity1D:
        """Temperature correction term C for heat capacity in J/(mol*K)."""
        return self._predict(gani_gcm, "Cp_C").to("J/(mol*K)")

    @cached_property
    def dS_fus(self) -> types.Quantity1D:
        """Fusion entropy for each compound in the fuel mixture in J/(mol*K)."""
        return self._predict(boehm_gcm, "dS_fus").to("J/(mol*K)")

    @cached_property
    def RD_coeffs(self) -> tuple[types.Quantity1D, types.Quantity1D, types.Quantity1D]:
        """Ruzicka-Domalski correlation coefficients for heat capacity."""
        rd_A = self._predict(gani_gcm, "rd_A")
        rd_B = self._predict(gani_gcm, "rd_B")
        rd_D = self._predict(gani_gcm, "rd_D")
        return rd_A, rd_B, rd_D

    @cached_property
    def phi(self) -> types.Quantity1D:
        """Alibakhshi phi terms for each compound in the fuel mixture in K."""
        return self._predict(gani_gcm, "alibakhshi_phi").to("K")

    # -------------------------------------------------------------------------
    # Derived properties
    # -------------------------------------------------------------------------
    @cached_property
    def Lv_stp(self) -> types.Quantity1D:
        """Latent heat of vaporization (at STP) for each compound in J/kg."""
        return (self.Hv_stp / self.MW).to("J/kg")

    @cached_property
    def epsilonByKB(self) -> types.Quantity1D:
        """Depth of the Lennard-Jones potential well divided by Boltzmann constant."""
        return (0.7915 + 0.1693 * self.omega.magnitude) * self.Tc.to("K")

    @cached_property
    def sigma(self) -> types.Quantity1D:
        """Lennard-Jones collision diameter for each compound in m."""
        sigma = (2.3551 - 0.0874 * self.omega.magnitude) * (
            self.Tc.to("K").magnitude / self.Pc.to("atm").magnitude
        ) ** (1.0 / 3)
        return Units.Quantity(sigma, "angstrom").to("m")

    # -------------------------------------------------------------------------
    # Molecular structure
    # -------------------------------------------------------------------------
    @cached_property
    def rdkit_mols(self) -> list[Mol]:
        """RDKit `Mol` objects for the compounds in the fuel mixture."""
        return [mol.from_smiles(smiles) for smiles in self.smiles]

    @property
    def nC(self) -> list[int]:
        """Number of carbon atoms for each compound in the fuel mixture."""
        return [mol.atom_counts(m).get("C", 0) for m in self.rdkit_mols]

    @property
    def nH(self) -> list[int]:
        """Number of hydrogen atoms for each compound in the fuel mixture."""
        return [mol.atom_counts(m).get("H", 0) for m in self.rdkit_mols]

    @property
    def MW(self) -> types.Quantity1D:
        """Molecular weights of the compounds in the fuel mixture in kg/mol."""
        return Units.Q([mol.molecular_weight(m) for m in self.rdkit_mols], "g/mol").to(
            "kg/mol"
        )

    @property
    def num_compounds(self) -> int:
        """Number of compounds in the fuel mixture."""
        return len(self.compounds)

    @property
    def formulas(self) -> list[str]:
        """Molecular formulas for each compound in the fuel mixture."""
        return [mol.hill_formula(m) for m in self.rdkit_mols]

    @cached_property
    def hc_type(self) -> list[str]:
        """Hydrocarbon type for each compound in the fuel mixture.

        Possible values (in ascending priority) are:
        * "n-alkane"
        * "iso-alkane"
        * "alkene"
        * "cyclo-alkane"
        * "aromatic"
        """
        hc_types = []
        for m in self.rdkit_mols:
            if mol.has_aromatic(m):
                hc_types.append("aromatic")
            elif mol.has_ring(m):
                hc_types.append("cyclo-alkane")
            elif mol.has_double_bond(m):
                hc_types.append("alkene")
            elif mol.has_branch(m):
                hc_types.append("iso-alkane")
            else:
                hc_types.append("n-alkane")
        return hc_types

    @property
    def fam(self) -> types.Array1D:
        """Hydrocarbon family codes for thermal conductivity.

        ==== ==================
        Code Hydrocarbon Family
        ==== ==================
        0    saturated
        1    aromatics
        2    cycloparaffins
        3    olefins
        ==== ==================

        Raises:
            ValueError: If an unknown hydrocarbon type is encountered.
        """
        fam_codes = []
        for hc in self.hc_type:
            if hc == "n-alkane" or hc == "iso-alkane":
                fam_codes.append(0)
            elif hc == "aromatic":
                fam_codes.append(1)
            elif hc == "cyclo-alkane":
                fam_codes.append(2)
            elif hc == "alkene":
                fam_codes.append(3)
            else:
                msg = f"Unknown hydrocarbon type '{hc}' encountered."
                raise ValueError(msg)
        return np.array(fam_codes, dtype=int)


__all__ = ["ComponentProperties"]
