"""Fuel class for Group Contribution Method calculations."""

import os
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyparsing as pp
from rdkit.Chem import Mol
from scipy.optimize import curve_fit

from .data.locator import (
    DEFAULT_FUELDATA_DIR,
    file_path,
    get_metadata_decomp_name,
    validate_fuel_data_dir,
)
from .gcm import GCMRegistry
from .rdk import mol
from .utils.constants import T_STP, EpsilonByKB_g, MW_g, Sigma_g
from .utils.logger import logger
from .utils.types import FloatVector, IntVector, PintScalar, PintVector, StrVector
from .utils.units import PintUnits
from .utils.utility import mixing_rule

NumCompounds = int


class Fuel:
    """
    Class for handling group contribution calculations of thermodynamic and mixture properties.

    :param name: Name of the mixture as it appears in its gcData file.
    :type name: str
    :param decompName: Name of the groupDecomposition file if different from name. Defaults to None.
    :type decompName: str, optional
    :param fuelDataDir: Directory where the fuel data is stored. If None, uses built-in embedded data.
    :type fuelDataDir: str, optional
    """

    #: Name of the fuel/mixture
    name: str
    #: Name of the groupDecomposition file if different from name
    decompName: str | None = None
    #: Directory where the fuel data is stored
    fuelDataDir: str | Path = DEFAULT_FUELDATA_DIR

    def __init__(
        self,
        name: str,
        decompName: str | None = None,
        fuelDataDir: str | Path | None = DEFAULT_FUELDATA_DIR,
    ) -> None:
        """
        Initialize the fuel object and calculate GCM properties.

        :param name: Name of the mixture as it appears in its gcData file.
        :type name: str
        :param decompName: Name of the groupDecomposition file if different from name.
        :type decompName: str, optional
        :param fuelDataDir: Directory where the fuel data is stored. If None, uses built-in embedded data.
        :type fuelDataDir: str, optional
        """
        logger.info("Initializing Fuel class for %s", name)

        self.name = name
        # Validate and set the fuel data directory
        self.fuelDataDir = validate_fuel_data_dir(fuelDataDir or DEFAULT_FUELDATA_DIR)
        # Determine the decomposition name, defaulting to the metadata if not provided
        self.decompName = decompName or get_metadata_decomp_name(name, self.fuelDataDir)
        # Set the directories for the group contribution data, decomposition data, and properties data
        self.fuelDataGcDir = self.fuelDataDir / "gcData"
        self.fuelDataDecompDir = self.fuelDataDir / "groupDecompositionData"
        self.fuelDataPropsDir = self.fuelDataDir / "propertiesData"
        logger.info(
            "Legacy property calls (e.g., fuel.Tc) set using the Gani GCM method. To access other\n"
            "method predictions, use ``fuel.get_gcm_property(method_name, property_name)``."
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # (Mostly) Legacy functionalities for backwards compatibility                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Parsed data
    @cached_property
    def gcxgc_data(self) -> pd.DataFrame:
        """GC csv data as a pandas DataFrame."""
        self.gcxgcFile = file_path(
            self.fuelDataGcDir,
            f"{self.name}_init.csv",
            f"{self.name}.gcxgc.csv",
        )
        return pd.read_csv(self.gcxgcFile)

    @cached_property
    def gcxgc_bins(self) -> list[tuple[str, int, str]]:
        """Hydrocarbon bins for the compounds. Shape: (num_compounds,)

        :return: List of tuples containing the bin name, carbon number, and family for each compound.
        :rtype: list[tuple[str, int, str]]
        """
        if "Bin" in self.gcxgc_data.columns:
            bins = self.gcxgc_data["Bin"].to_list()
        elif "Compound" in self.gcxgc_data.columns:
            logger.info(
                "Hydrocarbon bin information missing; using compound names as fallback. Consider renaming\n"
                "`Compound` column to `Bin` if appropriate."
            )
            bins = self.gcxgc_data["Compound"].to_list()
        else:
            msg = "Hydrocarbon bin information missing and no compound names available in gc_data."
            raise ValueError(msg)
        carbon_parser = pp.Suppress("C") + pp.Word(pp.nums).setResultsName(
            "carbon_number"
        )
        family_parser = pp.Word(pp.alphanums).setResultsName("family")
        parser = (carbon_parser + pp.Suppress("-") + family_parser) | (
            family_parser + pp.Suppress("-") + carbon_parser
        )

        def split_bin(bin_name: str) -> tuple[str, int, str]:
            parsed = parser.parse_string(bin_name)
            return bin_name, int(parsed.carbon_number), parsed.family.lower()

        return [split_bin(bin_name) for bin_name in bins]

    @cached_property
    def compounds(self) -> list[str]:
        """Simplified representation of gcxgc_bins for backwards compatibility."""
        if "Bin" in self.gcxgc_data.columns:
            return self.gcxgc_data["Bin"].to_list()
        elif "Compound" in self.gcxgc_data.columns:
            logger.info(
                "Hydrocarbon bin information missing; using compound names as fallback. Consider renaming\n"
                "`Compound` column to `Bin` if appropriate."
            )
            return self.gcxgc_data["Compound"].to_list()
        msg = "Hydrocarbon bin information missing and no compound names available in gc_data."
        raise ValueError(msg)

    @property
    def reference_compounds(self) -> list[str] | None:
        """Get the list of reference compounds for the compounds, if available. Shape: (num_compounds,)"""
        if "Reference Compound" in self.gcxgc_data.columns:
            return self.gcxgc_data["Reference Compound"].to_list()
        return None

    @property
    def smiles(self) -> list[str]:
        """Get the list of SMILES strings for the compounds. Shape: (num_compounds,)"""
        if "SMILES" in self.gcxgc_data.columns:
            return self.gcxgc_data["SMILES"].to_list()
        msg = "Required 'SMILES' column missing in gc_data."
        raise ValueError(msg)

    @property
    def pelephysics_keys(self) -> np.ndarray | None:
        """PelePhysics keys for the compounds, if present. Shape: (num_compounds,)"""
        if "PelePhysics Key" in self.gcxgc_data.columns:
            return np.array(
                [key.strip() for key in self.gcxgc_data["PelePhysics Key"].to_list()]
            )
        return None

    @property
    def formulas(self) -> list[str]:
        """Chemical formulas for the compounds. Shape: (num_compounds,)"""

        def _hill_order(atom_counts: dict[str, int]) -> str:
            atoms = []
            c = atom_counts.pop("C", None)
            if c is not None:
                atoms.append((0, "C", c))
            h = atom_counts.pop("H", None)
            if h is not None:
                atoms.append((1, "H", h))
            for atom, count in atom_counts.items():
                atoms.append((2, atom, count))
            return "".join(f"{s}{c if c > 1 else ''}" for _, s, c in sorted(atoms))

        return [_hill_order(counts.copy()) for counts in self.atom_counts]

    @property
    def Y_0(self) -> np.ndarray:
        """Normalized initial mass fraction (Y_0) for each compound. Shape: (num_compounds,)"""
        if "Weight %" in self.gcxgc_data.columns:
            wts = self.gcxgc_data["Weight %"].to_numpy().flatten().astype(float)
            return wts / wts.sum()
        msg = "Initial mass fractions (Y_0) not available: 'Weight %' column missing in gc_data."
        raise ValueError(msg)

    # GCM (Group Contribution Method) properties for the compounds
    @property
    def gani_decomp(self) -> pd.DataFrame:
        """Gani decomposition for the compounds. Shape: (num_compounds, num_groups)"""
        groupDecompFile = file_path(
            self.fuelDataDecompDir,
            f"{self.decompName}.csv",
            f"{self.decompName}.gani.csv",
        )
        if not Path(groupDecompFile).exists():
            msg = f"Gani decomposition file not found: {groupDecompFile}"
            raise FileNotFoundError(msg)

        df = pd.read_csv(groupDecompFile, header=0, index_col=0)
        if df.shape[0] != self.num_compounds:
            raise ValueError(
                f"Insufficient mixture description:\n"
                f"The number of compounds in {groupDecompFile} does not "
                f"equal the number of compounds in {self.gcxgcFile}."
            )
        return df

    @cached_property
    def gcm_properties(self) -> pd.DataFrame:
        """Pre-computed GCM properties for the compounds. Shape: (num_properties, num_compounds)"""
        props = pd.DataFrame(columns=["Method", "Property"] + self.compounds)
        for gcm in GCMRegistry.methods:
            props = pd.concat([props, gcm.predict_all(self)], ignore_index=True)
        return props

    def get_gcm_property(self, method: str, property_name: str) -> PintVector:
        """
        Get a specific property from the GCM for each compound.

        :param method: The GCM method to use.
        :type method: str
        :param property_name: The name of the property to retrieve.
        :type property_name: str
        :return: Array of the requested property for each compound.
        :rtype: PintVector
        """
        method = method.lower()
        if method not in self.gcm_properties["Method"].values:
            msg = f"Method '{method}' not found in computed GCM properties."
            raise KeyError(msg)

        property_name = property_name.lower()
        if property_name not in self.gcm_properties["Property"].values:
            msg = f"Property '{property_name}' not found in computed GCM properties."
            raise KeyError(msg)

        row = self.gcm_properties[
            (self.gcm_properties["Method"] == method)
            & (self.gcm_properties["Property"] == property_name)
        ]
        # Convert vector of PintQuantities to PintVector
        row = row[self.compounds].to_numpy().flatten()
        return PintUnits.Quantity([q.magnitude for q in row], row[0].units)

    ## Legacy Fuel attributes for backward compatibility
    @cached_property
    def MW(self) -> PintVector:
        """Molecular weight for each compound. Shape: (n_compounds,)"""
        mw = [mol.molecular_weight(m) for m in self.rdkit_mols]
        return PintUnits.Quantity(mw, "g/mol").to("kg/mol")

    @property
    def Tc(self) -> PintVector:
        """Gani critical temperature (K) for each compound. Shape: (n_compounds,)"""
        return self.get_gcm_property("gani", "Tc").to("K")

    @property
    def Pc(self) -> PintVector:
        """Gani critical pressure (Pa) for each compound. Shape: (n_compounds,)"""
        return self.get_gcm_property("gani", "Pc").to("Pa")

    @property
    def Vc(self) -> PintVector:
        """Gani critical volume (m^3/mol) for each compound. Shape: (n_compounds,)"""
        return self.get_gcm_property("gani", "Vc").to("m^3/mol")

    @property
    def Tb(self) -> PintVector:
        """Gani boiling temperature (K) for each compound. Shape: (n_compounds,)"""
        return self.get_gcm_property("gani", "Tb").to("K")

    @property
    def Tm(self) -> PintVector:
        """Gani melting temperature (K) for each compound. Shape: (n_compounds,)"""
        return self.get_gcm_property("gani", "Tm").to("K")

    @property
    def Hf(self) -> PintVector:
        """Gani enthalpy of formation (J/mol) for each compound. Shape: (n_compounds,)"""
        return self.get_gcm_property("gani", "Hf").to("J/mol")

    @property
    def Gf(self) -> PintVector:
        """Gani Gibbs free energy of formation (J/mol) for each compound. Shape: (n_compounds,)"""
        return self.get_gcm_property("gani", "Gf").to("J/mol")

    @property
    def Hv_stp(self) -> PintVector:
        """Gani enthalpy of vaporization at STP (J/mol) for each compound. Shape: (n_compounds,)"""
        return self.get_gcm_property("gani", "Hv_stp").to("J/mol")

    @property
    def omega(self) -> PintVector:
        """Gani acentric factor (dimensionless) for each compound. Shape: (n_compounds,)"""
        return self.get_gcm_property("gani", "omega").to("")

    @property
    def Vm_stp(self) -> PintVector:
        """Gani molar volume at STP (m^3/mol) for each compound. Shape: (n_compounds,)"""
        return self.get_gcm_property("gani", "Vm_stp").to("m^3/mol")

    @property
    def Cp_stp(self) -> PintVector:
        """Gani heat capacity at STP (J/mol/K) for each compound. Shape: (n_compounds,)"""
        return self.get_gcm_property("gani", "Cp_stp").to("J/mol/K")

    @property
    def Cp_B(self) -> PintVector:
        """Gani heat capacity correction (J/mol/K) for each compound. Shape: (n_compounds,)"""
        return self.get_gcm_property("gani", "Cp_B").to("J/mol/K")

    @property
    def Cp_C(self) -> PintVector:
        """Gani heat capacity correction (J/mol/K) for each compound. Shape: (n_compounds,)"""
        return self.get_gcm_property("gani", "Cp_C").to("J/mol/K")

    @property
    def Lv_stp(self) -> PintVector:
        """Standard latent heat of vaporization for each compound. Shape: (n_compounds,)"""
        return (self.Hv_stp / self.MW).to("J/kg")

    @property
    def epsilonByKB(self) -> PintVector:
        """Epsilon divided by Boltzmann constant (K) for each compound. Shape: (n_compounds,)"""
        A = PintUnits.Quantity(0.7915, "")
        eps = (A + 0.1693 * self.omega) * self.Tc
        return PintUnits.Quantity(eps, "K")

    @property
    def sigma(self) -> PintVector:
        """Sigma parameter (m) for each compound. Shape: (n_compounds,)"""
        w = self.omega.to("").magnitude
        tc = self.Tc.to("K").magnitude
        pc = self.Pc.to("atm").magnitude
        sigma = (2.3551 - 0.0874 * w) * np.power((tc / pc), 1.0 / 3)
        return PintUnits.Quantity(sigma, "angstrom").to("m")

    # Component classification and informatics
    @property
    def num_compounds(self) -> int:
        """Number of compounds in the fuel mixture."""
        return len(self.compounds)

    @cached_property
    def rdkit_mols(self) -> list[Mol]:
        """RDKit Mol objects for the compounds. Shape: (n_compounds,)"""
        return [mol.from_smiles(smiles) for smiles in self.smiles]

    @property
    def hc_type(self) -> StrVector:
        """Hydrocarbon type for each compound. Shape: (n_compounds,)

        Classification hierarchy:
            Aromatic > Cyclo-alkane > Alkene > Iso-alkane > n-alkane
        """
        is_hc = np.array([mol.is_hydrocarbon(m) for m in self.rdkit_mols])
        hc_type = np.full_like(self.compounds, "n-alkane")
        hc_type[[mol.has_branched(m) for m in self.rdkit_mols]] = "iso-alkane"
        hc_type[[mol.count_olefins(m) > 0 for m in self.rdkit_mols]] = "alkene"
        hc_type[[mol.count_aliphatic_rings(m) > 0 for m in self.rdkit_mols]] = (
            "cyclo-alkane"
        )
        hc_type[[mol.count_aromatic_rings(m) > 0 for m in self.rdkit_mols]] = "aromatic"
        hc_type[~is_hc] = np.nan
        return hc_type

    @property
    def fam(self) -> IntVector:
        """Family classification for each compound. Shape: (n_compounds,)

        Classification mapping:
            Aromatic     : 1
            Cyclo-alkane : 2
            Alkene       : 3
            Otherwise    : 0
        """
        fam = np.zeros_like(self.compounds, dtype=np.int64)
        fam[np.where(self.hc_type == "aromatic")] = 1
        fam[np.where(self.hc_type == "cyclo-alkane")] = 2
        fam[np.where(self.hc_type == "alkene")] = 3
        return fam

    @property
    def atom_counts(self) -> list[dict[str, int]]:
        """Dictionary of atom counts for each compound. Shape: (n_compounds,)"""
        return [mol.atom_counts(m) for m in self.rdkit_mols]

    @property
    def nC(self) -> IntVector:
        """Number of carbon atoms for each compound. Shape: (n_compounds,)"""
        return np.array(
            [count.get("C", 0) for count in self.atom_counts], dtype=np.int_
        )

    @property
    def nH(self) -> IntVector:
        """Number of hydrogen atoms for each compound. Shape: (n_compounds,)"""
        return np.array(
            [count.get("H", 0) for count in self.atom_counts], dtype=np.int_
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Component-wise property correlations                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def molar_liquid_vol(
        self, T: PintScalar, comp_idx: int | None = None
    ) -> PintVector:
        """
        Compute molar liquid volume with temperature correction.

        :param T: Temperature in Kelvin.
        :type T: PintScalar
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Molar liquid volume in m^3/mol.
        :rtype: PintArray
        """
        Tc = self.Tc
        omega = self.omega
        Vm_stp = self.Vm_stp

        x = -np.power((1 - (T_STP / Tc)), 2.0 / 7.0)
        y = np.power((1 - (T / Tc)), 2.0 / 7.0) + x
        phi = np.where(T > Tc, x, y)

        z = 0.29056 - 0.08775 * omega.magnitude
        Vmi = Vm_stp.to("m^3/mol") * np.power(z, phi)
        return Vmi[comp_idx] if comp_idx is not None else Vmi

    def density(self, T: PintScalar, comp_idx: int | None = None) -> PintVector:
        """
        Calculate the density of each component at temperature T.

        :param T: Temperature of the mixture in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Density of each compound in kg/m^3.
        :rtype: np.ndarray
        """
        rho = (self.MW / self.molar_liquid_vol(T)).to("kg/m^3")
        return rho[comp_idx] if comp_idx is not None else rho

    def viscosity_kinematic(
        self, T: PintScalar, comp_idx: int | None = None
    ) -> PintVector:
        """
        Calculate the viscosity using Dutt's equation.

        :meta private: This uses Dutt's equation (4.23) from "Viscosity of Liquids".
        :meta private: The equation predicts viscosity in mm^2/s and is converted to SI units.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Viscosity of each component in m^2/s.
        :rtype: np.ndarray
        """
        # Convert temperature to Celsius
        T_cels = T.to("celsius").magnitude
        Tb_cels = self.Tb.to("celsius").magnitude

        # RHS of Dutt's equation (4.23) in Viscosity of Liquids
        rhs = -3.0171 + (442.78 + 1.6452 * Tb_cels) / (T_cels + 239 - 0.19 * Tb_cels)
        nu_i = PintUnits.Quantity(np.exp(rhs), "mm^2/s").to("m^2/s")
        return nu_i[comp_idx] if comp_idx is not None else nu_i

    def viscosity_dynamic(
        self, T: PintScalar, comp_idx: int | None = None
    ) -> PintVector:
        """
        Calculate liquid dynamic viscosity based on droplet temperature and density.

        :meta private: Uses Dutt's equation (4.23) for kinematic viscosity, combined with density.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Dynamic viscosity in Pa*s.
        :rtype: np.ndarray
        """
        mu_i = (self.viscosity_kinematic(T) * self.density(T)).to("Pa*s")
        return mu_i[comp_idx] if comp_idx is not None else mu_i

    def Cp(self, T: PintScalar, comp_idx: int | None = None) -> PintVector:
        """
        Compute molar specific heat capacity at a given temperature.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Molar specific heat capacity in J/mol/K.
        :rtype: np.ndarray
        """
        T = T.to("K")
        theta = (T - T_STP) / PintUnits.Quantity(700, "K")

        Cp_stp = self.Cp_stp
        Cp_B = self.Cp_B
        Cp_C = self.Cp_C

        cp = (Cp_stp + Cp_B * theta + Cp_C * theta**2).to("J/(mol*K)")
        return cp[comp_idx] if comp_idx is not None else cp

    def Cl(self, T: PintScalar, comp_idx: int | None = None) -> PintVector:
        """
        Compute liquid mass specific heat capacity in J/kg/K at a given temperature.

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Mass specific heat capacity in J/kg/K.
        :rtype: np.ndarray
        """
        cl = (self.Cp(T) / self.MW).to("J/(kg*K)")
        return cl[comp_idx] if comp_idx is not None else cl

    def psat(
        self,
        T: PintScalar,
        comp_idx: int | None = None,
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> PintVector:
        """
        Compute saturated vapor pressure.

        :meta private: Can use Ambrose-Walton or Lee-Kesler correlations (default Lee-Kesler).

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Saturated vapor pressure in Pa.
        :rtype: np.ndarray
        """
        Tr = T.to("K") / self.Tc.to("K")
        Pc = self.Pc.to("Pa")
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

        psat = (Pc * rhs).to("Pa")
        return psat[comp_idx] if comp_idx is not None else psat

    def psat_antoine_coeffs(
        self,
        Tvals: PintVector | None = None,
        units: Literal["mks", "cgs", "bar", "atm", "dyne/cm^2", "Pa"] = "mks",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate Antoine coefficients for vapor pressure of an individual compound.

        :param Tvals: Temperature range or nodes for Antoine fit in Kelvin (default [273.15, Tb_i]).
        :type Tvals: np.ndarray, optional
        :param units: Units for pressure in fit ("mks", "cgs", "bar", "atm", "dyne/cm^2", "Pa")
        :type units: str, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Coefficients A, B, C, D
        :rtype: 4 np.ndarrays
        """
        # Convert ``cgs`` and ``mks`` to actual units
        units = "dyne/cm^2" if units == "cgs" else "Pa" if units == "mks" else units
        # Define or get temperature nodes for fit
        if Tvals is None:
            print("Tvals not specified, using [273.15, Tb_i] for each compound.")
            # Initialize as zeros for now, calculated for each compound later
            T = PintUnits.Quantity(np.zeros(20), "K")
        elif len(Tvals) == 2:
            T = PintUnits.Quantity(np.linspace(Tvals[0], Tvals[1], 20), "K")
        elif len(Tvals) > 2:
            T = Tvals.to("K")
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        # Antoine equation log10(p) = A - B/(C + T)
        def antoine_eq(T, A, B, C):
            """Antoine equation for vapor pressure."""
            return A - B / (T + C)

        # Determine conversion factor for pressure in MKS, CGS, bar, or atm
        D = PintUnits.Quantity(1, "Pa").to(units)

        # Fit Antoine coefficients for each compound
        A = np.zeros(self.num_compounds)
        B = np.zeros(self.num_compounds)
        C = np.zeros(self.num_compounds)
        for i in range(self.num_compounds):
            # Update T if not specified
            if Tvals is None:
                T = PintUnits.Quantity(np.linspace(T_STP, self.Tb[i], 20), "K")
            T_mag = T.to("K").magnitude
            Pvals = np.zeros_like(T_mag)
            for k in range(len(T)):
                Pvals[k] = (
                    self.psat(T[k], correlation=correlation)[i].to(units).magnitude
                )

            logP = np.log10(Pvals)
            popt, _ = curve_fit(antoine_eq, T_mag, logP, p0=[1, 1e3, -1])
            A[i], B[i], C[i] = popt
        D = D.magnitude + np.zeros(self.num_compounds)  # make D an array
        return A, B, C, D

    def latent_heat_vaporization(
        self, T: PintScalar, comp_idx: int | None = None
    ) -> PintVector:
        """
        Calculate latent heat of vaporization adjusted for temperature.

        :param T: Temperature in Kelvin.
        :type T: PintScalar
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Latent heat of vaporization in J/kg.
        :rtype: PintArray
        """
        Tc = self.Tc
        Tb = self.Tb
        Lv_stp = self.Lv_stp

        # Reduced temperatures
        Tr = T.to("K") / Tc.to("K")
        Trb = Tb.to("K") / Tc.to("K")

        Lvi = PintUnits.Quantity(
            np.where(
                T.to("K") > Tc.to("K"),
                0.0,
                Lv_stp * np.power((1.0 - Tr) / (1.0 - Trb), 0.38),
            ),
            "J/kg",
        )

        return Lvi[comp_idx] if comp_idx is not None else Lvi

    def diffusion_coeff(
        self,
        p: PintScalar,
        T: PintScalar,
        sigma_gas: PintScalar = Sigma_g,
        epsilonByKB_gas: PintScalar = EpsilonByKB_g,
        MW_gas: PintScalar = MW_g,
        correlation: Literal["Tee", "Wilke"] = "Tee",
    ) -> PintVector:
        """
        Compute diffusion coefficients using Lennard-Jones parameters.

        :meta private: Uses Wilke and Lee method (Poling, equation 11-4.1).
        :meta private: Ambient gas defaults to air parameters.

        :param p: Pressure in Pa.
        :type p: float
        :param T: Temperature in Kelvin.
        :type T: float
        :param sigma_gas: Collision diameter in m.
        :type sigma_gas: float, optional
        :param epsilonByKB_gas: Well depth over Boltzmann constant, in K.
        :type epsilonByKB_gas: float, optional
        :param MW_gas: Mean molecular weight of ambient gas in kg/mol.
        :type MW_gas: float, optional
        :param correlation: Method to calculate sigma and epsilon ("Tee" or "Wilke").
        :type correlation: str, optional
        :return: Diffusion coefficient.
        :rtype: np.ndarray
        """

        # Method of Tee for calculating liquid sigma and epsilon
        if correlation.casefold() == "Tee".casefold():
            sigma_i = self.sigma.to("angstrom")
            epsilonByKB_i = self.epsilonByKB.to("K")
        else:
            # Method of Wilke & Lee calculating liquid sigma and epsilon
            Vmb_i = PintUnits.Quantity(np.zeros_like(self.Tb), "cm^3/mol")
            for n in range(self.num_compounds):
                Vmb_i[n] = self.molar_liquid_vol(self.Tb[n])[n].to("cm^3/mol")
            sigma_i = PintUnits.Quantity(
                1.18 * np.power(Vmb_i.magnitude, 1 / 3), "angstrom"
            )  # Angstroms, Poling (11-4.2)
            epsilonByKB_i = 1.15 * self.Tb.to("K")  # K , Poling (11-4.3)

        # Compute binary sigma and epsilon
        sigma_gas = sigma_gas.to("angstrom")
        sigmaAB_i = (sigma_gas + sigma_i) / 2  # Angstroms, Poling (11-3.5)
        epsilonAB_byKB_i = np.sqrt(
            epsilonByKB_gas.to("K") * epsilonByKB_i
        )  # K, Poling (11-3.4)

        # Dimensionless collision integral for diffusion: Poling (11-3.6)
        Tstar_i = T / epsilonAB_byKB_i  # [1]
        A = 1.06036
        B = 0.15610
        C = 0.193
        D = 0.47635
        E = 1.03587
        F = 1.52996
        G = 1.76474
        H = 3.89411
        omegaD_i = (
            A / np.power(Tstar_i, B)
            + C / np.exp(D * Tstar_i)
            + E / np.exp(F * Tstar_i)
            + G / np.exp(H * Tstar_i)
        )

        # Convert molecular weights from kg/mol to g/mol then calculate M_AB
        MW_gas = MW_gas.to("g/mol")
        MW_i = self.MW.to("g/mol")
        M_AB_i = 2 * (MW_i * MW_gas) / (MW_i + MW_gas)  # g/mol, see Poling (11-3.1)

        # Convert pressure from Pa to bar
        p = p.to("bar")  # bar

        # Binary diffusion coefficients, Poling (11-4.1)
        D_AB_i = PintUnits.Quantity(
            1e-3
            * (3.03 - 0.98 / np.sqrt(M_AB_i.magnitude))
            * np.power(T.magnitude, 1.5)
            / (
                p.magnitude
                * np.sqrt(M_AB_i.magnitude)
                * np.power(sigmaAB_i.magnitude, 2)
                * omegaD_i.magnitude
            ),
            "cm^2/s",
        ).to("m^2/s")

        return D_AB_i

    def surface_tension(
        self,
        T: PintScalar,
        comp_idx: int | None = None,
        correlation: Literal["Pitzer", "Brock-Bird"] = "Brock-Bird",
    ) -> PintVector:
        """
        Calculate surface tension of each compound at a given temperature.

        :meta private: Uses Brock-Bird (default) or Pitzer correlations (Poling 12-3.5, 12-3.7).

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :param correlation: Correlation method ("Brock-Bird" or "Pitzer").
        :type correlation: Literal["Pitzer", "Brock-Bird"], optional
        :return: Surface tension in N/m.
        :rtype: np.ndarray
        """
        T = T.to("K")
        Tc = self.Tc.to("K")
        Pc = self.Pc.to("bar")
        Tb = self.Tb.to("K")

        Tr = T / Tc

        if correlation.casefold() == "Brock-Bird".casefold():
            Tbr = Tb / Tc
            Q = (
                0.1196 * (1.0 + (Tbr * np.log(Pc.magnitude / 1.01325)) / (1.0 - Tbr))
                - 0.279
            )
        else:
            w = self.omega.magnitude
            Q = (
                (1.86 + 1.18 * w)
                / 19.05
                * (np.power((3.75 + 0.91 * w) / (0.291 - 0.08 * w), 2.0 / 3.0))
            )

        st = PintUnits.Quantity(
            np.power(Pc.magnitude, 2.0 / 3.0)
            * np.power(Tc.magnitude, 1.0 / 3.0)
            * Q.magnitude
            * np.power(1 - Tr.magnitude, 11.0 / 9.0),
            "dyn/cm",
        ).to("N/m")

        return st[comp_idx] if comp_idx is not None else st

    def thermal_conductivity(
        self, T: PintScalar, comp_idx: int | None = None
    ) -> PintVector:
        """
        Calculate thermal conductivity at a given temperature.

        :meta private: Uses Latini et al. method (Poling equation 10-9.1).

        :param T: Temperature in Kelvin.
        :type T: float
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Thermal conductivity in W/m/K.
        :rtype: np.ndarray
        """
        T = T.to("K")
        Tc = self.Tc.to("K")
        Tb = self.Tb.to("K")
        fam = self.fam

        Astar = np.where(
            fam == 1,
            0.0346,
            np.where(
                fam == 2,
                0.0310,
                np.where(
                    fam == 3,
                    0.0361,
                    0.00350,
                ),
            ),
        )
        beta = np.where(
            fam == 1,
            1.0,
            np.where(
                fam == 2,
                1.0,
                np.where(
                    fam == 3,
                    1.0,
                    0.5,
                ),
            ),
        )
        MW = np.power(
            self.MW.to("g/mol").magnitude,
            beta,  ## Sqrt iff fam !E {1, 2, 3}
        )

        Tr = (T / Tc).magnitude
        A = Astar * np.power(Tb.magnitude, 1.2) / (MW * np.power(Tc.magnitude, 0.167))
        tc = PintUnits.Quantity(
            A * np.power(1 - Tr, 0.38) / np.power(Tr, 1.0 / 6.0), "W/(m*K)"
        )

        return tc[comp_idx] if comp_idx is not None else tc

    def liquid_heat_capacity(self, T: PintScalar) -> PintVector:
        """
        Calculate Ruzicka-Domalski liquid heat capacity at a given temperature.

        :param T: Temperature in Kelvin.
        :type T: float
        :return: Liquid heat capacity in J/kg/K.
        :rtype: np.ndarray
        """
        R = PintUnits.Quantity(1, "R").to("J/(mol*K)")
        Tr = T.to("K") / 100.0
        Cpl_A = self.get_gcm_property("gani", "rd_A")
        Cpl_B = self.get_gcm_property("gani", "rd_B")
        Cpl_D = self.get_gcm_property("gani", "rd_D")

        cp_molar = R * (Cpl_A + Cpl_B * Tr + Cpl_D * Tr**2)
        return (cp_molar / self.MW.to("kg/mol")).to("J/kg/K")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Mixture property correlations                                                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def mean_molecular_weight(self, Yi: FloatVector) -> PintScalar:
        """
        Calculate the mean molecular weight of the mixture.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :return: Mean molecular weight of the mixture in kg/mol.
        :rtype: PintScalar
        """
        if np.sum(Yi) != 0:
            Mbar = 1 / np.sum(Yi / self.MW.to("kg/mol"))
        else:
            Mbar = 0.0
        return PintUnits.Quantity(Mbar, "kg/mol")

    def mixture_density(self, Yi: FloatVector, T: PintScalar) -> PintScalar:
        """
        Calculate mixture density at a given temperature.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: PintScalar
        :return: Mixture density in kg/m^3.
        :rtype: PintScalar
        """
        MW = self.MW.to("kg/mol")
        Vmi = self.molar_liquid_vol(T).to("m^3/mol")
        rho: PintScalar = Yi @ (MW / Vmi)  # ty: ignore[invalid-assignment]
        return rho.to("kg/m^3")

    def mixture_kinematic_viscosity(
        self,
        Yi: FloatVector,
        T: PintScalar,
        correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> PintScalar:
        """
        Calculate kinematic viscosity of the mixture.

        :meta private: Uses Kendall-Monroe (default) or Arrhenius mixing correlations.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: PintScalar
        :param correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
        :type correlation: str, optional
        :return: Mixture kinematic viscosity in m^2/s.
        :rtype: PintScalar
        """
        nu_i = (
            self.viscosity_kinematic(T).to("m^2/s").magnitude
        )  # Viscosities of individual components

        # Calculate mole fractions for each species
        Xi = self.Y2X(Yi)

        if correlation.casefold() == "Arrhenius".casefold():
            # Arrhenius mixing correlation
            nu = np.exp(np.sum(Xi * np.log(nu_i)))
        else:
            # Default: Kendall-Monroe mixing correlation
            nu = np.power(np.sum(Xi * np.power(nu_i, 1.0 / 3.0)), 3.0)

        return PintUnits.Quantity(nu, "m^2/s")

    def mixture_dynamic_viscosity(
        self,
        Yi: FloatVector,
        T: PintScalar,
        correlation: Literal["Kendall-Monroe", "Arrhenius"] = "Kendall-Monroe",
    ) -> PintScalar:
        """
        Calculate dynamic viscosity of the mixture.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: PintScalar
        :param correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
        :type correlation: str, optional
        :return: Mixture dynamic viscosity in Pa*s.
        :rtype: PintScalar
        """
        nu = self.mixture_kinematic_viscosity(Yi, T, correlation=correlation)
        rho = self.mixture_density(Yi, T)
        return (rho * nu).to("Pa*s")

    def mixture_vapor_pressure(
        self,
        Yi: FloatVector,
        T: PintScalar,
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> PintScalar:
        """
        Calculate vapor pressure of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: PintScalar
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Mixture vapor pressure in Pa.
        :rtype: PintScalar
        """
        # Mole fraction for each compound
        Xi = self.Y2X(Yi)
        # Saturated vapor pressure for each compound (Pa)
        p_sati = self.psat(T, correlation=correlation).to("Pa")
        # Mixture vapor pressure via Raoult's law
        return (p_sati @ Xi).to("Pa")

    def mixture_vapor_pressure_antoine_coeffs(
        self,
        Yi: FloatVector,
        Tvals: PintVector | None = None,
        units: Literal["mks", "cgs", "bar", "atm", "dyne/cm^2", "Pa"] = "mks",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ):
        """
        Estimate Antoine coefficients for vapor pressure of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param Tvals: Temperature range or nodes for Antoine fit in Kelvin (default [273.15, min(Tb)]).
        :type Tvals: np.ndarray, optional
        :param units: Units for pressure in fit ("mks", "cgs", "bar", "atm")
        :type units: str, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Coefficients A, B, C, D
        :rtype: float
        """
        Tvals = Tvals.to("K") if Tvals is not None else None
        units = "dyne/cm^2" if units == "cgs" else "Pa" if units == "mks" else units
        # Define or get temperature nodes for fit
        if Tvals is None:
            print("Tvals not specified, using [273.15, min(Tb_mix)] for mixture.")
            X = self.Y2X(Yi)
            Tb = mixing_rule(self.Tb, X)
            T = PintUnits.Quantity(np.linspace(T_STP, np.min(Tb), 20), "K")
        elif len(Tvals) == 2:
            T = PintUnits.Quantity(np.linspace(Tvals[0], Tvals[1], 20), "K")
        elif len(Tvals) > 2:
            T = Tvals.to("K")
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        # Antoine equation log10(p) = A - B/(C + T)
        def antoine_eq(T, A, B, C):
            """Antoine equation for vapor pressure."""
            return A - B / (T + C)

        # Determine conversion factor for pressure in MKS, CGS, bar, or atm
        D = PintUnits.Quantity(1, "Pa").to(units)

        T_mag = T.to("K").magnitude
        Pvals = np.zeros_like(T_mag)
        for k in range(len(T)):
            Pvals[k] = (
                self.mixture_vapor_pressure(Yi, T[k], correlation=correlation)
                .to(units)
                .magnitude
            )

        logP = np.log10(Pvals)
        popt, _ = curve_fit(antoine_eq, T_mag, logP, p0=[1, 1e3, -1])  # initial guess
        A, B, C = popt

        return A, B, C, D.magnitude

    def mixture_surface_tension(
        self,
        Yi: FloatVector,
        T: PintScalar,
        correlation: Literal["Pitzer", "Brock-Bird"] = "Brock-Bird",
    ) -> PintScalar:
        """
        Calculate surface tension of the mixture.

        :meta private: Uses arithmetic pseudo-property method recommended by Hugill and van Welsenes (1986).

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :param correlation: Correlation method ("Pitzer" or "Brock-Bird").
        :type correlation: str, optional
        :return: Mixture surface tension in N/m.
        :rtype: pint.Quantity
        """
        # Mole fraction for each compound
        Xi = self.Y2X(Yi)
        # Surface tension for each compound (N/m)
        sti = self.surface_tension(T, correlation=correlation)
        # Mixture surface tension via arithmetic mean, Poling (12-5.2)
        return mixing_rule(sti, Xi, "arithmetic").to("N/m")

    def mixture_thermal_conductivity(
        self, Yi: FloatVector, T: PintScalar
    ) -> PintScalar:
        """
        Calculate thermal conductivity of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param T: Temperature in Kelvin.
        :type T: float
        :return: Thermal conductivity in W/m/K.
        :rtype: pint.Quantity
        """
        tc = self.thermal_conductivity(T)
        return np.power(np.sum(Yi * np.power(tc, -2)), -0.5).to("W/(m*K)")

    def mixture_heat_of_combustion(
        self, Yi: FloatVector, basis: Literal["mass", "mole"] = "mass"
    ) -> PintScalar:
        """
        Calculate the heat of combustion of the mixture.

        :meta public: Hess cycle with the Const-Gani ideal-gas formation enthalpy and a
        liquid-phase correction. Result is an engineering estimate related to ASTM
        D4809/D3338 heating-value characterization--it is not a simulated bomb-
        calorimeter measurement.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param basis: Whether to return the heat of combustion in "mass" (MJ/kg) or
        "mole" (kJ/mol).
        :type basis: str, optional
        :return: Mixture heat of combustion in kJ/mol or MJ/kg.
        :rtype: pint.Quantity
        """
        if np.any(self.hc_type == "NaN"):
            msg = "Heat of combustion is only compatible with hydrocarbon compounds."
            raise ValueError(msg)

        Hf_liq = self.Hf - self.Hv_stp
        Lhv_hess = -1.0 * (
            self.nC * PintUnits.Quantity(-393.51, "kJ/mol")
            + (self.nH / 2) * PintUnits.Quantity(-241.83, "kJ/mol")
            - Hf_liq.to("kJ/mol")
        )
        if basis == "mass":
            return PintUnits.Quantity(np.sum(Yi * Lhv_hess / self.MW.to("kg/mol"))).to(
                "J/kg"
            )
        Xi = self.Y2X(Yi)
        return PintUnits.Quantity(np.sum(Xi * Lhv_hess)).to("J/mol")

    def mixture_freezing_point(
        self,
        Yi: FloatVector,
        method: Literal["Boehm2022"] = "Boehm2022",
        alpha: float = 1.0,
        n_iter: int = 8,
    ) -> PintScalar:
        """
        Calculate the freezing point of the mixture.

        :meta public: Freezing point calculation using the Boehm et al. (2022) method
        for ideal solutions. This is an equilibrium-based estimation and does not model
        cooling rate, supercooling, crystal kinetics, or detailed solid-phase non-
        ideality.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param method: Method to use for calculating the freezing point. Currently only "Boehm2022" is supported.
        :type method: str, optional
        :param alpha: Non-ideality parameter for the mixture.
        :type alpha: float, optional
        :param n_iter: Number of iterations for the freezing point calculation.
        :type n_iter: int, optional
        :return: Freezing point of the mixture in Kelvin.
        :rtype: pint.Quantity
        :raises ValueError: If the calculation fails due to non-physical values.
        """
        R = PintUnits.Quantity(1, "R").to("J/(mol*K)")
        Xi = np.clip(self.Y2X(Yi), 1e-6, 1.0 - 1e-6)
        Tm = self.get_gcm_property("gani", "Tm").to("K")
        dH_fus = self.get_gcm_property("boehm", "dH_fus").to("kJ/mol")
        dS_fus = self.get_gcm_property("boehm", "dS_fus").to("kJ/(mol*K)")
        dS_mix = -R / Xi * ((1.0 - Xi) * np.log(1.0 - Xi) + Xi * np.log(Xi))
        dCpl = -0.35 * self.liquid_heat_capacity(T_STP) * self.MW

        T_j = Tm * np.ones_like(Xi)
        for _ in range(n_iter):
            T_j = np.maximum(T_j, PintUnits.Quantity(1.0, "K"))
            num = dH_fus + Xi * dCpl * (Tm - T_j)
            denom = dS_fus + Xi * dCpl * np.log(T_j / Tm) + alpha * dS_mix
            T_j = num / (denom + PintUnits.Quantity(1e-30, "J/(mol*K)"))
            #NOTE: Root-finding problem

        Tf_j = np.where(T_j.magnitude > 0, T_j.magnitude, -np.inf)
        Tf = np.max(np.where(Xi > 1e-6, Tf_j, -np.inf))
        return PintUnits.Quantity(Tf, "K")

    def mixture_flash_point(
        self,
        Yi: FloatVector,
        method: Literal["Alibakhshi", "Alqaheem"] = "Alibakhshi",
        mixing: Literal["Liaw", "linear"] = "Liaw",
    ) -> PintScalar:
        """
        Calculate the flash point of the mixture.

        :meta public: Pure-component values use either the Alibakhshi et al. (2015)
        group contribution model or the Alqaheem-Riazi correlation. Mixtures use the
        ideal Liaw-Chiu modified Le Chatelier criterion by default, with a mole-
        fraction-linear rule available as a simpler alternative.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: np.ndarray
        :param method: Pure-component method. Options are "Alibakhshi" or "Alqaheem".
        :type method: str, optional
        :param mixing: Mixing rule. Options are "Liaw" or "linear".
        :type mixing: str, optional
        :return: Flash point of the mixture in Kelvin.
        :rtype: pint.Quantity
        :raises ValueError: If the calculation fails due to non-physical values.
        """

        """
        def _fp_alibakhshi(Tb, phi_sum):
            return 12.14 + 0.73 * Tb + phi_sum
        def _fp_alqaheem(Tb):
            return 0.70 * Tb
        if Yi is None:
            Yi = self.Y_0
        if method.casefold() == "alibakhshi":
            component_flash_points = _fp_alibakhshi(self.Tb_astm, self.alibakhshi_phi)
        elif method.casefold() == "alqaheem":
            component_flash_points = _fp_alqaheem(self.Tb_astm)
        else:
            raise NotImplementedError(
                f"flash_point method '{method}' not supported "
                "(use 'Alibakhshi' or 'Alqaheem')."
            )

        Xi = self.Y2X(np.asarray(Yi, dtype=float))
        if mixing.casefold() == "linear":
            return float(np.sum(Xi * component_flash_points))
        if mixing.casefold() == "liaw":
            return float(
                _fp_liaw_ideal_iter(
                    Xi,
                    component_flash_points,
                    self.Tc,
                    self.Pc,
                    self.omega_astm,
                )
            )
        raise NotImplementedError(
            f"flash_point mixing rule '{mixing}' not supported "
            "(use 'Liaw' or 'linear')."
        )
        """
        # Placeholder implementation
        return PintUnits.Quantity(298.15, "K")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Utility functions                                                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def mass2Y(self, mass: PintVector) -> FloatVector:
        """
        Calculate the mass fractions from the mass of each component.

        :param mass: Mass of each compound.
        :type mass: np.ndarray
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        mass = mass.to("kg")
        # Normalize to get group mole fractions
        total_mass = np.sum(mass)
        if total_mass != 0:
            return (mass / total_mass).magnitude
        else:
            return np.zeros_like(self.MW.magnitude)

    def mass2X(self, mass: PintVector) -> FloatVector:
        """
        Calculate the mole fractions from the mass of each component.

        :param mass: Mass of each compound.
        :type mass: pint.Quantity[np.ndarray]
        :return: Mole fractions of the compounds (shape: num_comps,).
        :rtype: np.ndarray
        """
        # Calculate the number of moles for each compound
        num_mole = mass.to("kg") / self.MW.to("kg/mol")
        # Normalize to get group mole fractions
        total_moles = np.sum(num_mole)
        if total_moles != 0:
            return (num_mole / total_moles).magnitude
        else:
            return np.zeros_like(self.MW.magnitude)

    def X2Y(self, Xi: FloatVector) -> FloatVector:
        """
        Calculate the mass fractions from the mole fractions of each component.

        :param Xi: Mole fractions of each compound.
        :type Xi: np.ndarray
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        # Calculate the mass for each compound
        mass = Xi * self.MW.to("kg/mol").magnitude

        # Normalize to get group mass fractions
        total_mass = np.sum(mass)
        if total_mass != 0:
            return mass / total_mass
        else:
            return np.zeros_like(self.MW.magnitude)

    def Y2X(self, Yi: FloatVector) -> FloatVector:
        """
        Calculate the mole fractions from the mass fractions of each component.

        :param Yi: Mass fractions of each compound.
        :type Yi: np.ndarray
        :return: Mole fractions of the compounds (shape: num_compounds,).
        :rtype: np.ndarray
        """
        if np.sum(Yi) != 0:
            return (self.mean_molecular_weight(Yi) * Yi / self.MW).magnitude
        else:
            return np.zeros_like(self.MW, dtype=np.float64)


__all__ = ["Fuel"]
