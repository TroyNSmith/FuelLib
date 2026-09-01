"""Fuel class for Group Contribution Method calculations."""

import os
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
import unxt as u
from jax import Array
from scipy.optimize import curve_fit

from ._data_locator import (
    get_fueldata_decomp_dir,
    get_fueldata_dir,
    get_fueldata_gc_dir,
    get_fueldata_props_dir,
    get_gcmtable_dir,
    get_metadata_decomp_name,
)
from .convert import K2C
from .units import load_quantity
from .utility import mixing_rule


def _ustrip(q: u.AbstractQuantity, unit: str) -> Array:
    """Strip units to a bare array; narrows ty's overly-wide inference of unxt's dispatched `Quantity.ustrip`."""
    return q.ustrip(unit)  # ty: ignore[invalid-return-type]


def _atleast_col(x: Array) -> Array:
    """Add a trailing axis so an array of temperatures broadcasts against a per-compound axis; scalars pass through."""
    return x[:, None] if jnp.ndim(x) > 0 else x


def _molar_liquid_vol_core(T_K: Array, Tc: Array, omega: Array, Vm_stp: Array) -> Array:
    """Elementwise molar liquid volume, shape-agnostic (no outer-product broadcasting)."""
    Tstp = 298.0
    phi = jnp.where(
        T_K > Tc,
        -((1 - (Tstp / Tc)) ** (2.0 / 7.0)),
        ((1 - (T_K / Tc)) ** (2.0 / 7.0)) - ((1 - (Tstp / Tc)) ** (2.0 / 7.0)),
    )
    z = 0.29056 - 0.08775 * omega
    return Vm_stp * jnp.power(z, phi)


class fuel:
    """
    Class for handling group contribution calculations of thermodynamic and mixture properties.

    :param name: Name of the mixture as it appears in its gcData file.
    :type name: str
    :param decompName: Name of the groupDecomposition file if different from name. Defaults to None.
    :type decompName: str, optional
    :param fuelDataDir: Directory where the fuel data is stored. If None, uses built-in embedded data.
    :type fuelDataDir: str, optional
    """

    # Type annotations for documented attributes
    #: Root directory for fuel data (custom or embedded)
    fuelDataDir: str

    #: Directory containing GCxGC compositional data files
    fuelDataGcDir: str

    #: Directory containing functional group decomposition files
    fuelDataDecompDir: str

    #: Directory containing experimental property data (may be None)
    fuelDataPropsDir: str

    #: Name of the fuel/mixture
    name: str

    #: List of compound names in the mixture
    compounds: list

    #: Molecular formulas for each compound
    formulas: np.ndarray | None

    #: Mass fractions of each compound. Shape: (num_compounds,)
    Y_0: Array

    #: Functional group decomposition matrix. Shape: (num_compounds, num_groups)
    Nij: Array

    #: Number of compounds in the mixture
    num_compounds: int

    #: Number of functional groups in the decomposition
    num_groups: int

    #: Molecular weights in kg/mol. Shape: (num_compounds,)
    MW: u.Quantity

    #: Critical temperatures in K. Shape: (num_compounds,)
    Tc: u.Quantity

    #: Critical pressures in Pa. Shape: (num_compounds,)
    Pc: u.Quantity

    #: Critical volumes in m³/mol. Shape: (num_compounds,)
    Vc: u.Quantity

    #: Boiling temperatures in K. Shape: (num_compounds,)
    Tb: u.Quantity

    #: Melting temperatures in K. Shape: (num_compounds,)
    Tm: u.Quantity

    #: Enthalpy of formation in J/mol. Shape: (num_compounds,)
    Hf: u.Quantity

    #: Gibbs free energy in J/mol. Shape: (num_compounds,)
    Gf: u.Quantity

    #: Enthalpy of vaporization at 298 K in J/mol. Shape: (num_compounds,)
    Hv_stp: u.Quantity

    #: Latent heat of vaporization at 298 K in J/kg. Shape: (num_compounds,)
    Lv_stp: u.Quantity

    #: Molar specific heat at 298 K in J/mol/K. Shape: (num_compounds,)
    Cp_stp: u.Quantity

    #: Molar liquid volume at 298 K in m³/mol. Shape: (num_compounds,)
    Vm_stp: u.Quantity

    #: Acentric factors (dimensionless). Shape: (num_compounds,)
    omega: u.Quantity

    #: Lennard-Jones collision diameters in m. Shape: (num_compounds,)
    sigma: u.Quantity

    #: Lennard-Jones well depths in K. Shape: (num_compounds,)
    epsilonByKB: u.Quantity

    #: Hydrocarbon types ("n-alkane", "iso-alkane", "cyclo-alkane", "aromatic", "alkene")
    hc_type: np.ndarray

    #: Family codes for thermal conductivity (0: saturated, 1: aromatic, 2: cycloparaffin, 3: olefin)
    fam: Array

    #: Carbon numbers. Shape: (num_compounds,)
    nC: Array

    #: Hydrogen numbers. Shape: (num_compounds,)
    nH: Array

    #: PelePhysics keys for each compound (if available)
    pelephysics_keys: np.ndarray | None

    # Number of first and second order groups from Constantinou and Gani
    N_g1 = 78
    N_g2 = 43

    @property
    def num_compounds(self) -> int:
        """
        Return the number of compounds in the mixture.

        :return: Number of compounds.
        :rtype: int
        """
        return self.Nij.shape[0]

    @property
    def num_groups(self) -> int:
        """
        Return the number of functional groups in the decomposition.

        :return: Number of functional groups.
        :rtype: int
        """
        return self.Nij.shape[1]

    def __init__(
        self, name: str, decompName: str | None = None, fuelDataDir: str | None = None
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

        self.name = name
        if decompName is None:
            # Try to get decomposition name from metadata
            decompName = get_metadata_decomp_name(name, fuelDataDir)

        # Determine and set data directories for this fuel instance
        if fuelDataDir is None:
            # Use built-in embedded data
            self.fuelDataDir = get_fueldata_dir()
            self.fuelDataGcDir = get_fueldata_gc_dir()
            self.fuelDataDecompDir = get_fueldata_decomp_dir()
            self.fuelDataPropsDir = get_fueldata_props_dir()
        else:
            # Validate and use custom fuel directory
            from ._data_locator import (
                _get_props_dir_for_fueldata,
                _validate_fuel_data_dir,
            )

            _validate_fuel_data_dir(fuelDataDir)
            self.fuelDataDir = fuelDataDir
            self.fuelDataGcDir = os.path.join(fuelDataDir, "gcData")
            self.fuelDataDecompDir = os.path.join(fuelDataDir, "groupDecompositionData")
            self.fuelDataPropsDir = _get_props_dir_for_fueldata(fuelDataDir)

        # Get GCM table directory (always from built-in data)
        gcmtable_dir = get_gcmtable_dir()

        self.groupDecompFile = os.path.join(self.fuelDataDecompDir, f"{decompName}.csv")
        self.gcxgcFile = os.path.join(self.fuelDataGcDir, f"{name}_init.csv")
        self.gcmTableFile = os.path.join(gcmtable_dir, "gcmTable.csv")

        # Read functional group data for mixture (num_compounds,num_groups)
        df_Nij = pd.read_csv(self.groupDecompFile)
        self.Nij = jnp.array(df_Nij.iloc[:, 1:].to_numpy())

        # Classify hydrocarbon by family (used in thermal conductivity)
        # 0: saturated hydrocarbons
        # 1: aromatics
        # 2: cycloparaffins
        # 3: olefins
        aromatics = 10  # starting index for aromatic groups
        num_aromatics = 5
        branching = 78  # starting index for branching groups (Group j (CH3)2CH through C(CH3)2C(CH3)2)
        num_branching = 5  # groups 78-82 inclusive
        cyclos = 83  # starting index for membered ring groups (3-7 membered rings)
        num_cyclos = 5
        olefins = 4  # starting index for double bound groups
        num_olefins = 6

        # Vectorized per-compound group membership (jax arrays can't be mutated in a loop)
        has_aromatic = (
            jnp.sum(self.Nij[:, aromatics : aromatics + num_aromatics], axis=1) > 0
        )
        has_cyclo = jnp.sum(self.Nij[:, cyclos : cyclos + num_cyclos], axis=1) > 0
        has_olefin = jnp.sum(self.Nij[:, olefins : olefins + num_olefins], axis=1) > 0
        has_branching = (
            jnp.sum(self.Nij[:, branching : branching + num_branching], axis=1) > 0
        )

        self.fam = jnp.where(
            has_aromatic, 1, jnp.where(has_cyclo, 2, jnp.where(has_olefin, 3, 0))
        )

        # Classify hydrocarbon by type (n-alkane, iso-alkane, cyclo-alkane, aromatic)
        # Based on group decompositions from Constantinou-Gani method
        # hc_type holds strings, so it stays a numpy object array (jax has no string dtype)
        self.hc_type = np.full(self.num_compounds, "n-alkane", dtype=object)
        self.hc_type[np.asarray(has_branching)] = "iso-alkane"
        self.hc_type[np.asarray(has_olefin)] = "alkene"
        self.hc_type[np.asarray(has_cyclo)] = "cyclo-alkane"
        self.hc_type[np.asarray(has_aromatic)] = "aromatic"

        # Calculate carbon and hydrogen numbers from first-order group decomposition
        # For jet fuels, use only alkyl (0-3) and aromatic (10-14) groups
        # Alkyl: CH3=1C,3H; CH2=1C,2H; CH=1C,1H; C=1C,0H
        # Aromatic: ACH=1C,1H; AC=1C,0H; ACCH3=2C,3H; ACCH2=2C,2H; ACCH=2C,1H
        alkyl_carbons = jnp.array([1, 1, 1, 1])  # groups 0-3
        alkyl_hydrogens = jnp.array([3, 2, 1, 0])
        # Olefinic: group 4 appears to represent 2 carbons with 3 hydrogens in UNIFAC-based system
        olefinic_carbons = jnp.array([2, 1, 1, 0, 0, 0])  # groups 4-9
        olefinic_hydrogens = jnp.array([3, 1, 0, 0, 0, 0])
        aromatic_carbons = jnp.array([1, 1, 2, 2, 2])  # groups 10-14
        aromatic_hydrogens = jnp.array([1, 0, 3, 2, 1])

        # Alkyl + olefinic + aromatic contributions, vectorized over all compounds
        self.nC = (
            self.Nij[:, 0:4] @ alkyl_carbons
            + self.Nij[:, 4:10] @ olefinic_carbons
            + self.Nij[:, 10:15] @ aromatic_carbons
        )
        self.nH = (
            self.Nij[:, 0:4] @ alkyl_hydrogens
            + self.Nij[:, 4:10] @ olefinic_hydrogens
            + self.Nij[:, 10:15] @ aromatic_hydrogens
        )

        # Read GCxGC/compound data
        df_gcxgc = pd.read_csv(self.gcxgcFile)

        self.compounds = [
            compound.strip() for compound in df_gcxgc["Compound"].to_numpy()
        ]

        # Load molecular formulas if available
        if "Formula" in df_gcxgc.columns:
            self.formulas = np.array(
                [
                    formula.strip() if pd.notna(formula) else None
                    for formula in df_gcxgc["Formula"].to_numpy()
                ]
            )
        else:
            self.formulas = None

        if "PelePhysics Key" in df_gcxgc.columns:
            # Stays a numpy object array: jax has no string dtype support
            self.pelephysics_keys = np.array(
                [key.strip() for key in df_gcxgc["PelePhysics Key"].to_numpy()]
            )
        else:
            self.pelephysics_keys = None

        self.Y_0 = jnp.array(df_gcxgc["Weight %"].to_numpy().flatten().astype(float))
        self.Y_0 /= jnp.sum(self.Y_0)

        # Make sure mixture data is consistent:
        if self.num_groups < self.N_g1:
            raise ValueError(
                f"Insufficient mixture description:\n"
                f"The number of columns in {self.groupDecompFile} is less than "
                f"the required number of first-order groups (N_g1 = {self.N_g1})."
            )
        if self.Y_0.shape[0] != self.num_compounds:
            raise ValueError(
                f"Insufficient mixture description:\n"
                f"The number of compounds in {self.groupDecompFile} does not "
                f"equal the number of compounds in {self.gcxgcFile}."
            )

        # Read and store GCM table properties
        df_table = pd.read_csv(self.gcmTableFile)
        df_table = df_table.drop(columns=["Units"])

        def get_row(property_name: str) -> np.ndarray:
            """
            Get property row from GCM table.

            :param property_name: Name of the property to retrieve.
            :type property_name: str
            :return: Property values for all functional groups.
            :rtype: np.ndarray
            :raises ValueError: If property not found in GCM table.
            """
            row = df_table[df_table["Property"] == property_name]
            if row.empty:
                raise ValueError(f"Property '{property_name}' not found in GCM table.")
            return row.iloc[:, 1:].to_numpy().flatten()

        # Table data for functional groups (num_compounds,)
        Tck = get_row("tck")  # critical temperature (1)
        Pck = get_row("pck")  # critical pressure (bar)
        Vck = get_row("vck")  # critical volume (m^3/kmol)
        Tbk = get_row("tbk")  # boiling temperature (1)
        Tmk = get_row("tmk")  # melting point temperature (1)
        hfk = get_row("hfk")  # enthalpy of formation, (kJ/mol)
        gfk = get_row("gfk")  # Gibbs energy (kJ/mol)
        hvk = get_row("hvk")  # latent heat of vaporization (kJ/mol)
        wk = get_row("wk")  # accentric factor (1)
        Vmk = get_row("vmk")  # liquid molar volume fraction (m^3/kmol)
        cpak = get_row("CpAk")  # specific heat values (J/mol/K)
        cpbk = get_row("CpBk")  # specific heat values (J/mol/K)
        cpck = get_row("CpCk")  # specific heat values (J/mol/K)
        mwk = get_row("MW")  # molecular weights (g/mol)

        # --- Compute critical properties at standard temp (num_compounds,)
        # Molecular weights
        self.MW = load_quantity(jnp.matmul(self.Nij, mwk), "g/mole", "kg/mole")

        # T_c (critical temperature)
        self.Tc = load_quantity(181.128 * jnp.log(jnp.matmul(self.Nij, Tck)), "K")

        # p_c (critical pressure)
        self.Pc = load_quantity(
            1.3705 + (jnp.matmul(self.Nij, Pck) + 0.10022) ** (-2), "bar", "Pa"
        )

        # V_c (critical volume)
        self.Vc = load_quantity(
            -0.00435 + (jnp.matmul(self.Nij, Vck)), "m^3/kmol", "m^3/mol"
        )

        # T_b (boiling temperature)
        self.Tb = load_quantity(204.359 * jnp.log(jnp.matmul(self.Nij, Tbk)), "K")

        # T_m (melting temperature)
        self.Tm = load_quantity(102.425 * jnp.log(jnp.matmul(self.Nij, Tmk)), "K")

        # H_f (enthalpy of formation)
        Hf = 10.835 + jnp.matmul(self.Nij, hfk)  # kJ/mol
        self.Hf = load_quantity(Hf * 1e3, "J/mol")

        # G_f (Gibbs free energy)
        Gf = -14.828 + jnp.matmul(self.Nij, gfk)  # kJ/mol
        self.Gf = load_quantity(Gf * 1e3, "J/mol")

        # H_v,stp (enthalpy of vaporization at 298 K)
        Hv_stp = 6.829 + (jnp.matmul(self.Nij, hvk))  # kJ/mol
        self.Hv_stp = load_quantity(Hv_stp * 1e3, "J/mol")

        # omega (accentric factor, dimensionless)
        omega = 0.4085 * jnp.log(jnp.matmul(self.Nij, wk) + 1.1507) ** (1.0 / 0.5050)
        self.omega = load_quantity(omega, "")

        # V_m (molar liquid volume at 298 K)
        Vm_stp = 0.01211 + jnp.matmul(self.Nij, Vmk)  # m^3/kmol
        self.Vm_stp = load_quantity(Vm_stp * 1e-3, "m^3/mol")

        # C_p,stp (molar specific heat at 298 K)
        Cp_stp = jnp.matmul(self.Nij, cpak) - 19.7779  # J/mol/K
        self.Cp_stp = load_quantity(Cp_stp, "J/(mol*K)")

        # Temperature corrections for C_p
        self.Cp_B = load_quantity(jnp.matmul(self.Nij, cpbk), "J/(mol*K)")
        self.Cp_C = load_quantity(jnp.matmul(self.Nij, cpck), "J/(mol*K)")

        # L_v,stp (latent heat of vaporization at 298 K): unit-homogeneous, divide directly
        self.Lv_stp = (self.Hv_stp / self.MW).uconvert("J/kg")

        # Lennard-Jones parameters (Tee et al. 1966): empirical fit tied to K/atm
        # magnitudes, not dimensionally homogeneous - strip, compute, rewrap
        Tc_K = _ustrip(self.Tc, "K")
        Pc_atm = (
            _ustrip(self.Pc, "Pa") / 101325
        )  # unxt/astropy has no "atm" unit string
        self.epsilonByKB = load_quantity((0.7915 + 0.1693 * omega) * Tc_K, "K")
        sigma = (2.3551 - 0.0874 * omega) * (Tc_K / Pc_atm) ** (1.0 / 3)  # Angstroms
        self.sigma = load_quantity(sigma * 1e-10, "m")

    # -------------------------------------------------------------------------
    # Member functions
    # -------------------------------------------------------------------------
    def mean_molecular_weight(self, Yi: Array) -> u.Quantity:
        """
        Calculate the mean molecular weight of the mixture.

        :param Yi: Mass fractions of each compound.
        :type Yi: jax.Array
        :return: Mean molecular weight of the mixture in kg/mol.
        :rtype: unxt.Quantity
        """
        MW = _ustrip(self.MW, "kg/mol")
        if jnp.sum(Yi) != 0:
            Mbar = 1 / jnp.sum(Yi / MW)  # mean molar weight of the mixture
        else:
            Mbar = 0.0

        return load_quantity(Mbar, "kg/mol")

    def mass2Y(self, mass: u.Quantity) -> Array:
        """
        Calculate the mass fractions from the mass of each component.

        :param mass: Mass of each compound.
        :type mass: unxt.Quantity
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: Array
        """
        mass: Array = _ustrip(mass, "kg")
        # Normalize to get group mole fractions
        total_mass = jnp.sum(mass)
        if total_mass != 0:
            Yi = mass / total_mass
        else:
            Yi = jnp.zeros_like(mass)

        return Yi

    def mass2X(self, mass: u.Quantity) -> Array:
        """
        Calculate the mole fractions from the mass of each component.

        :param mass: Mass of each compound.
        :type mass: u.Quantity
        :return: Mole fractions of the compounds (shape: num_compounds,).
        :rtype: Array
        """
        # Calculate the number of moles for each compound
        num_mole = _ustrip(mass, "kg") / _ustrip(self.MW, "kg/mol")

        # Normalize to get group mole fractions
        total_moles = jnp.sum(num_mole)
        if total_moles != 0:
            Xi = num_mole / total_moles
        else:
            Xi = jnp.zeros_like(num_mole)

        return Xi

    def X2Y(self, Xi: Array) -> Array:
        """
        Calculate the mass fractions from the mole fractions of each component.

        :param Xi: Mole fractions of each compound.
        :type Xi: Array
        :return: Mass fractions of the compounds (shape: num_compounds,).
        :rtype: Array
        """
        # Calculate the mass for each compound
        MW = _ustrip(self.MW, "kg/mol")
        mass = Xi * MW

        # Normalize to get group mass fractions
        total_mass = jnp.sum(mass)
        if total_mass != 0:
            Yi = mass / total_mass
        else:
            Yi = jnp.zeros_like(MW)

        return Yi

    def Y2X(self, Yi: Array) -> Array:
        """
        Calculate the mole fractions from the mass fractions of each component.

        :param Yi: Mass fractions of each compound.
        :type Yi: Array
        :return: Mole fractions of the compounds (shape: num_compounds,).
        :rtype: Array
        """
        Mbar = self.mean_molecular_weight(Yi)  # Quantity(kg/mol)
        if jnp.sum(Yi) != 0:
            Xi = _ustrip(Mbar * Yi / self.MW, "")
        else:
            Xi = jnp.zeros_like(_ustrip(self.MW, "kg/mol"))

        return Xi

    def density(self, T: u.Quantity, comp_idx: int | None = None) -> u.Quantity:
        """
        Calculate the density of each component at temperature T.

        :param T: Temperature of the mixture in Kelvin.
        :type T: unxt.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Density of each compound in kg/m^3.
        :rtype: unxt.Quantity
        """
        if comp_idx is None:
            MW = self.MW  # kg/mol
            Vm = self.molar_liquid_vol(T)  # m^3/mol
        else:
            MW = self.MW[comp_idx]  # kg/mol
            Vm = self.molar_liquid_vol(T, comp_idx=comp_idx)  # m^3/mol

        rho = MW / Vm  # kg/m^3, unit-homogeneous
        return rho.uconvert("kg/m^3")

    def viscosity_kinematic(
        self, T: u.Quantity, comp_idx: int | None = None
    ) -> u.Quantity:
        """
        Calculate the viscosity using Dutt's equation.

        :meta private: This uses Dutt's equation (4.23) from "Viscosity of Liquids".
        :meta private: The equation predicts viscosity in mm^2/s and is converted to SI units.

        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Viscosity of each component in m^2/s.
        :rtype: unxt.Quantity
        """

        # Convert temperature to Celsius (plain float - Celsius is an offset unit)
        T_cels = K2C(T)
        if comp_idx is None:
            Tb_cels = K2C(self.Tb)
            T_cels = _atleast_col(
                T_cels
            )  # allow array T to broadcast over all compounds
        else:
            Tb_cels = K2C(self.Tb[comp_idx])

        # RHS of Dutt's equation (4.23) in Viscosity of Liquids
        rhs = -3.0171 + (442.78 + 1.6452 * Tb_cels) / (T_cels + 239 - 0.19 * Tb_cels)
        nu_i = jnp.exp(rhs)  # Viscosity in mm^2/s

        # Convert to SI (m^2/s)
        nu_i = nu_i * 1e-6

        return load_quantity(nu_i, "m^2/s")

    def viscosity_dynamic(
        self, T: u.Quantity, comp_idx: int | None = None
    ) -> u.Quantity:
        """
        Calculate liquid dynamic viscosity based on droplet temperature and density.

        :meta private: Uses Dutt's equation (4.23) for kinematic viscosity, combined with density.

        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Dynamic viscosity in Pa*s.
        :rtype: unxt.Quantity
        """

        nu_i = self.viscosity_kinematic(T, comp_idx=comp_idx)  # m^2/s
        rho_i = self.density(T, comp_idx=comp_idx)  # kg/m^3
        mu_i = nu_i * rho_i  # Pa*s, unit-homogeneous
        return mu_i.uconvert("Pa*s")

    def Cp(self, T: u.Quantity, comp_idx: int | None = None) -> u.Quantity:
        """
        Compute molar specific heat capacity at a given temperature.

        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Molar specific heat capacity in J/mol/K.
        :rtype: unxt.Quantity
        """

        theta = (_ustrip(T, "K") - 298) / 700
        if comp_idx is None:
            theta = _atleast_col(theta)  # allow array T to broadcast over all compounds
            Cp_stp = _ustrip(self.Cp_stp, "J/(mol*K)")
            Cp_B = _ustrip(self.Cp_B, "J/(mol*K)")
            Cp_C = _ustrip(self.Cp_C, "J/(mol*K)")
        else:
            Cp_stp = _ustrip(self.Cp_stp[comp_idx], "J/(mol*K)")
            Cp_B = _ustrip(self.Cp_B[comp_idx], "J/(mol*K)")
            Cp_C = _ustrip(self.Cp_C[comp_idx], "J/(mol*K)")

        cp = Cp_stp + Cp_B * theta + Cp_C * theta**2

        return load_quantity(cp, "J/(mol*K)")

    def Cl(self, T: u.Quantity, comp_idx: int | None = None) -> u.Quantity:
        """
        Compute liquid mass specific heat capacity in J/kg/K at a given temperature.

        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Mass specific heat capacity in J/kg/K.
        :rtype: unxt.Quantity
        """
        if comp_idx is None:
            MW = self.MW
        else:
            MW = self.MW[comp_idx]
        cp = self.Cp(T, comp_idx=comp_idx)
        return (cp / MW).uconvert("J/(kg*K)")

    def psat(
        self,
        T: u.Quantity,
        comp_idx: int | None = None,
        correlation: str = "Lee-Kesler",
    ) -> u.Quantity:
        """
        Compute saturated vapor pressure.

        :meta private: Can use Ambrose-Walton or Lee-Kesler correlations (default Lee-Kesler).

        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Saturated vapor pressure in Pa.
        :rtype: unxt.Quantity
        """

        T_K = _ustrip(T, "K")
        if comp_idx is None:
            Tc = _ustrip(self.Tc, "K")
            Pc = self.Pc
            omega = _ustrip(self.omega, "")
            T_K = _atleast_col(T_K)  # allow array T to broadcast over all compounds
        else:
            Tc = _ustrip(self.Tc[comp_idx], "K")
            Pc = self.Pc[comp_idx]
            omega = _ustrip(self.omega[comp_idx], "")
        Tr = T_K / Tc

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
            rhs = jnp.exp(f0 + omega * f1 + omega**2 * f2)

        else:  # Default correlation is Lee-Kesler
            f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * jnp.log(Tr) + 0.169347 * (Tr**6)
            f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * jnp.log(Tr) + 0.43577 * (Tr**6)
            rhs = jnp.exp(f0 + omega * f1)

        psat = Pc * rhs  # Quantity(Pa) * dimensionless multiplier
        return psat

    def psat_antoine_coeffs(
        self,
        Tvals: np.ndarray | None = None,
        units: str = "mks",
        correlation: str = "Lee-Kesler",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate Antoine coefficients for vapor pressure of an individual compound.

        :param Tvals: Temperature range or nodes for Antoine fit in Kelvin (default [273.15, Tb_i]).
        :type Tvals: np.ndarray, optional
        :param units: Units for pressure in fit ("mks", "cgs", "bar", "atm")
        :type units: str, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Coefficients A, B, C, D
        :rtype: 4 np.ndarrays
        """

        # Define or get temperature nodes for fit
        if Tvals is None:
            print("Tvals not specified, using [273.15, Tb_i] for each compound.")
            # Initialize as zeros for now, calculated for each compound later
            T = np.zeros(20)
        elif len(Tvals) == 2:
            T = np.linspace(Tvals[0], Tvals[1], 20)
        elif len(Tvals) > 2:
            T = Tvals
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        # Antoine equation log10(p) = A - B/(C + T)
        def antoine_eq(
            T: float | np.ndarray, A: float, B: float, C: float
        ) -> float | np.ndarray:
            """Antoine equation for vapor pressure."""
            return A - B / (T + C)

        # Determine conversion factor for pressure in MKS, CGS, bar, or atm
        D = 1  # default is Pa
        if units.lower() == "bar":
            D = 1e5
        elif units.lower() == "atm":
            D = 1.01325e5
        elif units.lower() == "cgs":
            D = 1 / 10  # dyne/cm^2

        # Fit Antoine coefficients for each compound
        # A/B/C/D and T/Pvals stay numpy: curve_fit requires numpy, not jax arrays
        A = np.zeros(self.num_compounds)
        B = np.zeros(self.num_compounds)
        C = np.zeros(self.num_compounds)
        for i in range(self.num_compounds):
            # Update T if not specified
            if Tvals is None:
                T = np.linspace(273.15, float(np.asarray(_ustrip(self.Tb[i], "K"))), 20)
            # One vectorized call over all T points instead of a per-T loop
            psat_Pa = _ustrip(
                self.psat(u.Q(T, "K"), comp_idx=i, correlation=correlation), "Pa"
            )
            Pvals = np.asarray(psat_Pa) / D

            logP = np.log10(Pvals)
            popt, _ = curve_fit(antoine_eq, T, logP, p0=[1, 1e3, -1])
            A[i], B[i], C[i] = popt
        D = D + np.zeros(self.num_compounds)  # make D an array
        return A, B, C, D

    def molar_liquid_vol(
        self, T: u.Quantity, comp_idx: int | None = None
    ) -> u.Quantity:
        """
        Compute molar liquid volume with temperature correction.

        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Molar liquid volume in m^3/mol.
        :rtype: unxt.Quantity
        """

        T_K = _ustrip(T, "K")
        if comp_idx is None:
            Tc = _ustrip(self.Tc, "K")
            omega = _ustrip(self.omega, "")
            Vm_stp = _ustrip(self.Vm_stp, "m^3/mol")
            T_K = _atleast_col(T_K)  # allow array T to broadcast over all compounds
        else:
            Tc = _ustrip(self.Tc[comp_idx], "K")
            omega = _ustrip(self.omega[comp_idx], "")
            Vm_stp = _ustrip(self.Vm_stp[comp_idx], "m^3/mol")
        Vmi = _molar_liquid_vol_core(T_K, Tc, omega, Vm_stp)
        return load_quantity(Vmi, "m^3/mol")

    def latent_heat_vaporization(
        self, T: u.Quantity, comp_idx: int | None = None
    ) -> u.Quantity:
        """
        Calculate latent heat of vaporization adjusted for temperature.

        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Latent heat of vaporization in J/kg.
        :rtype: unxt.Quantity
        """
        T_K = _ustrip(T, "K")
        if comp_idx is None:
            Tc = _ustrip(self.Tc, "K")
            Tb = _ustrip(self.Tb, "K")
            Lv_stp = _ustrip(self.Lv_stp, "J/kg")
            T_K = _atleast_col(T_K)  # allow array T to broadcast over all compounds
        else:
            Tc = _ustrip(self.Tc[comp_idx], "K")
            Tb = _ustrip(self.Tb[comp_idx], "K")
            Lv_stp = _ustrip(self.Lv_stp[comp_idx], "J/kg")

        # Reduced temperatures
        Tr = T_K / Tc
        Trb = Tb / Tc

        Lvi = jnp.where(T_K > Tc, 0.0, Lv_stp * (((1.0 - Tr) / (1.0 - Trb)) ** 0.38))

        return load_quantity(Lvi, "J/kg")

    def diffusion_coeff(
        self,
        p: u.Quantity,
        T: u.Quantity,
        sigma_gas: float = 3.62e-10,
        epsilonByKB_gas: float = 97.0,
        MW_gas: float = 28.97e-3,
        correlation: str = "Tee",
    ) -> u.Quantity:
        """
        Compute diffusion coefficients using Lennard-Jones parameters.

        :meta private: Uses Wilke and Lee method (Poling, equation 11-4.1).
        :meta private: Ambient gas defaults to air parameters.

        :param p: Pressure in Pa.
        :type p: unxt.Quantity
        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :param sigma_gas: Collision diameter in m.
        :type sigma_gas: float, optional
        :param epsilonByKB_gas: Well depth over Boltzmann constant, in K.
        :type epsilonByKB_gas: float, optional
        :param MW_gas: Mean molecular weight of ambient gas in kg/mol.
        :type MW_gas: float, optional
        :param correlation: Method to calculate sigma and epsilon ("Tee" or "Wilke").
        :type correlation: str, optional
        :return: Diffusion coefficient.
        :rtype: unxt.Quantity
        """

        T_K = _atleast_col(
            _ustrip(T, "K")
        )  # allow array T to broadcast over all compounds
        p_Pa = _atleast_col(_ustrip(p, "Pa"))

        # Method of Tee for calculating liquid sigma and epsilon
        if correlation.casefold() == "Tee".casefold():
            sigma_i = _ustrip(self.sigma, "m") * 1e10  # convert from m to Angstroms
            epsilonByKB_i = _ustrip(self.epsilonByKB, "K")
        else:
            # Method of Wilke & Lee calculating liquid sigma and epsilon
            # Bypass molar_liquid_vol's public broadcast: each compound at its own Tb, not an outer product
            Vmb_i = (
                _molar_liquid_vol_core(
                    _ustrip(self.Tb, "K"),
                    _ustrip(self.Tc, "K"),
                    _ustrip(self.omega, ""),
                    _ustrip(self.Vm_stp, "m^3/mol"),
                )
                * 1e6
            )  # cm^3/mol
            sigma_i = 1.18 * Vmb_i ** (1 / 3)  # Angstroms, Poling (11-4.2)
            epsilonByKB_i = 1.15 * _ustrip(self.Tb, "K")  # K , Poling (11-4.3)

        # Compute binary sigma and epsilon
        sigma_gas = sigma_gas * 1e10  # convert from m to Angstroms
        sigmaAB_i = (sigma_gas + sigma_i) / 2  # Angstroms, Poling (11-3.5)
        epsilonAB_byKB_i = (
            epsilonByKB_gas * epsilonByKB_i
        ) ** 0.5  # K, Poling (11-3.4)

        # Dimensionless collision integral for diffusion: Poling (11-3.6)
        Tstar_i = T_K / epsilonAB_byKB_i  # [1]
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
            + C / jnp.exp(D * Tstar_i)
            + E / jnp.exp(F * Tstar_i)
            + G / jnp.exp(H * Tstar_i)
        )

        # Convert molecular weights from kg/mol to g/mol then calculate M_AB
        MW_gas = MW_gas * 1e3
        MW_i = _ustrip(self.MW, "kg/mol") * 1e3
        M_AB_i = 2 * (MW_i * MW_gas) / (MW_i + MW_gas)  # g/mol, see Poling (11-3.1)

        # Convert pressure from Pa to bar
        p_bar = p_Pa * 1e-5  # bar

        # Binary diffusion coefficients, Poling (11-4.1)
        D_AB_i = (
            1e-3
            * (3.03 - 0.98 / (M_AB_i**0.5))
            * (T_K**1.5)
            / (p_bar * M_AB_i**0.5 * sigmaAB_i**2 * omegaD_i)
        )  # cm^2/s
        D_AB_i = D_AB_i * 1e-4  # Convert to m^2/s

        return load_quantity(D_AB_i, "m^2/s")

    def surface_tension(
        self,
        T: u.Quantity,
        comp_idx: int | None = None,
        correlation: str = "Brock-Bird",
    ) -> u.Quantity:
        """
        Calculate surface tension of each compound at a given temperature.

        :meta private: Uses Brock-Bird (default) or Pitzer correlations (Poling 12-3.5, 12-3.7).

        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :param correlation: Correlation method ("Brock-Bird" or "Pitzer").
        :type correlation: str, optional
        :return: Surface tension in N/m.
        :rtype: unxt.Quantity
        """
        T_K = _ustrip(T, "K")
        if comp_idx is None:
            Tc = _ustrip(self.Tc, "K")
            Pc = _ustrip(self.Pc, "Pa")
            Tb = _ustrip(self.Tb, "K")
            omega = _ustrip(self.omega, "")
            T_K = _atleast_col(T_K)  # allow array T to broadcast over all compounds
        else:
            Tc = _ustrip(self.Tc[comp_idx], "K")
            Pc = _ustrip(self.Pc[comp_idx], "Pa")
            Tb = _ustrip(self.Tb[comp_idx], "K")
            omega = _ustrip(self.omega[comp_idx], "")
        Tr = T_K / Tc
        Pc_bar = Pc * 1e-5  # convert from Pa to bar

        if correlation.casefold() == "Brock-Bird".casefold():
            Tbr = Tb / Tc
            Q = 0.1196 * (1.0 + (Tbr * jnp.log(Pc_bar / 1.01325)) / (1.0 - Tbr)) - 0.279
        else:
            w = omega
            Q = (
                (1.86 + 1.18 * w)
                / 19.05
                * (((3.75 + 0.91 * w) / (0.291 - 0.08 * w)) ** (2.0 / 3.0))
            )

        st = Pc_bar ** (2.0 / 3.0) * Tc ** (1.0 / 3.0) * Q * (1 - Tr) ** (11.0 / 9.0)

        st = st * 1e-3  # Convert from dyn/cm to N/m

        return load_quantity(st, "N/m")

    def thermal_conductivity(
        self, T: u.Quantity, comp_idx: int | None = None
    ) -> u.Quantity:
        """
        Calculate thermal conductivity at a given temperature.

        :meta private: Uses Latini et al. method (Poling equation 10-9.1).

        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :param comp_idx: Index of compound to calculate property for.
        :type comp_idx: int, optional
        :return: Thermal conductivity in W/m/K.
        :rtype: unxt.Quantity
        """
        T_K = _ustrip(T, "K")
        if comp_idx is None:
            MW = _ustrip(self.MW, "kg/mol")
            Tc = _ustrip(self.Tc, "K")
            Tb = _ustrip(self.Tb, "K")
            fam = self.fam
            T_K = _atleast_col(T_K)  # allow array T to broadcast over all compounds
        else:
            MW = _ustrip(self.MW[comp_idx], "kg/mol")
            Tc = _ustrip(self.Tc[comp_idx], "K")
            Tb = _ustrip(self.Tb[comp_idx], "K")
            fam = self.fam[comp_idx]

        alpha = 1.2
        gamma = 0.167
        # Family-specific constants (1: aromatic, 2: cycloparaffin, 3: olefin; else saturated)
        Astar = jnp.where(
            fam == 1,
            0.0346,
            jnp.where(fam == 2, 0.0310, jnp.where(fam == 3, 0.0361, 0.00350)),
        )
        beta = jnp.where(fam == 0, 0.5, 1.0)
        MW_beta = (MW * 1e3) ** beta  # convert from kg/mol to g/mol
        Tr = T_K / Tc

        A = Astar * Tb**alpha / (MW_beta * Tc**gamma)
        tc = A * (1 - Tr) ** (0.38) / (Tr ** (1 / 6))

        return load_quantity(tc, "W/(m*K)")

    # --- Mixture functions ---
    def mixture_density(self, Yi: Array, T: u.Quantity) -> u.Quantity:
        """
        Calculate mixture density at a given temperature.

        :param Yi: Mass fractions of each compound.
        :type Yi: jax.Array
        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :return: Mixture density in kg/m^3.
        :rtype: unxt.Quantity
        """
        MW = _ustrip(self.MW, "kg/mol")  # Molecular weights of each component
        Vmi = _ustrip(self.molar_liquid_vol(T), "m^3/mol")  # Molar volume

        # Calculate density (kg/m^3); axis=-1 sums over compounds, keeping a leading T axis if present
        rho = jnp.sum(Yi * (MW / Vmi), axis=-1)

        return load_quantity(rho, "kg/m^3")

    def mixture_kinematic_viscosity(
        self, Yi: Array, T: u.Quantity, correlation: str = "Kendall-Monroe"
    ) -> u.Quantity:
        """
        Calculate kinematic viscosity of the mixture.

        :meta private: Uses Kendall-Monroe (default) or Arrhenius mixing correlations.

        :param Yi: Mass fractions of each compound.
        :type Yi: jax.Array
        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :param correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
        :type correlation: str, optional
        :return: Mixture kinematic viscosity in m^2/s.
        :rtype: unxt.Quantity
        """
        nu_i = _ustrip(self.viscosity_kinematic(T), "m^2/s")

        # Calculate mole fractions for each species
        Xi = self.Y2X(Yi)

        if correlation.casefold() == "Arrhenius".casefold():
            # Arrhenius mixing correlation
            nu = jnp.exp(jnp.sum(Xi * jnp.log(nu_i), axis=-1))
        else:
            # Default: Kendall-Monroe mixing correlation
            nu = jnp.sum(Xi * (nu_i ** (1.0 / 3.0)), axis=-1) ** (3.0)

        return load_quantity(nu, "m^2/s")

    def mixture_dynamic_viscosity(
        self, Yi: Array, T: u.Quantity, correlation: str = "Kendall-Monroe"
    ) -> u.Quantity:
        """
        Calculate dynamic viscosity of the mixture.

        :param Yi: Mass fractions of each compound.
        :type Yi: jax.Array
        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :param correlation: Mixing model ("Kendall-Monroe" or "Arrhenius").
        :type correlation: str, optional
        :return: Mixture dynamic viscosity in Pa*s.
        :rtype: unxt.Quantity
        """

        nu = self.mixture_kinematic_viscosity(Yi, T, correlation=correlation)
        rho = self.mixture_density(Yi, T)

        return (rho * nu).uconvert("Pa*s")

    def mixture_vapor_pressure(
        self, Yi: Array, T: u.Quantity, correlation: str = "Lee-Kesler"
    ) -> u.Quantity:
        """
        Calculate vapor pressure of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: jax.Array
        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Mixture vapor pressure in Pa.
        :rtype: unxt.Quantity
        """

        # Mole fraction for each compound
        Xi = self.Y2X(Yi)

        # Saturated vapor pressure for each compound (Pa)
        p_sati = _ustrip(self.psat(T, correlation=correlation), "Pa")

        # Mixture vapor pressure via Raoult's law; axis=-1 sums over compounds
        p_v = jnp.sum(p_sati * Xi, axis=-1)

        return load_quantity(p_v, "Pa")

    def mixture_vapor_pressure_antoine_coeffs(
        self,
        Yi: Array,
        Tvals: np.ndarray | None = None,
        units: str = "mks",
        correlation: str = "Lee-Kesler",
    ) -> tuple[float, float, float, float]:
        """
        Estimate Antoine coefficients for vapor pressure of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: jax.Array
        :param Tvals: Temperature range or nodes for Antoine fit in Kelvin (default [273.15, min(Tb)]).
        :type Tvals: np.ndarray, optional
        :param units: Units for pressure in fit ("mks", "cgs", "bar", "atm")
        :type units: str, optional
        :param correlation: Correlation method ("Ambrose-Walton" or "Lee-Kesler").
        :type correlation: str, optional
        :return: Coefficients A, B, C, D
        :rtype: float
        """

        # Define or get temperature nodes for fit
        # T/Pvals stay numpy: curve_fit requires numpy, not jax arrays
        if Tvals is None:
            print("Tvals not specified, using [273.15, min(Tb_mix)] for mixture.")
            # Initialize as zeros for now, calculated for each compound later
            X = self.Y2X(Yi)
            Tb = mixing_rule(_ustrip(self.Tb, "K"), X)
            T = np.linspace(273.15, float(jnp.min(Tb)), 20)
        elif len(Tvals) == 2:
            T = np.linspace(Tvals[0], Tvals[1], 20)
        elif len(Tvals) > 2:
            T = Tvals
        else:
            raise ValueError("Tvals must be None, length 2, or length > 2.")

        # Antoine equation log10(p) = A - B/(C + T)
        def antoine_eq(
            T: float | np.ndarray, A: float, B: float, C: float
        ) -> float | np.ndarray:
            """
            Antoine equation for vapor pressure.

            :param T: Temperature.
            :type T: float or np.ndarray
            :param A: Antoine coefficient A.
            :type A: float
            :param B: Antoine coefficient B.
            :type B: float
            :param C: Antoine coefficient C.
            :type C: float
            :return: log10(pressure).
            :rtype: float or np.ndarray
            """
            return A - B / (T + C)

        # Determine conversion factor for pressure in MKS, CGS, bar, or atm
        D = 1  # default is Pa
        if units.lower() == "bar":
            D = 1e5
        elif units.lower() == "atm":
            D = 1.01325e5
        elif units.lower() == "cgs":
            D = 1 / 10  # dyne/cm^2

        Pvals = (
            np.asarray(
                _ustrip(
                    self.mixture_vapor_pressure(
                        Yi, u.Q(T, "K"), correlation=correlation
                    ),
                    "Pa",
                )
            )
            / D
        )

        logP = np.log10(Pvals)
        popt, _ = curve_fit(antoine_eq, T, logP, p0=[1, 1e3, -1])  # initial guess
        A, B, C = popt

        return A, B, C, D

    def mixture_surface_tension(
        self,
        Yi: Array,
        T: u.Quantity,
        correlation: Literal["Pitzer", "Brock-Bird"] = "Brock-Bird",
    ) -> u.Quantity:
        """
        Calculate surface tension of the mixture.

        :meta private: Uses arithmetic pseudo-property method recommended by Hugill and van Welsenes (1986).

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: Array
        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :param correlation: Correlation method ("Pitzer" or "Brock-Bird").
        :type correlation: Literal["Pitzer", "Brock-Bird"], optional
        :return: Mixture surface tension in N/m.
        :rtype: u.Quantity
        """

        # Mole fraction for each compound
        Xi = self.Y2X(Yi)

        # Surface tension for each compound (N/m)
        sti = _ustrip(self.surface_tension(T, correlation=correlation), "N/m")

        # Mixture surface tension via arithmetic mean, Poling (12-5.2)
        st = mixing_rule(sti, Xi, "arithmetic")

        return load_quantity(st, "N/m")

    def mixture_thermal_conductivity(self, Yi: Array, T: u.Quantity) -> u.Quantity:
        """
        Calculate thermal conductivity of the mixture.

        :param Yi: Mass fractions of each compound in the mixture.
        :type Yi: Array
        :param T: Temperature in Kelvin.
        :type T: unxt.Quantity
        :return: Thermal conductivity in W/m/K.
        :rtype: u.Quantity
        """
        tc = _ustrip(self.thermal_conductivity(T), "W/(m*K)")
        result = jnp.sum(Yi * tc ** (-2), axis=-1) ** (-0.5)
        return load_quantity(result, "W/(m*K)")


__all__ = ["fuel"]
