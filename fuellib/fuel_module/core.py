from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pandas as pd
import quaxed.numpy as qnp
from jax import Array
from scipy.optimize import curve_fit
from unxt import AbstractQuantity, Quantity

from ..constants import EPSILON_BY_KB_GAS, MW_GAS, SIGMA_GAS
from ..units import convert_pressure_to_atm, convert_temperature
from .correlation import tee_epsilon, tee_sigma
from .gcm import GaniGCM
from .locator import DEFAULT_DATA_DIR

# Front-load initialization to avoid overhead of loading GCM parameters for each Fuel instance
gani = GaniGCM()  # Constantinou-Gani method


def _check_valid_property(
    value: Array | AbstractQuantity, expected_length: int, name: str
) -> None:
    """
    Check if the length of the array matches the expected length.

    :param value: Array to check.
    :type value: Array | AbstractQuantity
    :param expected_length: Expected length of the array.
    :type expected_length: int
    :param name: Name of the variable for error messages.
    :type name: str
    :raises ValueError: If the array is not 1D.
    :raises ValueError: If the length of the array does not match the expected length.
    """
    value = jnp.array(value.value) if isinstance(value, AbstractQuantity) else value
    if value.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if value.shape[0] != expected_length:
        raise ValueError(
            f"{name} must have length {expected_length}, but has length {value.shape[0]}."
        )


def _atleast_col(x: AbstractQuantity) -> AbstractQuantity:
    """
    Add a trailing axis so an array of temperatures broadcasts against a per-compound axis; scalars pass through.

    Idempotent: only 1D arrays are expanded, so calling this again on an already-expanded
    array (e.g. because a caller already column-expanded T before passing it to another
    method that also calls this) is a no-op instead of adding another axis.

    :param x: Input array.
    :type x: Array
    :return: Array with at least one trailing axis.
    :rtype: Array
    """
    return x[:, None] if qnp.ndim(x) == 1 else x


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

    fuelDataDir: str | Path
    fuelDataGcDir: str | Path
    fuelDataDecompDir: str | Path
    fuelDataPropsDir: str | Path

    # Fuel / component properties
    name: str
    compounds: list[str]
    formulas: list[str] | None = None
    smiles: list[str] | None = None
    pelephysics_keys: list[str] | None = None

    # GCM descriptions
    gani_groups: npt.NDArray[np.str_]  # Constantinou-Gani group names
    gani_decomp: npt.NDArray[np.int_]  # Constantinou-Gani decomposition matrix

    # Critical properties of the mixture components
    _Y_0: Array  # Mass fractions of the compounds in the mixture
    _MW: AbstractQuantity  # Molecular weights in kg/mol
    _Tc: AbstractQuantity  # Critical temperatures in K
    _Pc: AbstractQuantity  # Critical pressures in Pa
    _Vc: AbstractQuantity  # Critical volumes in m^3/mol
    _Tb: AbstractQuantity  # Boiling points in K
    _Tm: AbstractQuantity  # Melting points in K
    _Hf: AbstractQuantity  # Heat of formation in J/mol
    _Gf: AbstractQuantity  # Gibbs free energy in J/mol
    _Hv_stp: AbstractQuantity  # Heat of vaporization at STP in J/mol
    _Cp_stp: AbstractQuantity  # Heat capacity at STP in J/mol/K
    _Cp_B: AbstractQuantity  # Temperature correction for heat capacity in J/mol/K
    _Cp_C: AbstractQuantity  # Temperature correction for heat capacity in J/mol/K
    _Vm_stp: AbstractQuantity  # Molar volume at STP in m^3/mol
    _omega: AbstractQuantity  # Acentric factor (dimensionless)
    # Derived properties
    _Lv_stp: AbstractQuantity  # Latent heat of vaporization at STP in J/mol
    _sigma: AbstractQuantity  # Surface tension in N/m

    def __init__(
        self,
        name: str,
        decompName: str | None = None,
        fuelDataDir: str | Path | None = None,
    ) -> None:
        """Initialize the Fuel object.

        :param name: Name of the mixture as it appears in its gcData file.
        :type name: str
        :param decompName: Name of the groupDecomposition file if different from name. Defaults to None.
        :type decompName: str, optional
        :param fuelDataDir: Directory where the fuel data is stored. If None, uses built-in embedded data.
        :type fuelDataDir: str, optional
        """
        self.name = name

        # Set data directories, using embedded data if no directory is provided
        self.fuelDataDir = Path(fuelDataDir or DEFAULT_DATA_DIR)
        self.fuelDataGcDir = self.fuelDataDir / "gcData"
        self.fuelDataDecompDir = self.fuelDataDir / "groupDecompositionData"
        self.fuelDataPropsDir = self.fuelDataDir / "propertiesData"

        # Load gc data
        self._init_gc_data(self.fuelDataGcDir / f"{name}_init.csv")

        # Load group decomposition data
        self._init_gani_decomp(self.fuelDataDecompDir / f"{decompName or name}.csv")

        # Load default critical properties
        self._init_critical_properties()

    def _init_gc_data(self, csv_path: str | Path) -> None:
        """
        Load the gcxgc from a CSV file.

        :param gcm_name: Name of the GCM to load.
        :type gcm_name: str
        """
        df = pd.read_csv(csv_path, skipinitialspace=True)
        self.compounds = [c.strip() for c in df["Compound"].tolist()]

        if "Formula" in df.columns:
            self.formulas = [
                f.strip() if pd.notna(f) else "" for f in df["Formula"].tolist()
            ]
        if "PelePhysics Key" in df.columns:
            self.pelephysics_keys = [
                k.strip() if pd.notna(k) else "" for k in df["PelePhysics Key"].tolist()
            ]
        if "SMILES" in df.columns:
            self.smiles = [
                s.strip() if pd.notna(s) else "" for s in df["SMILES"].tolist()
            ]

        _wts = df["Weight %"].to_numpy(dtype=float)
        self._Y_0 = jnp.array(_wts / np.sum(_wts), dtype=float)

    def _init_gani_decomp(self, csv_path: str | Path) -> None:
        """
        Load the Constantinou-Gani group decomposition from a CSV file.

        :param csv_path: Path to the CSV file containing the group decomposition.
        :type csv_path: str or Path
        :raises ValueError: If the CSV file does not contain any compounds, or
            contains groups not in ``self.groups``.
        """
        gani_gcm = GaniGCM()
        compounds, self.gani_groups, decomp = gani_gcm.load_fuel_decomposition(csv_path)
        unknown_compounds = set(self.compounds) - set(compounds)
        if unknown_compounds:
            raise ValueError(
                f"{csv_path} contains compounds not in the Gani "
                f"decomposition data: {sorted(unknown_compounds)}"
            )
        if compounds != self.compounds:
            # Reorder (and subset) decomposition rows to match the gcData compound order
            order = [compounds.index(c) for c in self.compounds]
            decomp = decomp[order]

        self.gani_decomp = decomp

    def _init_critical_properties(self) -> None:
        """Initialize critical properties from group contribution method(s)."""

        # Group contribution critical properties
        self._MW = gani.MW(self, unit="kg/mol")
        self._Tc = gani.Tc(self, unit="K")
        self._Pc = gani.Pc(self, unit="Pa")
        self._Vc = gani.Vc(self, unit="m^3/mol")
        self._Tb = gani.Tb(self, unit="K")
        self._Tm = gani.Tm(self, unit="K")
        self._Hf = gani.Hf(self, unit="J/mol")
        self._Gf = gani.Gf(self, unit="J/mol")
        self._Hv_stp = gani.Hv_stp(self, unit="J/mol")
        self._Cp_stp = gani.Cp_stp(self, unit="J/(mol*K)")
        self._Cp_B = gani.Cp_B(self, unit="J/(mol*K)")
        self._Cp_C = gani.Cp_C(self, unit="J/(mol*K)")
        self._Vm_stp = gani.Vm_stp(self, unit="m^3/mol")
        self._omega = gani.omega(self)

    # Basic properties of the mixture
    @property
    def num_compounds(self) -> int:
        """
        Return the number of compounds in the mixture.

        :return: Number of compounds.
        :rtype: int
        """
        return self.gani_decomp.shape[0]

    @property
    def num_groups(self) -> int:
        """
        Return the number of functional groups in the decomposition.

        :return: Number of functional groups.
        :rtype: int
        """
        return self.gani_decomp.shape[1]

    @property
    def num_carbons(self) -> int:
        """
        Return the number of carbon atoms in each component of the mixture.

        :return: Number of carbon atoms.
        :rtype: int
        """
        raise NotImplementedError("num_carbons property is not implemented yet.")

    @property
    def num_hydrogens(self) -> int:
        """
        Return the number of hydrogen atoms in each component of the mixture.

        :return: Number of hydrogen atoms.
        :rtype: int
        """
        raise NotImplementedError("num_hydrogens property is not implemented yet.")

    # Derived properties
    ## Cannot cache these because they depend on properties that can be updated (e.g., MW, Tc, Pc, etc.)
    @property
    def Lv_stp(self) -> AbstractQuantity:
        """Return the latent heat of vaporization at STP for each compound in the mixture."""
        return self._Hv_stp / self._MW

    @property
    def sigma(self) -> AbstractQuantity:
        """Calculate the diffusion coefficient of fuel compounds in a gas using the Tee correlation."""
        return tee_sigma(self)

    @property
    def epsilonByKB(self) -> AbstractQuantity:
        """Return the Lennard-Jones potential well depth (epsilon/k) for each compound in the mixture."""
        return tee_epsilon(self)

    # Compound property correlations
    def molar_liquid_vol(
        self, T: AbstractQuantity, *, unit: str = "m^3/mol"
    ) -> Quantity:
        """
        Calculate the molar liquid volume of fuel compounds over a range of temperatures.

        :param T: Temperature at which to calculate the molar liquid volume.
        :type T: AbstractQuantity
        :param unit: Desired unit for the output. Defaults to "m^3/mol".
        :type unit: str
        :return: Molar liquid volume of fuel compounds at the specified temperature.
        :rtype: Quantity
        """
        Tstp = Quantity(298.15, "K")

        T = _atleast_col(T)  # Ensure T has a trailing axis for broadcasting
        T = convert_temperature(T, "K")
        Tc = convert_temperature(self.Tc, "K")

        # Strip units from T and broadcast for comparison
        condition = qnp.array(T.value) > qnp.array(Tc.value)
        x = -qnp.power(1 - (Tstp / Tc), 2.0 / 7.0)
        y = qnp.power(1 - (T / Tc), 2.0 / 7.0) + x

        phi: Array = qnp.where(condition, x, y).value  # ty: ignore[unresolved-attribute]

        z1 = 0.29056
        z2 = 0.08775
        z: Array = (z1 - z2 * self.omega).value  # ty: ignore[invalid-assignment]

        return (self.Vm_stp * qnp.power(z, phi)).to(unit)

    def density(self, T: AbstractQuantity, *, unit: str = "kg/m^3") -> AbstractQuantity:
        """
        Calculate the density of fuel compounds over a range of temperatures.

        :param T: Temperatures at which to calculate the density.
        :type T: AbstractQuantity
        :param unit: Desired unit for the output. Defaults to "kg/m^3".
        :type unit: str
        :return: Density of fuel compounds at the specified temperature.
        :rtype: AbstractQuantity
        """
        T = _atleast_col(T)  # Ensure T has a trailing axis for broadcasting
        T = convert_temperature(T, "K")
        return (self.MW / self.molar_liquid_vol(T, unit="m^3/mol")).to(unit)

    def viscosity_kinematic(
        self, T: AbstractQuantity, *, unit: str = "mm^2/s"
    ) -> AbstractQuantity:
        """
        Calculate the kinematic viscosity of fuel compounds over a range of temperatures.

        :param T: Temperatures at which to calculate the kinematic viscosity.
        :type T: AbstractQuantity
        :param unit: Desired unit for the output. Defaults to "mm^2/s".
        :type unit: str
        :return: Kinematic viscosity of fuel compounds at the specified temperature.
        :rtype: AbstractQuantity
        """
        T = _atleast_col(T)  # Ensure T has a trailing axis for broadcasting
        T = convert_temperature(T, "Celsius")
        Tb = convert_temperature(self.Tb, "Celsius")

        num = Quantity(442.78, "Celsius") + 1.6452 * Tb
        denom = T + Quantity(239.0, "Celsius") - 0.19 * Tb
        return Quantity(qnp.exp(-3.0171 + (num / denom)).value, "mm^2/s").to(unit)

    def viscosity_dynamic(
        self, T: AbstractQuantity, *, unit: str = "Pa*s"
    ) -> AbstractQuantity:
        """
        Calculate the dynamic viscosity of fuel compounds over a range of temperatures.

        :param T: Temperatures at which to calculate the dynamic viscosity.
        :type T: AbstractQuantity
        :param unit: Desired unit for the output. Defaults to "Pa*s".
        :type unit: str
        :return: Dynamic viscosity of fuel compounds at the specified temperature.
        :rtype: AbstractQuantity
        """
        # Skipping conversion and broadcasting here because viscosity_kinematic and density already handle it
        return (self.viscosity_kinematic(T) * self.density(T)).to(unit)

    def Cp(self, T: AbstractQuantity, *, unit: str = "J/(mol*K)") -> AbstractQuantity:
        """
        Calculate the heat capacity of fuel compounds over a range of temperatures.

        :param T: Temperatures at which to calculate the heat capacity.
        :type T: AbstractQuantity
        :param unit: Desired unit for the output. Defaults to "J/(mol*K)".
        :type unit: str
        :return: Heat capacity of fuel compounds at the specified temperature.
        :rtype: AbstractQuantity
        """
        Tstp = Quantity(298.0, "K")
        T = _atleast_col(T)  # Ensure T has a trailing axis for broadcasting
        T = convert_temperature(T, "K")
        theta = (T - Tstp) / Quantity(700, "K")
        return (self.Cp_stp + self.Cp_B * theta + self.Cp_C * qnp.power(theta, 2)).to(
            unit
        )

    def Cl(self, T: AbstractQuantity, *, unit: str = "J/(kg*K)") -> AbstractQuantity:
        """
        Calculate the specific heat capacity of fuel compounds over a range of temperatures.

        :param T: Temperatures at which to calculate the specific heat capacity.
        :type T: AbstractQuantity
        :param unit: Desired unit for the output. Defaults to "J/(kg*K)".
        :type unit: str
        :return: Specific heat capacity of fuel compounds at the specified temperature.
        :rtype: AbstractQuantity
        """
        # Skipping conversion and broadcasting here because Cp already handles it
        return (self.Cp(T) / self.MW).to(unit)

    def psat(
        self,
        T: AbstractQuantity,
        *,
        unit: str = "Pa",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> AbstractQuantity:
        """
        Compute the saturation pressure of fuel compounds over a range of temperatures.

        :param T: Temperatures at which to calculate the saturation pressure.
        :type T: AbstractQuantity
        :param unit: Desired unit for the output. Defaults to "Pa".
        :type unit: str
        :param correlation: The correlation method to use for calculating saturation pressure.
            Options are "Ambrose-Walton" or "Lee-Kesler". Defaults to "Lee-Kesler".
        :type correlation: Literal["Ambrose-Walton", "Lee-Kesler"], optional
        :return: Saturation pressure of fuel compounds at the specified temperature.
        :rtype: AbstractQuantity
        """
        T = _atleast_col(T)  # Ensure T has a trailing axis for broadcasting
        T = convert_temperature(T, "K")
        Tc = convert_temperature(self.Tc, "K")
        Pc = self.Pc.to("Pa")
        omega = self.omega

        Tr = T / Tc

        if correlation == "Ambrose-Walton":
            tau = 1 - Tr
            f0 = (
                -5.97616 * tau
                + 1.29874 * qnp.power(tau, 1.5)
                - 0.60394 * qnp.power(tau, 2.5)
                - 1.06841 * qnp.power(tau, 5.0)
            ) / Tr
            f1 = (
                -5.03365 * tau
                + 1.11505 * qnp.power(tau, 1.5)
                - 5.41217 * qnp.power(tau, 2.5)
                - 7.46628 * qnp.power(tau, 5.0)
            ) / Tr
            f2 = (
                -0.64771 * tau
                + 2.41539 * qnp.power(tau, 1.5)
                - 4.26979 * qnp.power(tau, 2.5)
                - 3.25259 * qnp.power(tau, 5.0)
            ) / Tr
            return (Pc * qnp.exp(f0 + omega * f1 + omega**2 * f2)).to(unit)

        elif correlation == "Lee-Kesler":
            f0 = (
                5.92714
                - (6.09648 / Tr)
                - 1.28862 * qnp.log(Tr)
                + 0.169347 * qnp.power(Tr, 6)
            )
            f1 = (
                15.2518
                - (15.6875 / Tr)
                - 13.4721 * qnp.log(Tr)
                + 0.43577 * qnp.power(Tr, 6)
            )
            return (Pc * qnp.exp(f0 + omega * f1)).to(unit)

        raise ValueError(
            f"Invalid correlation '{correlation}'. Must be 'Ambrose-Walton' or 'Lee-Kesler'."
        )

    def psat_antoine_coeffs(
        self,
        T: AbstractQuantity,
        *,
        unit: Literal["mks", "cgs", "bar", "atm"] = "mks",
        correlation: Literal["Ambrose-Walton", "Lee-Kesler"] = "Lee-Kesler",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the Antoine coefficients for the saturation pressure of fuel compounds over a range of temperatures.

        :param T: Temperatures at which to calculate the Antoine coefficients.
        :type T: AbstractQuantity
        :param unit: Desired units for the output. Options are "mks", "cgs", "bar", or "atm". Defaults to "mks".
        :type unit: Literal["mks", "cgs", "bar", "atm"], optional
        :param correlation: The correlation method to use for calculating Antoine coefficients.
            Options are "Ambrose-Walton" or "Lee-Kesler". Defaults to "Lee-Kesler".
        :type correlation: Literal["Ambrose-Walton", "Lee-Kesler"], optional
        :return: Tuple containing the Antoine coefficients (A, B, C) and the conversion factor (D).
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        if qnp.shape(T)[0] < 3:
            raise ValueError(
                "At least three temperature points are required to calculate Antoine coefficients."
            )

        def _antoine_eq(T, A, B, C):
            """Antoine equation for vapor pressure."""
            return 10 ** (A - (B / (T + C)))

        # Hard-coded unit conversions for pressure because curve_fit doesn't handle unxt quantities
        # and unxt doesn't some of recognize the unit strings used here (e.g., "atm")
        conversions = {
            "mks": 1.0,  # Pa
            "cgs": 1e-1,  # dyn/cm^2
            "bar": 1.0e5,  # bar
            "atm": 1.01325e5,  # atm
        }

        T = convert_temperature(T, "K")

        A = np.zeros(self.num_compounds)
        B = np.zeros(self.num_compounds)
        C = np.zeros(self.num_compounds)
        D = np.zeros(self.num_compounds) + conversions[unit]  # Store conversion factor
        for i in range(self.num_compounds):
            psat = (
                self.psat(T, unit="Pa", correlation=correlation).value[:, i]
                / conversions[unit]
            )
            popt, _ = curve_fit(_antoine_eq, T.value, psat, p0=[1, 1e3, -1])
            A[i], B[i], C[i] = popt

        return A, B, C, D

    def latent_heat_vaporization(
        self, T: AbstractQuantity, *, unit: str = "J/kg"
    ) -> AbstractQuantity:
        """
        Calculate the latent heat of vaporization of fuel compounds over a range of temperatures.

        :param T: Temperatures at which to calculate the latent heat of vaporization.
        :type T: AbstractQuantity
        :param unit: Desired unit for the output. Defaults to "J/kg".
        :type unit: str
        :return: Latent heat of vaporization of fuel compounds at the specified temperature.
        :rtype: AbstractQuantity
        """
        T = _atleast_col(T)  # Ensure T has a trailing axis for broadcasting
        T = convert_temperature(T, "K")
        Tc = convert_temperature(self.Tc, "K")
        Tb = convert_temperature(self.Tb, "K")

        # Reduced temperatures
        Tr = T / Tc
        Trb = Tb / Tc

        # Strip units from T and broadcast for comparison
        condition = qnp.array(T.value) > qnp.array(Tc.value)
        x = Quantity(0.0, "J/kg")
        y = self.Lv_stp * qnp.power((1.0 - Tr) / (1.0 - Trb), 0.38)
        return qnp.where(condition, x, y).to(unit)  # ty: ignore[invalid-argument-type]

    def diffusion_coeff(
        self,
        P: AbstractQuantity,
        T: AbstractQuantity,
        *,
        unit: str = "m^2/s",
        correlation: Literal["Tee", "Wilke"] = "Tee",
        sigma_gas: AbstractQuantity = SIGMA_GAS,
        epsilonByKB_gas: AbstractQuantity = EPSILON_BY_KB_GAS,
        MW_gas: AbstractQuantity = MW_GAS,
    ) -> AbstractQuantity:
        """
        Calculate the diffusion coefficient of fuel compounds over a range of temperatures and pressures.

        If both ``T`` and ``P`` are 1D arrays, the result is broadcast over the full
        (T, P) grid, giving shape ``(len(T), len(P), num_compounds)``. If only one of
        ``T`` or ``P`` is a 1D array, the result has shape ``(len(T or P), num_compounds)``.

        :param P: Pressures at which to calculate the diffusion coefficient.
        :type P: AbstractQuantity
        :param T: Temperatures at which to calculate the diffusion coefficient.
        :type T: AbstractQuantity
        :param unit: Desired unit for the output. Defaults to "m^2/s".
        :type unit: str
        :param correlation: Correlation method to use for calculating the diffusion coefficient. Defaults to "Tee".
        :type correlation: Literal["Tee", "Wilke"]
        :param sigma_gas: Lennard-Jones collision diameter of the gas. Defaults to 3.62e-10 m.
        :type sigma_gas: AbstractQuantity
        :param epsilonByKB_gas: Lennard-Jones energy parameter divided by Boltzmann constant for the gas. Defaults to 97.0 K.
        :type epsilonByKB_gas: AbstractQuantity
        :param MW_gas: Molecular weight of the gas. Defaults to 28.97e-3 kg/mol.
        :type MW_gas: AbstractQuantity
        :return: Diffusion coefficient of fuel compounds at the specified temperature(s) and pressure(s).
            The output shape is ``(len(T), len(P), num_compounds)`` if both ``T`` and ``P`` are 1D arrays, or
            ``(len(T or P), num_compounds)`` if only one of ``T`` or ``P`` is a 1D array.
            If both ``T`` and ``P`` are scalars, the output shape is ``(num_compounds,)``.
        :rtype: AbstractQuantity
        """
        if not (
            qnp.size(sigma_gas) == 1
            and qnp.size(epsilonByKB_gas) == 1
            and qnp.size(MW_gas) == 1
        ):
            raise ValueError(
                "sigma_gas, epsilonByKB_gas, and MW_gas must be scalar quantities."
            )

        if qnp.ndim(T) == 1 and qnp.ndim(P) == 1:
            # Both T and P vary independently, so give each its own axis (ahead of the
            # per-compound axis) to broadcast a full (T, P) grid instead of colliding on axis 0
            T = T[:, None, None]
            P = P[None, :, None]
        else:
            T = _atleast_col(T)  # Ensure T has a trailing axis for broadcasting
            P = _atleast_col(P)  # Ensure P has a trailing axis for broadcasting

        T = convert_temperature(T, "K")
        P = P.to("Pa")

        if correlation == "Tee":
            sigma_i = self.sigma.to("Angstrom")
            epsilonByKB_i = convert_temperature(self.epsilonByKB, "K")
        elif correlation == "Wilke":
            Vmb_i = self.molar_liquid_vol(self.Tb, unit="m^3/mol").to("m^3/mol")
            sigma_i = 1.18 * qnp.power(Vmb_i, 1 / 3).to("Angstrom")
            epsilonByKB_i = 1.15 * convert_temperature(self.Tb, "K")
        else:
            raise ValueError(
                f"Invalid correlation '{correlation}'. Must be 'Tee' or 'Wilke'."
            )

        sigmaAB_i = (sigma_gas + sigma_i) / 2.0
        epsilonAB_byKB_i = qnp.sqrt(epsilonByKB_gas * epsilonByKB_i)

        # T and epsilonAB_byKB_i are both temperatures, so Tstar_i is dimensionless
        Tstar_i = T / epsilonAB_byKB_i
        omegaD_i = (
            1.06036 / qnp.power(Tstar_i, 0.15610)
            + 0.193 / qnp.exp(0.47635 * Tstar_i)
            + 1.03587 / qnp.exp(1.52996 * Tstar_i)
            + 1.76474 / qnp.exp(3.89411 * Tstar_i)
        )

        MW_AB_i = 2 * (self.MW * MW_gas) / (self.MW + MW_gas)

        # Wilke-Lee correlation is calibrated to a mixed unit system
        C = Quantity(1e-3, "bar*cm^2*Angstrom^2*g^(1/2)/(mol^(1/2)*s*K^(3/2))")
        MW_const = Quantity(0.98, "g^(1/2)/mol^(1/2)")

        D_AB_i = (
            C
            * (3.03 - MW_const / qnp.sqrt(MW_AB_i))
            * qnp.power(T, 1.5)
            / (P * qnp.sqrt(MW_AB_i) * qnp.square(sigmaAB_i) * omegaD_i)
        )

        return (D_AB_i).to(unit)

    def surface_tension(
        self,
        T: AbstractQuantity,
        *,
        unit: str = "N/m",
        correlation: Literal["Brock-Bird", "Pitzer"] = "Brock-Bird",
    ) -> AbstractQuantity:
        """
        Calculate the surface tension of fuel compounds over a range of temperatures.

        :param T: Temperatures at which to calculate the surface tension.
        :type T: AbstractQuantity
        :param unit: Desired unit for the output. Defaults to "N/m".
        :type unit: str
        :return: Surface tension of fuel compounds at the specified temperature.
        :rtype: AbstractQuantity
        :param correlation: Correlation method to use for calculating surface tension.
            Options are "Brock-Bird" or "Pitzer". Defaults to "Brock-Bird".
        :type correlation: Literal["Brock-Bird", "Pitzer"], optional
        """
        T = _atleast_col(T)  # Ensure T has a trailing axis for broadcasting
        T = convert_temperature(T, "K")
        Tc = convert_temperature(self.Tc, "K")
        Tr = T / Tc

        if correlation == "Brock-Bird":
            Tbr = convert_temperature(self.Tb, "K") / Tc
            Q = (
                0.1196
                * (
                    1.0
                    + (Tbr * qnp.log(self.Pc / Quantity(1.01325, "bar"))) / (1.0 - Tbr)
                )
                - 0.279
            )
        elif correlation == "Pitzer":
            Q = ((1.86 + 1.18 * self.omega) / 19.05) * qnp.power(
                (3.75 + 0.91 * self.omega) / (0.291 - 0.08 * self.omega), 2.0 / 3.0
            )
        else:
            raise ValueError(
                f"Invalid correlation '{correlation}'. Must be 'Brock-Bird' or 'Pitzer'."
            )

        # Brock-Bird/Pitzer correlation is calibrated to Pc in atm and Tc in K, giving dyn/cm
        Pc_atm = convert_pressure_to_atm(self.Pc)
        C = Quantity(1.0, "dyn/(cm*atm^(2/3)*K^(1/3))")

        st = (
            C
            * qnp.power(Pc_atm, 2.0 / 3.0)
            * qnp.power(Tc, 1.0 / 3.0)
            * Q
            * qnp.power(1 - Tr, 11.0 / 9.0)
        )

        return st.to(unit)

    # Mixture property correlations
    def mean_molecular_weight(self, *, unit: str = "kg/mol") -> AbstractQuantity:
        """
        Calculate the mean molecular weight of the mixture.

        :return: Mean molecular weight of the mixture.
        :rtype: AbstractQuantity
        """
        return qnp.sum(self.MW / self.Y_0).to(unit)

    # Utility functions
    def mass2Y(self, mass: AbstractQuantity) -> Array:
        """
        Convert component masses to mass fractions of the mixture.

        :param mass: Masses of the components in the mixture.
        :type mass: AbstractQuantity
        :return: Mass fractions of the components in the mixture.
        :rtype: Array
        """
        _check_valid_property(mass, self.num_compounds, "mass")
        Y = mass / qnp.sum(mass)
        return jnp.array(Y.value)

    def mass2X(self, mass: AbstractQuantity) -> Array:
        """
        Convert component masses to mole fractions of the mixture.

        :param mass: Masses of the components in the mixture.
        :type mass: AbstractQuantity
        :return: Mole fractions of the components in the mixture.
        :rtype: Array
        """
        _check_valid_property(mass, self.num_compounds, "mass")
        X = (mass / self.MW) / qnp.sum(mass / self.MW)
        return jnp.array(X.value)

    def X2Y(self, X: Array) -> Array:
        """
        Convert mole fractions to mass fractions of the mixture.

        :param X: Mole fractions of the components in the mixture.
        :type X: Array
        :return: Mass fractions of the components in the mixture.
        :rtype: Array
        """
        _check_valid_property(X, self.num_compounds, "X")
        Y = (X * self.MW) / qnp.sum(X * self.MW)
        return jnp.array(Y.value)

    def Y2X(self, Y: Array) -> Array:
        """
        Convert mass fractions to mole fractions of the mixture.

        :param Y: Mass fractions of the components in the mixture.
        :type Y: Array
        :return: Mole fractions of the components in the mixture.
        :rtype: Array
        """
        _check_valid_property(Y, self.num_compounds, "Y")
        X = (Y / self.MW) / qnp.sum(Y / self.MW)
        return jnp.array(X.value)

    # Critical property getters and setters
    @property
    def Y_0(self) -> Array:
        """
        Return the mass fractions of the compounds in the mixture.

        :return: Mass fractions of the compounds.
        :rtype: Array
        """
        return self._Y_0

    @Y_0.setter
    def Y_0(self, value: Array) -> None:
        """
        Set the mass fractions of the compounds in the mixture.

        :param value: Mass fractions of the compounds.
        :type value: Array
        """
        _check_valid_property(value, self.num_compounds, "Y_0")
        if not jnp.isclose(jnp.sum(value), 1.0, rtol=5e-2):
            raise ValueError("Y_0 must sum to 1.00 +/- 0.05.")

        self._Y_0 = value

    @property
    def MW(self) -> AbstractQuantity:
        """
        Return the molecular weights of the compounds in the mixture.

        :return: Molecular weights of the compounds.
        :rtype: AbstractQuantity
        """
        return self._MW

    @MW.setter
    def MW(self, value: AbstractQuantity) -> None:
        """
        Set the molecular weights of the compounds in the mixture.

        :param value: Molecular weights of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "MW")
        self._MW = value.to("kg/mol")

    @property
    def Tc(self) -> AbstractQuantity:
        """
        Return the critical temperatures of the compounds in the mixture.

        :return: Critical temperatures of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Tc

    @Tc.setter
    def Tc(self, value: AbstractQuantity) -> None:
        """
        Set the critical temperatures of the compounds in the mixture.

        :param value: Critical temperatures of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Tc")
        self._Tc = convert_temperature(value, "K")

    @property
    def Pc(self) -> AbstractQuantity:
        """
        Return the critical pressures of the compounds in the mixture.

        :return: Critical pressures of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Pc

    @Pc.setter
    def Pc(self, value: AbstractQuantity) -> None:
        """
        Set the critical pressures of the compounds in the mixture.

        :param value: Critical pressures of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Pc")
        self._Pc = value.to("Pa")

    @property
    def Vc(self) -> AbstractQuantity:
        """
        Return the critical volumes of the compounds in the mixture.

        :return: Critical volumes of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Vc

    @Vc.setter
    def Vc(self, value: AbstractQuantity) -> None:
        """
        Set the critical volumes of the compounds in the mixture.

        :param value: Critical volumes of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Vc")
        self._Vc = value.to("m^3/mol")

    @property
    def Tb(self) -> AbstractQuantity:
        """
        Return the boiling points of the compounds in the mixture.

        :return: Boiling points of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Tb

    @Tb.setter
    def Tb(self, value: AbstractQuantity) -> None:
        """
        Set the boiling points of the compounds in the mixture.

        :param value: Boiling points of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Tb")
        self._Tb = convert_temperature(value, "K")

    @property
    def Tm(self) -> AbstractQuantity:
        """
        Return the melting points of the compounds in the mixture.

        :return: Melting points of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Tm

    @Tm.setter
    def Tm(self, value: AbstractQuantity) -> None:
        """
        Set the melting points of the compounds in the mixture.

        :param value: Melting points of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Tm")
        self._Tm = convert_temperature(value, "K")

    @property
    def Hf(self) -> AbstractQuantity:
        """
        Return the heat of formation of the compounds in the mixture.

        :return: Heat of formation of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Hf

    @Hf.setter
    def Hf(self, value: AbstractQuantity) -> None:
        """
        Set the heat of formation of the compounds in the mixture.

        :param value: Heat of formation of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Hf")
        self._Hf = value.to("J/mol")

    @property
    def Gf(self) -> AbstractQuantity:
        """
        Return the Gibbs free energy of the compounds in the mixture.

        :return: Gibbs free energy of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Gf

    @Gf.setter
    def Gf(self, value: AbstractQuantity) -> None:
        """
        Set the Gibbs free energy of the compounds in the mixture.

        :param value: Gibbs free energy of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Gf")
        self._Gf = value.to("J/mol")

    @property
    def Hv_stp(self) -> AbstractQuantity:
        """
        Return the heat of vaporization at STP of the compounds in the mixture.

        :return: Heat of vaporization at STP of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Hv_stp

    @Hv_stp.setter
    def Hv_stp(self, value: AbstractQuantity) -> None:
        """
        Set the heat of vaporization at STP of the compounds in the mixture.

        :param value: Heat of vaporization at STP of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Hv_stp")
        self._Hv_stp = value.to("J/mol")

    @property
    def Lv_stp(self) -> AbstractQuantity:
        """
        Return the latent heat of vaporization at STP of the compounds in the mixture.

        :return: Latent heat of vaporization at STP of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Lv_stp

    @Lv_stp.setter
    def Lv_stp(self, value: AbstractQuantity) -> None:
        """
        Set the latent heat of vaporization at STP of the compounds in the mixture.

        :param value: Latent heat of vaporization at STP of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Lv_stp")
        self._Lv_stp = value.to("J/mol")

    @property
    def Cp_stp(self) -> AbstractQuantity:
        """
        Return the heat capacity at STP of the compounds in the mixture.

        :return: Heat capacity at STP of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Cp_stp

    @Cp_stp.setter
    def Cp_stp(self, value: AbstractQuantity) -> None:
        """
        Set the heat capacity at STP of the compounds in the mixture.

        :param value: Heat capacity at STP of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Cp_stp")
        self._Cp_stp = value.to("J/mol/K")

    @property
    def Cp_B(self) -> AbstractQuantity:
        """
        Return the temperature correction B for heat capacity of the compounds in the mixture.

        :return: Temperature correction B for heat capacity of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Cp_B

    @Cp_B.setter
    def Cp_B(self, value: AbstractQuantity) -> None:
        """
        Set the temperature correction B for heat capacity of the compounds in the mixture.

        :param value: Temperature correction B for heat capacity of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Cp_B")
        self._Cp_B = value.to("J/(mol*K)")

    @property
    def Cp_C(self) -> AbstractQuantity:
        """
        Return the temperature correction C for heat capacity of the compounds in the mixture.

        :return: Temperature correction C for heat capacity of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Cp_C

    @Cp_C.setter
    def Cp_C(self, value: AbstractQuantity) -> None:
        """
        Set the temperature correction C for heat capacity of the compounds in the mixture.

        :param value: Temperature correction C for heat capacity of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Cp_C")
        self._Cp_C = value.to("J/(mol*K)")

    @property
    def Vm_stp(self) -> AbstractQuantity:
        """
        Return the molar volume at STP of the compounds in the mixture.

        :return: Molar volume at STP of the compounds.
        :rtype: AbstractQuantity
        """
        return self._Vm_stp

    @Vm_stp.setter
    def Vm_stp(self, value: AbstractQuantity) -> None:
        """
        Set the molar volume at STP of the compounds in the mixture.

        :param value: Molar volume at STP of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "Vm_stp")
        self._Vm_stp = value.to("m^3/mol")

    @property
    def omega(self) -> AbstractQuantity:
        """
        Return the acentric factor of the compounds in the mixture.

        :return: Acentric factor of the compounds.
        :rtype: AbstractQuantity
        """
        return self._omega

    @omega.setter
    def omega(self, value: AbstractQuantity) -> None:
        """
        Set the acentric factor of the compounds in the mixture.

        :param value: Acentric factor of the compounds.
        :type value: AbstractQuantity
        """
        _check_valid_property(value, self.num_compounds, "omega")
        self._omega = value
