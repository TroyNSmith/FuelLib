import argparse
import os

import numpy as np
import pandas as pd

import fuellib as fl
from fuellib.utils.units import ureg

# Default data directory - use fuellib's embedded data
FUELDATA_DIR = fl.get_fueldata_dir()

"""
Script that exports mixture properties over large temperature range for use in
Converge simulations.

This script is designed to be run from the command line and will create
a file named "mixturePropsGCM_<fuel_name>.csv" in the specified directory.
The file contains mixture properties for the fuel, formatted for Converge.

Usage:
    fl-export-converge -f <fuel_name>

For detailed options, run:
    fl-export-converge -h
"""


class UnitConverter:
    """Unit conversion factors and labels for different unit systems."""

    def __init__(self, units: str):
        """
        Initialize converter for specified unit system.

        :param units: Unit system ('cgs' or 'mks').
        :type units: str
        """
        self.units = units.lower()
        self._set_conversion_factors()
        self._set_labels()

    def _set_conversion_factors(self):
        """Set conversion factors based on unit system."""
        if self.units == "cgs":
            # Convert from MKS to CGS
            self.mw = 1e3  # kg/mol to g/mol
            self.mu = 1e2  # Pa*s to Poise
            self.surface_tension = 1e7  # N/m to dyne/cm
            self.latent_heat = 1e4  # J/kg to erg/g
            self.pressure = 1e1  # Pa to dyne/cm^2
            self.rho = 1e3  # kg/m^3 to g/cm^3
            self.specific_heat_mass = 1e4  # J/kg/K to erg/g/K
            self.thermal_conductivity = 1e5  # W/m/K to erg/cm/s/K
        else:
            # MKS units (no conversion)
            self.mw = 1
            self.mu = 1
            self.surface_tension = 1
            self.latent_heat = 1
            self.pressure = 1
            self.rho = 1
            self.specific_heat_mass = 1
            self.thermal_conductivity = 1

    def _set_labels(self):
        """Set unit labels for DataFrame columns."""
        if self.units == "cgs":
            self.labels = {
                "temperature": "Temperature (K)",
                "critical_temp": "Critical Temperature (K)",
                "viscosity": "Viscosity (Poise)",
                "surface_tension": "Surface Tension (dyne/cm)",
                "heat_vaporization": "Heat of Vaporization (erg/g)",
                "vapor_pressure": "Vapor Pressure (dyne/cm^2)",
                "density": "Density (g/cm^3)",
                "specific_heat": "Specific Heat (erg/g/K)",
                "thermal_conductivity": "Thermal Conductivity (erg/cm/s/K)",
                "molecular_weight": "Molecular Weight (g/mol)",
            }
        else:
            self.labels = {
                "temperature": "Temperature (K)",
                "critical_temp": "Critical Temperature (K)",
                "viscosity": "Viscosity (Pa*s)",
                "surface_tension": "Surface Tension (N/m)",
                "heat_vaporization": "Heat of Vaporization (J/kg)",
                "vapor_pressure": "Vapor Pressure (Pa)",
                "density": "Density (kg/m^3)",
                "specific_heat": "Specific Heat (J/kg/K)",
                "thermal_conductivity": "Thermal Conductivity (W/m/K)",
                "molecular_weight": "Molecular Weight (kg/mol)",
            }

    def create_data_dict(
        self,
        T,
        T_crit,
        mu,
        surface_tension,
        latent_heat,
        pv,
        rho,
        specific_heat_mass,
        thermal_conductivity,
    ):
        """
        Create a data dictionary with converted units and appropriate labels.

        :param T: Temperature array.
        :type T: np.ndarray
        :param T_crit: Critical temperature.
        :type T_crit: float
        :param mu: Viscosity array.
        :type mu: np.ndarray
        :param surface_tension: Surface tension array.
        :type surface_tension: np.ndarray
        :param latent_heat: Heat of vaporization array.
        :type latent_heat: np.ndarray
        :param pv: Vapor pressure array.
        :type pv: np.ndarray
        :param rho: Density array.
        :type rho: np.ndarray
        :param specific_heat_mass: Specific heat array.
        :type specific_heat_mass: np.ndarray
        :param thermal_conductivity: Thermal conductivity array.
        :type thermal_conductivity: np.ndarray
        :return: Dictionary with converted properties and labels.
        :rtype: dict
        """
        return {
            self.labels["temperature"]: T,
            self.labels["critical_temp"]: T_crit + np.zeros_like(T),
            self.labels["viscosity"]: mu * self.mu,
            self.labels["surface_tension"]: surface_tension * self.surface_tension,
            self.labels["heat_vaporization"]: latent_heat * self.latent_heat,
            self.labels["vapor_pressure"]: pv * self.pressure,
            self.labels["density"]: rho * self.rho,
            self.labels["specific_heat"]: specific_heat_mass * self.specific_heat_mass,
            self.labels["thermal_conductivity"]: thermal_conductivity
            * self.thermal_conductivity,
        }


def export_converge(
    fuel,
    path=None,
    units="mks",
    temp_min=0,
    temp_max=1000,
    temp_step=10,
    export_mix=False,
):
    """
    Export mixture fuel properties to csv files for Converge simulations.

    :param fuel: Fuel object containing properties to export.
    :type fuel: fl.fuel

    :param path: Directory to save the input file.
    :type path: str, optional (default: current working directory)

    :param units: Units for the properties ("mks" for SI, "cgs" for CGS).
    :type units: str, optional (default: "mks")

    :param temp_min: Minimum temperature (K) for the property calculations.
    :type temp_min: float, optional (default: 0)

    :param temp_max: Maximum temperature (K)for the property calculations.
    :type temp_max: float, optional (default: 1000)

    :param temp_step: Step size for temperature (K).
    :type temp_step: int, optional (default: 10)

    :param export_mix: Whether to export individual component or mixture properties.
    :type export_mix: bool, optional (default: False)

    :return: None
    :rtype: None

    :raises ValueError: If input parameters are invalid
    :raises TypeError: If fuel object is not a FuelLib fuel instance
    """
    if path is None:
        path = os.getcwd()

    # Input validation
    if not hasattr(fuel, "compounds") or not hasattr(fuel, "initial_mass_fractions"):
        raise TypeError("fuel parameter must be a valid FuelLib fuel object")

    if units.lower() not in ["mks", "cgs"]:
        raise ValueError(f"Units must be 'mks' or 'cgs', got '{units}'")

    if temp_min < 0:
        raise ValueError(f"temp_min must be non-negative, got {temp_min}")

    if temp_max <= temp_min:
        raise ValueError(
            f"temp_max ({temp_max}) must be greater than temp_min ({temp_min})"
        )

    if temp_step <= 0:
        raise ValueError(f"temp_step must be positive, got {temp_step}")

    # Ensure output directory exists
    if not os.path.exists(path):
        os.makedirs(path)

    if export_mix:
        # Export mixture properties only
        file_name = os.path.join(path, f"mixturePropsGCM_{fuel.name}.csv")
        components = [fuel.name]
    else:
        # Export individual component properties and composition
        path = os.path.join(path, fuel.name)
        components = fuel.compounds

    # Initialize unit converter
    converter = UnitConverter(units)

    def nearest_temp(x, base=temp_step):
        """
        Round to nearest multiple of temp_step.

        :param x: Temperature value to round.
        :type x: float
        :param base: Base value for rounding (temp_step).
        :type base: float
        :return: Rounded temperature.
        :rtype: float
        """
        return base * round(x / base)

    def nearest_floor(array, value):
        """
        Find the largest value in the array that is less than or equal to the given value.

        :param array: Array of temperature values.
        :type array: np.ndarray
        :param value: Reference value.
        :type value: float
        :return: Largest array value <= reference value.
        :rtype: float
        :raises ValueError: If no array value is <= reference value.
        """
        if np.any(array <= value):
            return array[array <= value].max()
        else:
            raise ValueError(
                f"No temperature in the array is less than or equal to the critical point {value}. Choose a lower temp_min"
            )

    def nearest_ceil(array, value):
        """
        Find the smallest value in the array that is greater than or equal to the given value.

        :param array: Array of temperature values.
        :type array: np.ndarray
        :param value: Reference value.
        :type value: float
        :return: Smallest array value >= reference value.
        :rtype: float
        :raises ValueError: If no array value is >= reference value.
        """
        if np.any(array >= value):
            return array[array >= value].min()
        else:
            raise ValueError(
                f"No temperature in the array is greater than or equal the freezing point {value}. Choose a higher temp_max"
            )

    def validate_temperature_range(T_array, T_freeze, T_crit, is_mixture=True):
        """
        Validate and adjust temperature range based on freezing and critical temperatures.

        :param T_array: Array of temperature values.
        :type T_array: np.ndarray
        :param T_freeze: Freezing temperature.
        :type T_freeze: float
        :param T_crit: Critical temperature.
        :type T_crit: float
        :param is_mixture: Whether this is for mixture properties.
        :type is_mixture: bool
        :return: Tuple of (T_min_allowed, T_max_allowed, adjusted_T_array).
        :rtype: tuple
        """
        T_min_allowed = nearest_temp(T_freeze)
        T_max_allowed = T_crit

        # Handle minimum temperature warnings
        if np.any(T_array < T_min_allowed):
            T_min_allowed = nearest_ceil(T_array, T_min_allowed)
            compound_type = "mixture" if is_mixture else "compound"
            print("!" * 88)
            print(
                "   Warning: Some compounds have freezing temperatures above the estimated"
            )
            print(
                f"   freezing temperature of the {compound_type} ({T_freeze:.2f} K). All properties calculated"
            )
            print(
                f"   below {T_min_allowed} K will be set using a temperature of {T_min_allowed} K."
            )
            print("!" * 88)

        # Handle maximum temperature warnings for mixtures
        if is_mixture:
            T_max_allowed = Tc_arr.min()
            if np.any(T_array > T_max_allowed):
                T_max_allowed = nearest_floor(T_array, T_max_allowed)
                print("!" * 88)
                print(
                    "   Warning: Some compounds have critical temperatures below the estimated"
                )
                print(
                    f"   critical temperature of the mixture ({T_crit:.2f} K). All properties will be"
                )
                print(f"   calculated up to {T_max_allowed} K.")
                print("!" * 88)

        # Filter temperature array to allowed range
        adjusted_T = T_array[(T_array >= T_min_allowed) & (T_array <= T_max_allowed)]

        return T_min_allowed, T_max_allowed, adjusted_T

    def calculate_mixture_properties(T_array, fuel):
        """
        Calculate mixture properties for a range of temperatures.

        :param T_array: Array of temperature values.
        :type T_array: np.ndarray
        :param fuel: Fuel object.
        :type fuel: fl.fuel
        :return: Tuple of property arrays (mu, surface_tension, latent_heat, pv, rho, specific_heat_mass, thermal_conductivity).
        :rtype: tuple
        """
        # Initialize property arrays
        mu = np.zeros_like(T_array)
        surface_tension = np.zeros_like(T_array)
        latent_heat = np.zeros_like(T_array)
        pv = np.zeros_like(T_array)
        rho = np.zeros_like(T_array)
        specific_heat_mass = np.zeros_like(T_array)
        thermal_conductivity = np.zeros_like(T_array)

        for k, Temp in enumerate(T_array):
            mass_fractions = fuel.initial_mass_fractions
            mole_fractions = fl.fuel.mass_fraction_to_mole_fraction(
                fuel, mass_fractions
            )
            T_q = Temp * ureg.K

            # Standard mixing rules for properties
            rho[k] = (
                fl.fuel.mixture_density(fuel, mass_fractions, T_q)
                .to("kg/m**3")
                .magnitude
            )
            mu[k] = (
                fl.fuel.mixture_dynamic_viscosity(fuel, mass_fractions, T_q)
                .to("Pa*s")
                .magnitude
            )
            pv[k] = (
                fl.fuel.mixture_vapor_pressure(fuel, mass_fractions, T_q)
                .to("Pa")
                .magnitude
            )
            surface_tension[k] = (
                fl.fuel.mixture_surface_tension(fuel, mass_fractions, T_q)
                .to("N/m")
                .magnitude
            )
            thermal_conductivity[k] = (
                fl.fuel.mixture_thermal_conductivity(fuel, mass_fractions, T_q)
                .to("W/m/K")
                .magnitude
            )

            # Generic mixing rules for latent heat and specific heat
            latent_heat[k] = fl.helpers.mixing_rule(
                fl.fuel.latent_heat_vaporization(fuel, T_q).to("J/kg").magnitude,
                mole_fractions,
            )  # J/kg
            specific_heat_mass[k] = fl.helpers.mixing_rule(
                fl.fuel.mass_heat_capacity(fuel, T_q).to("J/kg/K").magnitude,
                mole_fractions,
            )  # J/kg/K

        return (
            mu,
            surface_tension,
            latent_heat,
            pv,
            rho,
            specific_heat_mass,
            thermal_conductivity,
        )

    def calculate_component_properties(T_array, fuel, comp_idx):
        """
        Calculate individual component properties for a range of temperatures.

        :param T_array: Array of temperature values.
        :type T_array: np.ndarray
        :param fuel: Fuel object.
        :type fuel: fl.fuel
        :param comp_idx: Index of the component.
        :type comp_idx: int
        :return: Tuple of property arrays (mu, surface_tension, latent_heat, pv, rho, specific_heat_mass, thermal_conductivity).
        :rtype: tuple
        """
        # Initialize property arrays
        mu = np.zeros_like(T_array)
        surface_tension = np.zeros_like(T_array)
        latent_heat = np.zeros_like(T_array)
        pv = np.zeros_like(T_array)
        rho = np.zeros_like(T_array)
        specific_heat_mass = np.zeros_like(T_array)
        thermal_conductivity = np.zeros_like(T_array)

        for k, Temp in enumerate(T_array):
            T_q = Temp * ureg.K
            rho[k] = (
                fl.fuel.density(fuel, T_q, comp_idx=comp_idx).to("kg/m**3").magnitude
            )
            mu[k] = (
                fl.fuel.viscosity_dynamic(fuel, T_q, comp_idx=comp_idx)
                .to("Pa*s")
                .magnitude
            )
            pv[k] = (
                fl.fuel.saturation_pressure(fuel, T_q, comp_idx=comp_idx)
                .to("Pa")
                .magnitude
            )
            surface_tension[k] = (
                fl.fuel.surface_tension(fuel, T_q, comp_idx=comp_idx)
                .to("N/m")
                .magnitude
            )
            thermal_conductivity[k] = (
                fl.fuel.thermal_conductivity(fuel, T_q, comp_idx=comp_idx)
                .to("W/m/K")
                .magnitude
            )
            latent_heat[k] = (
                fl.fuel.latent_heat_vaporization(fuel, T_q, comp_idx=comp_idx)
                .to("J/kg")
                .magnitude
            )
            specific_heat_mass[k] = (
                fl.fuel.mass_heat_capacity(fuel, T_q, comp_idx=comp_idx)
                .to("J/kg/K")
                .magnitude
            )

        return (
            mu,
            surface_tension,
            latent_heat,
            pv,
            rho,
            specific_heat_mass,
            thermal_conductivity,
        )

    def export_properties_to_csv(file_path, data_dict, overwrite=True):
        """
        Export properties data to CSV file.

        :param file_path: Path to the output CSV file.
        :type file_path: str
        :param data_dict: Dictionary containing property data.
        :type data_dict: dict
        :param overwrite: Whether to overwrite existing file.
        :type overwrite: bool
        """
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Remove existing file if overwrite is True
        if overwrite and os.path.exists(file_path):
            os.remove(file_path)

        # Create and save DataFrame
        df = pd.DataFrame(data_dict)
        df.to_csv(file_path, index=False)

    if export_mix:
        # Vector of evenly spaced temperatures
        nT = int((temp_max - temp_min) / temp_step) + 1
        T = np.linspace(temp_min, temp_max, nT)

        # Estimate freezing point and critical temp of mixture
        Tm_arr = fuel.melting_temperature.to("K").magnitude
        Tc_arr = fuel.critical_temperature.to("K").magnitude
        mole_fractions_init = fl.fuel.mass_fraction_to_mole_fraction(
            fuel, fuel.initial_mass_fractions
        )
        T_freeze = fl.helpers.mixing_rule(Tm_arr, mole_fractions_init)
        T_crit = fl.helpers.mixing_rule(Tc_arr, mole_fractions_init)

        print(f"\nEstimated mixture freezing temp: {T_freeze:.2f} K")
        print(f"Min freezing temp min(Tm_i): {Tm_arr.min():.2f} K")
        print(f"Max freezing temp max(Tm_i): {Tm_arr.max():.2f} K")
        print(f"Estimated mixture critical temp: {T_crit:.2f} K")
        print(f"Min critical temp min(Tc_i): {Tc_arr.min():.2f} K")
        print(f"Max critical temp max(Tc_i): {Tc_arr.max():.2f} K")

        # Validate and adjust temperature range
        T_min_allowed, T_max_allowed, T = validate_temperature_range(
            T, T_freeze, T_crit, is_mixture=True
        )

    for comp_idx, compound in enumerate(components):
        if not export_mix:
            # Get component-specific temperature limits
            T_freeze = fuel.melting_temperature[comp_idx].to("K").magnitude
            T_crit = fuel.critical_temperature[comp_idx].to("K").magnitude
            T_min_allowed = nearest_temp(T_freeze)

            # Create temperature array up to critical temperature
            maxtemps = np.array(
                [
                    nearest_temp(T_crit) - temp_step,
                    nearest_temp(T_crit),
                    nearest_temp(T_crit) + temp_step,
                ]
            )
            T_nearest_floor = nearest_floor(maxtemps, T_crit)
            nT = int((T_nearest_floor - T_min_allowed) / temp_step) + 1
            T = np.linspace(T_min_allowed, T_nearest_floor, nT)
            T = np.append(T, T_crit)
            T_max_allowed = T_crit
        # Calculate GCM properties for a range of temperatures
        comp_text = "" if export_mix else f"for {compound}"
        print(
            f"\nCalculating properties {comp_text} over {len(T)} temperatures from {T_min_allowed} K to {T_max_allowed} K..."
        )

        if export_mix:
            (
                mu,
                surface_tension,
                latent_heat,
                pv,
                rho,
                specific_heat_mass,
                thermal_conductivity,
            ) = calculate_mixture_properties(T, fuel)
        else:
            (
                mu,
                surface_tension,
                latent_heat,
                pv,
                rho,
                specific_heat_mass,
                thermal_conductivity,
            ) = calculate_component_properties(T, fuel, comp_idx)

        # Create data dictionary with converted units
        data = converter.create_data_dict(
            T,
            T_crit,
            mu,
            surface_tension,
            latent_heat,
            pv,
            rho,
            specific_heat_mass,
            thermal_conductivity,
        )

        # Export the properties to CSV file
        if export_mix:
            print(f"\nWriting mixture properties to {file_name}")
        else:
            file_name = os.path.join(path, f"{comp_idx}_{compound}.csv")
            print(f"\nWriting properties for {compound} to {file_name}")

        export_properties_to_csv(file_name, data)

    if not export_mix:
        # Also export the initial mass fractions
        composition_file = os.path.join(path, f"composition_{fuel.name}.csv")
        print(f"\nWriting mass fractions for {fuel.name} to {composition_file}")
        composition_data = {
            "Index": range(len(fuel.compounds)),
            "Component": fuel.compounds,
            "Mass Fraction": fuel.initial_mass_fractions,
            "Mole Fraction": fl.fuel.mass_fraction_to_mole_fraction(
                fuel, fuel.initial_mass_fractions
            ),
            converter.labels["molecular_weight"]: fuel.molecular_weight.to(
                "kg/mol"
            ).magnitude
            * converter.mw,
        }
        export_properties_to_csv(composition_file, composition_data)


def main():
    """
    Main function to execute the export process.

    :param --fuel_name: Name of the fuel (mandatory).
    :type --fuel_name: str

    :param --fuel_data_dir: Directory where fuel data files are located.
    :type --fuel_data_dir: str, optional (default: FuelLib/fuelData)

    :param --units: Units for critical properties. Options are mks or cgs.
    :type --units: str, optional (default: mks)

    :param --temp_min: Minimum temperature (K) for the property calculations.
    :type --temp_min: float, optional (default: 0 K)

    :param --temp_max: Maximum temperature (K) for the property calculations.
    :type --temp_max: float, optional (default: 1000 K)

    :param --temp_step: Step size for temperature (K).
    :type --temp_step: float, optional (default: 10 K)

    :param --export_dir: Directory to export the properties.
    :type --export_dir: str, optional (default: current working directory)

    :param --export_mix: Whether to export individual component or mixture properties.
    :type --export_mix: bool, optional (default: False)

    :raises FileNotFoundError: If required files for the specified fuel are not found.
    """

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Export mixture fuel properties for Converge simulations."
    )

    # Mandatory argument for fuel name
    parser.add_argument(
        "-f",
        "--fuel_name",
        required=True,
        metavar="NAME",
        help="Name of the fuel (mandatory).",
    )

    # Optional argument for fuel data directory
    parser.add_argument(
        "-dir",
        "--fuel_data_dir",
        default=FUELDATA_DIR,
        metavar="PATH",
        help="Directory where fuel data files are located (optional, default: FuelLib/fuelData).",
    )

    # Optional argument for units
    # Default is 'mks', but can be set to 'cgs'
    parser.add_argument(
        "-u",
        "--units",
        default="mks",
        metavar="{mks,cgs}",
        help="Units for critical properties (optional, default: mks).",
    )

    # Optional argument for minimum temperature
    parser.add_argument(
        "-t",
        "--temp_min",
        type=float,
        default=0,
        metavar="K",
        help="Minimum temperature for property calculations (optional, default: 0).",
    )

    # Optional argument for maximum temperature
    parser.add_argument(
        "-T",
        "--temp_max",
        type=float,
        default=1000,
        metavar="K",
        help="Maximum temperature for property calculations (optional, default: 1000).",
    )

    # Optional argument for temperature step size
    parser.add_argument(
        "-s",
        "--temp_step",
        type=int,
        default=10,
        metavar="K",
        help="Step size for temperature (optional, default: 10).",
    )

    # Optional argument for export directory
    parser.add_argument(
        "-o",
        "--export_dir",
        default=os.getcwd(),
        metavar="PATH",
        help="Directory to export the properties (optional, default: current working directory).",
    )

    # Optional argument for exporting mixture properties
    parser.add_argument(
        "-m",
        "--export_mix",
        type=lambda x: str(x).lower() in ["true", "1"],
        default=False,
        metavar="{true,false}",
        help="Export mixture properties of the fuel (optional, default: false).",
    )

    # Parse arguments
    args = parser.parse_args()
    fuel_name = args.fuel_name
    fuel_data_dir = args.fuel_data_dir
    units = args.units.lower()
    temp_min = args.temp_min
    temp_max = args.temp_max
    temp_step = args.temp_step
    export_dir = args.export_dir
    export_mix = args.export_mix

    # Print the parsed arguments
    print("Preparing to export mixture properties:")
    print(f"    Fuel name: {fuel_name}")
    if export_mix:
        print("    Exporting mixture properties: True")
    print(f"    Units: {units}")
    print(f"    Minimum temperature: {temp_min} K")
    print(f"    Maximum temperature: {temp_max} K")
    print(f"    Temperature step size: {temp_step} K")
    print(f"    Export directory: {export_dir}")
    print(f"    Fuel data directory: {fuel_data_dir}")

    # Create the fuel object
    fuel = fl.Fuel.from_name(fuel_name, fuel_data_dir=fuel_data_dir)

    # Export properties for Converge
    export_converge(
        fuel,
        path=export_dir,
        units=units,
        temp_min=temp_min,
        temp_max=temp_max,
        temp_step=temp_step,
        export_mix=export_mix,
    )

    print("\nExport completed successfully!")


if __name__ == "__main__":
    main()
