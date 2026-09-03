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

import os
from pathlib import Path
from typing import Literal, LiteralString

import click
import quaxed.numpy as qnp
from jax import Array
from unxt import AbstractQuantity, Quantity

from fuellib.fuel import Fuel, mixing_rule
from fuellib.fuel.locator import DEFAULT_DATA_DIR
from fuellib.units import convert_temperature


class Units:
    """Default units and labels for supported unit systems."""

    _supported_systems = ("cgs", "mks")

    def __init__(self, system: Literal["cgs", "mks"] = "mks"):
        """
        Initialize the Units class with the specified unit system.

        :param system: Unit system ('cgs' or 'mks').
        :type system: str
        """
        self._system: LiteralString = system.lower()

        self._mks_units = {
            "temperature": "K",
            "critical_temp": "K",
            "viscosity": "Poise",
            "surface_tension": "dyne/cm",
            "heat_vaporization": "erg/g",
            "vapor_pressure": "dyne/cm^2",
            "density": "g/cm^3",
            "specific_heat": "erg/g/K",
            "thermal_conductivity": "erg/cm/s/K",
            "molecular_weight": "g/mol",
        }

        self._cgs_units = {
            "temperature": "K",
            "critical_temp": "K",
            "viscosity": "Pa*s",
            "surface_tension": "N/m",
            "heat_vaporization": "J/kg",
            "vapor_pressure": "Pa",
            "density": "kg/m^3",
            "specific_heat": "J/kg/K",
            "thermal_conductivity": "W/m/K",
            "molecular_weight": "kg/mol",
        }

    @property
    def system(self) -> LiteralString:
        """Get the current unit system."""
        return self._system

    @system.setter
    def system(self, value: Literal["cgs", "mks"]):
        """Set the unit system."""
        if value.lower() not in self._supported_systems:
            raise ValueError(
                f"Unsupported unit system: {value}. Supported systems: {self._supported_systems}"
            )
        self._system = value.lower()

    @property
    def units(self) -> dict[str, str]:
        """Get the unit labels for the current unit system."""
        if self._system == "cgs":
            return self._cgs_units
        elif self._system == "mks":
            return self._mks_units
        else:
            raise ValueError(
                f"Unsupported unit system: {self._system}. Supported systems: {self._supported_systems}"
            )

    @property
    def labels(self) -> dict[str, str]:
        """Return a dictionary of unit labels for the current unit system."""
        units = self.units
        return {
            "temperature": f"Temperature ({units['temperature']})",
            "critical_temp": f"Critical Temperature ({units['critical_temp']})",
            "viscosity": f"Viscosity ({units['viscosity']})",
            "surface_tension": f"Surface Tension ({units['surface_tension']})",
            "heat_vaporization": f"Heat of Vaporization ({units['heat_vaporization']})",
            "vapor_pressure": f"Vapor Pressure ({units['vapor_pressure']})",
            "density": f"Density ({units['density']})",
            "specific_heat": f"Specific Heat ({units['specific_heat']})",
            "thermal_conductivity": f"Thermal Conductivity ({units['thermal_conductivity']})",
            "molecular_weight": f"Molecular Weight ({units['molecular_weight']})",
        }


def _validate_temperature_range(
    T_array: AbstractQuantity,
    T_freeze: AbstractQuantity,
    T_crit: AbstractQuantity,
    is_mixture: bool,
):
    """
    Validate and adjust temperature range based on freezing and critical temperatures.

    :param T_array: Array of temperature values.
    :type T_array: AbstractQuantity
    :param T_freeze: Freezing temperature.
    :type T_freeze: AbstractQuantity
    :param T_crit: Critical temperature.
    :type T_crit: AbstractQuantity
    :param is_mixture: Whether this is for mixture properties.
    :type is_mixture: bool
    :return: Tuple of (T_min_allowed, T_max_allowed, adjusted_T_array).
    :rtype: tuple
    """
    if is_mixture:
        T_max = T_array[T_array <= T_crit].max()
        raise ValueError(T_max)
    abs_freeze_diff = qnp.abs(T_freeze[:, qnp.newaxis] - T_array)
    raise ValueError(abs_freeze_diff)
    T_min_allowed = T_freeze[abs_freeze_diff.argmin(axis=1)]

    raise ValueError(T_min_allowed)

    T_max_allowed = T_crit

    # Handle minimum temperature warnings
    if False:
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
    if False:
        T_max_allowed = min(Tc_K)
        if jnp.any(T_array > T_max_allowed):
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


@click.command()
@click.option(
    "fuel_name",
    "-f",
    "--fuel",
    required=True,
    type=str,
    help="Name of the fuel to export.",
)
@click.option(
    "fuel_data_dir",
    "-dir",
    "--fuel_data_dir",
    default=DEFAULT_DATA_DIR,
    required=False,
    type=click.Path(),
    help="Path to the fuel data directory. (default: fuellib/data/fuelData)",
)
@click.option(
    "units",
    "-u",
    "--units",
    default="mks",
    type=click.Choice(["cgs", "mks"], case_sensitive=False),
    help="Unit system for the exported data (default: mks).",
)
@click.option(
    "temp_min",
    "-t",
    "--temp_min",
    required=False,
    default=0,
    type=float,
    help="Minimum temperature for the export (default: 0 K).",
)
@click.option(
    "temp_max",
    "-T",
    "--temp_max",
    required=False,
    default=1000,
    type=float,
    help="Maximum temperature for the export (default: 1000 K).",
)
@click.option(
    "temp_step",
    "-s",
    "--temp_step",
    type=click.FloatRange(min=0.01),
    default=10,
    help="Temperature step size for the export (default: 10 K).",
)
@click.option(
    "temp_units",
    "-U",
    "--temp_units",
    default="K",
    type=click.Choice(["K", "Celsius", "Fahrenheit"], case_sensitive=False),
    help="Input temperature units (default: K).",
)
@click.option(
    "export_dir",
    "-o",
    "--export_dir",
    default=os.getcwd(),
    type=click.Path(),
    help="Directory to export the CSV file (default: current working directory).",
)
@click.option(
    "export_mix",
    "-m",
    "--export_mix",
    is_flag=True,
    default=False,
    type=bool,
    help="Whether to export the fuel mix (default: False).",
)
def export_converge(
    fuel_name: str,
    fuel_data_dir: str | Path = DEFAULT_DATA_DIR,
    units: Literal["cgs", "mks"] = "mks",
    temp_min: float = 0,
    temp_max: float = 1000,
    temp_step: float = 10,
    temp_units: Literal["K", "Celsius", "Fahrenheit"] = "K",
    export_mix: bool = False,
    export_dir: str | Path | None = None,
) -> None:
    """
    Export mixture properties for a given fuel to a CSV file formatted for Converge.
    """
    # Initialize variables
    T = Quantity(qnp.arange(temp_min, temp_max + temp_step, temp_step), temp_units)

    if qnp.any(convert_temperature(T, "K") < 0):
        raise ValueError(
            f"Error: Invalid value for `-t` / `--temp_min`: {temp_min} is not in the range T>=0 K."
        )
    if temp_min >= temp_max:
        raise ValueError("Minimum temperature must be less than maximum temperature.")

    export_dir = Path(export_dir) if export_dir is not None else Path.cwd()

    if export_mix:
        export_dir = Path(export_dir) / "mix"

    U = Units(units)

    fuel = Fuel(fuel_name, fuelDataDir=fuel_data_dir)

    # Component properties
    Tc = convert_temperature(fuel.Tc, temp_units)
    Tm = convert_temperature(fuel.Tm, temp_units)

    if export_mix:
        Tc_mixture = convert_temperature(
            mixing_rule(Tc, fuel.Y2X(fuel.Y_0)), temp_units
        )
        Tf_mixture = convert_temperature(
            mixing_rule(Tm, fuel.Y2X(fuel.Y_0)), temp_units
        )

        print(f"\nEstimated mixture freezing temp: {Tf_mixture:.2f}")
        print(f"Min freezing temp min(Tm_i): {min(Tm):.2f}")
        print(f"Max freezing temp max(Tm_i): {max(Tm):.2f}")
        print(f"Estimated mixture critical temp: {Tc_mixture:.2f}")
        print(f"Min critical temp min(Tc_i): {min(Tc):.2f}")
        print(f"Max critical temp max(Tc_i): {max(Tc):.2f}")
        print()


if __name__ == "__main__":
    export_converge()
