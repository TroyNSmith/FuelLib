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

import logging
import os
from pathlib import Path
from typing import ClassVar, Literal

import click
import pandas as pd
import quaxed.numpy as qnp
from unxt import Quantity

from fuellib.fuel import Fuel, mixing_rule
from fuellib.fuel.locator import DEFAULT_DATA_DIR
from fuellib.units import convert_temperature

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _warn_clipped_range(subject: str, low, high) -> None:
    """Log a banner-style warning that temperatures were clipped to a valid range."""
    msg = f"\n\033[1;31mWARNING\033[0m: Clipping temperatures for {subject} to the valid range [{low:.2f}, {high:.2f}]"
    logger.warning(msg)


class Units:
    """
    Per-quantity units and display labels for the 'mks' or 'cgs' system.

    Usage: ``u = Units("cgs"); u.unit("rho"); u.label("rho")``.
    """

    # key -> (display name, mks unit, cgs unit)
    _QUANTITIES: ClassVar[dict[str, tuple[str, str, str]]] = {
        "T": ("Temperature", "K", "K"),
        "Tc": ("Critical Temperature", "K", "K"),
        "mu": ("Viscosity", "Pa*s", "poise"),
        "st": ("Surface Tension", "N/m", "dyne/cm"),
        "lv": ("Heat of Vaporization", "J/kg", "erg/g"),
        "pv": ("Vapor Pressure", "Pa", "dyne/cm^2"),
        "rho": ("Density", "kg/m^3", "g/cm^3"),
        "cl": ("Specific Heat", "J/(kg*K)", "erg/(g*K)"),
        "tc": ("Thermal Conductivity", "W/(m*K)", "erg/(cm*s*K)"),
        "mw": ("Molecular Weight", "kg/mol", "g/mol"),
    }

    def __init__(self, system: Literal["cgs", "mks"] = "mks"):
        """
        Initialize the Units class with the specified unit system.

        :param system: Unit system ('cgs' or 'mks').
        :type system: str
        """
        self.system = system

    @property
    def system(self) -> str:
        """Get the current unit system."""
        return self._system

    @system.setter
    def system(self, value: Literal["cgs", "mks"]):
        """Set the unit system."""
        normalized = value.lower()
        if normalized not in ("mks", "cgs"):
            raise ValueError(
                f"Unsupported unit system: {value}. Supported systems: ('cgs', 'mks')"
            )
        self._system = normalized

    def unit(self, quantity: str) -> str:
        """Get the unit string for `quantity` in the current system."""
        _, mks_unit, cgs_unit = self._QUANTITIES[quantity]
        return mks_unit if self._system == "mks" else cgs_unit

    def label(self, quantity: str) -> str:
        """Get the display label (name + unit) for `quantity`."""
        name, *_ = self._QUANTITIES[quantity]
        return f"{name} ({self.unit(quantity)})"


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
    logger.info("\033[1;34mPreparing to export mixture properties:\033[0m")
    logger.info(f"\tFuel name: {fuel_name}")
    if export_mix:
        logger.info("\tExporting mixture properties: True")
    logger.info(f"\tUnits: {units}")
    logger.info(f"\tMinimum temperature: {temp_min} {temp_units}")
    logger.info(f"\tMaximum temperature: {temp_max} {temp_units}")
    logger.info(f"\tTemperature step size: {temp_step} {temp_units}")
    logger.info(f"\tExport directory: {export_dir}")
    logger.info(f"\tFuel data directory: {fuel_data_dir}")

    # Initialize variables
    T = Quantity(qnp.arange(temp_min, temp_max + temp_step, temp_step), temp_units)

    if qnp.any(convert_temperature(T, "K") < 0):
        raise ValueError(
            f"Error: Invalid value for `-t` / `--temp_min`: {temp_min} is not in the range T>=0 K."
        )
    if temp_min >= temp_max:
        raise ValueError("Minimum temperature must be less than maximum temperature.")

    export_dir = Path(export_dir) if export_dir is not None else Path.cwd()

    U = Units(units)

    fuel = Fuel(fuel_name, fuelDataDir=fuel_data_dir)

    # Component properties
    Tc = convert_temperature(fuel.Tc, temp_units)
    Tm = convert_temperature(fuel.Tm, temp_units)

    if export_mix:
        X = fuel.Y2X(fuel.Y_0)
        Tc_mixture = convert_temperature(mixing_rule(Tc, X), temp_units)
        Tf_mixture = convert_temperature(mixing_rule(Tm, X), temp_units)

        logger.info(f"\nEstimated mixture freezing temp: {Tf_mixture:.2f}")
        logger.info(f"Min freezing temp min(Tm_i): {min(Tm):.2f}")
        logger.info(f"Max freezing temp max(Tm_i): {max(Tm):.2f}")
        logger.info(f"Estimated mixture critical temp: {Tc_mixture:.2f}")
        logger.info(f"Min critical temp min(Tc_i): {min(Tc):.2f}")
        logger.info(f"Max critical temp max(Tc_i): {max(Tc):.2f}")

        # Compute all mixture properties over the full temperature range at once
        logger.info(
            f"\nCalculating mixture properties over {qnp.size(T)} temperatures "
            f"from {temp_min} to {temp_max} {temp_units}..."
        )
        rho = fuel.mixture_density(T, unit=U.unit("rho"))
        mu = fuel.mixture_dynamic_viscosity(T, unit=U.unit("mu"))
        pv = fuel.mixture_vapor_pressure(T, unit=U.unit("pv"))
        st = fuel.mixture_surface_tension(T, unit=U.unit("st"))
        tc = fuel.mixture_thermal_conductivity(T, unit=U.unit("tc"))
        lv = mixing_rule(fuel.latent_heat_vaporization(T, unit=U.unit("lv")), X)
        cl = mixing_rule(fuel.Cl(T, unit=U.unit("cl")), X)

        val = (T.value >= Tf_mixture.value) & (T.value <= Tc_mixture.value)
        if qnp.size(T[val]) != qnp.size(T):
            _warn_clipped_range("the mixture", Tf_mixture.value, Tc_mixture.value)

        df = pd.DataFrame(
            {
                U.label("T"): T[val].value,
                U.label("Tc"): qnp.zeros_like(T[val].value) + Tc_mixture.value,
                U.label("mu"): mu[val].value,
                U.label("st"): st[val].value,
                U.label("lv"): lv[val].value,
                U.label("pv"): pv[val].value,
                U.label("rho"): rho[val].value,
                U.label("cl"): cl[val].value,
                U.label("tc"): tc[val].value,
            },
        )
        export_dir.mkdir(parents=True, exist_ok=True)
        file_name = export_dir / f"mixturePropsGCM_{fuel_name}.csv"
        logger.info(f"\nWriting mixture properties to {file_name}")
        df.to_csv(file_name, index=False)

    else:
        component_dir = export_dir / fuel_name
        component_dir.mkdir(parents=True, exist_ok=True)

        # Compute all component properties over the full temperature range at once
        logger.info(
            f"\nCalculating properties for {len(fuel.compounds)} compounds over "
            f"{qnp.size(T)} temperatures from {temp_min} to {temp_max} {temp_units}..."
        )
        rho = fuel.density(T, unit=U.unit("rho"))
        mu = fuel.viscosity_dynamic(T, unit=U.unit("mu"))
        pv = fuel.psat(T, unit=U.unit("pv"))
        st = fuel.surface_tension(T, unit=U.unit("st"))
        tc = fuel.thermal_conductivity(T, unit=U.unit("tc"))
        lv = fuel.latent_heat_vaporization(T, unit=U.unit("lv"))
        cl = fuel.Cl(T, unit=U.unit("cl"))

        for i, comp in enumerate(fuel.compounds):
            val = (T.value >= Tm[i].value) & (T.value <= Tc[i].value)  # Valid temps
            if qnp.size(T[val]) != qnp.size(T):
                _warn_clipped_range(f"component {comp}", Tm[i].value, Tc[i].value)
            df = pd.DataFrame(
                {
                    U.label("T"): T[val].value,
                    U.label("Tc"): qnp.zeros_like(T[val].value) + Tc[i].value,
                    U.label("mu"): mu[val, i].value,
                    U.label("st"): st[val, i].value,
                    U.label("lv"): lv[val, i].value,
                    U.label("pv"): pv[val, i].value,
                    U.label("rho"): rho[val, i].value,
                    U.label("cl"): cl[val, i].value,
                    U.label("tc"): tc[val, i].value,
                },
            )
            file_name = component_dir / f"{i}_{comp}.csv"
            logger.info(f"\nWriting properties for {comp} to {file_name}")
            df.to_csv(file_name, index=False)

        # Composition (mass/mole fractions and molecular weight) for the mixture
        composition = pd.DataFrame(
            {
                "Index": range(len(fuel.compounds)),
                "Component": fuel.compounds,
                "Mass Fraction": fuel.Y_0,
                "Mole Fraction": fuel.Y2X(fuel.Y_0),
                U.label("mw"): fuel.MW.to(U.unit("mw")).value,
            }
        )
        composition_file = component_dir / f"composition_{fuel_name}.csv"
        logger.info(f"\nWriting mass fractions for {fuel_name} to {composition_file}")
        composition.to_csv(composition_file, index=False)

    logger.info("\n\033[92mExport completed successfully!\033[0m\n")


if __name__ == "__main__":
    export_converge()
