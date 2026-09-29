#!/usr/bin/env python3
"""Generate updated baseline predictions for Fuel properties."""

import sys

from pathlib import Path

import pandas as pd
from fuellib import Fuel
from fuellib.utils import types

baseline_dir = Path(__file__).parent
if str(baseline_dir.parent) not in sys.path:
    sys.path.insert(0, str(baseline_dir.parent))

from get_pred_and_data import get_pred_and_data, get_pred_and_data_compound


def _prep_quantity(quantity: types.Quantity1D) -> list[str | float]:
    """Convert a Quantity1D to a list of float values with units in 0th index."""
    return [str(quantity.units)] + quantity.magnitude.tolist()


fuel_names = ["heptane", "decane", "dodecane", "posf10264", "posf10325", "posf10289"]
properties = {
    "Density": "g/cm^3",
    "Viscosity": "mm^2/s",
    "VaporPressure": "kPa",
    "SurfaceTension": "N/m",
    "ThermalConductivity": "W/m/K",
}

compound_fuel_name = "refCompounds"
compound_properties = {
    "Tb": "K",
    "Tm": "K",
    "omega": "dimensionless",
}


def generate_compound_baseline():
    """Generate updated baseline predictions for compound-specific properties.

    Predictions are included for every compound in the fuel mixture, even if
    there is no corresponding experimental data (in which case the error
    column will be NaN for that compound).
    """
    out_file = baseline_dir / f"{compound_fuel_name}.csv"
    fuel = Fuel(compound_fuel_name)

    df_combined = pd.DataFrame()
    for prop_name, prop_unit in compound_properties.items():
        compounds, data, pred = get_pred_and_data_compound(
            compound_fuel_name, prop_name
        )

        pred_conv = pred.to(prop_unit)
        error = abs(data.to(prop_unit) - pred_conv)

        df_prop = pd.DataFrame({
            "Compound": ["Units"] + compounds,
            prop_name: _prep_quantity(pred_conv),
            f"Error_{prop_name}": _prep_quantity(error),
        })

        if df_combined.empty:
            df_combined = df_prop
        else:
            df_combined = pd.merge(
                df_combined, df_prop, on="Compound", how="outer", sort=False
            )

    # Preserve the fuel's compound ordering, with the "Units" row first.
    compound_order = {c: i for i, c in enumerate(fuel.compounds)}
    sort_key = df_combined["Compound"].map(
        lambda c: -1 if c == "Units" else compound_order.get(c, len(compound_order))
    )
    df_combined = (
        df_combined
        .assign(_sort_key=sort_key)
        .sort_values("_sort_key")
        .drop(columns="_sort_key")
        .reset_index(drop=True)
    )

    df_combined.to_csv(out_file, index=False)


def main():
    """Generate updated baseline predictions for Fuel properties."""
    for fuel_name in fuel_names:
        out_file = baseline_dir / f"{fuel_name}.csv"

        df_combined = pd.DataFrame()
        for prop_name, prop_unit in properties.items():
            T, data, pred = get_pred_and_data(fuel_name, prop_name)

            df_prop = pd.DataFrame({
                "Temperature": _prep_quantity(T.to("celsius")),
                prop_name: _prep_quantity(pred.to(prop_unit)),
                f"Error_{prop_name}": _prep_quantity(
                    abs(data.to(prop_unit) - pred.to(prop_unit))
                ),
            })

            if df_combined.empty:
                df_combined = df_prop
            else:
                df_combined = pd.merge(
                    df_combined, df_prop, on="Temperature", how="outer"
                )

        is_numeric = pd.to_numeric(df_combined["Temperature"], errors="coerce").notna()
        units_row = df_combined[~is_numeric]
        data_rows = df_combined[is_numeric].sort_values(
            by="Temperature", key=lambda s: s.astype(float)
        )
        df_combined = pd.concat([units_row, data_rows], ignore_index=True)

        df_combined.to_csv(out_file, index=False)

    generate_compound_baseline()


if __name__ == "__main__":
    main()
