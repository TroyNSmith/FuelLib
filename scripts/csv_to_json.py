"""Convert legacy per-fuel CSV/YAML data into the self-contained per-fuel JSON schema.

Merges gcData/<name>_init.csv (compound list, weight fractions, optional formula/
PelePhysics key), groupDecompositionData/<decomp_name>.csv (functional-group counts,
matched to the gcData rows positionally), and propertiesData/<name>.csv (optional
experimental validation data) into a single "<name>.json" file consumable by
``fuellib.Fuel.from_json`` / ``fuellib.Fuel.from_name``.

Usage:
    python scripts/csv_to_json.py [--fuel-data-dir PATH] [--force] [fuel_name ...]

With no fuel names given, converts every fuel listed in fuel_metadata.yaml. Existing
"<name>.json" files are skipped unless --force is passed.
"""

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import yaml


def _read_gc_data(gc_dir: Path, name: str) -> pd.DataFrame:
    """Read the gcData/<name>_init.csv compound table."""
    df = pd.read_csv(gc_dir / f"{name}_init.csv")
    df.columns = [c.strip() for c in df.columns]
    return df


def _read_decomposition(decomp_dir: Path, decomp_name: str) -> pd.DataFrame:
    """Read the groupDecompositionData/<decomp_name>.csv functional-group matrix."""
    return pd.read_csv(decomp_dir / f"{decomp_name}.csv")


def _build_components(df_gc: pd.DataFrame, df_decomp: pd.DataFrame) -> dict:
    """Build the "components" JSON block by matching gcData and decomposition rows positionally."""
    if len(df_gc) != len(df_decomp):
        raise ValueError(
            f"Row count mismatch: gcData has {len(df_gc)} compounds, "
            f"decomposition file has {len(df_decomp)} rows."
        )

    group_cols = list(df_decomp.columns[1:])
    components = {}
    for i in range(len(df_gc)):
        gc_row = df_gc.iloc[i]
        compound = str(gc_row["Compound"]).strip()
        if compound in components:
            raise ValueError(f"Duplicate compound name '{compound}' in gcData table.")

        entry = {"weight_percent": float(gc_row["Weight %"])}

        if "Formula" in df_gc.columns and pd.notna(gc_row["Formula"]):
            entry["formula"] = str(gc_row["Formula"]).strip()
        if "PelePhysics Key" in df_gc.columns and pd.notna(gc_row["PelePhysics Key"]):
            entry["pelephysics_key"] = str(gc_row["PelePhysics Key"]).strip()

        decomp_row = df_decomp.iloc[i]
        entry["decomposition"] = {
            g: int(decomp_row[g])
            for g in group_cols
            if pd.notna(decomp_row[g]) and float(decomp_row[g]) != 0
        }
        components[compound] = entry

    return components


def _build_properties(props_file: Path) -> dict | None:
    """Build the "properties" JSON block from a wide-format propertiesData/<name>.csv file."""
    if not props_file.exists():
        return None

    with open(props_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        names = next(reader)
        units = next(reader)

    df = pd.read_csv(props_file, skiprows=[1])
    temperature_unit = units[0]

    properties = {"temperature_unit": temperature_unit}
    for col, unit in zip(names[1:], units[1:]):
        sub = df[["Temperature", col]].dropna()
        if sub.empty:
            continue
        properties[col] = {"unit": unit, "data": sub.to_numpy().tolist()}

    return properties


def convert_fuel(
    fuel_data_dir: Path, name: str, decomp_name: str, metadata: dict | None = None
) -> dict:
    """Build the full per-fuel JSON payload for a single fuel."""
    df_gc = _read_gc_data(fuel_data_dir / "gcData", name)
    df_decomp = _read_decomposition(
        fuel_data_dir / "groupDecompositionData", decomp_name
    )

    payload = {"components": _build_components(df_gc, df_decomp)}

    properties = _build_properties(fuel_data_dir / "propertiesData" / f"{name}.csv")
    if properties is not None:
        payload["properties"] = properties

    if metadata:
        payload["metadata"] = metadata

    return payload


def main():
    """Entry point for the CSV/YAML -> JSON fuel data converter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fuel_names",
        nargs="*",
        help="Fuel names to convert (default: every fuel in fuel_metadata.yaml).",
    )
    parser.add_argument(
        "--fuel-data-dir",
        default=Path(__file__).resolve().parents[1] / "fuellib" / "data" / "fuel",
        type=Path,
        help="Fuel data directory (default: embedded fuellib/data/fuel).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing <name>.json files (default: skip them).",
    )
    args = parser.parse_args()

    metadata = yaml.safe_load((args.fuel_data_dir / "fuel_metadata.yaml").read_text())[
        "fuels"
    ]

    fuel_names = args.fuel_names or sorted(metadata.keys())

    for name in fuel_names:
        out_file = args.fuel_data_dir / f"{name}.json"
        if out_file.exists() and not args.force:
            print(f"skip  {name} (already exists, use --force to overwrite)")
            continue

        fuel_metadata = metadata.get(name, {})
        decomp_name = fuel_metadata.get("decomp_name", name)
        payload = convert_fuel(
            args.fuel_data_dir, name, decomp_name, metadata=fuel_metadata
        )
        out_file.write_text(json.dumps(payload, indent=4))
        print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
