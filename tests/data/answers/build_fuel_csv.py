"""Build a CSV file for the FuelLib Fuel class tests from current codebase."""

from pathlib import Path

import pandas as pd

from fuellib.fuel import Fuel
from fuellib.utils.units import strip

FILE_DIR = Path(__file__).parent
FUEL_DIR = FILE_DIR.parent

name = "jet-a"
fuel = Fuel(name, decompName="refCompounds", fuelDataDir=FUEL_DIR)

csv_file = FILE_DIR / f"fuel-props-{name.replace('-', '_')}.csv"

df = pd.DataFrame(
    {
        "Formulas": fuel.formulas,
        "MW": strip(fuel.MW),
        "num_carbons": fuel.num_carbons,
        "num_hydrogens": fuel.num_hydrogens,
        "hydrocarbon": fuel._hydrocarbon,
        "aromatic": fuel._aromatic,
        "cyclic": fuel._cyclic,
        "branched": fuel._branched,
        "alkene": fuel._alkene,
        "hydrocarbon_types": fuel.hydrocarbon_types,
        "Lv_stp": strip(fuel.Lv_stp),
        "sigma": strip(fuel.sigma),
        "epsilonByKB": strip(fuel.epsilonByKB),
    }
)
df.to_csv(csv_file, index=False)
