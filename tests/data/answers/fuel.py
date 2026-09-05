"""Answers for the FuelLib Fuel class tests."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FILE_DIR = Path(__file__).parent


@dataclass
class JetAProperties:
    """Answers for the JetA fuel class tests."""

    _csv_data = pd.read_csv(FILE_DIR / "fuel-props-jet_a.csv")

    formulas = _csv_data["Formulas"].tolist()
    MW = _csv_data["MW"].to_numpy()
    num_carbons = _csv_data["num_carbons"].to_numpy()
    num_hydrogens = _csv_data["num_hydrogens"].to_numpy()
    hydrocarbon = _csv_data["hydrocarbon"].to_numpy()
    aromatic = _csv_data["aromatic"].to_numpy()
    cyclic = _csv_data["cyclic"].to_numpy()
    branched = _csv_data["branched"].to_numpy()
    alkene = _csv_data["alkene"].to_numpy()
    hydrocarbon_types = _csv_data["hydrocarbon_types"].tolist()
    Lv_stp = _csv_data["Lv_stp"].to_numpy()
    sigma = _csv_data["sigma"].to_numpy()
    epsilonByKB = _csv_data["epsilonByKB"].to_numpy()
