"""Answers for the Group Contribution Method (GCM) tests in FuelLib."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import quaxed.numpy as qnp

FILE_DIR = Path(__file__).parent


@dataclass
class Gani:
    """Jet-A predictions using the Gani Group Contribution Method (GCM) in FuelLib."""

    _csv_data = pd.read_csv(FILE_DIR / "gcm-gani-jet_a.csv")

    Tc = qnp.array(_csv_data["Tc"].values)
    Pc = qnp.array(_csv_data["Pc"].values)
    Vc = qnp.array(_csv_data["Vc"].values)
    Tb = qnp.array(_csv_data["Tb"].values)
    Tm = qnp.array(_csv_data["Tm"].values)
    Hf = qnp.array(_csv_data["Hf"].values)
    Gf = qnp.array(_csv_data["Gf"].values)
    Hv_stp = qnp.array(_csv_data["Hv_stp"].values)
    Cp_stp = qnp.array(_csv_data["Cp_stp"].values)
    Cp_B = qnp.array(_csv_data["Cp_B"].values)
    Cp_C = qnp.array(_csv_data["Cp_C"].values)
    Vm_stp = qnp.array(_csv_data["Vm_stp"].values)
    omega = qnp.array(_csv_data["omega"].values)
