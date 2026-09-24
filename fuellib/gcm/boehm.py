"""Boehm et al. (2022) method for freezing point calculation of ideal solutions."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..utils.types import PintVector
from ..utils.units import PintUnits
from . import astm
from .core import GCMRegistry

if TYPE_CHECKING:
    from ..fuel import Fuel

FUSION_TABLE = pd.read_csv(
    Path(__file__).with_suffix(".fusion.csv"), header=0, index_col=0
)
boehm_gcm = GCMRegistry.register("boehm", property_fns=[])


@boehm_gcm.register_property
def dS_fus(fuel: "Fuel") -> PintVector:
    """Predict the entropy of fusion (dS_fus) for the given fuel's components."""
    ds_fus = np.full(fuel.num_compounds, 56.5)
    for i, (_, c_num, family) in enumerate(fuel.gcxgc_bins):
        if family not in FUSION_TABLE.index:
            continue

        coeff: float = FUSION_TABLE.loc[family, "dSfus_A"]
        slope: float = FUSION_TABLE.loc[family, "dSfus_B"]
        ref_c_num: int = FUSION_TABLE.loc[family, "C_ref"]
        ds_fus[i] = max(coeff + slope * (c_num - ref_c_num), 20.0)

    return PintUnits.Quantity(ds_fus, "J/(mol*K)")


@boehm_gcm.register_property
def dH_fus(fuel: "Fuel") -> PintVector:
    """Predict the enthalpy of fusion (dH_fus) for the given fuel's components."""
    return (dS_fus(fuel) * astm.Tm(fuel)).to("J/mol")
