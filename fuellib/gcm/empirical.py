"""Empirical data for group contribution methods."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..rdk import mol
from ..utils.logger import logger
from ..utils.types import PintVector
from ..utils.units import PintUnits
from . import astm
from .core import GCMRegistry

if TYPE_CHECKING:
    from ..fuel import Fuel

YSI_TABLE = pd.read_csv(Path(__file__).with_suffix(".ysi.v2.csv"), header=0)
DCN_TABLE = pd.read_csv(Path(__file__).with_suffix(".dcn.unknown.csv"), header=0)
empirical_gcm = GCMRegistry.register("empirical", property_fns=[])


@empirical_gcm.register_property
def ysi(fuel: "Fuel") -> PintVector:
    """Extract empirical Yale Sooting Index (YSI) for the given fuel's components."""
    ysi_values = np.zeros(fuel.num_compounds, dtype=np.float64)
    for i, (name, (bin, _, _), rdk_mol) in enumerate(
        zip(fuel.compounds, fuel.gcxgc_bins, fuel.rdkit_mols, strict=True)
    ):
        inchi_str = mol.inchi(rdk_mol)
        row = YSI_TABLE.loc[YSI_TABLE["InChI"] == inchi_str]
        if row.empty:
            logger.info(
                "Attempting to match YSI entry for %s by Bin as fallback.", name
            )
            row = YSI_TABLE.loc[YSI_TABLE["Bin"] == bin]
            if row.empty:
                logger.info(
                    "No matching YSI entry found for %s with InChI %s and Bin %s",
                    name,
                    inchi_str,
                    bin,
                )
                continue
        ysi_values[i] = row["Unified YSI"].values[0]
    return PintUnits.Quantity(ysi_values, "dimensionless")


@empirical_gcm.register_property
def dcn(fuel: "Fuel") -> PintVector:
    """Extract empirical Derived Cetane Number (DCN) for the given fuel's components."""
    dcn_values = np.zeros(fuel.num_compounds, dtype=np.float64)
    for i, (name, (bin, _, _), rdk_mol) in enumerate(
        zip(fuel.compounds, fuel.gcxgc_bins, fuel.rdkit_mols, strict=True)
    ):
        inchi_str = mol.inchi(rdk_mol)
        row = DCN_TABLE.loc[DCN_TABLE["InChI"] == inchi_str]
        if row.empty:
            logger.info(
                "Attempting to match DCN entry for %s by Bin as fallback.", name
            )
            row = DCN_TABLE.loc[DCN_TABLE["Bin"] == bin]
            if row.empty:
                logger.info(
                    "No matching DCN entry found for %s with InChI %s and Bin %s",
                    name,
                    inchi_str,
                    bin,
                )
                continue
        dcn_values[i] = row["DCN"].values[0]
    return PintUnits.Quantity(dcn_values, "dimensionless")
