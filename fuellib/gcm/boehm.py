"""Boehm (2022) group contribution parameters."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import pandas as pd

from ..rdk import mol
from ..utils import Units, types
from .core import GCMRegistry

if TYPE_CHECKING:
    from ..fuel import Fuel

TABLE = pd.read_csv(Path(__file__).with_suffix(".csv"), header=0, index_col=0)
boehm_gcm = GCMRegistry.register("boehm", property_fns=[])


def _identify_families(fuel: "Fuel") -> list[str]:
    """Identify the families of a fuel's components.

    =============== ==========================================
    Family          Description
    =============== ==========================================
    n-alkane        Straight-chain alkanes
    iso-alkane      Branched-chain alkanes
    alkene          Unsaturated hydrocarbons with double bonds
    monocyclic      Single-ring hydrocarbons
    dicyclic        Fused two-ring hydrocarbons
    tricyclic       Fused three-ring hydrocarbons
    alkylbenzene    Alkyl-substituted benzene derivatives
    cycloaromatic   Fused cyclic-aromatic hydrocarbons
    diaromatic      Fused two-ring aromatic hydrocarbons
    =============== ==========================================

    Args:
        fuel: The fuel object to identify the family for.

    Returns:
        The family names as a list of strings.

    Raises:
        ValueError: If the family cannot be identified.
    """
    families = []
    for m in fuel.rdkit_mols:
        aromatic = mol.has_aromatic(m)
        fused = mol.has_fused_rings(m)
        fused_two = mol.has_fused_rings(m, number_of_rings=2)
        fused_three = mol.has_fused_rings(m, number_of_rings=3)
        n_aromatic_rings = mol.count_aromatic_rings(m)

        if fused_two and n_aromatic_rings >= 2:
            family = "diaromatic"
        elif fused_two and n_aromatic_rings == 1:
            family = "cycloaromatic"
        elif aromatic and not fused:
            family = "alkylbenzene"
        elif fused_three and not aromatic:
            family = "tricyclic"
        elif fused_two and not aromatic:
            family = "dicyclic"
        elif fused:
            msg = f"Cannot identify family for molecule: {mol.smiles(m)}"
            raise ValueError(msg)
        elif mol.has_ring(m):
            family = "monocyclic"
        elif mol.has_double_bond(m):
            family = "alkene"
        elif mol.has_branch(m):
            family = "iso-alkane"
        else:
            family = "n-alkane"
        families.append(family)
    return families


@boehm_gcm.register_property
def dsFus(fuel: "Fuel") -> types.Quantity1D:
    """Return the fusion entropy (dSfus) for each component in the fuel.

    Args:
        fuel: Fuel object.

    Returns:
        The predicted fusion entropy (dSfus) values in J/(mol*K).

    Raises:
        ValueError: If fusion entropy data is not available for a family.
    """
    families = _identify_families(fuel)
    result = []
    for i in range(fuel.num_compounds):
        family = families[i]
        nC = fuel.nC[i]
        try:
            row = TABLE.loc[family]
            dsFus_i = float(row["dSfus_A"]) + float(row["dSfus_B"]) * (
                nC - int(row["C_ref"])
            )
            result.append(max(dsFus_i, 20.0))
        except KeyError:
            msg = f"No dSfus data available for family: {family}"
            raise ValueError(msg)
    return Units.Quantity(result, "J/(mol*K)")
