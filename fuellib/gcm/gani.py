"""Constantinou-Gani Group Contribution Method."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import pandas as pd

from ..utils.types import FloatVector, IntMatrix, PintVector
from ..utils.units import PintUnits
from .core import GCMRegistry

if TYPE_CHECKING:
    from ..fuel import Fuel

TABLE = pd.read_csv(Path(__file__).with_suffix(".csv"), header=0, index_col=0)
gani_gcm = GCMRegistry.register("gani", property_fns=[])


def _get_row(property_name: str) -> FloatVector:
    """Get property row from GCM table."""
    row = TABLE.loc[property_name]
    if row.empty:
        msg = f"Property '{property_name}' not found in GCM table."
        raise KeyError(msg)
    return row.to_numpy().flatten()


def _get_decomp(fuel: "Fuel") -> IntMatrix:
    """Organize the decomposition matrix for the given fuel and convert it to a numpy array."""
    decomp = fuel.gani_decomp.reindex(columns=TABLE.columns)
    return decomp.to_numpy()


def _check_compatible_dims(
    cj: Annotated[FloatVector, "GCM contribution vector"],
    ij: Annotated[IntMatrix, "Group decomposition matrix"],
) -> None:
    """Check if the dimensions of the GCM contribution vector and the group decomposition matrix are compatible."""
    if ij.shape[1] != cj.shape[0]:
        msg = (
            f"Incompatible dimensions: GCM contribution vector has length {cj.shape[0]}, "
            f"but group decomposition matrix has {ij.shape[1]} columns."
        )
        raise ValueError(msg)


@gani_gcm.register_property
def Tc(fuel: "Fuel") -> PintVector:
    """Predict the critical temperature (Tc) for the given fuel's components."""
    nij = _get_decomp(fuel)
    tcj = _get_row("tck")
    _check_compatible_dims(tcj, nij)
    tci: FloatVector = 181.128 * np.log(np.matmul(nij, tcj))
    return PintUnits.Quantity(tci, "K")


@gani_gcm.register_property
def Pc(fuel: "Fuel") -> PintVector:
    """Predict the critical pressure (Pc) for the given fuel's components."""
    nij = _get_decomp(fuel)
    pcj = _get_row("pck")
    _check_compatible_dims(pcj, nij)
    pci: FloatVector = 1.3705 + np.power(np.matmul(nij, pcj) + 0.10022, -2.0)
    return PintUnits.Quantity(pci, "bar")


@gani_gcm.register_property
def Vc(fuel: "Fuel") -> PintVector:
    """Predict the critical volume (Vc) for the given fuel's components."""
    nij = _get_decomp(fuel)
    vcj = _get_row("vck")
    _check_compatible_dims(vcj, nij)
    vci: FloatVector = -0.00435 + (np.matmul(nij, vcj))
    return PintUnits.Quantity(vci, "m^3/kmol")


@gani_gcm.register_property
def Tb(fuel: "Fuel") -> PintVector:
    """Predict the normal boiling temperature (Tb) for the given fuel's components."""
    nij = _get_decomp(fuel)
    tbj = _get_row("tbk")
    _check_compatible_dims(tbj, nij)
    tbi: FloatVector = 204.359 * np.log(np.matmul(nij, tbj))
    return PintUnits.Quantity(tbi, "K")


@gani_gcm.register_property
def Tm(fuel: "Fuel") -> PintVector:
    """Predict the melting temperature (Tm) for the given fuel's components."""
    nij = _get_decomp(fuel)
    tmj = _get_row("tmk")
    _check_compatible_dims(tmj, nij)
    tmi: FloatVector = 102.425 * np.log(np.matmul(nij, tmj))
    return PintUnits.Quantity(tmi, "K")


@gani_gcm.register_property
def Hf(fuel: "Fuel") -> PintVector:
    """Predict the standard enthalpy of formation (Hf) for the given fuel's components."""
    nij = _get_decomp(fuel)
    hfj = _get_row("hfk")
    _check_compatible_dims(hfj, nij)
    hfi: FloatVector = 10.835 + np.matmul(nij, hfj)
    return PintUnits.Quantity(hfi, "kJ/mol")


@gani_gcm.register_property
def Gf(fuel: "Fuel") -> PintVector:
    """Predict the standard Gibbs free energy of formation (Gf) for the given fuel's components."""
    nij = _get_decomp(fuel)
    gfj = _get_row("gfk")
    _check_compatible_dims(gfj, nij)
    gfi: FloatVector = -14.828 + np.matmul(nij, gfj)
    return PintUnits.Quantity(gfi, "kJ/mol")


@gani_gcm.register_property
def Hv_stp(fuel: "Fuel") -> PintVector:
    """Predict the standard enthalpy of vaporization at standard temperature and pressure (Hv_stp) for the given fuel's components."""
    nij = _get_decomp(fuel)
    hvj = _get_row("hvk")
    _check_compatible_dims(hvj, nij)
    hvi: FloatVector = 6.829 + (np.matmul(nij, hvj))
    return PintUnits.Quantity(hvi, "kJ/mol")


@gani_gcm.register_property
def omega(fuel: "Fuel") -> PintVector:
    """Predict the acentric factor (omega) for the given fuel's components."""
    nij = _get_decomp(fuel)
    wj = _get_row("wk")
    _check_compatible_dims(wj, nij)
    wi: FloatVector = 0.4085 * np.power(
        np.log(np.matmul(nij, wj) + 1.1507), (1.0 / 0.5050)
    )
    return PintUnits.Quantity(wi, "dimensionless")


@gani_gcm.register_property
def Vm_stp(fuel: "Fuel") -> PintVector:
    """Predict the standard molar volume at standard temperature and pressure (Vm_stp) for the given fuel's components."""
    nij = _get_decomp(fuel)
    vmj = _get_row("vmk")
    _check_compatible_dims(vmj, nij)
    vmi: FloatVector = 0.01211 + np.matmul(nij, vmj)
    return PintUnits.Quantity(vmi, "m^3/kmol")


@gani_gcm.register_property
def Cp_stp(fuel: "Fuel") -> PintVector:
    """Predict the standard molar heat capacity at standard temperature and pressure (Cp_stp) for the given fuel's components."""
    nij = _get_decomp(fuel)
    cpj = _get_row("CpAk")
    _check_compatible_dims(cpj, nij)
    cpi: FloatVector = np.matmul(nij, cpj) - 19.7779
    return PintUnits.Quantity(cpi, "J/(mol*K)")


@gani_gcm.register_property
def Cp_B(fuel: "Fuel") -> PintVector:
    """Predict the molar heat capacity at the boiling point (Cp_B) for the given fuel's components."""
    nij = _get_decomp(fuel)
    cpbj = _get_row("CpBk")
    _check_compatible_dims(cpbj, nij)
    cpbi: FloatVector = np.matmul(nij, cpbj)
    return PintUnits.Quantity(cpbi, "J/(mol*K)")


@gani_gcm.register_property
def Cp_C(fuel: "Fuel") -> PintVector:
    """Predict the molar heat capacity at the critical point (Cp_C) for the given fuel's components."""
    nij = _get_decomp(fuel)
    cpcj = _get_row("CpCk")
    _check_compatible_dims(cpcj, nij)
    cpci: FloatVector = np.matmul(nij, cpcj)
    return PintUnits.Quantity(cpci, "J/(mol*K)")
