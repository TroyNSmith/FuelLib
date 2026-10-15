"""Nannoolal group contribution parameters."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import pandas as pd

from ..utils import Units, types
from .core import GCMRegistry

if TYPE_CHECKING:
    from ..fuel import Fuel

TABLE = pd.read_csv(Path(__file__).with_suffix(".csv"), header=0, index_col=0)
nannoolal_gcm = GCMRegistry.register("nannoolal", property_fns=[])


def _get_row(property_name: str) -> types.Array1D:
    """Get property row from GCM table.

    Args:
        property_name: Name of the property to retrieve from the GCM table.

    Returns:
        The property row as a 1D numpy array.

    Raises:
        KeyError: If the property is not found in the GCM table.
    """
    row = TABLE.loc[property_name]
    if row.empty:
        msg = f"Property '{property_name}' not found in GCM table."
        raise KeyError(msg)
    return row.to_numpy().flatten()


def _get_decomp(fuel: "Fuel") -> types.Array2D:
    """Organize the decomposition matrix for the fuel and convert it to a numpy array.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The decomposition matrix as a 2D numpy array.
    """
    decomp = fuel.nannoolalDecomp.reindex(columns=TABLE.columns, fill_value=0)
    return decomp.to_numpy()


def _check_compatible_dims(
    xj: Annotated[types.Array1D, "GCM property vector"],
    Aij: Annotated[types.Array2D, "Group decomposition matrix"],
) -> None:
    """Check if the shape of the GCM properties and group decomposition matrix match.

    Args:
        xj: The GCM property vector.
        Aij: The group decomposition matrix.

    Raises:
        ValueError: If the dimensions of the GCM contribution vector and the group
            decomposition matrix are incompatible.
    """
    if Aij.shape[1] != xj.shape[0]:
        msg = (
            f"Incompatible dimensions: GCM contribution vector has length {xj.shape[0]}"
            f", but group decomposition matrix has {Aij.shape[1]} columns."
        )
        raise ValueError(msg)


@nannoolal_gcm.register_property
def dBv(fuel: "Fuel") -> types.Quantity1D:
    """Get the saturated liquid viscosity curve slope for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The dimensionless group contribution sums for the saturated liquid
        viscosity curve slope (dBv) of the fuel's components.
        Converting these sums into dBv requires Nannoolal Eq. (7).
    """
    nij = _get_decomp(fuel)
    dBvj = _get_row("dBv")
    _check_compatible_dims(dBvj, nij)
    # Only absent groups are zeroed; NaN propagates for used, unsupported groups
    dBvi = np.where(nij != 0, nij * dBvj, 0.0).sum(axis=1)
    return Units.Quantity(dBvi, "")


@nannoolal_gcm.register_property
def Tv(fuel: "Fuel") -> types.Quantity1D:
    """Get the saturated liquid viscosity reference temperature for a fuel's components.

    Args:
        fuel: The fuel object containing the decomposition matrix.

    Returns:
        The dimensionless group contribution sums for the saturated liquid
        viscosity reference temperature (Tv) of the fuel's components.
        Converting these sums into Tv in K requires Nannoolal Eq. (8).
    """
    nij = _get_decomp(fuel)
    Tvj = _get_row("Tv")
    _check_compatible_dims(Tvj, nij)
    Tvi = np.where(nij != 0, nij * Tvj, 0.0).sum(axis=1)
    return Units.Quantity(Tvi, "")
