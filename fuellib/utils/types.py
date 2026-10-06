"""Type definitions for FuelLib."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pint
from numpy.typing import ArrayLike

# NumPy
Array1D: TypeAlias = np.ndarray[tuple[int,]]
"""One-dimensional NumPy array."""
Array2D: TypeAlias = np.ndarray[tuple[int, int]]
"""Two-dimensional NumPy array."""

# Pint
Units = pint.UnitRegistry()
"""Shared Pint unit registry used for all FuelLib quantities."""

Quantity0D: TypeAlias = pint.Quantity[float]
"""Unit-aware scalar quantity with a float/zero-dimensional array magnitude."""
Quantity1D: TypeAlias = pint.Quantity[Array1D]
"""Unit-aware quantity with a one-dimensional array magnitude."""
Quantity2D: TypeAlias = pint.Quantity[Array2D]
"""Unit-aware quantity with a two-dimensional array magnitude."""


__all__ = [
    "Array1D",
    "Array2D",
    "ArrayLike",
    "Quantity0D",
    "Quantity1D",
    "Quantity2D",
    "Units",
]
