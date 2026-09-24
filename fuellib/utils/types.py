"""Type definitions for FuelLib."""

import numpy as np
import pint

# Aliases for readability
NumCompounds = int

# NumPy
IntVector = np.ndarray[tuple[NumCompounds,], np.dtype[np.int64]]
IntMatrix = np.ndarray[tuple[NumCompounds, int], np.dtype[np.int64]]
FloatVector = np.ndarray[tuple[NumCompounds,], np.dtype[np.float64]]
FloatMatrix = np.ndarray[tuple[NumCompounds, int], np.dtype[np.float64]]
StrVector = np.ndarray[tuple[NumCompounds,], np.dtype[np.str_]]

# Pint
PintScalar = pint.Quantity[float]
PintVector = pint.Quantity[FloatVector]
PintMatrix = pint.Quantity[FloatMatrix]
