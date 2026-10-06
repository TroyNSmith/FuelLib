"""Unitless, backend-agnostic correlation kernels.

Kernels are pure functions of plain arrays in SI units with no `Fuel` object or
unit tracking. They are the single source of truth for each correlation: the
unit-aware `fuellib.correlate.components`/`fuellib.correlate.mixture` functions
strip units and delegate here, while downstream packages (e.g., inverse design)
can call them directly with JAX arrays inside `jax.jit`/`jax.grad`, passing
component properties that were computed once up front.

Only correlations needed for differentiable workflows live here; everything
else remains NumPy/Pint-only in the unit-aware modules.
"""

from . import mixture

__all__ = ["mixture"]
