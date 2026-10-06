"""Array backend dispatch for unitless correlation kernels.

Kernels in `fuellib.correlate.kernels` are written once against a NumPy-like
namespace and dispatched on their inputs: JAX arrays (including tracers under
`jax.jit`/`jax.grad`) are evaluated with `jax.numpy`, everything else with
NumPy. JAX is never imported by FuelLib itself, so it remains optional and the
default FuelLib path keeps NumPy's float64 precision.
"""

import sys
from types import ModuleType

import numpy as np


def get_namespace(*arrays: object) -> ModuleType:
    """Return the array namespace matching the given inputs.

    Args:
        *arrays: Inputs to a kernel.

    Returns:
        `jax.numpy` if any input is a JAX array or tracer, otherwise `numpy`.
    """
    jax = sys.modules.get("jax")
    if jax is not None and any(isinstance(a, jax.Array) for a in arrays):
        return jax.numpy
    return np


__all__ = ["get_namespace"]
