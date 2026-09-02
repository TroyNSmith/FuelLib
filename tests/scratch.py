"""Scratch."""

import jax.numpy as jnp
import numpy as np
import quaxed.numpy as qnp
from unxt import Quantity

from fuellib import Fuel
from fuellib.units import convert_temperature

fuel1 = Fuel("decane")
fuel2 = Fuel("heptane-decane")
fuel3 = Fuel("jet-a")

print(
    fuel1.diffusion_coeff(
        P=Quantity([1.0], "atm"),
        T=Quantity([25.0], "Celsius"),
        unit="m^2/s",
        correlation="Tee",
    )
)
