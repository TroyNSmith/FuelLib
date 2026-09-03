"""Scratch."""

import jax.numpy as jnp
import numpy as np
import quaxed.numpy as qnp
from unxt import Quantity

from fuellib import Fuel
from fuellib.units import convert_temperature

fuel = Fuel(name="jet-a")

print(fuel.mixture_thermal_conductivity(Quantity(25.0, "Celsius"), unit="W/(m*K)"))
