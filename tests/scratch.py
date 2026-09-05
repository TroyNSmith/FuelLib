"""Scratch."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import quaxed.numpy as qnp
from unxt import Quantity

from fuellib import Fuel
from fuellib.gcm import GaniGCM
from fuellib.utils.units import convert_temperature

fuel = Fuel(name="jet-a")
gani = GaniGCM()

df = pd.DataFrame(
    {
        "Tc": gani.Tc(fuel).value,
        "Pc": gani.Pc(fuel).value,
        "Vc": gani.Vc(fuel).value,
        "Tb": gani.Tb(fuel).value,
        "Tm": gani.Tm(fuel).value,
        "Hf": gani.Hf(fuel).value,
        "Gf": gani.Gf(fuel).value,
        "Hv_stp": gani.Hv_stp(fuel).value,
        "Cp_stp": gani.Cp_stp(fuel).value,
        "Cp_B": gani.Cp_B(fuel).value,
        "Cp_C": gani.Cp_C(fuel).value,
        "Vm_stp": gani.Vm_stp(fuel).value,
        "omega": gani.omega(fuel).value,
    }
)

df.to_csv(
    "/Users/tsmith5/Documents/coding/FuelLib/tests/data/answers/gcm-gani-jet_a.csv",
    index=False,
)
