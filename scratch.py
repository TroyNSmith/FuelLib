"""Scratch."""

import numpy as np
import pandas as pd
from pint import UnitRegistry

from fuellib import Component, FuelNew
from fuellib.gcm import ConstantinouMethod

ureg = UnitRegistry()

comp = Component(name="test", decomposition={("CH3", 2), ("CH2", 1)})

method = ConstantinouMethod()

fuel = FuelNew.from_json("fuellib/data/fuelData/decane.json")
print(fuel)
