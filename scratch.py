"""Scratch."""

from pint import UnitRegistry

import fuellib as fl
from fuellib import Component, Fuel
from fuellib.gcm import ConstantinouMethod

ureg = UnitRegistry()

comp = Component(name="test", decomposition={("CH3", 2), ("CH2", 1)})

method = ConstantinouMethod()

fuel = Fuel.from_json("fuellib/data/fuel/decane.json")
print(fl.fuel.mixture_density(fuel, Yi=fuel.Y_0, T=350 * ureg("degC")))
