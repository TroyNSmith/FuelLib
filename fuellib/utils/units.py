"""FuelLib units."""

from pint import UnitRegistry

#: Shared UnitRegistry instance for FuelLib.
ureg = UnitRegistry()

#: Shorthand for creating Quantity objects.
Q_ = ureg.Quantity

__all__ = ["Q_", "ureg"]
