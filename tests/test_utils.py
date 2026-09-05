"""Tests for the utils module in the FuelLib library."""

import astropy.units as apyu
import quaxed.numpy as qnp
from unxt import Quantity

# NOTE: Ensure that the custom units are registered with unxt before running the tests.
from fuellib.utils import units


class TestUnits:
    """Tests for the Units class in the utils module of FuelLib."""

    def test_defined_units(self) -> None:
        """Test that the custom units are defined correctly."""
        assert Quantity(1, "atm") == Quantity(101325, "Pa")
        assert Quantity(1, "dyne/cm^2") == Quantity(0.1, "Pa")
        assert Quantity(1, "cgs") == Quantity(0.1, "Pa")
        assert Quantity(1, "mks") == Quantity(1, "Pa")
        assert Quantity(1, "Fahrenheit") == Quantity(1, apyu.imperial.deg_F)

    def test_strip_units_from_quantity(self) -> None:
        """Test stripping units from an array of quantities."""
        arr = Quantity([1, 2, 3], "m")
        stripped = units.strip(arr)
        assert qnp.allclose(stripped, qnp.array([1, 2, 3]))

    def test_convert_from_celsius(self) -> None:
        """Test conversion from Celsius to different temperature units."""
        deg_c = Quantity(0.0, "Celsius")
        deg_f = units.convert_temperature(deg_c, "Fahrenheit")
        kelvin = units.convert_temperature(deg_c, "K")

        assert qnp.allclose(units.strip(deg_f), qnp.array(32.0))
        assert qnp.allclose(units.strip(kelvin), qnp.array(273.15))

    def test_convert_from_fahrenheit(self) -> None:
        """Test conversion from Fahrenheit to different temperature units."""
        deg_f = Quantity(32.0, "Fahrenheit")
        deg_c = units.convert_temperature(deg_f, "Celsius")
        kelvin = units.convert_temperature(deg_f, "K")

        assert qnp.allclose(units.strip(deg_c), qnp.array(0.0))
        assert qnp.allclose(units.strip(kelvin), qnp.array(273.15))

    def test_convert_from_kelvin(self) -> None:
        """Test conversion from Kelvin to different temperature units."""
        kelvin = Quantity(273.15, "K")
        deg_c = units.convert_temperature(kelvin, "Celsius")
        deg_f = units.convert_temperature(kelvin, "Fahrenheit")

        assert qnp.allclose(units.strip(deg_c), qnp.array(0.0))
        assert qnp.allclose(units.strip(deg_f), qnp.array(32.0))
