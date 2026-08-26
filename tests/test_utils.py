"""Tests for small utility modules: convert and helpers."""

import numpy as np
import pytest

from fuellib.utils.constants import N_A, k_B
from fuellib.utils.convert import epsilon_to_characteristic_temperature
from fuellib.utils.helpers import droplet_mass, droplet_volume, mixing_rule
from fuellib.utils.units import ureg


class TestEpsilonToCharacteristicTemperature:
    def test_matches_direct_formula(self):
        epsilon = 1500.0
        expected = (epsilon / N_A) / k_B
        assert epsilon_to_characteristic_temperature(epsilon) == pytest.approx(expected)

    def test_zero_epsilon_is_zero_kelvin(self):
        assert epsilon_to_characteristic_temperature(0.0) == 0.0


class TestMixingRule:
    def test_arithmetic_mean_two_equal_components(self):
        values = np.array([1.0, 3.0])
        fractions = np.array([0.5, 0.5])
        # arithmetic: sum_i sum_j x_i x_j (v_i+v_j)/2
        expected = 0.25 * (1 + 1) / 2 + 0.25 * (1 + 3) / 2 * 2 + 0.25 * (3 + 3) / 2
        assert mixing_rule(values, fractions) == pytest.approx(expected)

    def test_geometric_mean(self):
        values = np.array([4.0, 9.0])
        fractions = np.array([1.0, 0.0])
        # With one fraction 0, only the (0,0) term (sqrt(4*4)) contributes.
        result = mixing_rule(values, fractions, pseudo_prop="geometric")
        assert result == pytest.approx(4.0)

    def test_geometric_is_case_insensitive(self):
        values = np.array([1.0, 1.0])
        fractions = np.array([0.5, 0.5])
        result = mixing_rule(values, fractions, pseudo_prop="Geometric")
        assert result == pytest.approx(1.0)


class TestDropletHelpers:
    def test_droplet_volume_matches_sphere_formula(self):
        radius = 2.0
        expected = 4.0 / 3.0 * np.pi * radius**3
        assert droplet_volume(radius) == pytest.approx(expected)

    def test_droplet_mass_zero_radius_returns_zeros(self):
        from fuellib.fuel import Fuel

        fuel = Fuel.from_name("decane")
        mass = droplet_mass(
            fuel,
            0.0,
            np.array(fuel.initial_mass_fractions),
            298.0 * ureg.K,
        )
        assert mass.units == ureg.kg
        assert np.allclose(mass.magnitude, 0.0)

    def test_droplet_mass_positive_radius_is_positive(self):
        from fuellib.fuel import Fuel

        fuel = Fuel.from_name("decane")
        mass = droplet_mass(
            fuel,
            1e-5,
            np.array(fuel.initial_mass_fractions),
            298.0 * ureg.K,
        )
        assert np.all(mass.magnitude > 0)
