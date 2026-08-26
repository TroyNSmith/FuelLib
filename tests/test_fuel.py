"""Tests for the Fuel class and fuel-level (fuel) module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest
from pint import Quantity

from fuellib import comp as comp_module
from fuellib import fuel as fuel_module
from fuellib.comp import Component
from fuellib.fuel import Fuel
from fuellib.gcm.constantinou import ConstantinouMethod
from fuellib.utils.helpers import mixing_rule
from fuellib.utils.units import ureg

DECANE_DECOMP = {("CH3", 2), ("CH2", 8)}
HEPTANE_DECOMP = {("CH3", 2), ("CH2", 5)}

BASELINE_DIR = Path(__file__).parent / "baseline_predictions"
BASELINE_FUEL_NAMES = [
    "decane",
    "dodecane",
    "heptane",
    "posf10264",
    "posf10289",
    "posf10325",
]

# Property columns in the baseline CSVs mapped to the mixture function/unit that
# should reproduce them.
BASELINE_PROPERTY_FUNCS = {
    "Density": (fuel_module.mixture_density, "g/cm**3"),
    "Viscosity": (fuel_module.mixture_kinematic_viscosity, "mm**2/s"),
    "VaporPressure": (fuel_module.mixture_vapor_pressure, "kPa"),
    "SurfaceTension": (fuel_module.mixture_surface_tension, "N/m"),
    "ThermalConductivity": (fuel_module.mixture_thermal_conductivity, "W/m/K"),
}


def _load_baseline(name: str) -> pd.DataFrame:
    """Load a baseline_predictions CSV, dropping its units row (row index 0)."""
    df = pd.read_csv(BASELINE_DIR / f"{name}.csv", header=0)
    df = df.iloc[1:].reset_index(drop=True)
    return df.apply(pd.to_numeric, errors="coerce")


@pytest.fixture
def cg_method() -> ConstantinouMethod:
    return ConstantinouMethod()


@pytest.fixture
def decane_component(cg_method: ConstantinouMethod) -> Component:
    return Component(name="decane", decomposition=DECANE_DECOMP, method=cg_method)


@pytest.fixture
def heptane_component(cg_method: ConstantinouMethod) -> Component:
    return Component(name="heptane", decomposition=HEPTANE_DECOMP, method=cg_method)


@pytest.fixture
def small_fuel(
    decane_component: Component,
    heptane_component: Component,
    cg_method: ConstantinouMethod,
) -> Fuel:
    """A synthetic two-component blend for fast, deterministic checks."""
    return Fuel(
        name="test-blend",
        components=[(decane_component, 0.6), (heptane_component, 0.4)],
        method=cg_method,
    )


# --- Fuel loading ---


class TestFuelLoading:
    def test_from_name_loads_single_component_fuel(self):
        fuel = Fuel.from_name("decane")
        assert fuel.num_compounds == 1
        assert fuel.compounds == ["n-C10"]
        assert fuel.initial_mass_fractions == pytest.approx([1.0])

    def test_from_name_loads_multi_component_fuel(self):
        fuel = Fuel.from_name("posf10264")
        assert fuel.num_compounds > 1
        assert sum(fuel.initial_mass_fractions) == pytest.approx(1.0)

    def test_from_json_assigns_method_to_components_missing_one(self, tmp_path):
        data = {
            "components": {
                "compA": {
                    "smiles": "CCCC",
                    "decomposition": {"CH3": 2, "CH2": 2},
                    "weight_percent": 60.0,
                },
                "compB": {
                    "smiles": "CCCCC",
                    "decomposition": {"CH3": 2, "CH2": 3},
                    "weight_percent": 40.0,
                },
            },
            "properties": {},
        }
        json_path = tmp_path / "blend.json"
        json_path.write_text(json.dumps(data))

        fuel = Fuel.from_json(json_path)
        assert fuel.name == "blend"
        assert fuel.num_compounds == 2
        for comp, _ in fuel.components:
            assert comp.method is fuel.method

    def test_initial_mass_fractions_are_normalized(
        self, decane_component: Component, heptane_component: Component
    ):
        fuel = Fuel(
            name="unnormalized",
            components=[(decane_component, 3.0), (heptane_component, 1.0)],
        )
        assert fuel.initial_mass_fractions == pytest.approx([0.75, 0.25])

    def test_pelephysics_keys_is_none_when_unset(self, small_fuel: Fuel):
        assert small_fuel.pelephysics_keys is None

    def test_pelephysics_keys_returned_when_set(self, cg_method: ConstantinouMethod):
        comp = Component(
            name="decane",
            decomposition=DECANE_DECOMP,
            method=cg_method,
            pelephysics_key="NC10H22",
        )
        fuel = Fuel(name="f", components=[(comp, 1.0)], method=cg_method)
        assert fuel.pelephysics_keys is not None
        assert list(fuel.pelephysics_keys) == ["NC10H22"]


# --- Aggregate GCM properties ---


class TestAggregateProperties:
    @pytest.mark.parametrize(
        "prop_name",
        [
            "molecular_weight",
            "critical_temperature",
            "critical_pressure",
            "critical_volume",
            "boiling_temperature",
            "melting_temperature",
            "enthalpy_of_formation",
            "gibbs_free_energy",
            "enthalpy_of_vaporization_stp",
            "latent_heat_vaporization_stp",
            "heat_capacity_stp",
            "heat_capacity_coeff_b",
            "heat_capacity_coeff_c",
            "carbon_number",
            "molar_liquid_volume_stp",
            "acentric_factor",
            "lennard_jones_diameter",
            "epsilon_over_kb",
        ],
    )
    def test_array_matches_stacked_component_values(
        self, small_fuel: Fuel, prop_name: str
    ):
        fuel_values = getattr(small_fuel, prop_name).magnitude
        comp_values = [
            getattr(comp, prop_name).magnitude for comp, _ in small_fuel.components
        ]
        np.testing.assert_allclose(fuel_values, comp_values, rtol=1e-9)

    def test_hc_type_matches_components(self, small_fuel: Fuel):
        expected = [comp.hc_type for comp, _ in small_fuel.components]
        assert list(small_fuel.hc_type) == expected

    def test_family_code_matches_components(self, small_fuel: Fuel):
        expected = [comp.family_code for comp, _ in small_fuel.components]
        np.testing.assert_array_equal(small_fuel.family_code, expected)


# --- Fraction conversions ---


class TestFractionConversions:
    def test_mean_molecular_weight_harmonic_mean(self, small_fuel: Fuel):
        mass_fractions = small_fuel.initial_mass_fractions
        mw = small_fuel.molecular_weight.magnitude
        expected = 1.0 / np.sum(mass_fractions / mw)

        result = fuel_module.mean_molecular_weight(small_fuel, mass_fractions)
        assert result.magnitude == pytest.approx(expected, rel=1e-9)

    def test_mean_molecular_weight_zero_mass_returns_zero(self, small_fuel: Fuel):
        result = fuel_module.mean_molecular_weight(
            small_fuel, np.zeros(small_fuel.num_compounds)
        )
        assert result.magnitude == 0.0

    def test_mass_to_mass_fraction_normalizes(self, small_fuel: Fuel):
        mass = np.array([2.0, 6.0])
        result = fuel_module.mass_to_mass_fraction(small_fuel, mass)
        np.testing.assert_allclose(result, [0.25, 0.75])

    def test_mass_to_mass_fraction_zero_total_returns_zeros(self, small_fuel: Fuel):
        result = fuel_module.mass_to_mass_fraction(
            small_fuel, np.zeros(small_fuel.num_compounds)
        )
        np.testing.assert_array_equal(result, np.zeros(small_fuel.num_compounds))

    def test_mass_to_mole_fraction_normalizes_by_molecular_weight(
        self, small_fuel: Fuel
    ):
        mass = np.array([2.0, 6.0])
        mw = small_fuel.molecular_weight.magnitude
        expected_moles = mass / mw
        expected = expected_moles / expected_moles.sum()

        result = fuel_module.mass_to_mole_fraction(small_fuel, mass)
        np.testing.assert_allclose(result, expected)

    def test_mole_to_mass_and_back_round_trip(self, small_fuel: Fuel):
        mole_fractions = np.array([0.7, 0.3])
        mass_fractions = fuel_module.mole_fraction_to_mass_fraction(
            small_fuel, mole_fractions
        )
        round_trip = fuel_module.mass_fraction_to_mole_fraction(
            small_fuel, mass_fractions
        )
        np.testing.assert_allclose(round_trip, mole_fractions, rtol=1e-9)

    def test_mole_fraction_to_mass_fraction_zero_total_returns_zeros(
        self, small_fuel: Fuel
    ):
        result = fuel_module.mole_fraction_to_mass_fraction(
            small_fuel, np.zeros(small_fuel.num_compounds)
        )
        np.testing.assert_array_equal(result, np.zeros(small_fuel.num_compounds))

    def test_mass_fraction_to_mole_fraction_zero_sum_returns_zeros(
        self, small_fuel: Fuel
    ):
        result = fuel_module.mass_fraction_to_mole_fraction(
            small_fuel, np.zeros(small_fuel.num_compounds)
        )
        np.testing.assert_array_equal(result, np.zeros(small_fuel.num_compounds))


# --- Per-component temperature-dependent functions ---

_SIMPLE_FUNC_PAIRS = [
    ("density", "kg/m**3"),
    ("viscosity_kinematic", "m**2/s"),
    ("viscosity_dynamic", "Pa*s"),
    ("molar_heat_capacity", "J/mol/K"),
    ("mass_heat_capacity", "J/kg/K"),
    ("molar_liquid_vol", "m**3/mol"),
    ("latent_heat_vaporization", "J/kg"),
    ("thermal_conductivity", "W/m/K"),
]


class TestPerComponentTemperatureFunctions:
    @pytest.mark.parametrize(("func_name", "unit"), _SIMPLE_FUNC_PAIRS)
    def test_all_components_matches_stacked_per_component(
        self, small_fuel: Fuel, func_name: str, unit: str
    ):
        temperature = 350.0 * ureg.K
        fuel_func = getattr(fuel_module, func_name)
        comp_func = getattr(comp_module, func_name)

        all_result = fuel_func(small_fuel, temperature).to(unit).magnitude
        expected = [
            comp_func(comp, temperature).to(unit).magnitude
            for comp, _ in small_fuel.components
        ]
        np.testing.assert_allclose(all_result, expected, rtol=1e-9)

    @pytest.mark.parametrize(("func_name", "unit"), _SIMPLE_FUNC_PAIRS)
    def test_comp_idx_selects_single_component(
        self, small_fuel: Fuel, func_name: str, unit: str
    ):
        temperature = 350.0 * ureg.K
        fuel_func = getattr(fuel_module, func_name)
        comp_func = getattr(comp_module, func_name)

        for idx, (comp, _) in enumerate(small_fuel.components):
            result = fuel_func(small_fuel, temperature, comp_idx=idx).to(unit).magnitude
            expected = comp_func(comp, temperature).to(unit).magnitude
            assert result == pytest.approx(expected, rel=1e-9)

    def test_saturation_pressure_delegates_per_component(self, small_fuel: Fuel):
        temperature = 350.0 * ureg.K
        for correlation in ("Lee-Kesler", "Ambrose-Walton"):
            all_result = (
                fuel_module.saturation_pressure(
                    small_fuel, temperature, correlation=correlation
                )
                .to("Pa")
                .magnitude
            )
            expected = [
                comp_module.saturation_pressure(
                    comp, temperature, correlation=correlation
                )
                .to("Pa")
                .magnitude
                for comp, _ in small_fuel.components
            ]
            np.testing.assert_allclose(all_result, expected, rtol=1e-9)

    def test_saturation_pressure_comp_idx_selects_single_component(
        self, small_fuel: Fuel
    ):
        temperature = 350.0 * ureg.K
        for idx, (comp, _) in enumerate(small_fuel.components):
            result = (
                fuel_module.saturation_pressure(small_fuel, temperature, comp_idx=idx)
                .to("Pa")
                .magnitude
            )
            expected = (
                comp_module.saturation_pressure(comp, temperature).to("Pa").magnitude
            )
            assert result == pytest.approx(expected, rel=1e-9)

    def test_surface_tension_delegates_per_component(self, small_fuel: Fuel):
        temperature = 350.0 * ureg.K
        for correlation in ("Brock-Bird", "Pitzer"):
            all_result = (
                fuel_module.surface_tension(
                    small_fuel, temperature, correlation=correlation
                )
                .to("N/m")
                .magnitude
            )
            expected = [
                comp_module.surface_tension(comp, temperature, correlation=correlation)
                .to("N/m")
                .magnitude
                for comp, _ in small_fuel.components
            ]
            np.testing.assert_allclose(all_result, expected, rtol=1e-9)

    def test_surface_tension_comp_idx_selects_single_component(self, small_fuel: Fuel):
        temperature = 350.0 * ureg.K
        for idx, (comp, _) in enumerate(small_fuel.components):
            result = (
                fuel_module.surface_tension(small_fuel, temperature, comp_idx=idx)
                .to("N/m")
                .magnitude
            )
            expected = (
                comp_module.surface_tension(comp, temperature).to("N/m").magnitude
            )
            assert result == pytest.approx(expected, rel=1e-9)


# --- Mixture functions ---


class TestMixtureFunctions:
    def test_mixture_density_is_mass_weighted_specific_volume(self, small_fuel: Fuel):
        temperature = 320.0 * ureg.K
        mass_fractions = small_fuel.initial_mass_fractions
        mw = small_fuel.molecular_weight.magnitude
        vm = fuel_module.molar_liquid_vol(small_fuel, temperature).magnitude
        expected = mass_fractions @ (mw / vm)

        result = fuel_module.mixture_density(small_fuel, mass_fractions, temperature)
        assert result.units == ureg("kg/m**3").units
        assert result.magnitude == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("correlation", ["Kendall-Monroe", "Arrhenius"])
    def test_mixture_kinematic_viscosity_matches_mixing_rule(
        self, small_fuel: Fuel, correlation: Literal["Kendall-Monroe", "Arrhenius"]
    ):
        temperature = 320.0 * ureg.K
        mass_fractions = small_fuel.initial_mass_fractions
        nu_arr = fuel_module.viscosity_kinematic(small_fuel, temperature).magnitude
        mole_fractions = fuel_module.mass_fraction_to_mole_fraction(
            small_fuel, mass_fractions
        )

        if correlation == "Arrhenius":
            expected = np.exp(np.sum(mole_fractions * np.log(nu_arr)))
        else:
            expected = np.sum(mole_fractions * (nu_arr ** (1.0 / 3.0))) ** 3.0

        result = fuel_module.mixture_kinematic_viscosity(
            small_fuel, mass_fractions, temperature, correlation=correlation
        )
        assert result.magnitude == pytest.approx(expected, rel=1e-9)

    def test_mixture_dynamic_viscosity_is_density_times_kinematic(
        self, small_fuel: Fuel
    ):
        temperature = 320.0 * ureg.K
        mass_fractions = small_fuel.initial_mass_fractions
        nu = fuel_module.mixture_kinematic_viscosity(
            small_fuel, mass_fractions, temperature
        )
        rho = fuel_module.mixture_density(small_fuel, mass_fractions, temperature)
        expected = (rho * nu).to("Pa*s")

        result = fuel_module.mixture_dynamic_viscosity(
            small_fuel, mass_fractions, temperature
        )
        assert result.magnitude == pytest.approx(expected.magnitude, rel=1e-9)

    def test_mixture_vapor_pressure_follows_raoults_law(self, small_fuel: Fuel):
        temperature = 320.0 * ureg.K
        mass_fractions = small_fuel.initial_mass_fractions
        mole_fractions = fuel_module.mass_fraction_to_mole_fraction(
            small_fuel, mass_fractions
        )
        psat = fuel_module.saturation_pressure(small_fuel, temperature).magnitude
        expected = psat @ mole_fractions

        result = fuel_module.mixture_vapor_pressure(
            small_fuel, mass_fractions, temperature
        )
        assert result.magnitude == pytest.approx(expected, rel=1e-9)

    def test_mixture_surface_tension_uses_arithmetic_mixing_rule(
        self, small_fuel: Fuel
    ):
        temperature = 320.0 * ureg.K
        mass_fractions = small_fuel.initial_mass_fractions
        mole_fractions = fuel_module.mass_fraction_to_mole_fraction(
            small_fuel, mass_fractions
        )
        st_arr = fuel_module.surface_tension(small_fuel, temperature).magnitude
        expected = mixing_rule(st_arr, mole_fractions, "arithmetic")

        result = fuel_module.mixture_surface_tension(
            small_fuel, mass_fractions, temperature
        )
        assert result.magnitude == pytest.approx(expected, rel=1e-9)

    def test_mixture_thermal_conductivity_matches_formula(self, small_fuel: Fuel):
        temperature = 320.0 * ureg.K
        mass_fractions = small_fuel.initial_mass_fractions
        k_arr = fuel_module.thermal_conductivity(small_fuel, temperature).magnitude
        expected = np.sum(mass_fractions * k_arr ** (-2)) ** (-0.5)

        result = fuel_module.mixture_thermal_conductivity(
            small_fuel, mass_fractions, temperature
        )
        assert result.magnitude == pytest.approx(expected, rel=1e-9)


# --- Special functions ---


class TestSpecialFunctions:
    @pytest.mark.parametrize("correlation", ["Tee", "Wilke"])
    def test_diffusion_coeff_is_positive_and_finite(
        self, small_fuel: Fuel, correlation: Literal["Tee", "Wilke"]
    ):
        result = fuel_module.diffusion_coeff(
            small_fuel,
            pressure=1.0 * ureg.atm,
            temperature=300.0 * ureg.K,
            correlation=correlation,
        )
        assert result.units == ureg("m**2/s").units
        assert np.all(np.isfinite(result.magnitude))
        assert np.all(result.magnitude > 0)

    @pytest.mark.parametrize(
        ("units", "factor"),
        [("mks", 1), ("bar", 1e5), ("atm", 1.01325e5), ("cgs", 0.1)],
    )
    def test_saturation_pressure_antoine_coeffs_unit_factor(
        self,
        small_fuel: Fuel,
        units: Literal["mks", "cgs", "bar", "atm"],
        factor: float,
    ):
        _, _, _, unit_factor_arr = fuel_module.saturation_pressure_antoine_coeffs(
            small_fuel, units=units
        )
        np.testing.assert_allclose(unit_factor_arr, factor)

    def test_saturation_pressure_antoine_coeffs_reconstructs_pressure(
        self, small_fuel: Fuel
    ):
        coeff_a, coeff_b, coeff_c, unit_factor = (
            fuel_module.saturation_pressure_antoine_coeffs(small_fuel)
        )
        for i, (comp, _) in enumerate(small_fuel.components):
            temperature_k = 0.9 * comp.boiling_temperature.magnitude
            actual = (
                comp_module.saturation_pressure(comp, temperature_k * ureg.K).magnitude
                / unit_factor[i]
            )
            predicted = 10 ** (coeff_a[i] - coeff_b[i] / (temperature_k + coeff_c[i]))
            assert predicted == pytest.approx(actual, rel=0.05)

    @pytest.mark.parametrize(
        "temperature_values",
        [(280.0, 400.0), np.linspace(280.0, 400.0, 8)],
        ids=["two_element_range", "explicit_array"],
    )
    def test_saturation_pressure_antoine_coeffs_accepts_temperature_values(
        self, small_fuel: Fuel, temperature_values
    ):
        coeff_a, coeff_b, coeff_c, unit_factor = (
            fuel_module.saturation_pressure_antoine_coeffs(
                small_fuel, temperature_values=temperature_values
            )
        )
        assert coeff_a.shape == (small_fuel.num_compounds,)
        assert np.all(np.isfinite(coeff_a))
        assert np.all(np.isfinite(coeff_b))
        assert np.all(np.isfinite(coeff_c))
        assert np.all(unit_factor == 1)

    def test_mixture_vapor_pressure_antoine_coeffs_reconstructs_pressure(
        self, small_fuel: Fuel
    ):
        mass_fractions = small_fuel.initial_mass_fractions
        coeff_a, coeff_b, coeff_c, unit_factor = (
            fuel_module.mixture_vapor_pressure_antoine_coeffs(
                small_fuel, mass_fractions
            )
        )
        temperature_k = 320.0
        actual = (
            fuel_module.mixture_vapor_pressure(
                small_fuel, mass_fractions, temperature_k * ureg.K
            ).magnitude
            / unit_factor
        )
        predicted = 10 ** (coeff_a - coeff_b / (temperature_k + coeff_c))
        assert predicted == pytest.approx(actual, rel=0.05)

    @pytest.mark.parametrize(
        "temperature_values",
        [(280.0, 400.0), np.linspace(280.0, 400.0, 8)],
        ids=["two_element_range", "explicit_array"],
    )
    def test_mixture_vapor_pressure_antoine_coeffs_accepts_temperature_values(
        self, small_fuel: Fuel, temperature_values
    ):
        mass_fractions = small_fuel.initial_mass_fractions
        coeff_a, coeff_b, coeff_c, unit_factor = (
            fuel_module.mixture_vapor_pressure_antoine_coeffs(
                small_fuel, mass_fractions, temperature_values=temperature_values
            )
        )
        assert all(np.isfinite(v) for v in (coeff_a, coeff_b, coeff_c))
        assert unit_factor == 1


# --- Experimental (validation) property data ---


class TestExperimentalProperty:
    def test_returns_converted_temperature_and_values(self):
        fuel = Fuel.from_name("decane")
        temperature, values = fuel_module.experimental_property(fuel, "Density")

        assert temperature.units == ureg("K").units
        assert values.units == ureg("g/cm**3").units
        assert len(temperature) == len(values)
        # first data point in decane.json's Density block is -29.65 degC
        assert temperature[0].to("degC").magnitude == pytest.approx(-29.65)

    def test_unknown_property_raises_key_error(self):
        fuel = Fuel.from_name("decane")
        with pytest.raises(KeyError, match="No experimental data"):
            fuel_module.experimental_property(fuel, "NotAProperty")


# --- Baseline-prediction regression tests ---


class TestBaselinePredictions:
    @pytest.mark.parametrize("fuel_name", BASELINE_FUEL_NAMES)
    def test_mixture_properties_within_baseline_tolerance(self, fuel_name: str):
        fuel = Fuel.from_name(fuel_name)
        mass_fractions = fuel.initial_mass_fractions
        baseline = _load_baseline(fuel_name)

        failures = []
        for row in baseline.itertuples(index=False):
            row_temperature = cast(float, getattr(row, "Temperature"))  # noqa: B009
            temperature = cast(Quantity, Quantity(row_temperature, "degC"))
            for prop_name, (func, unit) in BASELINE_PROPERTY_FUNCS.items():
                baseline_value = getattr(row, prop_name)
                error = getattr(row, f"Error_{prop_name}")
                if np.isnan(baseline_value) or np.isnan(error):
                    continue
                predicted = func(fuel, mass_fractions, temperature).to(unit).magnitude
                if abs(predicted - baseline_value) > error:
                    failures.append(
                        f"{fuel_name}: {prop_name} @ {row_temperature} C -> "
                        f"predicted={predicted:.6g}, "
                        f"baseline={baseline_value:.6g} +/- {error:.3g}"
                    )

        assert not failures, "\n".join(failures)
