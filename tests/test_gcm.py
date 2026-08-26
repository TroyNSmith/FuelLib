"""Tests for the group contribution method (GCM) module."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pint import Quantity

from fuellib.comp import Component
from fuellib.gcm.constantinou import ConstantinouMethod
from fuellib.gcm.core import BaseMethod
from fuellib.utils.units import ureg

DECANE_DECOMP = {("CH3", 2), ("CH2", 8)}
HEPTANE_DECOMP = {("CH3", 2), ("CH2", 5)}


@pytest.fixture
def cg_method() -> ConstantinouMethod:
    return ConstantinouMethod()


@pytest.fixture
def decane_component(cg_method: ConstantinouMethod) -> Component:
    return Component(name="decane", decomposition=DECANE_DECOMP, method=cg_method)


@pytest.fixture
def heptane_component(cg_method: ConstantinouMethod) -> Component:
    return Component(name="heptane", decomposition=HEPTANE_DECOMP, method=cg_method)


def _group_sum(
    method: ConstantinouMethod, prop: str, decomp: set[tuple[str, int]]
) -> float:
    """Independently recompute the additive group-contribution sum for a property."""
    groups, counts = zip(*decomp)
    values = method.groups.loc[prop, list(groups)].to_numpy(dtype=float)
    return float(np.dot(values, counts))


# --- load_groups ---


class TestLoadGroups:
    def test_loads_dataframe_with_required_index(self, cg_method: ConstantinouMethod):
        assert isinstance(cg_method.groups, pd.DataFrame)
        assert not cg_method.groups.empty
        assert set(cg_method._contributions.keys()).issubset(cg_method.groups.index)

    def test_missing_required_index_raises(self):
        bad_df = pd.DataFrame({"CH3": [1.0]}, index=["pck"])  # missing "tck" etc.
        with patch("fuellib.gcm.constantinou.pd.read_csv", return_value=bad_df):
            with pytest.raises(ValueError, match="missing required index columns"):
                ConstantinouMethod()


# --- get_contributions / _sum_contributions ---


class TestGetContributions:
    def test_returns_quantity_with_correct_length_and_values(
        self, cg_method: ConstantinouMethod
    ):
        result = cg_method.get_contributions(
            Component(name="decane", decomposition=DECANE_DECOMP), "MW"
        )
        assert isinstance(result, Quantity)
        assert result.units == ureg("gram/mole").units
        assert result.magnitude.size == sum(count for _, count in DECANE_DECOMP)

    @pytest.mark.parametrize(
        ("prop", "unit"),
        [
            ("pck", "bar**(-0.5)"),
            ("vck", "meter**3 / kilomole"),
            ("MW", "gram / mole"),
        ],
    )
    def test_units_match_declared_contribution(
        self, cg_method: ConstantinouMethod, prop: str, unit: str
    ):
        result = cg_method.get_contributions(
            Component(name="decane", decomposition=DECANE_DECOMP), prop
        )
        assert result.units == ureg(unit).units

    def test_invalid_property_raises_value_error(self, cg_method: ConstantinouMethod):
        with pytest.raises(ValueError, match="is not a valid property"):
            cg_method.get_contributions(
                Component(name="decane", decomposition=DECANE_DECOMP), "not_a_property"
            )

    def test_unknown_group_raises_key_error(self, cg_method: ConstantinouMethod):
        comp = Component(name="bogus", decomposition={("BOGUS", 1)})
        with pytest.raises(KeyError):
            cg_method.get_contributions(comp, "MW")

    def test_repeat_logic_for_single_group(self, cg_method: ConstantinouMethod):
        comp = Component(name="tri-methyl", decomposition={("CH3", 3)})
        total = cg_method._sum_contributions(comp, "MW").magnitude
        expected = 3 * float(cg_method.groups.loc["MW", "CH3"])
        assert total == pytest.approx(expected)


# --- calc_* property formulas ---


class TestPropertyCalculations:
    @pytest.mark.parametrize(
        "decomp", [DECANE_DECOMP, HEPTANE_DECOMP], ids=["decane", "heptane"]
    )
    def test_molecular_weight(self, cg_method: ConstantinouMethod, decomp):
        comp = Component(name="c", decomposition=decomp)
        expected = (_group_sum(cg_method, "MW", decomp) * ureg("g/mol")).to("kg/mol")
        assert cg_method.calc_molecular_weight(comp).to(
            "kg/mol"
        ).magnitude == pytest.approx(expected.magnitude, rel=1e-6)

    @pytest.mark.parametrize(
        "decomp", [DECANE_DECOMP, HEPTANE_DECOMP], ids=["decane", "heptane"]
    )
    def test_critical_temperature(self, cg_method: ConstantinouMethod, decomp):
        comp = Component(name="c", decomposition=decomp)
        expected = 181.128 * np.log(_group_sum(cg_method, "tck", decomp))
        assert cg_method.calc_critical_temperature(comp).to(
            "K"
        ).magnitude == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize(
        "decomp", [DECANE_DECOMP, HEPTANE_DECOMP], ids=["decane", "heptane"]
    )
    def test_critical_pressure(self, cg_method: ConstantinouMethod, decomp):
        comp = Component(name="c", decomposition=decomp)
        pck_sum = _group_sum(cg_method, "pck", decomp)
        expected_bar = 1.3705 + (pck_sum + 0.10022) ** (-2)
        expected = (expected_bar * ureg.bar).to("Pa").magnitude
        assert cg_method.calc_critical_pressure(comp).to(
            "Pa"
        ).magnitude == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize(
        "decomp", [DECANE_DECOMP, HEPTANE_DECOMP], ids=["decane", "heptane"]
    )
    def test_critical_volume(self, cg_method: ConstantinouMethod, decomp):
        comp = Component(name="c", decomposition=decomp)
        vck_sum = _group_sum(cg_method, "vck", decomp)
        expected = ((-0.00435 + vck_sum) * ureg("m**3/kmol")).to("m**3/mol")
        assert cg_method.calc_critical_volume(comp).to(
            "m**3/mol"
        ).magnitude == pytest.approx(expected.magnitude, rel=1e-6)

    @pytest.mark.parametrize(
        "decomp", [DECANE_DECOMP, HEPTANE_DECOMP], ids=["decane", "heptane"]
    )
    def test_boiling_temperature(self, cg_method: ConstantinouMethod, decomp):
        comp = Component(name="c", decomposition=decomp)
        expected = 204.359 * np.log(_group_sum(cg_method, "tbk", decomp))
        assert cg_method.calc_boiling_temperature(comp).to(
            "K"
        ).magnitude == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize(
        "decomp", [DECANE_DECOMP, HEPTANE_DECOMP], ids=["decane", "heptane"]
    )
    def test_melting_temperature(self, cg_method: ConstantinouMethod, decomp):
        comp = Component(name="c", decomposition=decomp)
        expected = 102.425 * np.log(_group_sum(cg_method, "tmk", decomp))
        assert cg_method.calc_melting_temperature(comp).to(
            "K"
        ).magnitude == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize(
        "decomp", [DECANE_DECOMP, HEPTANE_DECOMP], ids=["decane", "heptane"]
    )
    def test_enthalpy_of_formation(self, cg_method: ConstantinouMethod, decomp):
        comp = Component(name="c", decomposition=decomp)
        expected = (
            (10.835 + _group_sum(cg_method, "hfk", decomp)) * ureg("kJ/mol")
        ).to("J/mol")
        assert cg_method.calc_enthalpy_of_formation(comp).to(
            "J/mol"
        ).magnitude == pytest.approx(expected.magnitude, rel=1e-6)

    @pytest.mark.parametrize(
        "decomp", [DECANE_DECOMP, HEPTANE_DECOMP], ids=["decane", "heptane"]
    )
    def test_gibbs_free_energy(self, cg_method: ConstantinouMethod, decomp):
        comp = Component(name="c", decomposition=decomp)
        expected = (
            (-14.828 + _group_sum(cg_method, "gfk", decomp)) * ureg("kJ/mol")
        ).to("J/mol")
        assert cg_method.calc_gibbs_free_energy(comp).to(
            "J/mol"
        ).magnitude == pytest.approx(expected.magnitude, rel=1e-6)

    @pytest.mark.parametrize(
        "decomp", [DECANE_DECOMP, HEPTANE_DECOMP], ids=["decane", "heptane"]
    )
    def test_enthalpy_of_vaporization_stp(self, cg_method: ConstantinouMethod, decomp):
        comp = Component(name="c", decomposition=decomp)
        expected = ((6.829 + _group_sum(cg_method, "hvk", decomp)) * ureg("kJ/mol")).to(
            "J/mol"
        )
        assert cg_method.calc_enthalpy_of_vaporization_stp(comp).to(
            "J/mol"
        ).magnitude == pytest.approx(expected.magnitude, rel=1e-6)

    @pytest.mark.parametrize(
        "decomp", [DECANE_DECOMP, HEPTANE_DECOMP], ids=["decane", "heptane"]
    )
    def test_acentric_factor(self, cg_method: ConstantinouMethod, decomp):
        comp = Component(name="c", decomposition=decomp)
        wk_sum = _group_sum(cg_method, "wk", decomp)
        expected = 0.4085 * np.log(wk_sum + 1.1507) ** (1.0 / 0.5050)
        assert cg_method.calc_acentric_factor(comp).magnitude == pytest.approx(
            expected, rel=1e-6
        )
        assert cg_method.calc_acentric_factor(comp).units == ureg.dimensionless

    @pytest.mark.parametrize(
        "decomp", [DECANE_DECOMP, HEPTANE_DECOMP], ids=["decane", "heptane"]
    )
    def test_molar_liquid_volume_stp(self, cg_method: ConstantinouMethod, decomp):
        comp = Component(name="c", decomposition=decomp)
        vmk_sum = _group_sum(cg_method, "vmk", decomp)
        expected = ((0.01211 + vmk_sum) * ureg("m**3/kmol")).to("m**3/mol")
        assert cg_method.calc_molar_liquid_volume_stp(comp).to(
            "m**3/mol"
        ).magnitude == pytest.approx(expected.magnitude, rel=1e-6)

    @pytest.mark.parametrize(
        "decomp", [DECANE_DECOMP, HEPTANE_DECOMP], ids=["decane", "heptane"]
    )
    def test_heat_capacity_coeffs(self, cg_method: ConstantinouMethod, decomp):
        comp = Component(name="c", decomposition=decomp)
        cp_a, cp_b, cp_c = cg_method.calc_heat_capacity_coeffs(comp)

        expected_a = _group_sum(cg_method, "CpAk", decomp) - 19.7779
        expected_b = _group_sum(cg_method, "CpBk", decomp)
        expected_c = _group_sum(cg_method, "CpCk", decomp)

        assert cp_a.to("J/mol/K").magnitude == pytest.approx(expected_a, rel=1e-6)
        assert cp_b.to("J/mol/K").magnitude == pytest.approx(expected_b, rel=1e-6)
        assert cp_c.to("J/mol/K").magnitude == pytest.approx(expected_c, rel=1e-6)
        for coeff in (cp_a, cp_b, cp_c):
            assert coeff.units == ureg("J/mol/K").units

    def test_carbon_number_decane(self, cg_method: ConstantinouMethod):
        comp = Component(name="decane", decomposition=DECANE_DECOMP)
        assert cg_method.calc_carbon_number(comp).magnitude == pytest.approx(10)

    def test_carbon_number_heptane(self, cg_method: ConstantinouMethod):
        comp = Component(name="heptane", decomposition=HEPTANE_DECOMP)
        assert cg_method.calc_carbon_number(comp).magnitude == pytest.approx(7)


# --- Component integration ---


class TestComponentIntegration:
    def test_missing_method_raises(self):
        comp = Component(name="no-method", decomposition=DECANE_DECOMP, method=None)
        with pytest.raises(ValueError, match="has no GCM method set"):
            _ = comp.molecular_weight

    def test_property_matches_method_calculation(
        self, decane_component: Component, cg_method: ConstantinouMethod
    ):
        expected = cg_method.calc_molecular_weight(decane_component)
        assert decane_component.molecular_weight.to(
            "kg/mol"
        ).magnitude == pytest.approx(expected.to("kg/mol").magnitude)

    def test_property_is_cached(self, decane_component: Component):
        with patch.object(
            ConstantinouMethod,
            "calc_molecular_weight",
            wraps=decane_component.method.calc_molecular_weight,
        ) as mocked:
            _ = decane_component.molecular_weight
            _ = decane_component.molecular_weight
            assert mocked.call_count == 1
        assert "molecular_weight" in decane_component._cache


# --- BaseMethod structure ---


class TestBaseMethodStructure:
    def test_constantinou_is_subclass_of_base_method(self):
        assert issubclass(ConstantinouMethod, BaseMethod)

    @pytest.mark.parametrize(
        "method_name",
        [
            "calc_molecular_weight",
            "calc_critical_temperature",
            "calc_critical_pressure",
            "calc_critical_volume",
            "calc_boiling_temperature",
            "calc_melting_temperature",
            "calc_enthalpy_of_formation",
            "calc_gibbs_free_energy",
            "calc_enthalpy_of_vaporization_stp",
            "calc_acentric_factor",
            "calc_molar_liquid_volume_stp",
            "calc_heat_capacity_coeffs",
            "calc_carbon_number",
        ],
    )
    def test_all_abstract_methods_implemented(self, method_name: str):
        assert callable(getattr(ConstantinouMethod, method_name))
