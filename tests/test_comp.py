"""Tests for the fuel component (comp) module."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest

from fuellib import comp as comp_module
from fuellib.comp import Component
from fuellib.gcm.constantinou import ConstantinouMethod
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


# --- molar_liquid_vol / density ---


class TestMolarLiquidVolAndDensity:
    def test_at_stp_matches_stp_property(self, decane_component: Component):
        result = comp_module.molar_liquid_vol(decane_component, 298.0 * ureg.K)
        assert result.units == ureg("m**3/mol").units
        assert result.magnitude == pytest.approx(
            decane_component.molar_liquid_volume_stp.magnitude, rel=1e-6
        )

    def test_below_stp_uses_temperature_correction(self, decane_component: Component):
        result = comp_module.molar_liquid_vol(decane_component, 250.0 * ureg.K)

        tc = decane_component.critical_temperature.magnitude
        w = decane_component.acentric_factor.magnitude
        vstp = decane_component.molar_liquid_volume_stp.magnitude
        phi = (1 - 250.0 / tc) ** (2.0 / 7.0) - (1 - 298.0 / tc) ** (2.0 / 7.0)
        z = 0.29056 - 0.08775 * w
        expected = vstp * z**phi

        assert result.magnitude == pytest.approx(expected, rel=1e-6)

    def test_above_critical_temperature_uses_negative_phi(
        self, decane_component: Component
    ):
        tc = decane_component.critical_temperature.magnitude
        result = comp_module.molar_liquid_vol(decane_component, (tc + 50.0) * ureg.K)

        w = decane_component.acentric_factor.magnitude
        vstp = decane_component.molar_liquid_volume_stp.magnitude
        phi = -((1 - 298.0 / tc) ** (2.0 / 7.0))
        z = 0.29056 - 0.08775 * w
        expected = vstp * z**phi

        assert result.magnitude == pytest.approx(expected, rel=1e-6)

    def test_density_matches_molecular_weight_over_molar_volume(
        self, decane_component: Component
    ):
        temperature = 320.0 * ureg.K
        rho = comp_module.density(decane_component, temperature)
        vm = comp_module.molar_liquid_vol(decane_component, temperature)
        expected = (decane_component.molecular_weight / vm).to("kg/m**3")

        assert rho.units == ureg("kg/m**3").units
        assert rho.magnitude == pytest.approx(expected.magnitude, rel=1e-6)


# --- viscosity ---


class TestViscosity:
    def test_kinematic_viscosity_matches_dutt_equation(
        self, decane_component: Component
    ):
        temperature = 310.0 * ureg.K
        result = comp_module.viscosity_kinematic(decane_component, temperature)

        temp_c = temperature.to("degC").magnitude
        tb_c = decane_component.boiling_temperature.magnitude - 273.15
        rhs = -3.0171 + (442.78 + 1.6452 * tb_c) / (temp_c + 239 - 0.19 * tb_c)
        expected = (np.exp(rhs) * ureg("mm**2/s")).to("m**2/s")

        assert result.units == ureg("m**2/s").units
        assert result.magnitude == pytest.approx(expected.magnitude, rel=1e-6)

    def test_dynamic_viscosity_is_kinematic_times_density(
        self, decane_component: Component
    ):
        temperature = 310.0 * ureg.K
        nu = comp_module.viscosity_kinematic(decane_component, temperature)
        rho = comp_module.density(decane_component, temperature)
        expected = (nu * rho).to("Pa*s")

        result = comp_module.viscosity_dynamic(decane_component, temperature)
        assert result.units == ureg("Pa*s").units
        assert result.magnitude == pytest.approx(expected.magnitude, rel=1e-6)


# --- heat capacity ---


class TestHeatCapacity:
    def test_molar_heat_capacity_at_stp_matches_coeff_a(
        self, decane_component: Component
    ):
        result = comp_module.molar_heat_capacity(decane_component, 298.0 * ureg.K)
        assert result.units == ureg("J/mol/K").units
        assert result.magnitude == pytest.approx(
            decane_component.heat_capacity_stp.magnitude, rel=1e-6
        )

    def test_molar_heat_capacity_matches_quadratic_formula(
        self, decane_component: Component
    ):
        temperature = 450.0 * ureg.K
        result = comp_module.molar_heat_capacity(decane_component, temperature)

        theta = (450.0 - 298) / 700
        cp_a, cp_b, cp_c = decane_component.heat_capacity_coeffs
        expected = cp_a.magnitude + cp_b.magnitude * theta + cp_c.magnitude * theta**2

        assert result.magnitude == pytest.approx(expected, rel=1e-6)

    def test_mass_heat_capacity_divides_by_molecular_weight(
        self, decane_component: Component
    ):
        temperature = 400.0 * ureg.K
        molar = comp_module.molar_heat_capacity(decane_component, temperature)
        expected = (molar / decane_component.molecular_weight).to("J/kg/K")

        result = comp_module.mass_heat_capacity(decane_component, temperature)
        assert result.units == ureg("J/kg/K").units
        assert result.magnitude == pytest.approx(expected.magnitude, rel=1e-6)


# --- saturation pressure ---


class TestSaturationPressure:
    def test_lee_kesler_matches_formula(self, decane_component: Component):
        temperature = 350.0 * ureg.K
        result = comp_module.saturation_pressure(
            decane_component, temperature, correlation="Lee-Kesler"
        )

        tr = 350.0 / decane_component.critical_temperature.magnitude
        w = decane_component.acentric_factor.magnitude
        f0 = 5.92714 - (6.09648 / tr) - 1.28862 * np.log(tr) + 0.169347 * (tr**6)
        f1 = 15.2518 - (15.6875 / tr) - 13.4721 * np.log(tr) + 0.43577 * (tr**6)
        expected = decane_component.critical_pressure.magnitude * np.exp(f0 + w * f1)

        assert result.units == ureg("Pa").units
        assert result.magnitude == pytest.approx(expected, rel=1e-6)

    def test_ambrose_walton_matches_formula(self, decane_component: Component):
        temperature = 350.0 * ureg.K
        result = comp_module.saturation_pressure(
            decane_component, temperature, correlation="Ambrose-Walton"
        )

        tr = 350.0 / decane_component.critical_temperature.magnitude
        w = decane_component.acentric_factor.magnitude
        tau = 1 - tr
        f0 = (
            -5.97616 * tau
            + 1.29874 * tau**1.5
            - 0.60394 * tau**2.5
            - 1.06841 * tau**5.0
        ) / tr
        f1 = (
            -5.03365 * tau
            + 1.11505 * tau**1.5
            - 5.41217 * tau**2.5
            - 7.46628 * tau**5.0
        ) / tr
        f2 = (
            -0.64771 * tau
            + 2.41539 * tau**1.5
            - 4.26979 * tau**2.5
            - 3.25259 * tau**5.0
        ) / tr
        expected = decane_component.critical_pressure.magnitude * np.exp(
            f0 + w * f1 + w**2 * f2
        )

        assert result.magnitude == pytest.approx(expected, rel=1e-6)

    def test_correlation_is_case_insensitive(self, decane_component: Component):
        temperature = 350.0 * ureg.K
        lower = comp_module.saturation_pressure(
            decane_component,
            temperature,
            correlation=cast(Literal["Lee-Kesler"], "lee-kesler"),
        )
        upper = comp_module.saturation_pressure(
            decane_component, temperature, correlation="Lee-Kesler"
        )
        assert lower.magnitude == pytest.approx(upper.magnitude, rel=1e-9)


# --- latent heat of vaporization ---


class TestLatentHeatVaporization:
    def test_zero_above_critical_temperature(self, decane_component: Component):
        tc = decane_component.critical_temperature.magnitude
        result = comp_module.latent_heat_vaporization(
            decane_component, (tc + 10.0) * ureg.K
        )
        assert result.magnitude == pytest.approx(0.0)
        assert result.units == ureg("J/kg").units

    def test_matches_watson_correlation_below_critical_temperature(
        self, decane_component: Component
    ):
        temperature = 350.0 * ureg.K
        result = comp_module.latent_heat_vaporization(decane_component, temperature)

        tc = decane_component.critical_temperature.magnitude
        tb = decane_component.boiling_temperature.magnitude
        tr = 350.0 / tc
        trb = tb / tc
        expected = decane_component.latent_heat_vaporization_stp.magnitude * (
            ((1.0 - tr) / (1.0 - trb)) ** 0.38
        )

        assert result.magnitude == pytest.approx(expected, rel=1e-6)

    def test_latent_heat_vaporization_stp_matches_enthalpy_over_mw(
        self, decane_component: Component
    ):
        expected = (
            decane_component.enthalpy_of_vaporization_stp
            / decane_component.molecular_weight
        ).to("J/kg")
        assert decane_component.latent_heat_vaporization_stp.magnitude == pytest.approx(
            expected.magnitude, rel=1e-9
        )


# --- surface tension ---


class TestSurfaceTension:
    def test_brock_bird_matches_formula(self, decane_component: Component):
        temperature = 350.0 * ureg.K
        result = comp_module.surface_tension(
            decane_component, temperature, correlation="Brock-Bird"
        )

        tc = decane_component.critical_temperature.magnitude
        pc_bar = decane_component.critical_pressure.to("bar").magnitude
        tb = decane_component.boiling_temperature.magnitude
        tr = 350.0 / tc
        tbr = tb / tc
        q = 0.1196 * (1.0 + (tbr * np.log(pc_bar / 1.01325)) / (1.0 - tbr)) - 0.279
        st_dyncm = (
            pc_bar ** (2.0 / 3.0) * tc ** (1.0 / 3.0) * q * (1 - tr) ** (11.0 / 9.0)
        )
        expected = (st_dyncm * ureg("dyn/cm")).to("N/m")

        assert result.units == ureg("N/m").units
        assert result.magnitude == pytest.approx(expected.magnitude, rel=1e-6)

    def test_pitzer_matches_formula(self, decane_component: Component):
        temperature = 350.0 * ureg.K
        result = comp_module.surface_tension(
            decane_component, temperature, correlation="Pitzer"
        )

        tc = decane_component.critical_temperature.magnitude
        pc_bar = decane_component.critical_pressure.to("bar").magnitude
        w = decane_component.acentric_factor.magnitude
        tr = 350.0 / tc
        q = (
            (1.86 + 1.18 * w)
            / 19.05
            * (((3.75 + 0.91 * w) / (0.291 - 0.08 * w)) ** (2.0 / 3.0))
        )
        st_dyncm = (
            pc_bar ** (2.0 / 3.0) * tc ** (1.0 / 3.0) * q * (1 - tr) ** (11.0 / 9.0)
        )
        expected = (st_dyncm * ureg("dyn/cm")).to("N/m")

        assert result.magnitude == pytest.approx(expected.magnitude, rel=1e-6)


# --- hc_type / family_code ---


class TestHcTypeAndFamilyCode:
    @pytest.mark.parametrize(
        ("decomp", "expected"),
        [
            ({("ACH", 6)}, "aromatic"),
            ({("5 membered ring", 1)}, "cyclo-alkane"),
            ({("CH2=CH", 1)}, "alkene"),
            ({("(CH3)2CH", 1)}, "iso-alkane"),
            (DECANE_DECOMP, "n-alkane"),
        ],
        ids=["aromatic", "cyclo-alkane", "alkene", "iso-alkane", "n-alkane"],
    )
    def test_classification_by_group_type(
        self, cg_method: ConstantinouMethod, decomp: set[tuple[str, int]], expected: str
    ):
        comp = Component(name="c", decomposition=decomp, method=cg_method)
        assert comp.hc_type == expected

    def test_aromatic_takes_priority_over_other_types(
        self, cg_method: ConstantinouMethod
    ):
        comp = Component(
            name="c",
            decomposition={("ACH", 1), ("CH2=CH", 1), ("(CH3)2CH", 1)},
            method=cg_method,
        )
        assert comp.hc_type == "aromatic"

    @pytest.mark.parametrize(
        ("hc_type", "expected_code"),
        [
            ("aromatic", 1),
            ("cyclo-alkane", 2),
            ("alkene", 3),
            ("iso-alkane", 0),
            ("n-alkane", 0),
        ],
    )
    def test_family_code_mapping(
        self, decane_component: Component, hc_type: str, expected_code: int
    ):
        decane_component._cache["hc_type"] = hc_type
        assert decane_component.family_code == expected_code


# --- thermal conductivity ---


class TestThermalConductivity:
    @pytest.mark.parametrize(
        ("hc_type", "coeff_star", "beta"),
        [
            ("aromatic", 0.0346, 1.0),
            ("cyclo-alkane", 0.0310, 1.0),
            ("alkene", 0.0361, 1.0),
            ("n-alkane", 0.00350, 0.5),
        ],
    )
    def test_family_specific_coefficients(
        self,
        decane_component: Component,
        hc_type: str,
        coeff_star: float,
        beta: float,
    ):
        decane_component._cache["hc_type"] = hc_type
        temperature = 350.0 * ureg.K
        result = comp_module.thermal_conductivity(decane_component, temperature)

        mw = decane_component.molecular_weight.to("g/mol").magnitude
        tc = decane_component.critical_temperature.magnitude
        tb = decane_component.boiling_temperature.magnitude
        tr = 350.0 / tc
        coeff = coeff_star * tb**1.2 / (mw**beta * tc**0.167)
        expected = coeff * (1 - tr) ** 0.38 / (tr ** (1 / 6))

        assert result.units == ureg("W/m/K").units
        assert result.magnitude == pytest.approx(expected, rel=1e-6)


# --- Lennard-Jones derived properties ---


class TestLennardJonesProperties:
    def test_diameter_matches_tee_correlation(self, decane_component: Component):
        pc_atm = decane_component.critical_pressure.to("atm").magnitude
        tc = decane_component.critical_temperature.magnitude
        w = decane_component.acentric_factor.magnitude
        expected_angstrom = (2.3551 - 0.0874 * w) * (tc / pc_atm) ** (1.0 / 3)

        result = decane_component.lennard_jones_diameter.to("angstrom").magnitude
        assert result == pytest.approx(expected_angstrom, rel=1e-6)

    def test_epsilon_over_kb_matches_tee_correlation(self, decane_component: Component):
        w = decane_component.acentric_factor.magnitude
        tc = decane_component.critical_temperature.magnitude
        expected = (0.7915 + 0.1693 * w) * tc

        result = decane_component.epsilon_over_kb.magnitude
        assert result == pytest.approx(expected, rel=1e-6)


# --- Component.from_csv ---


class TestFromCsv:
    def test_parses_names_smiles_and_decomposition(self, tmp_path):
        csv_path = tmp_path / "components.csv"
        csv_path.write_text("name,smiles,CH3,CH2\ncompA,CCCC,2,1\ncompB,CCCCC,0,3\n")

        components = Component.from_csv(csv_path)

        assert [c.name for c in components] == ["compA", "compB"]
        assert components[0].smiles == "CCCC"
        assert components[0].decomposition == {("CH3", 2), ("CH2", 1)}
        # a zero count for CH3 must be excluded from the decomposition
        assert components[1].decomposition == {("CH2", 3)}

    def test_from_csv_reads_dataframe_with_index_col(self, tmp_path):
        df = pd.DataFrame(
            {"smiles": ["C"], "CH4": [1]}, index=pd.Index(["methane"], name="name")
        )
        csv_path = tmp_path / "single.csv"
        df.to_csv(csv_path)

        components = Component.from_csv(csv_path)
        assert len(components) == 1
        assert components[0].name == "methane"
        assert components[0].decomposition == {("CH4", 1)}


# --- Component integration / error handling ---


class TestComponentErrorHandling:
    def test_missing_method_raises_value_error(self):
        comp = Component(name="no-method", decomposition=DECANE_DECOMP, method=None)
        with pytest.raises(ValueError, match="has no GCM method set"):
            _ = comp.molecular_weight

    def test_property_is_cached_across_accesses(self, decane_component: Component):
        first = decane_component.critical_temperature
        second = decane_component.critical_temperature
        assert first is second or first.magnitude == pytest.approx(second.magnitude)
        assert "critical_temperature" in decane_component._cache
