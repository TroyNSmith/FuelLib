"""Tests for the Converge properties exporter."""

import sys

import numpy as np
import pandas as pd
import pytest

from fuellib.exporters import converge
from fuellib.fuel import Fuel


@pytest.fixture
def decane_fuel() -> Fuel:
    return Fuel.from_name("decane")


@pytest.fixture
def blend_fuel() -> Fuel:
    """Two-compound blend so mixture Tm/Tc differ from either pure compound."""
    return Fuel.from_name("heptane-decane")


class TestUnitConverter:
    def test_mks_has_identity_factors(self):
        conv = converge.UnitConverter("mks")
        assert conv.mw == 1
        assert conv.pressure == 1
        assert conv.labels["temperature"] == "Temperature (K)"

    def test_cgs_has_scaled_factors(self):
        conv = converge.UnitConverter("CGS")
        assert conv.mw == 1e3
        assert conv.pressure == 1e1
        assert conv.labels["viscosity"] == "Viscosity (Poise)"

    def test_create_data_dict_applies_conversion(self):
        conv = converge.UnitConverter("cgs")
        data = conv.create_data_dict(
            T=np.array([300.0]),
            T_crit=500.0,
            mu=np.array([1.0]),
            surface_tension=np.array([1.0]),
            latent_heat=np.array([1.0]),
            pv=np.array([1.0]),
            rho=np.array([1.0]),
            specific_heat_mass=np.array([1.0]),
            thermal_conductivity=np.array([1.0]),
        )
        assert data["Viscosity (Poise)"][0] == pytest.approx(1e2)
        assert data["Critical Temperature (K)"][0] == pytest.approx(500.0)


class TestExportConvergeValidation:
    def test_invalid_fuel_raises_type_error(self):
        with pytest.raises(TypeError, match="valid FuelLib fuel object"):
            converge.export_converge(object())

    def test_invalid_units_raises_value_error(self, decane_fuel):
        with pytest.raises(ValueError, match="Units must be"):
            converge.export_converge(decane_fuel, units="bogus")

    def test_negative_temp_min_raises_value_error(self, decane_fuel):
        with pytest.raises(ValueError, match="temp_min must be non-negative"):
            converge.export_converge(decane_fuel, temp_min=-1)

    def test_temp_max_not_greater_than_min_raises_value_error(self, decane_fuel):
        with pytest.raises(ValueError, match="must be greater than temp_min"):
            converge.export_converge(decane_fuel, temp_min=300, temp_max=300)

    def test_non_positive_temp_step_raises_value_error(self, decane_fuel):
        with pytest.raises(ValueError, match="temp_step must be positive"):
            converge.export_converge(decane_fuel, temp_step=0)


class TestExportConvergeSuccess:
    def test_export_mix_writes_mixture_csv(self, decane_fuel, tmp_path):
        converge.export_converge(
            decane_fuel,
            path=str(tmp_path),
            temp_min=280,
            temp_max=320,
            temp_step=20,
            export_mix=True,
        )
        out_file = tmp_path / f"mixturePropsGCM_{decane_fuel.name}.csv"
        assert out_file.exists()
        df = pd.read_csv(out_file)
        assert "Temperature (K)" in df.columns
        assert "Density (kg/m^3)" in df.columns
        assert len(df) > 0

    def test_export_components_writes_composition_and_component_csv(
        self, decane_fuel, tmp_path
    ):
        converge.export_converge(
            decane_fuel,
            path=str(tmp_path),
            temp_min=280,
            temp_max=320,
            temp_step=20,
            export_mix=False,
        )
        component_dir = tmp_path / decane_fuel.name
        composition_file = component_dir / f"composition_{decane_fuel.name}.csv"
        assert composition_file.exists()
        comp_df = pd.read_csv(composition_file)
        assert "Mass Fraction" in comp_df.columns

        component_file = component_dir / "0_n-C10.csv"
        assert component_file.exists()

    def test_cgs_units_change_column_labels(self, decane_fuel, tmp_path):
        converge.export_converge(
            decane_fuel,
            path=str(tmp_path),
            units="cgs",
            temp_min=280,
            temp_max=320,
            temp_step=20,
            export_mix=True,
        )
        out_file = tmp_path / f"mixturePropsGCM_{decane_fuel.name}.csv"
        df = pd.read_csv(out_file)
        assert "Density (g/cm^3)" in df.columns

    def test_mixture_temperature_range_is_adjusted_with_warning(
        self, blend_fuel, tmp_path, capsys
    ):
        tm_arr = blend_fuel.melting_temperature.to("K").magnitude
        tc_arr = blend_fuel.critical_temperature.to("K").magnitude
        # Span a range that starts below the coldest melting point and ends
        # above the lowest critical temperature so both warning branches fire.
        temp_min = max(0.0, float(tm_arr.min()) - 50.0)
        temp_max = float(tc_arr.min()) + 50.0

        converge.export_converge(
            blend_fuel,
            path=str(tmp_path),
            temp_min=temp_min,
            temp_max=temp_max,
            temp_step=20,
            export_mix=True,
        )
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        out_file = tmp_path / f"mixturePropsGCM_{blend_fuel.name}.csv"
        assert out_file.exists()


class TestMain:
    def test_main_exports_mixture_properties(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "fl-export-converge",
                "-f",
                "decane",
                "-o",
                str(tmp_path),
                "-t",
                "280",
                "-T",
                "320",
                "-s",
                "20",
                "-m",
                "true",
            ],
        )
        converge.main()
        assert (tmp_path / "mixturePropsGCM_decane.csv").exists()
