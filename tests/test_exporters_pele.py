"""Tests for the Pele properties exporter."""

import json
import subprocess
import sys
import urllib.error
from unittest.mock import patch

import pandas as pd
import pytest

from fuellib.exporters import pele
from fuellib.fuel import Fuel


@pytest.fixture
def decane_fuel() -> Fuel:
    return Fuel.from_name("decane")


@pytest.fixture
def posf_fuel() -> Fuel:
    """Multi-compound blend with PelePhysics keys set."""
    return Fuel.from_name("posf10264")


class TestUnitConverter:
    def test_mks_identity_factors(self):
        conv = pele.UnitConverter("mks")
        assert conv.mw == 1.0
        assert conv.pressure == 1.0

    def test_cgs_scaled_factors(self):
        conv = pele.UnitConverter("CGS")
        assert conv.mw == 1e3
        assert conv.vm == 1e6

    def test_invalid_units_raises_value_error(self):
        with pytest.raises(ValueError, match="Units must be"):
            pele.UnitConverter("bogus")


class TestVecToStr:
    def test_list_of_strings(self):
        assert pele.vec_to_str(["a", "b", "c"]) == "a b c"

    def test_series_of_numbers(self):
        series = pd.Series([1.0, 2.0, 3.0])
        assert pele.vec_to_str(series) == "1.0 2.0 3.0"


class TestGetFilename:
    @pytest.mark.parametrize(
        ("model", "export_mix", "expected"),
        [
            ("gcm", False, "sprayPropsGCM_decane.inp"),
            ("gcm", True, "sprayPropsGCM_mixture_decane.inp"),
            ("mp", False, "sprayPropsMP_decane.inp"),
            ("mp", True, "sprayPropsMP_mixture_decane.inp"),
        ],
    )
    def test_generates_expected_name(self, model, export_mix, expected, tmp_path):
        result = pele.get_filename("decane", model, export_mix, str(tmp_path))
        assert result == str(tmp_path / expected)


class TestGetGitInfo:
    def test_uses_git_subprocess_when_available(self):
        with patch(
            "subprocess.check_output",
            side_effect=[b"abc123\n", b"git@example.com:repo.git\n"],
        ):
            commit, remote = pele.get_git_info()
        assert commit == "abc123"
        assert remote == "git@example.com:repo.git"

    def test_falls_back_to_package_version_and_pypi(self):
        with (
            patch(
                "subprocess.check_output",
                side_effect=subprocess.CalledProcessError(1, "git"),
            ),
            patch("fuellib.__version__", "1.2.3", create=True),
            patch.object(
                pele, "_get_pypi_repo_url", return_value="https://pypi/fallback"
            ),
        ):
            commit, remote = pele.get_git_info()
        assert commit == "1.2.3"
        assert remote == "https://pypi/fallback"


class TestGetPypiRepoUrl:
    def _fake_response(self, payload):
        class _Resp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *args):
                return False

            def read(self_inner):
                return json.dumps(payload).encode("utf-8")

        return _Resp()

    def test_returns_repository_url_from_project_urls(self):
        payload = {"info": {"project_urls": {"Repository": "https://repo.example"}}}
        with patch("urllib.request.urlopen", return_value=self._fake_response(payload)):
            assert pele._get_pypi_repo_url() == "https://repo.example"

    def test_falls_back_to_home_page(self):
        payload = {"info": {"project_urls": {}, "home_page": "https://home.example"}}
        with patch("urllib.request.urlopen", return_value=self._fake_response(payload)):
            assert pele._get_pypi_repo_url() == "https://home.example"

    def test_falls_back_to_pypi_project_url_on_network_error(self):
        with (
            patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.URLError("no network"),
            ),
            pytest.warns(UserWarning, match="Failed to fetch repository URL"),
        ):
            result = pele._get_pypi_repo_url()
        assert result.startswith("https://pypi.org/project/fuellib/")


class TestExportPeleValidation:
    def test_invalid_fuel_raises_type_error(self):
        with pytest.raises(TypeError, match="valid FuelLib fuel object"):
            pele.export_pele(object())

    def test_invalid_liq_prop_model_raises_value_error(self, decane_fuel):
        with pytest.raises(ValueError, match="liq_prop_model must be"):
            pele.export_pele(decane_fuel, liq_prop_model="bogus")

    def test_dep_fuel_names_length_mismatch_raises_value_error(
        self, posf_fuel, tmp_path
    ):
        with pytest.raises(ValueError, match="Length of dep_fuel_names"):
            pele.export_pele(
                posf_fuel,
                path=str(tmp_path),
                dep_fuel_names=["a", "b"],
                use_pp_keys=False,
            )

    def test_compound_name_with_space_raises_value_error(self, decane_fuel, tmp_path):
        decane_fuel.components[0][0].name = "bad name"
        with pytest.raises(ValueError, match="cannot accept compounds with spaces"):
            pele.export_pele(decane_fuel, path=str(tmp_path), use_pp_keys=False)


class TestExportPeleSuccess:
    def test_gcm_individual_compounds_with_pp_keys(self, posf_fuel, tmp_path):
        pele.export_pele(
            posf_fuel,
            path=str(tmp_path),
            liq_prop_model="gcm",
            use_pp_keys=True,
        )
        out_file = tmp_path / "sprayPropsGCM_posf10264.inp"
        assert out_file.exists()
        content = out_file.read_text()
        assert "particles.fuel_species" in content
        assert "particles.Y_0" in content

    def test_gcm_individual_compounds_without_pp_keys(self, decane_fuel, tmp_path):
        pele.export_pele(
            decane_fuel,
            path=str(tmp_path),
            liq_prop_model="gcm",
            use_pp_keys=False,
        )
        out_file = tmp_path / "sprayPropsGCM_decane.inp"
        content = out_file.read_text()
        assert "n-C10" in content

    def test_gcm_mixture_export_uppercases_posf_name(self, posf_fuel, tmp_path):
        pele.export_pele(
            posf_fuel,
            path=str(tmp_path),
            liq_prop_model="gcm",
            export_mix=True,
        )
        out_file = tmp_path / "sprayPropsGCM_mixture_posf10264.inp"
        content = out_file.read_text()
        assert "POSF10264" in content

    def test_mp_model_with_antoine_coeffs(self, decane_fuel, tmp_path):
        pele.export_pele(
            decane_fuel,
            path=str(tmp_path),
            liq_prop_model="mp",
            use_pp_keys=False,
            psat_antoine=True,
        )
        out_file = tmp_path / "sprayPropsMP_decane.inp"
        content = out_file.read_text()
        assert "particles.fuel_ref_temp" in content
        assert "_psat" in content

    def test_mp_model_without_antoine_coeffs(self, decane_fuel, tmp_path):
        pele.export_pele(
            decane_fuel,
            path=str(tmp_path),
            liq_prop_model="mp",
            use_pp_keys=False,
            psat_antoine=False,
        )
        out_file = tmp_path / "sprayPropsMP_decane.inp"
        content = out_file.read_text()
        assert "_psat" not in content

    def test_dep_fuel_names_broadcast_single_value(self, posf_fuel, tmp_path):
        pele.export_pele(
            posf_fuel,
            path=str(tmp_path),
            dep_fuel_names=["single-dep"],
            use_pp_keys=False,
        )
        out_file = tmp_path / "sprayPropsGCM_posf10264.inp"
        content = out_file.read_text()
        assert "particles.dep_fuel_species" in content

    def test_cgs_units(self, decane_fuel, tmp_path):
        pele.export_pele(
            decane_fuel,
            path=str(tmp_path),
            units="cgs",
            use_pp_keys=False,
        )
        out_file = tmp_path / "sprayPropsGCM_decane.inp"
        assert out_file.exists()


class TestMain:
    def test_main_exports_gcm_properties(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "fl-export-pele",
                "-f",
                "decane",
                "-o",
                str(tmp_path),
                "-pp",
                "false",
            ],
        )
        pele.main()
        assert (tmp_path / "sprayPropsGCM_decane.inp").exists()
