"""Tests for the data locator module."""

import fuellib as fl
from fuellib import _data_locator


class TestDataDirs:
    def test_get_data_dir_exists(self):
        assert fl.get_data_dir().endswith("data")

    def test_get_gcmtable_dir_is_under_data_dir(self):
        assert fl.get_gcmtable_dir().startswith(fl.get_data_dir())
        assert fl.get_gcmtable_dir().endswith("gcmTableData")

    def test_get_fueldata_dir_is_under_data_dir(self):
        assert fl.get_fueldata_dir().startswith(fl.get_data_dir())
        assert fl.get_fueldata_dir().endswith("fuel")


class TestFuelDataPropsDir:
    def test_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_data_locator, "get_fueldata_dir", lambda: str(tmp_path))
        assert _data_locator.get_fueldata_props_dir() is None

    def test_returns_path_when_present(self, tmp_path, monkeypatch):
        props_dir = tmp_path / "propertiesData"
        props_dir.mkdir()
        monkeypatch.setattr(_data_locator, "get_fueldata_dir", lambda: str(tmp_path))
        assert _data_locator.get_fueldata_props_dir() == str(props_dir)


class TestListFuelNames:
    def test_lists_json_stems_from_custom_dir(self, tmp_path):
        (tmp_path / "decane.json").write_text("{}")
        (tmp_path / "heptane.json").write_text("{}")
        (tmp_path / "notes.txt").write_text("")

        assert _data_locator.list_fuel_names(str(tmp_path)) == ["decane", "heptane"]

    def test_defaults_to_embedded_fuel_dir(self):
        names = _data_locator.list_fuel_names()
        assert "decane" in names
