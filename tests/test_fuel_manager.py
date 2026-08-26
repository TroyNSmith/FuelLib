"""Tests for the fl-fuels CLI (fuel_manager)."""

import pytest

import fuellib as fl
from fuellib.cli import fuel_manager


@pytest.fixture
def fuel_dir(tmp_path):
    (tmp_path / "decane.json").write_text("{}")
    (tmp_path / "heptane.json").write_text("{}")
    return tmp_path


@pytest.fixture
def fuel_dir_with_metadata(fuel_dir):
    (fuel_dir / "fuel_metadata.yaml").write_text(
        "fuels:\n"
        "  decane:\n"
        "    category: Conventional\n"
        "    source: Some Source\n"
        "    description: A test note\n"
    )
    return fuel_dir


class TestLoadFuelMetadata:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        assert fuel_manager.load_fuel_metadata(str(tmp_path)) == {}

    def test_loads_fuels_key_from_yaml(self, fuel_dir_with_metadata):
        metadata = fuel_manager.load_fuel_metadata(str(fuel_dir_with_metadata))
        assert metadata["decane"]["source"] == "Some Source"

    def test_none_dir_loads_embedded_metadata(self):
        metadata = fuel_manager.load_fuel_metadata(None)
        assert "posf10264" in metadata

    def test_returns_empty_dict_when_yaml_unavailable(
        self, monkeypatch, fuel_dir_with_metadata
    ):
        monkeypatch.setattr(fuel_manager, "HAS_YAML", False)
        assert fuel_manager.load_fuel_metadata(str(fuel_dir_with_metadata)) == {}

    def test_malformed_yaml_warns_and_returns_empty_dict(self, tmp_path):
        (tmp_path / "fuel_metadata.yaml").write_text("fuels: [unterminated")
        with pytest.warns(UserWarning, match="Failed to parse fuel metadata"):
            assert fuel_manager.load_fuel_metadata(str(tmp_path)) == {}

    def test_unreadable_file_warns_and_returns_empty_dict(self, tmp_path, monkeypatch):
        (tmp_path / "fuel_metadata.yaml").write_text("fuels: {}")

        def _raise_os_error(*args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr("builtins.open", _raise_os_error)
        with pytest.warns(UserWarning, match="Failed to read fuel metadata"):
            assert fuel_manager.load_fuel_metadata(str(tmp_path)) == {}


class TestListFuelsMain:
    def test_default_embedded_dir_lists_fuels(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["fl-fuels"])
        fuel_manager.list_fuels_main()
        captured = capsys.readouterr()
        assert "decane" in captured.out
        assert "Available Fuels in FuelLib" in captured.out

    def test_custom_dir_simple_listing_shows_source(
        self, monkeypatch, capsys, fuel_dir_with_metadata
    ):
        monkeypatch.setattr(
            "sys.argv", ["fl-fuels", "-dir", str(fuel_dir_with_metadata)]
        )
        fuel_manager.list_fuels_main()
        captured = capsys.readouterr()
        assert "decane" in captured.out
        assert "[Some Source]" in captured.out
        assert "heptane" in captured.out

    def test_custom_dir_verbose_shows_metadata_fields(
        self, monkeypatch, capsys, fuel_dir_with_metadata
    ):
        monkeypatch.setattr(
            "sys.argv",
            ["fl-fuels", "-dir", str(fuel_dir_with_metadata), "--verbose"],
        )
        fuel_manager.list_fuels_main()
        captured = capsys.readouterr()
        assert "Category:      Conventional" in captured.out
        assert "Source:        Some Source" in captured.out
        assert "Note:          A test note" in captured.out

    def test_custom_dir_without_metadata_omits_brackets(
        self, monkeypatch, capsys, fuel_dir
    ):
        monkeypatch.setattr("sys.argv", ["fl-fuels", "-dir", str(fuel_dir)])
        fuel_manager.list_fuels_main()
        captured = capsys.readouterr()
        assert "decane" in captured.out
        assert "[" not in captured.out

    def test_missing_directory_exits_with_error(self, monkeypatch, capsys, tmp_path):
        missing_dir = tmp_path / "does-not-exist"
        monkeypatch.setattr("sys.argv", ["fl-fuels", "-dir", str(missing_dir)])
        with pytest.raises(SystemExit) as exc_info:
            fuel_manager.list_fuels_main()
        assert exc_info.value.code == 1
        assert "Fuel data directory not found" in capsys.readouterr().out

    def test_empty_directory_exits_cleanly(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr("sys.argv", ["fl-fuels", "-dir", str(tmp_path)])
        with pytest.raises(SystemExit) as exc_info:
            fuel_manager.list_fuels_main()
        assert exc_info.value.code == 0
        assert "No fuels found" in capsys.readouterr().out

    def test_unexpected_error_exits_with_message(self, monkeypatch, capsys, fuel_dir):
        monkeypatch.setattr("sys.argv", ["fl-fuels", "-dir", str(fuel_dir)])

        def _raise(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(fl, "list_fuel_names", _raise)
        with pytest.raises(SystemExit) as exc_info:
            fuel_manager.list_fuels_main()
        assert exc_info.value.code == 1
        assert "Error listing fuels: boom" in capsys.readouterr().out
