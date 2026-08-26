"""Tests for the transport property CLI converter."""

import pytest

from fuellib.cli.transport_props_converter import epsilon_to_kelvin_main
from fuellib.utils.convert import epsilon_to_characteristic_temperature


class TestEpsilonToKelvinMain:
    def test_prints_characteristic_temperature(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["fl-eps2K", "1500.0"])
        epsilon_to_kelvin_main()

        expected = epsilon_to_characteristic_temperature(1500.0)
        captured = capsys.readouterr()
        assert f"{expected:.3f} K" in captured.out

    def test_requires_epsilon_argument(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["fl-eps2K"])
        with pytest.raises(SystemExit):
            epsilon_to_kelvin_main()
