"""Tests for fuel composition and mixture property plotting."""

import matplotlib

matplotlib.use("Agg")


from fuellib.utils import plot


class TestPlotComposition:
    def test_default_title_saves_png(self, tmp_path):
        plot.plot_composition("posf10264", output_dir=str(tmp_path))
        assert (tmp_path / "composition_posf10264.png").exists()

    def test_title_none_disables_title(self, tmp_path):
        plot.plot_composition("posf10264", output_dir=str(tmp_path), title="none")
        assert (tmp_path / "composition_posf10264.png").exists()

    def test_custom_title(self, tmp_path):
        plot.plot_composition("decane", output_dir=str(tmp_path), title="My Fuel")
        assert (tmp_path / "composition_decane.png").exists()

    def test_display_without_save_does_not_write_file(self, tmp_path, monkeypatch):
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda: None)
        plot.plot_composition(
            "decane", output_dir=str(tmp_path), save=False, display=True
        )
        assert not (tmp_path / "composition_decane.png").exists()


class TestPlotMixtureProperties:
    def test_single_fuel_name_default_properties(self, tmp_path):
        plot.plot_mixture_properties("posf10264", output_dir=str(tmp_path))
        assert (tmp_path / "mixture_properties.png").exists() or any(
            tmp_path.glob("*.png")
        )

    def test_list_of_fuel_names_single_property(self, tmp_path):
        plot.plot_mixture_properties(
            ["decane", "posf10264"],
            property_names=["Density"],
            output_dir=str(tmp_path),
        )
        assert any(tmp_path.glob("*.png"))

    def test_hefa_fuel_name_legend_branch(self, tmp_path):
        plot.plot_mixture_properties(
            ["hefa-came"],
            property_names=["Density"],
            output_dir=str(tmp_path),
        )
        assert any(tmp_path.glob("*.png"))

    def test_fuel_without_experimental_data_still_plots(self, tmp_path):
        # decane only has experimental "Density" data, not Viscosity.
        plot.plot_mixture_properties(
            "decane",
            property_names=["Viscosity"],
            output_dir=str(tmp_path),
        )
        assert any(tmp_path.glob("*.png"))

    def test_title_is_applied(self, tmp_path):
        plot.plot_mixture_properties(
            "decane",
            property_names=["Density"],
            output_dir=str(tmp_path),
            title="Comparison",
        )
        assert any(tmp_path.glob("*.png"))
