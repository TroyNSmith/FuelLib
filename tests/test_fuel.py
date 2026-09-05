"""Tests for the Fuel module in the FuelLib library."""

from pathlib import Path

import pytest
import quaxed.numpy as qnp
from unxt import unit

from fuellib.fuel import Fuel
from fuellib.utils.units import strip

from .data.answers import fuel as fuel_answers

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def jet_a() -> Fuel:
    """Fixture for the Jet-A fuel instance."""
    return Fuel(name="jet-a", decompName="refCompounds", fuelDataDir=DATA_DIR)


class TestFuelInit:
    """Tests for the initialization of the Fuel class in the FuelLib library."""

    def test_initialization(self, jet_a: Fuel) -> None:
        """Test that the Fuel class initializes correctly."""
        num_compounds = 67
        num_gani_groups = 121

        assert jet_a.name == "jet-a"
        assert len(jet_a.compounds) == num_compounds
        assert len(jet_a.smiles) == num_compounds
        # We are assuming that gani_decomp correctly reformats the decomposition matrix.
        assert jet_a.gani_decomp.shape == (num_compounds, num_gani_groups)
        # This .csv file contains PelePhysics keys for each compound.
        assert jet_a.pelephysics_keys is not None
        assert len(jet_a.pelephysics_keys) == num_compounds
        assert len(jet_a.formulas) == num_compounds

    def test_initialization_with_no_name(self) -> None:
        """Test that the Fuel class fails to initialize correctly when no name is provided."""
        with pytest.raises(TypeError):
            Fuel(decompName="refCompounds", fuelDataDir=DATA_DIR)  # ty: ignore[missing-argument]

    def test_initialization_with_name_not_found(self) -> None:
        """Test that the Fuel class fails to initialize correctly when the name is not found."""
        with pytest.raises(FileNotFoundError):
            Fuel(name="non-existent", decompName="refCompounds", fuelDataDir=DATA_DIR)

    def test_initialization_with_no_compounds(self) -> None:
        """Test that the Fuel class fails to initialize correctly when there are no compounds."""
        with pytest.raises(KeyError, match="No compounds found"):
            Fuel(name="no-compound", decompName="refCompounds", fuelDataDir=DATA_DIR)

    def test_initialization_with_no_smiles(self) -> None:
        """Test that the Fuel class fails to initialize correctly when there are no SMILES strings."""
        with pytest.raises(KeyError, match="No SMILES strings found"):
            Fuel(name="no-smiles", decompName="refCompounds", fuelDataDir=DATA_DIR)

    def test_initialization_with_no_weight_percentages(self) -> None:
        """Test that the Fuel class fails to initialize correctly when there are no weight percentages."""
        with pytest.raises(KeyError, match="No weight percentages found"):
            Fuel(name="no-weights", decompName="refCompounds", fuelDataDir=DATA_DIR)

    def test_initialization_with_no_pelephysics_keys(self) -> None:
        """Test that the Fuel class initializes correctly when there are no PelePhysics keys."""
        fuel = Fuel(
            name="no-pele-keys", decompName="refCompounds", fuelDataDir=DATA_DIR
        )
        assert fuel.pelephysics_keys is None


class TestFuelPropertyEstimations:
    """Test the properties of the Fuel class against previous estimations."""

    def test_formulas_correctly_initialized(self, jet_a: Fuel) -> None:
        """Test that the Hill formulas are correctly determined."""
        assert jet_a.formulas == fuel_answers.JetAProperties.formulas

    def test_molecular_weights_correctly_initialized(self, jet_a: Fuel) -> None:
        """Test that the molecular weights are correctly determined."""
        assert jet_a.MW.unit == unit("g/mol")
        assert qnp.allclose(strip(jet_a.MW), fuel_answers.JetAProperties.MW)

    def test_num_carbons_correctly_initialized(self, jet_a: Fuel) -> None:
        """Test that the number of carbon atoms is correctly determined."""
        assert qnp.all(jet_a.num_carbons == fuel_answers.JetAProperties.num_carbons)

    def test_num_hydrogens_correctly_initialized(self, jet_a: Fuel) -> None:
        """Test that the number of hydrogen atoms is correctly determined."""
        assert qnp.all(jet_a.num_hydrogens == fuel_answers.JetAProperties.num_hydrogens)

    def test_hydrocarbon_correctly_initialized(self, jet_a: Fuel) -> None:
        """Test that the hydrocarbon flag is correctly determined."""
        assert qnp.all(jet_a._hydrocarbon == fuel_answers.JetAProperties.hydrocarbon)

    def test_aromatic_correctly_initialized(self, jet_a: Fuel) -> None:
        """Test that the aromatic flag is correctly determined."""
        assert qnp.all(jet_a._aromatic == fuel_answers.JetAProperties.aromatic)

    def test_cyclic_correctly_initialized(self, jet_a: Fuel) -> None:
        """Test that the cyclic flag is correctly determined."""
        assert qnp.all(jet_a._cyclic == fuel_answers.JetAProperties.cyclic)

    def test_branched_correctly_initialized(self, jet_a: Fuel) -> None:
        """Test that the branched flag is correctly determined."""
        assert qnp.all(jet_a._branched == fuel_answers.JetAProperties.branched)

    def test_alkene_correctly_initialized(self, jet_a: Fuel) -> None:
        """Test that the alkene flag is correctly determined."""
        assert qnp.all(jet_a._alkene == fuel_answers.JetAProperties.alkene)

    def test_hydrocarbon_types_correctly_initialized(self, jet_a: Fuel) -> None:
        """Test that the hydrocarbon types are correctly determined."""
        assert (
            list(jet_a.hydrocarbon_types)
            == fuel_answers.JetAProperties.hydrocarbon_types
        )

    def test_Lv_stp_correctly_initialized(self, jet_a: Fuel) -> None:
        """Test that the Lv_stp values are correctly determined."""
        assert qnp.allclose(strip(jet_a.Lv_stp), fuel_answers.JetAProperties.Lv_stp)

    def test_sigma_correctly_initialized(self, jet_a: Fuel) -> None:
        """Test that the sigma values are correctly determined."""
        assert qnp.allclose(strip(jet_a.sigma), fuel_answers.JetAProperties.sigma)

    def test_epsilonByKB_correctly_initialized(self, jet_a: Fuel) -> None:
        """Test that the epsilonByKB values are correctly determined."""
        assert qnp.allclose(
            strip(jet_a.epsilonByKB), fuel_answers.JetAProperties.epsilonByKB
        )
