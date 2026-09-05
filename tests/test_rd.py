"""Tests for the RDKit module in FuelLib."""

import pytest
from rdkit.Chem import Mol

from fuellib import rd


@pytest.fixture
def water():
    return rd.mol.from_smiles("O")


@pytest.fixture
def ethane():
    return rd.mol.from_smiles("CC")


@pytest.fixture
def ethene():
    return rd.mol.from_smiles("C=C")


@pytest.fixture
def isobutane():
    return rd.mol.from_smiles("CC(C)C")


@pytest.fixture
def toluene():
    return rd.mol.from_smiles("Cc1ccccc1")


class TestMol:
    """Tests for the Mol class in the RDKit module of FuelLib."""

    def test_mol_from_smiles(self) -> None:
        """Test the creation of a Mol object from a SMILES string."""
        mol = rd.mol.from_smiles("CC")
        assert sorted([a.GetSymbol() for a in mol.GetAtoms()]) == ["C"] * 2 + ["H"] * 6

    def test_mol_to_smiles(self, ethane: Mol) -> None:
        """Test the conversion of a Mol object to a SMILES string."""
        smiles = rd.mol.smiles(ethane)
        assert smiles == "CC"

    def test_mol_from_inchi(self) -> None:
        """Test the creation of a Mol object from an InChI string."""
        inchi = "InChI=1S/C2H6/c1-2/h1-2H3"
        mol = rd.mol.from_inchi(inchi)
        assert sorted([a.GetSymbol() for a in mol.GetAtoms()]) == ["C"] * 2 + ["H"] * 6

    def test_mol_to_inchi(self, ethane: Mol) -> None:
        """Test the conversion of a Mol object to an InChI string."""
        inchi = rd.mol.inchi(ethane)
        assert inchi == "InChI=1S/C2H6/c1-2/h1-2H3"

    def test_mol_to_hill_formula(self, ethane: Mol) -> None:
        """Test the conversion of a Mol object to its Hill formula."""
        formula = rd.mol.hill_formula(ethane)
        assert formula == "C2H6"

    def test_has_coordinates(self, ethane: Mol) -> None:
        """Test whether a Mol object has 3D coordinates."""
        assert not rd.mol.has_coordinates(ethane)

    def test_add_coordinates(self, ethane: Mol) -> None:
        """Test adding 3D coordinates to a Mol object.

        Additionally covers `has_coordinates` method where evaluation
        is `True`.
        """
        ethane = rd.mol.add_coordinates(ethane)
        assert rd.mol.has_coordinates(ethane)

    def test_count_element(self, ethane: Mol) -> None:
        """Test counting the number of a specific element in a Mol object."""
        assert rd.mol.count_element(ethane, "C") == 2
        assert rd.mol.count_element(ethane, "H") == 6
        assert rd.mol.count_element(ethane, "O") == 0

    def test_is_hydrocarbon(self, ethane: Mol) -> None:
        """Test whether a Mol object is a hydrocarbon."""
        assert rd.mol.is_hydrocarbon(ethane)

    def test_is_not_hydrocarbon(self, water: Mol) -> None:
        """Test whether a Mol object is not a hydrocarbon."""
        assert not rd.mol.is_hydrocarbon(water)

    def test_has_aromatic(self, toluene: Mol) -> None:
        """Test whether a Mol object has aromatic rings."""
        assert rd.mol.has_aromatic(toluene)

    def test_has_no_aromatic(self, ethane: Mol) -> None:
        """Test whether a Mol object has no aromatic rings."""
        assert not rd.mol.has_aromatic(ethane)

    def test_has_ring(self, toluene: Mol) -> None:
        """Test whether a Mol object has rings."""
        assert rd.mol.has_ring(toluene)

    def test_has_no_ring(self, ethane: Mol) -> None:
        """Test whether a Mol object has no rings."""
        assert not rd.mol.has_ring(ethane)

    def test_has_branch(self, isobutane: Mol) -> None:
        """Test whether a Mol object has branches."""
        assert rd.mol.has_branch(isobutane)

    def test_has_no_branch(self, ethane: Mol) -> None:
        """Test whether a Mol object has no branches."""
        assert not rd.mol.has_branch(ethane)

    def test_has_alkene_bond(self, ethene: Mol) -> None:
        """Test whether a Mol object has alkene bonds."""
        assert rd.mol.has_alkene_bond(ethene)

    def test_has_no_alkene_bond(self, ethane: Mol) -> None:
        """Test whether a Mol object has no alkene bonds."""
        assert not rd.mol.has_alkene_bond(ethane)

    def test_molecular_weight(self, ethane: Mol) -> None:
        """Test calculating the molecular weight of a Mol object."""
        assert rd.mol.molecular_weight(ethane) == pytest.approx(30.07, rel=1e-2)
