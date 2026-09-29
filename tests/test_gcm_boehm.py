"""Test cases for the Boehm (2022) group contribution parameters module."""

from types import SimpleNamespace
from typing import cast

import pytest

from fuellib.fuel import Fuel
from fuellib.gcm.boehm import TABLE, _identify_families
from fuellib.rdk import mol


class TestIdentifyFamilies:
    """Test suite for the `_identify_families` helper function."""

    @pytest.mark.parametrize(
        "smiles, expected_family",
        [
            ("CCCCCCC", "n-alkane"),  # heptane, straight chain
            ("CC(C)CCCC", "iso-alkane"),  # 2-methylhexane, branched
            ("CCCCC=C", "alkene"),  # 1-hexene, unsaturated
            ("CC1CCCCC1", "monocyclic"),  # methylcyclohexane, single ring
            ("C1CCC2CCCCC2C1", "dicyclic"),  # decalin, fused two-ring
            ("C1CC2CCC3CCCC3C2C1", "tricyclic"),  # fused three-ring
            ("Cc1ccccc1", "alkylbenzene"),  # toluene, alkyl-substituted benzene
            ("c1ccc2c(c1)CCCC2", "cycloaromatic"),  # tetralin, fused cyclic-aromatic
            ("c1ccc2ccccc2c1", "diaromatic"),  # naphthalene, fused two-ring aromatic
        ],
    )
    def test_identify_families_classifies_known_smiles(
        self, smiles: str, expected_family: str
    ) -> None:
        """_identify_families should classify representative molecules correctly."""
        fake_fuel = cast(Fuel, SimpleNamespace(rdkit_mols=[mol.from_smiles(smiles)]))
        result = _identify_families(fake_fuel)
        assert result == [expected_family]

    def test_identify_families_handles_multiple_compounds_in_order(self) -> None:
        """_identify_families should return one family per compound, in order."""
        smiles_list = ["CCCCCCC", "Cc1ccccc1", "c1ccc2ccccc2c1"]
        fake_fuel = cast(
            Fuel,
            SimpleNamespace(rdkit_mols=[mol.from_smiles(s) for s in smiles_list]),
        )
        result = _identify_families(fake_fuel)
        assert result == ["n-alkane", "alkylbenzene", "diaromatic"]

    def test_identify_families_raises_for_unsupported_fused_aromatic_system(
        self,
    ) -> None:
        """_identify_families should raise ValueError for a fused three-ring aromatic system."""
        anthracene = mol.from_smiles("c1ccc2cc3ccccc3cc2c1")
        fake_fuel = cast(Fuel, SimpleNamespace(rdkit_mols=[anthracene]))
        with pytest.raises(ValueError):
            _identify_families(fake_fuel)

    def test_identify_families_raises_for_unsupported_fused_ring_size(self) -> None:
        """_identify_families should raise ValueError for a fused ring system of unsupported size."""
        steroid_core = mol.from_smiles("C1CCC2C(C1)CCC3C2CCC4C3CCCC4")
        fake_fuel = cast(Fuel, SimpleNamespace(rdkit_mols=[steroid_core]))
        with pytest.raises(ValueError):
            _identify_families(fake_fuel)
