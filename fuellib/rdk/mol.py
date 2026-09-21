"""RDKit Mol interface for FuelLib."""

from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, Mol


def from_smiles(smiles: str) -> Mol:
    """
    Instantiate an RDKit Mol object from a SMILES string.

    :param smiles: SMILES string representing the molecule.
    :type smiles: str
    :return: RDKit Mol object.
    :rtype: Mol
    """
    return Chem.MolFromSmiles(smiles)


def has_aromatic(mol: Mol) -> bool:
    """
    Check if an RDKit Mol object contains any aromatic atoms.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :return: True if the molecule contains any aromatic atoms, False otherwise.
    :rtype: bool
    """
    return any(atom.GetIsAromatic() for atom in mol.GetAtoms())


def has_ring(mol: Mol) -> bool:
    """
    Check if an RDKit Mol object contains any ring structures.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :return: True if the molecule contains any ring structures, False otherwise.
    :rtype: bool
    """
    return any(atom.IsInRing() for atom in mol.GetAtoms())


def has_branched(mol: Mol) -> bool:
    """
    Check if an RDKit Mol object contains any branched atoms.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :return: True if the molecule contains any branched atoms, False otherwise.
    :rtype: bool
    """
    mol = Chem.RemoveAllHs(mol)
    return any(atom.GetDegree() > 2 for atom in mol.GetAtoms())


def has_double_bond(mol: Mol) -> bool:
    """
    Check if an RDKit Mol object contains any double bonds.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :return: True if the molecule contains any double bonds, False otherwise.
    :rtype: bool
    """
    return any(
        bond.GetBondType() == Chem.rdchem.BondType.DOUBLE for bond in mol.GetBonds()
    )


def is_hydrocarbon(mol: Mol) -> bool:
    """
    Check if an RDKit Mol object represents a hydrocarbon.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :return: True if the molecule contains only carbon and hydrogen atoms, False otherwise.
    :rtype: bool
    """
    return all(atom.GetSymbol() in {"C", "H"} for atom in mol.GetAtoms())


def atom_counts(mol: Mol) -> dict[str, int]:
    """
    Count the number of each type of atom in an RDKit Mol object.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :return: Dictionary mapping atom symbols to their counts.
    :rtype: dict[str, int]
    """
    mol = Chem.AddHs(mol)
    return Counter(atom.GetSymbol() for atom in mol.GetAtoms())


def molecular_weight(mol: Mol, *, exact: bool = False) -> float:
    """
    Calculate the molecular weight of an RDKit Mol object.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :param exact: Whether to calculate the monoisotopic molecular weight.
    :type exact: bool
    :return: Molecular weight of the molecule in atomic mass units (amu).
    :rtype: float
    """
    if exact:
        return Descriptors.ExactMolWt(mol)  # ty: ignore[unresolved-attribute]
    return Descriptors.MolWt(mol)  # ty: ignore[unresolved-attribute]
