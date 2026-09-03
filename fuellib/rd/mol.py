"""RDKit Mol module."""

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdDistGeom import EmbedMolecule

from ..element import number


def from_smiles(smiles: str, *, with_coords: bool = False) -> Mol:
    """
    Create a molecule from a SMILES string.

    :param smiles: SMILES string representing the molecule.
    :type smiles: str
    :param with_coords: Whether to generate 3D coordinates for the molecule.
    :type with_coords: bool
    :return: RDKit molecule object.
    :rtype: rdkit.Chem.rdchem.Mol
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # Add hydrogens to the molecule
    if with_coords:
        add_coordinates(mol, in_place=True)

    return mol


def smiles(mol: Mol) -> str:
    """
    Get the SMILES representation of a molecule.

    :param mol: RDKit molecule object.
    :type mol: rdkit.Chem.rdchem.Mol
    :return: SMILES string representing the molecule.
    :rtype: str
    """
    return Chem.MolToSmiles(mol)


def from_inchi(inchi: str, *, with_coords: bool = False) -> Mol:
    """
    Create a molecule from an InChI string.

    :param inchi: InChI string representing the molecule.
    :type inchi: str
    :param with_coords: Whether to generate 3D coordinates for the molecule.
    :type with_coords: bool
    :return: RDKit molecule object.
    :rtype: rdkit.Chem.rdchem.Mol
    """
    mol = Chem.MolFromInchi(inchi, sanitize=False, removeHs=False)
    mol = Chem.AddHs(mol)  # Add hydrogens to the molecule
    if with_coords:
        add_coordinates(mol, in_place=True)

    return mol


def inchi(mol: Mol) -> str:
    """
    Get the InChI representation of a molecule.

    :param mol: RDKit molecule object.
    :type mol: rdkit.Chem.rdchem.Mol
    :return: InChI string representing the molecule.
    :rtype: str
    """
    molblock = Chem.rdmolfiles.MolToMolBlock(mol)
    return Chem.inchi.MolBlockToInchi(molblock)


def has_coordinates(mol: Mol) -> bool:
    """
    Check if a molecule has 3D coordinates.

    :param mol: RDKit molecule object.
    :type mol: rdkit.Chem.rdchem.Mol
    :return: True if the molecule has 3D coordinates, False otherwise.
    :rtype: bool
    """
    return bool(mol.GetNumConformers())


def add_coordinates(mol: Mol, *, in_place: bool = False) -> Mol:
    """
    Generate 3D coordinates for a molecule.

    :param mol: RDKit molecule object.
    :type mol: rdkit.Chem.rdchem.Mol
    :param in_place: Whether to modify the molecule in place or return a new one.
    :type in_place: bool
    :return: RDKit molecule object with 3D coordinates.
    :rtype: rdkit.Chem.rdchem.Mol
    """
    if has_coordinates(mol):
        return mol

    mol = mol if in_place else Mol(mol)  # Create a copy if not modifying in place
    EmbedMolecule(mol)  # Generate 3D coordinates
    return mol


def count_element(mol: Mol, element: str | int) -> int:
    """
    Count the number of atoms of a specific element in a molecule.

    :param mol: RDKit molecule object.
    :type mol: rdkit.Chem.rdchem.Mol
    :param element: Element symbol (str) or atomic number (int).
    :type element: str | int
    :return: Number of atoms of the specified element in the molecule.
    :rtype: int
    """
    z = number(element)  # Convert to atomic number if needed
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == z)


def is_hydrocarbon(mol: Mol) -> bool:
    """
    Check if a molecule is a hydrocarbon.

    :param mol: RDKit molecule object.
    :type mol: rdkit.Chem.rdchem.Mol
    :return: True if the molecule is a hydrocarbon, False otherwise.
    :rtype: bool
    """
    return all(atom.GetAtomicNum() in (1, 6) for atom in mol.GetAtoms())


def has_aromatic(mol: Mol) -> bool:
    """
    Check if a molecule is aromatic.

    :param mol: RDKit molecule object.
    :type mol: rdkit.Chem.rdchem.Mol
    :return: True if the molecule is aromatic, False otherwise.
    :rtype: bool
    """
    return any(atom.GetIsAromatic() for atom in mol.GetAtoms())


def has_ring(mol: Mol) -> bool:
    """
    Check if a molecule contains any rings.

    :param mol: RDKit molecule object.
    :type mol: rdkit.Chem.rdchem.Mol
    :return: True if the molecule contains rings, False otherwise.
    :rtype: bool
    """
    return mol.GetRingInfo().NumRings() > 0


def has_branch(mol: Mol) -> bool:
    """
    Check if a molecule is branched.

    :param mol: RDKit molecule object.
    :type mol: rdkit.Chem.rdchem.Mol
    :return: True if the molecule is branched, False otherwise.
    :rtype: bool
    """
    mol = Mol(mol)  # Create a copy to avoid modifying the original molecule
    mol = Chem.RemoveAllHs(mol)
    return any(atom.GetDegree() > 2 for atom in mol.GetAtoms())


def has_double_bond(mol: Mol) -> bool:
    """
    Check if a molecule contains any double bonds.

    :param mol: RDKit molecule object.
    :type mol: rdkit.Chem.rdchem.Mol
    :return: True if the molecule contains double bonds, False otherwise.
    :rtype: bool
    """
    return any(
        bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
        if not bond.GetIsAromatic()
        else False
        for bond in mol.GetBonds()
    )
