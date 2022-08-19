from rdkit import Chem
from rdkit.Chem.Lipinski import RotatableBondSmarts, NumRotatableBonds

from library.molfile.ligfile_parser import load_structure_file


def find_bond_groups(mol):
    """Find groups of contiguous rotatable bonds and return them sorted by decreasing size"""

    if type(mol) == str:    # if the input is a SMILES string
        mol = Chem.MolFromSmiles(mol)

    rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
    rot_bond_set = set([mol.GetBondBetweenAtoms(*ap).GetIdx() for ap in rot_atom_pairs])
    rot_bond_groups = []
    while (rot_bond_set):
        i = rot_bond_set.pop()
        connected_bond_set = set([i])
        stack = [i]
        while (stack):
            i = stack.pop()
            b = mol.GetBondWithIdx(i)
            bonds = []
            for a in (b.GetBeginAtom(), b.GetEndAtom()):
                bonds.extend([b.GetIdx() for b in a.GetBonds() if (
                    (b.GetIdx() in rot_bond_set) and (not (b.GetIdx() in connected_bond_set)))])
            connected_bond_set.update(bonds)
            stack.extend(bonds)
        rot_bond_set.difference_update(connected_bond_set)
        rot_bond_groups.append(tuple(connected_bond_set))
    return tuple(sorted(rot_bond_groups, reverse = True, key = lambda x: len(x)))


def create_rotbond_featvec(bond_groups, max_rotbond_count=50):
    """
    Creates a feature vector of length max_rotbond_count+1 (+1 for 0 contiguous rotbonds) which carries information
    about the size of contiguous rotatables bonds and their count in this molecule.
    :param bond_groups:
    :param max_rotbond_count:
    :return featvec: max_rotbond_count + 1 integers
    """

    featvec = [0]*(max_rotbond_count + 1)    # +1 to include 0 rotbonds
    for group in bond_groups:
        featvec[len(group)] += 1
    if len(bond_groups) == 0:
        featvec[0] += 1
    return featvec


def create_rotbond_featvec_from_mol(max_rotbond_count=50, include_numrotbonds=False):
    """
    Helper method to create the contiguous rotbond featvec from an RDKit Mol object.
    :param mol:
    :param max_rotbond_count:
    TODO: finish include_numrotbonds
    :return:
    """
    def _create_rotbond_featvec_from_mol(mol):

        return create_rotbond_featvec(find_bond_groups(mol), max_rotbond_count=max_rotbond_count)

    return _create_rotbond_featvec_from_mol
