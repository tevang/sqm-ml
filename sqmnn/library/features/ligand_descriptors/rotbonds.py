from rdkit import Chem
from rdkit.Chem.Lipinski import RotatableBondSmarts, NumRotatableBonds

from lib.molfile.ligfile_parser import load_structure_file


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


def create_rotbond_featvec_from_molfile(molfile, max_rotbond_count=50, include_numrotbonds=False):
    """
    Helper method to create the contiguous rotbond featvecs from a molfile containing one or more molecules.
    :param mol:
    :param max_rotbond_count:
    :return molname_rotbond_featvec_dict: a dict molname->featvec
    """
    molname_SMI_conf_mdict = load_structure_file(molfile, keep_structvar=True, get_SMILES=False, addHs=False)
    molname_rotbond_featvec_dict = {}
    for molname in molname_SMI_conf_mdict.keys():
        mol = molname_SMI_conf_mdict[molname]['SMI']
        molname_rotbond_featvec_dict[molname] = create_rotbond_featvec(find_bond_groups(mol),
                                                                       max_rotbond_count=max_rotbond_count)
        if include_numrotbonds:
            num_rotbonds = NumRotatableBonds(mol)   # a stricter definition of rotable bonds is used
                                                    # this excludes amides, esters, etc.
            molname_rotbond_featvec_dict[molname].insert(0, num_rotbonds)   # append at the beginning the number of rotbonds
    return molname_rotbond_featvec_dict


def get_num_rotbonds_from_molfile(molfile):
    """
    Method to computer the number of rotatable bonds for all the molecules within a file using the Strict
    criteria, which exclude amides, esters, etc.
    :param molfile:
    :return molname_num_rotbonds_dict:  dict molname->num_rotbonds
    """
    molname_SMI_conf_mdict = load_structure_file(molfile, keep_structvar=True, get_SMILES=False, addHs=False)
    molname_num_rotbonds_dict = {}
    for molname in molname_SMI_conf_mdict.keys():
        mol = molname_SMI_conf_mdict[molname]['SMI']
        molname_num_rotbonds_dict[molname] = NumRotatableBonds(mol)  # a stricter definition of rotable bonds
                                                                    # is used this excludes amides, esters, etc.
    return molname_num_rotbonds_dict