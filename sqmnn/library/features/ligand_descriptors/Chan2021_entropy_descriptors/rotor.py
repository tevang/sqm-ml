from rdkit import Chem


def compute_rotor_count(mol):
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")))