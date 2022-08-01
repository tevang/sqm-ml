from rdkit import Chem


def compute_terminal_CH3_count(mol):
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CH3]")))