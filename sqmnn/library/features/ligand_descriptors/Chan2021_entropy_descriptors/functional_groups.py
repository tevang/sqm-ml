from rdkit import Chem


def GetCyclicAmide(mol):
    """

    Return:

    count: int
    """
    amide = Chem.MolFromSmarts("[CX3](=O)@[NH]")
    matches = mol.GetSubstructMatches(amide)
    return matches


def GetCyclicEster(mol):
    """

    Return:

    count: int
    """
    ester = Chem.MolFromSmarts("[CX3](=O)@[O]")
    matches = mol.GetSubstructMatches(ester)
    return matches


def GetCyclicThioamide(mol):
    """
    Return:

    count: int
    """
    thioamide = Chem.MolFromSmarts("[CX3](=[SX1])@[NH]")
    matches = mol.GetSubstructMatches(thioamide)
    return matches


def compute_functional_group_count(mol):
    return len(GetCyclicThioamide(mol)) + len(GetCyclicAmide(mol)) + len(GetCyclicEster(mol))

