from rdkit.Chem import GetPeriodicTable
from library.chemlib.macrocycle import *
from library.utils.print_functions import ColorPrint


def get_all_property_vals(propname, *molname_SMILES_conf_mdicts):
    values = []
    for molname_SMILES_conf_mdict in molname_SMILES_conf_mdicts:
        for molname in molname_SMILES_conf_mdict.keys():
            for mol in molname_SMILES_conf_mdict[molname].values():
                if propname not in mol.GetPropNames():
                    ColorPrint("WARNING: molecule %s does not have property %s!" % (molname, propname),
                               "WARNING")
                    continue
                values.extend( get_mol_property_vals(mol, propname) )
    return np.array(values)

def is_mol_property_discretizable(mol, propname):
    if propname not in mol.GetPropNames():
        return False
    # If an atomic property
    if ',' in mol.GetProp(propname):
        for v in mol.GetProp(propname).split(','):
            if v in ['true', 'True', 'false', 'False']:
                continue
            try:
                float(v)
            except ValueError:
                return False

        return True
    else:   # If a molecular property
        if mol.GetProp(propname) in ['true', 'True', 'false', 'False']:
            return True
        try:
            float(mol.GetProp(propname))
        except ValueError:
            return False

        return True

def is_atomic_property(mol, propname):
    return ',' in mol.GetProp(propname)

def get_mol_property_vals(mol, propname):
    """
    Get this property's values of one molecule in the order of the atoms in the Mol.
    :param mol:
    :return:
    """
    if ',' in mol.GetProp(propname):
        return [float(c) for c in mol.GetProp(propname).split(',')]
    else:
        return [float(mol.GetProp(propname))]

class Discretize():
    """
    This class discretizes an iterable of scalar values and assigns a bin ID to each scalar value.
    """
    def __init__(self):
        self.bins = None

    def fit(self, X, bin_width=None, center_to=None):
        bin_num = 'auto'
        if bin_width != None:
            if center_to != None:
                self.bins = Discretize.centered_bins(X, bin_width=bin_width, center_to=center_to)
                ColorPrint("Discretizing this property to %i number of bins" % len(self.bins), "OKBLUE")
                return
            else:
                bin_num = np.arange(X.min(), X.max(), bin_width).size + 1
        self.bins = np.histogram_bin_edges(X, bins=bin_num)
        ColorPrint("Discretizing this property to %i number of bins" % len(self.bins), "OKBLUE")

    def fit_transform(self, X, bin_width=None, center_to=None):
        bin_num = 'auto'
        if bin_width != None:
            if center_to != None:
                self.bins = Discretize.centered_bins(X, bin_width=bin_width, center_to=center_to)
            else:
                bin_num = np.arange(X.min(), X.max(), bin_width).size + 1
        self.bins = np.histogram_bin_edges(X, bins=bin_num) # X is flattened, therefore it can't take multiple features
        return np.digitize(X, self.bins)

    def transform(self, X):
        return np.digitize(X, self.bins)

    def transform_mol(self, mol, propname):
        """
        Method to discretize the specified property of a Mol object, either molecular property or atomic property.
        :param mol:
        :param propname:
        :return:
        """
        try:
            discrete_vals = self.transform(get_mol_property_vals(mol, propname))
        except ValueError:
            ColorPrint("WARNING: failed to discretize property %s with value %s" %
                       (propname,), "WARNING")
            return mol
        if len(discrete_vals) == 1: # this is a molecular property
            mol.SetProp('discrete ' + propname, str(discrete_vals[0]))
        elif len(discrete_vals) > 1:    # this is an atomic property
            for atom, discrete_val in zip(list(mol.GetAtoms()), discrete_vals):
                atom.SetProp('discrete ' + propname, str(discrete_val))
        return mol

    @staticmethod
    def centered_bins(X, bin_width=None, center_to=None):
        """
        Creates histogram bins for vector X, which center at center_to and width bin_width
        :param X:
        :param bin_width:
        :param center_to:
        :return:
        """
        bins = [center_to]
        edge_max = max(X)
        edge_min = min(X)
        loc = center_to
        while edge_min <= loc:
            loc -= bin_width
            bins.append(loc)
        loc = center_to
        while loc <= edge_max:
            loc += bin_width
            bins.append(loc)
        bins.sort()
        return bins

def generateAtomInvariant(mol):
    """
    Method to generate the default ECFP atom invariants but also extra ones. ECFP default are:
    * atom number
    * total degree
    * total number of hydrogens
    * formal charge
    * mass difference between the mass of your isotope and the average atomic weight of the element
                                            (this will be 0 if you are not using a specific isotope)
    * the number of rings that the atom belongs to.
    On top of these 5 components, if includeRingMembership is true (the default) and the atom is part of one
    or more rings, there will be a 6th component equal to the number of rings that the atom belongs to.
    This 5 (or 6)-component vector is then hashed.

    CURRENTLY ONLY PARTIAL CHARGES ARE ADDED AS EXTRA INVARIANTS (IF PRESENT)!

    Some extra interesting atomic properties to add are:
    + GetExplicitValence
    + GetNumExplicitHs
    ++ GetHybridization (e.g. rdkit.Chem.rdchem.HybridizationType.SP2, needs to be converted to integer)
    +++ GetIsAromatic
    - IsInRing
    + IsInRingSize  # the if(ring_info.NumAtomRings(i)): is slightly better
    - NumAtomRings

    >>> generateAtomInvariant(Chem.MolFromSmiles("Cc1ncccc1"))
    [341294046, 3184205312, 522345510, 1545984525, 1545984525, 1545984525, 1545984525]

    :param mol:
    :return:
    """
    pt = GetPeriodicTable()
    num_atoms = mol.GetNumAtoms()
    invariants = [0]*num_atoms
    ring_info = mol.GetRingInfo()
    for i,a in enumerate(mol.GetAtoms()):
        descriptors=[]
        # DEFAULT ECFP DESCRIPTORS
        descriptors.append(a.GetAtomicNum())
        descriptors.append(a.GetTotalDegree())
        descriptors.append(a.GetTotalNumHs())
        descriptors.append(a.GetFormalCharge())
        descriptors.append(a.GetMass() - pt.GetAtomicWeight(a.GetSymbol()))
        if(ring_info.NumAtomRings(i)):    # better than a.IsInRing()
            descriptors.append(1)
        # EXTRA DESCRIPTORS
        # print("DEBUG: atomic properties:", list(a.GetPropNames()))
        # if(a.GetIsAromatic()):  # unclear if it is beneficial
        #     descriptors.append(1)
        # descriptors.append(ring_info.NumAtomRings(i)) # detrimental to the performance
        # if (Macrocycle.is_atom_in_macrocyclic_ring(mol=mol, atom=a)):
        #     descriptors.append(1)
        try:    # ADD PARTIAL CHARGES
            descriptors.append(int(a.GetProp('discrete partial charge')))
        except KeyError:
            pass
        # try:    # ADD CONFORMATIONAL ENTROPY OF THE WHOLE MOLECULE
        #     descriptors.append(int(mol.GetProp('discrete schrodinger_confS_Boltzmann_KT0.593')))
        # except KeyError:
        #     pass

        invariants[i]=hash(tuple(descriptors))& 0xffffffff
    return invariants

def generate_ECFP_Atom_Invariant(mol):
    """
    Method to generate the default ECFP atom invariants for validation.
    :param mol:
    :return:
    """
    pt = GetPeriodicTable()
    num_atoms = mol.GetNumAtoms()
    invariants = [0]*num_atoms
    ring_info = mol.GetRingInfo()
    for i,a in enumerate(mol.GetAtoms()):
        descriptors=[]
        descriptors.append(a.GetAtomicNum())
        descriptors.append(a.GetTotalDegree())
        descriptors.append(a.GetTotalNumHs())
        descriptors.append(a.GetFormalCharge())
        descriptors.append(a.GetMass() - pt.GetAtomicWeight(a.GetSymbol()))
        if(ring_info.NumAtomRings(i)):
            descriptors.append(1)
        invariants[i]=hash(tuple(descriptors))& 0xffffffff
    return invariants

def generate_CSFP_Atom_Invariant(mol):
    """
    Method to generate the extra atom invariants for the *CSFP class of fingerprints.
    :param mol:
    :return invariants: a dictionary with keys the position of each heavy atom and values list of integers.
    """
    pt = GetPeriodicTable()
    num_atoms = mol.GetNumAtoms()
    invariants = {}
    ring_info = mol.GetRingInfo()
    for i,a in enumerate(mol.GetAtoms()):
        descriptors=[]
        # # DEFAULT ECFP DESCRIPTORS
        # descriptors.append(a.GetAtomicNum())
        # descriptors.append(a.GetTotalDegree())
        # descriptors.append(a.GetTotalNumHs())
        # descriptors.append(a.GetFormalCharge())
        # descriptors.append(a.GetMass() - pt.GetAtomicWeight(a.GetSymbol()))
        # if(ring_info.NumAtomRings(i)):    # better than a.IsInRing()
        #     descriptors.append(1)
        # EXTRA DESCRIPTORS
        # print("DEBUG: atomic properties:", list(a.GetPropNames()))
        # if(a.GetIsAromatic()):  # unclear if it is beneficial
        #     descriptors.append(1)
        # descriptors.append(ring_info.NumAtomRings(i)) # detrimental to the performance
        # if (Macrocycle.is_atom_in_macrocyclic_ring(mol=mol, atom=a)):
        #     descriptors.append(1)
        try:    # ADD PARTIAL CHARGES
            descriptors.append(int(a.GetProp('discrete partial charge')))
        except KeyError:
            pass
        # try:    # ADD CONFORMATIONAL ENTROPY OF THE WHOLE MOLECULE
        #     descriptors.append(int(mol.GetProp('discrete schrodinger_confS_Boltzmann_KT0.593')))
        # except KeyError:
        #     pass

        invariants[i] = descriptors
    return invariants

def discretize_atomic_properties(*molname_SMILES_Mol_mdicts):
    discrete_molname_SMILES_Mol_mdicts = [None] * len(molname_SMILES_Mol_mdicts)
    # STEP1: retrieve all charges and discretize them
    sample_molname = list(molname_SMILES_Mol_mdicts[0].keys())[0]
    sample_mol = list(molname_SMILES_Mol_mdicts[0][sample_molname].values())[0]
    for propname in sample_mol.GetPropNames():
        if propname.startswith('discrete') or not is_mol_property_discretizable(sample_mol, propname):
            continue
        ColorPrint("Discretizing continuous property *** %s ***" % propname, "OKBLUE")
        property_values = get_all_property_vals(propname, *molname_SMILES_Mol_mdicts)
        discrete = Discretize()
        if '_confS_' in propname:   # use preset bind_width for conformational Entropy
            discrete.fit(property_values, bin_width=1.0, center_to=0)
        else:
            discrete.fit(property_values)
        # STEP2: add the discretized property to every molecule
        for i, molname_SMILES_mol_mdict in enumerate(molname_SMILES_Mol_mdicts):
            for molname in molname_SMILES_mol_mdict.keys():
                for SMILES in molname_SMILES_mol_mdict[molname].keys():
                    if is_mol_property_discretizable(molname_SMILES_mol_mdict[molname][SMILES], propname):
                        updated_mol = \
                            discrete.transform_mol(molname_SMILES_mol_mdict[molname][SMILES], propname)
                        molname_SMILES_mol_mdict[molname][SMILES] = updated_mol
            discrete_molname_SMILES_Mol_mdicts[i] = molname_SMILES_mol_mdict
    return discrete_molname_SMILES_Mol_mdicts

def discretize_ranks(X):
    """
    This method is useful when you have a score that has high density regions which are miss-interpreted by
    the conventional ranking method, which lead to spurious results if you sum the ranks of multiple alternative
    scores. It instead bins the values and returns their reversed bin assignment.
    The only difference with ranking is that the values are binned and the bin assignments are reversed, so
    that higher scores samples to have lower bin assignment==>higher rank.

    :param x:
    :return:
    """
    discretize = Discretize()
    return discretize.fit_transform(X=-1*X)    # -1x to reverse the bin assignment in order to be like ranking

def int_to_bitstring(x, n=0):
    """
    Get the binary representation of x.

    Parameters
    ----------
    x : int
    n : int
        Minimum number of digits. If x needs less digits in binary, the rest
        is filled with zeros.

    Returns
    -------
    str
    """
    return np.array(list(format(x, 'b').zfill(n)), dtype=int)

def ints_to_bitvector(X, n=16):
    """
    Convert a list of integers to bit representation.
    :param X: a vector of integer which will be converted to binary format. All together will form the final bit vector.
    :param n:
    :return:
    """
    # n = max([len(int_to_bitstring(x)) for x in X])
    bit_arrays = np.array([int_to_bitstring(x=x, n=n) for x in X])
    bit_vector = bit_arrays.flatten()
    return bit_vector


"""
SOME EXAMPLES:

# To reproduce the ECFP fingerprint with custom atom invariants do:
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import PeriodicTable, GetPeriodicTable, AllChem
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

for SMILES in ['Cc1ncccc1', 'CS(=O)(=O)N1CCc2c(C1)c(nn2CCCN1CCOCC1)c1ccc(Cl)c(C#Cc2ccc3C[C@H](NCc3c2)C(=O)N2CCCCC2)c1']:
    mol = Chem.MolFromSmiles(SMILES)
    mol = Chem.AddHs(mol)
    invariants = generate_ECFP_Atom_Invariant(mol)
    bi1={}
    bi2={}
    fp1 = rdMolDescriptors.GetMorganFingerprint(mol,radius=3,bitInfo=bi1)
    fp2 = rdMolDescriptors.GetMorganFingerprint(mol,radius=3,invariants=invariants,bitInfo=bi2)
    print("========")
    print(SMILES)
    nz1 = fp1.GetNonzeroElements()
    nz2 = fp2.GetNonzeroElements()
    print(len(nz1),len(nz1))
    print(sorted(bi1.values())==sorted(bi2.values()))
    print(nz1==nz2)

# You can get the SMILES of substructures that are extracted via `GetMorganFingerprint` function as follows.
# Then, you can append any labels to the SMILES string but not real numbers.

mol = Chem.MolFromSmiles('Cc1ncccc1')
info = {}
AllChem.GetMorganFingerprint(mol, radius=2, bitInfo=info)
radius, atom_id = list(info.values())[0][0][::-1]
env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_id)
sub_struct = Chem.PathToSubmol(mol, env)
type(sub_struct) #=> rdkit.Chem.rdchem.Mol
Chem.MolToSmiles(sub_struct, isomericSmiles=True, canonical=True, allBondsExplicit=True) #=>  'ccc'

"""