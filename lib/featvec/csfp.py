from lib.utils.print_functions import ColorPrint

try:
    import csfpy
except (ModuleNotFoundError, ImportError):
    ColorPrint("WARNING: module csfpy was not found, therefore CSFPy fingerprints won't be calculated.", "OKRED")
    pass

from lib.featvec.invariants import *

def is_same_mol(rdkit_mol, csfpy_mol):
    """
    Method to check if the two Mol objects contain the same set of atoms.

    :param rdkit_mol:
    :param csfpy_mol:
    :return:
    """
    csfpy_mol_atoms = csfpy_mol.atoms
    if len(rdkit_mol.GetAtoms()) != len(csfpy_mol_atoms):
        if len(rdkit_mol.GetAtoms()) < len(csfpy_mol_atoms):    # check if the extra atoms in csfpy_mol are protons wo number
            for a in csfpy_mol.atoms[len(rdkit_mol.GetAtoms()):]:
                if a.name != 'H':
                    print("DEBUG: the two mols have unequal number of atoms: rdkit %i != csfpy %i" %
                          (len(rdkit_mol.GetAtoms()), len(csfpy_mol.atoms)))
                    return False
            csfpy_mol_atoms = csfpy_mol.atoms[:len(rdkit_mol.GetAtoms())]

    rconf = rdkit_mol.GetConformer(0)
    for i, catom in enumerate(csfpy_mol_atoms):
        # Check for agreement in atomic number
        if catom.atomic_number != rdkit_mol.GetAtomWithIdx(i).GetAtomicNum():
            print("DEBUG: CSFPy atomic_num = %f , RDKit atomic_num = %f" %
                  (catom.atomic_number, rdkit_mol.GetAtomWithIdx(i).GetAtomicNum()))
            return False
        # Check for agreement in coordinates
        x,y,z = rconf.GetAtomPosition(i)
        X, Y, Z = catom.coords
        if (X,Y,Z) != (x,y,z):
            return False

    return True

def create_CSFP_type_fingeprint(molname_SMILES_conformersMol_mdict,
                                featvec_type="tCSFP",
                                as_array=True,
                                nBits=4096):

    if featvec_type.startswith("CSFP"):
        # fp_function = csfpy.csfp      OBSOLETE
        config = csfpy.getCSFPConfig()
        lower_bound = 2
        upper_bound = 6
    elif featvec_type.startswith("tCSFP"):
        # fp_function = csfpy.tcsfp     OBSOLETE
        config = csfpy.gettCSFPConfig()
        lower_bound = 2
        upper_bound = 8
    elif featvec_type.startswith("iCSFP"):
        # fp_function = csfpy.icsfp     OBSOLETE
        config = csfpy.getiCSFPConfig()
        lower_bound = 2
        upper_bound = 8
    elif featvec_type.startswith("fCSFP"):
        # fp_function = csfpy.fcsfp     OBSOLETE
        config = csfpy.getfCSFPConfig()
        lower_bound = 2
        upper_bound = 6
    elif featvec_type.startswith("pCSFP"):
        # fp_function = csfpy.pcsfp     OBSOLETE
        lower_bound = 2
        upper_bound = 6
    elif featvec_type.startswith("gCSFP"):
        # fp_function = csfpy.gcsfp     OBSOLETE
        config = csfpy.getgCSFPConfig()
        lower_bound = 2
        upper_bound = 8

    molname_CSFPmol_dict = {}  # stores the CSFP Mol objects
    # For security, load the mols from the SDF file with/without properties and control whether they are the same
    # stuctvars that were loaded to molname_SMILES_conformersMol_mdict
    # ATTENTION: DO NOT USE SMILES OR YOU WILL MIX THE ATOM ORDER!
    sample_molname = list(molname_SMILES_conformersMol_mdict.keys())[0]
    sample_mol = list(molname_SMILES_conformersMol_mdict[sample_molname].values())[0]
    assert "SDF_file" in sample_mol.GetPropNames(), \
        Debuginfo("ERROR: you have chosen to generate %s fingerprints but no SDF"
                    " files with invariant properties were supplied!" % featvec_type, fail=True)

    # NOTE: in SMILES the hydrogens are not loaded in the same atom order as you loaded the structures from the
    # NOTE: original MOL2/SDF file. Therefore load the structures to CSFPy MOL objects from the same files,
    # NOTE: never from SMILES! That way the invariant you will pass will correspond to the correct atoms in the Mol.

    CSFP_mols = csfpy.Molecule.from_file( sample_mol.GetProp("SDF_file") )
    for mol in CSFP_mols:
        molname = get_basemolname(mol.name).lower()   # remove also _pose[0-9] and _frm[0-9]
        # If this molname has been saved and the current structvar is not the No. 1, skip it. Otherwise save it.
        # ATTENTION: If a lowestE structvar file has been given, then "stereo1_ion1_tau1" is not always the dominant!
        # TODO: perhaps there is a major flaw here! The code loads only the 1st occurrence of each molecule, namely
        # TODO: only the 1st structval in the file.
        try:
            SMILES = list(molname_SMILES_conformersMol_mdict[molname].keys())[0]
        except IndexError:
            ColorPrint("FAIL: molecule %s is missing from molname_SMILES_conformersMol_mdict!" % molname, "OKRED")
            continue
        try:
            if molname in molname_SMILES_conformersMol_mdict.keys() \
                and molname not in molname_CSFPmol_dict.keys() \
                and is_same_mol(molname_SMILES_conformersMol_mdict[molname][SMILES], mol):        # keep only the structural variants that are loaded
                molname_CSFPmol_dict[molname] = mol
        except RuntimeError:
            ColorPrint("Some atom of %s did not have coordinates." % molname, "FAIL")
            continue
            
    # Finally, it's time to generate the designated CSFP fingerprint
    molname_fingerprint_dict = {}  # molname (without _iso[0-9] suffix)->fingerprint
    invariants = {}     # save here the invariants (if requested)
    for molname in molname_CSFPmol_dict.keys():
        SMILES = list(molname_SMILES_conformersMol_mdict[molname].keys())[0]  # only the dominant structvar must be there
        if featvec_type.endswith('Li'):  # If invariants must be created
            invariants = generate_CSFP_Atom_Invariant(mol=molname_SMILES_conformersMol_mdict[molname][SMILES])
        if featvec_type.startswith("pCSFP"):    #  this fingerprint doesn't support the csfpCustom()
            fp = csfpy.pcsfp(molname_CSFPmol_dict[molname], lower_bound, upper_bound)
        else:
            fp = csfpy.csfpCustom(molecule=molname_CSFPmol_dict[molname],
                                  lower_bound=lower_bound,
                                  upper_bound=upper_bound,
                                  config=config,
                                  invariants=invariants)
        if as_array == True:
            molname_fingerprint_dict[molname] = fp.toNumpyBitArray(nBits=nBits)
        else:
            molname_fingerprint_dict[molname] = fp  # NOT RECOMMENDED, it's not RDKit's Fingerprint object

    return molname_fingerprint_dict