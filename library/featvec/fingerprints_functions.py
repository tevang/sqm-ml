import numpy as np
from rdkit import Chem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import rdReducedGraphs, AllChem, MACCSkeys

from library.USRCAT_functions import fp_generator_2Dpp, fp_generator_3Dpp, Mol2Vec
from library.featvec.csfp import create_CSFP_type_fingeprint, generateAtomInvariant
from library.featvec.feature_scale_utils import minmax_scale_crossval_xtest
from library.featvec.featvec_array_functions import getNumpyArray
from library.featvec.physchem import calculate_physchem_descriptors_from_mols
from library.utils.print_functions import ColorPrint

try:
    import tmap as tm
    from map4 import MAP4Calculator
except (ModuleNotFoundError, ImportError):
    ColorPrint("WARNING: module map4 was not found.", "OKRED")
    pass
from scoop import futures
from rdkit.Chem import rdMolDescriptors as rdMD

def comp_ECFP(molname_SMILES_conformersMol_mdict,
               featvec_type="ECFP",
               as_array=True,
               nBits=4096,
               radius=3,
               maxPath=5,
               featvec_average_mode=1,
               useChirality=False):
    molname_fingerprint_dict = {}  # molname (without _iso[0-9] suffix)->fingerprint
    invariants = []
    if as_array == True:
        for molname in list(molname_SMILES_conformersMol_mdict.keys()):
            for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
                mol = molname_SMILES_conformersMol_mdict[molname][SMILES]
                if featvec_type.endswith('Li'):
                    invariants = generateAtomInvariant(mol)
                molname_fingerprint_dict[molname] = \
                    getNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol,
                                                                        useChirality=useChirality,
                                                                        radius=radius,
                                                                        nBits=nBits,
                                                                        invariants=invariants))
    else:
        for molname in list(molname_SMILES_conformersMol_mdict.keys()):
            for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
                mol = molname_SMILES_conformersMol_mdict[molname][SMILES]
                if featvec_type.endswith('Li'):
                    invariants = generateAtomInvariant(mol)
                molname_fingerprint_dict[molname] = \
                    AllChem.GetMorganFingerprint(mol,
                                                 useChirality=useChirality,
                                                 radius=radius,
                                                 invariants=invariants)  # this function does not take nBits
    return molname_fingerprint_dict

def comp_FCFP(molname_SMILES_conformersMol_mdict,
               featvec_type="FCFP",
               as_array=True,
               nBits=4096,
               radius=3,
               maxPath=5,
               featvec_average_mode=1,
               useChirality=False):
    molname_fingerprint_dict = {}  # molname (without _iso[0-9] suffix)->fingerprint
    invariants = []
    if as_array == True:
        for molname in list(molname_SMILES_conformersMol_mdict.keys()):
            for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
                mol = molname_SMILES_conformersMol_mdict[molname][SMILES]
                if featvec_type.endswith('Li'):
                    invariants = generateAtomInvariant(mol)
                molname_fingerprint_dict[molname] = \
                    getNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=useChirality,
                                                                        radius=radius, nBits=nBits,
                                                                        invariants=invariants, useFeatures=True))
    else:
        for molname in list(molname_SMILES_conformersMol_mdict.keys()):
            for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
                mol = molname_SMILES_conformersMol_mdict[molname][SMILES]
                if featvec_type.endswith('Li'):
                    invariants = generateAtomInvariant(mol)
                molname_fingerprint_dict[molname] = \
                    AllChem.GetMorganFingerprint(molname_SMILES_conformersMol_mdict[molname][SMILES],
                                                 radius=radius, invariants=invariants,
                                                 useChirality=useChirality,
                                                 useFeatures=True)  # this function does not take nBits
    return molname_fingerprint_dict

def comp_RDK5(molname_SMILES_conformersMol_mdict,
              featvec_type="RDK5",
              as_array=True,
              nBits=4096,
              radius=3,
              maxPath=5,
              featvec_average_mode=1,
              useChirality=False):
    molname_fingerprint_dict = {}  # molname (without _iso[0-9] suffix)->fingerprint
    if as_array == True:
        for molname in list(molname_SMILES_conformersMol_mdict.keys()):
            for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
                molname_fingerprint_dict[molname] = getNumpyArray(Chem.RDKFingerprint(molname_SMILES_conformersMol_mdict[molname][SMILES], fpSize=nBits, maxPath=maxPath))
    else:
        for molname in list(molname_SMILES_conformersMol_mdict.keys()):
            for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
                molname_fingerprint_dict[molname] = Chem.RDKFingerprint(molname_SMILES_conformersMol_mdict[molname][SMILES], fpSize=nBits, maxPath=maxPath)
    return molname_fingerprint_dict

def comp_CSFP(molname_SMILES_conformersMol_mdict,
              featvec_type="CSFP",
              as_array=True,
              nBits=4096,
              radius=3,
              maxPath=5,
              featvec_average_mode=1,
              useChirality=False):
    return create_CSFP_type_fingeprint(molname_SMILES_conformersMol_mdict,
                                                               featvec_type=featvec_type,
                                                               as_array=as_array,
                                                               nBits=nBits)

def comp_AP(molname_SMILES_conformersMol_mdict,
              featvec_type="AP",
              as_array=True,
              nBits=4096,
              radius=3,
              maxPath=5,
              featvec_average_mode=1,
              useChirality=False):
    molname_fingerprint_dict = {}  # molname (without _iso[0-9] suffix)->fingerprint
    if as_array == True:
        for molname in molname_SMILES_conformersMol_mdict.keys():
            for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
                molname_fingerprint_dict[molname] = getNumpyArray(
                    rdMD.GetHashedAtomPairFingerprintAsBitVect(molname_SMILES_conformersMol_mdict[molname][SMILES],
                                                               includeChirality=useChirality, nBits=nBits))
    return molname_fingerprint_dict

def comp_TT(molname_SMILES_conformersMol_mdict,
              featvec_type="TT",
              as_array=True,
              nBits=4096,
              radius=3,
              maxPath=5,
              featvec_average_mode=1,
              useChirality=False):
    molname_fingerprint_dict = {}  # molname (without _iso[0-9] suffix)->fingerprint
    if as_array == True:
        for molname in list(molname_SMILES_conformersMol_mdict.keys()):
            for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
                molname_fingerprint_dict[molname] = getNumpyArray(
                    rdMD.GetHashedTopologicalTorsionFingerprint(molname_SMILES_conformersMol_mdict[molname][SMILES],
                                                                includeChirality=useChirality, nBits=nBits,
                                                                targetSize=7))
    return molname_fingerprint_dict

def comp_AvalonFP(molname_SMILES_conformersMol_mdict,
              featvec_type="AvalonFP",
              as_array=True,
              nBits=4096,
              radius=3,
              maxPath=5,
              featvec_average_mode=1,
              useChirality=False):
    molname_fingerprint_dict = {}  # molname (without _iso[0-9] suffix)->fingerprint
    if as_array == True:
        for molname in list(molname_SMILES_conformersMol_mdict.keys()):
            for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
                molname_fingerprint_dict[molname] = getNumpyArray(
                    GetAvalonFP(molname_SMILES_conformersMol_mdict[molname][SMILES], nBits=nBits))
    return molname_fingerprint_dict

def comp_MACCS(molname_SMILES_conformersMol_mdict,
              featvec_type="MACCS",
              as_array=True,
              nBits=4096,
              radius=3,
              maxPath=5,
              featvec_average_mode=1,
              useChirality=False):
    molname_fingerprint_dict = {}  # molname (without _iso[0-9] suffix)->fingerprint
    if as_array == True:
        for molname in list(molname_SMILES_conformersMol_mdict.keys()):
            for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
                mol = molname_SMILES_conformersMol_mdict[molname][SMILES]
                fp = MACCSkeys.GenMACCSKeys(mol)
                molname_fingerprint_dict[molname] = np.array(
                    [int(b) for b in fp.ToBitString()])  # convert the MACCS fp to bit array
    return molname_fingerprint_dict

def comp_ErgFP(molname_SMILES_conformersMol_mdict,
              featvec_type="ErgFP",
              as_array=True,
              nBits=4096,
              radius=3,
              maxPath=5,
              featvec_average_mode=1,
              useChirality=False):
    molname_fingerprint_dict = {}  # molname (without _iso[0-9] suffix)->fingerprint
    if as_array == True:
        for molname in list(molname_SMILES_conformersMol_mdict.keys()):
            for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
                molname_fingerprint_dict[molname] = rdReducedGraphs.GetErGFingerprint(
                    molname_SMILES_conformersMol_mdict[molname][SMILES])
    return molname_fingerprint_dict

def comp_2Dpp(molname_SMILES_conformersMol_mdict,
              featvec_type="2Dpp",
              as_array=True,
              nBits=4096,
              radius=3,
              maxPath=5,
              featvec_average_mode=1,
              useChirality=False):
    molname_fingerprint_dict = {}  # molname (without _iso[0-9] suffix)->fingerprint
    if as_array == True:
        mol_list = []
        molname_list = []
        SMILES_list = []
        # ATTENTION: WE ASSUME THAT EACH MOLNAME HAS ONLY ONE SMILES STRING!
        for molname in list(molname_SMILES_conformersMol_mdict.keys()):
            for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
                mol = molname_SMILES_conformersMol_mdict[molname][SMILES]
                molname_list.append(molname)
                SMILES_list.append(SMILES)
                mol_list.append(mol)
        # Generate 2Dpp feature vectors in parallel for all molecules and conformers
        results = list(
            futures.map(fp_generator_2Dpp, mol_list, molname_list))  # list of the form: [ (molname_key, feat_vec), ...]
        for duplet in results:
            molname_key = duplet[0]
            feat_vec = duplet[1]
            molname_fingerprint_dict[molname_key] = feat_vec
        print("")  # just to change line
    return molname_fingerprint_dict

def comp_3Dpp(molname_SMILES_conformersMol_mdict,
              featvec_type="3Dpp",
              as_array=True,
              nBits=4096,
              radius=3,
              maxPath=5,
              featvec_average_mode=1,
              useChirality=False):
    molname_fingerprint_dict = {}  # molname (without _iso[0-9] suffix)->fingerprint
    if as_array == True:
        mol_list = []
        molname_list = []
        SMILES_list = []
        featvec_average_mode_list = []
        # TODO: ATTENTION: WE ASSUME THAT EACH MOLNAME HAS ONLY ONE SMILES STRING!
        for molname in list(molname_SMILES_conformersMol_mdict.keys()):
            for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
                molname_list.append(molname)
                SMILES_list.append(SMILES)
                mol_list.append( molname_SMILES_conformersMol_mdict[molname][SMILES] )
                featvec_average_mode_list.append(featvec_average_mode)
        # Generate 3Dpp feature vectors in parallel for all molecules and conformers
        results = list(futures.map(fp_generator_3Dpp, mol_list, molname_list,
                                   featvec_average_mode_list))  # list of the form: [ (molname_key, feat_vec), ...]
        for duplet in results:
            molname_key = duplet[0]
            feat_vec = duplet[1]
            molname_fingerprint_dict[molname_key] = feat_vec
        print("")  # just to change line
    return molname_fingerprint_dict

def comp_mol2vec(molname_SMILES_conformersMol_mdict,
              featvec_type="mol2vec",
              as_array=True,
              nBits=4096,
              radius=3,
              maxPath=5,
              featvec_average_mode=1,
              useChirality=False):
    mol2vec = Mol2Vec(model=featvec_type)
    return mol2vec.calc_mol2vec(molname_SMILES_conformersMol_mdict, get_dict=True)

def comp_physchem(molname_SMILES_conformersMol_mdict,
              featvec_type="physchem",
              as_array=True,
              nBits=4096,
              radius=3,
              maxPath=5,
              featvec_average_mode=1,
              useChirality=False):
    ## CALCULATE PHYSICOCHEMICAL DESCRIPTORS (best correlated with IC50s)
    crossval_molname_fingerprint_dict, xtest_molname_fingerprint_dict = {}, {}
    SMILES_args, molname_args = [], []
    for molname in list(molname_SMILES_conformersMol_mdict.keys()):
        SMILES = list(molname_SMILES_conformersMol_mdict[molname].keys())[0]
        SMILES_args.append(SMILES)
        molname_args.append(molname)
        # crossval_molname_fingerprint_dict[molname] = self.calculate_physicochemical_descriptors(SMILES)
    results = list(futures.map(calculate_physchem_descriptors_from_mols, SMILES_args, molname_args))
    molname_fingerprint_dict = {m: d for m, d in results}
    return minmax_scale_crossval_xtest(molname_fingerprint_dict)

def comp_MAP4(molname_SMILES_conformersMol_mdict,
              featvec_type="MAP4",
              as_array=True,
              nBits=4096,
              radius=3,
              maxPath=5,
              featvec_average_mode=1,
              useChirality=False):
    """
    # TODO: ATTENTION: Please note that the similarity/dissimilarity between two MinHashed fingerprints
    # TODO: cannot be assessed with "standard" Jaccard, Manhattan, or Cosine functions. Due to MinHashing, the
    # TODO: order of the features matters and the distance cannot be calculated "feature-wise". There is a well
    # TODO: written blog post that explains it:
    # TODO: https://aksakalli.github.io/2016/03/01/jaccard-similarity-with-minhash.html.
    # TODO: Therefore, a custom kernel/loss function needs to be implemented for machine learning applications
    # TODO: of MAP4 (e.g. using the distance function found in the test.py script).

    :param molname_SMILES_conformersMol_mdict:
    :param featvec_type:
    :param as_array:
    :param nBits:
    :param radius:
    :param maxPath:
    :param featvec_average_mode:
    :param useChirality:
    :return:
    """
    MAP4 = MAP4Calculator(dimensions=nBits, is_folded=True)
    ENC = tm.Minhash(nBits)
    mol_list, molname_list, SMILES_list = [], [], []
    # TODO: ATTENTION: WE KEEP ONLY THE FIRST SMILES FROM EACH MOLNAME!
    for molname in list(molname_SMILES_conformersMol_mdict.keys()):
        for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
            molname_list.append(molname)
            SMILES_list.append(SMILES)
            mol_list.append(molname_SMILES_conformersMol_mdict[molname][SMILES])
            break
    # use parallelized version
    fps = np.array(MAP4.calculate_many(mol_list), dtype=np.int)
    return {m: fp for m, fp in zip(molname_list, fps)}
