import numpy as np
from rdkit import Chem

from library.featvec.similarity_functions import calc_Fingerprint_sim, calc_Fingerprint_sim_fromArrays
from library.usrcat.toolkits.rd import generate_moments


def create_similarity_matrix(molname_SMILES_conformersMol_mdict, actives_list, featvec_type="Morgan3"):
    
    actives_list = [m.lower() for m in actives_list]  # lower case
    similarity_matrix = np.zeros([len(actives_list), len(actives_list)])
    for ind1,molname1 in enumerate(actives_list):
        for ind2, molname2 in enumerate(actives_list):
            if type(molname_SMILES_conformersMol_mdict[molname1]) == np.ndarray: # if this is a molname->fingeprint(array dict
                # print("DEBUG: fp1=", molname_SMILES_conformersMol_mdict[molname1].tolist()))
                # print("DEBUG: fp2=", molname_SMILES_conformersMol_mdict[molname1].tolist()))
                sim = calc_Fingerprint_sim_fromArrays(molname_SMILES_conformersMol_mdict[molname1],
                                                      molname_SMILES_conformersMol_mdict[molname2])
                similarity_matrix[ind1, ind2] = sim
            else:
                similarity_list = [] # list of fp similarity between all the isomers of molnames1 with all the isomers of molname2
                for SMILES1 in list(molname_SMILES_conformersMol_mdict[molname1].keys()):
                    for SMILES2 in list(molname_SMILES_conformersMol_mdict[molname2].keys()):
                        similarity_list.append(
                            calc_Fingerprint_sim(molname_SMILES_conformersMol_mdict[molname1][SMILES1],
                                                 molname_SMILES_conformersMol_mdict[molname2][SMILES2],
                                                 featvec_type=featvec_type))
                similarity_matrix[ind1, ind2] = np.max(similarity_list)
    
    return similarity_matrix


def calc_Fingeprint_sim_list(molname_SMILES_conformersMol_mdict, sorted_ligand_experimentalE_dict, query_molname, is_aveof=False,
                             query_molfile=None, return_molnames=False, featvec_type="Morgan3", moment_number=4, onlyshape=False):
    """
        FUNCTION to calculate the USRC distance from the lowest energy conformation (usually crystal conformation) of a query ligand, of all
        isomers and conformers of each target compound.
        RETURN:
        reordered_FPsim_list:   list of FP similarities (same order as reordered_experimentalE_list but without the query ligand)
        reordered_experimentalE_list:   list of Exp DeltaG (same order as reordered_FPsim_list but without the query ligand)
    """

    if query_molfile:
        qmol=Chem.MolFromMol2File(query_molfile)
        query_molname = qmol.GetProp('_Name')
        qmoment = generate_moments(qmol, moment_number=moment_number, onlyshape=onlyshape, ensemble_mode=1)
    else:
        if query_molname not in list(molname_SMILES_conformersMol_mdict.keys()):    # if this ligand is not in the sdf file
            print("ERROR: not conformation was found for query_molname", query_molname)
            return False
        query_SMILES_list = list(molname_SMILES_conformersMol_mdict[query_molname].keys())
        qmol = molname_SMILES_conformersMol_mdict[query_molname][query_SMILES_list[0]]  # keep the mol of the lowest energy conformer

    molname_FPsim_dict = {} # target molname-> FP similarity from query ligand
    for molname in list(molname_SMILES_conformersMol_mdict.keys()):
        for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
            target_mol = molname_SMILES_conformersMol_mdict[molname][SMILES]
            molname_FPsim_dict[molname] = calc_Fingerprint_sim(qmol, target_mol, featvec_type)

    molname_list = []   # contains the common molnames between molname_FPsim_dict and sorted_ligand_experimentalE_dict
    for k in list(sorted_ligand_experimentalE_dict.keys()):
        if k in list(molname_FPsim_dict.keys()):
            molname_list.append(k)
        # ADD A WARNING HERE ! # else:
        #     print("WARNING..."))
    if is_aveof == True:    # Deactivate the following for the MinRank Method
        if query_molname in molname_list:
            molname_list.remove(query_molname)
    if is_aveof == False:   # only from 'minrank' and 'consscortk' scoring schemes
        molname_FPsim_dict[query_molname] = -1.0 # set the similarity of query ligand to -1 to avoid bias in the ranking
    reordered_FPsim_list = []
    reordered_experimentalE_list = []
    for molname in molname_list:
        reordered_FPsim_list.append(molname_FPsim_dict[molname])
        reordered_experimentalE_list.append(sorted_ligand_experimentalE_dict[molname])

    if return_molnames == True:
        return reordered_FPsim_list, reordered_experimentalE_list, molname_list
    else:
        # return 2 lists in the same order, the FP similarities and the Exp DeltaG
        return reordered_FPsim_list, reordered_experimentalE_list