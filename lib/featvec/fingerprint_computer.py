from lib.ConsScoreTK_Statistics import remove_uniform_columns
from lib.featvec.variables import fingerprint_functions_dict
from lib.utils.print_functions import Debuginfo, ColorPrint


def calculate_fingerprints_from_RDKit_mols(molname_SMILES_conformersMol_mdict,
                                           featvec_type="ECFP",
                                           as_array=True,
                                           nBits=4096,
                                           radius=3,
                                           maxPath=5,
                                           featvec_average_mode=1,
                                           remove_nan_cols=False):
    """
        FUNCTION to calculate various types of fingerprints and save them as bit arrays. Note that topological fingerprints (ECFP,FCPF,RDK5,MACCS,2Dpp)
        will be the same no matter how many conformers each mol object has. Only 3Dpp will be different, therefore I have an if clause that returns
        the average fingerprint(array in case multiple conformers are present for a molecule. The mol objects in molname_SMILES_conformersMol_mdict
        must contains conformers only if featvec_type="3Dpp".

        ARGS:
        featvec_average_mode:    if '0', only the last conformer will be considered for 3Dpp fingerprint(calculation. If '1' then the 3Dpp bit arrays of
                            all the conformers of each molecule will be averaged. If '2' then each position will assume value 0 if the average value
                            in <0.5 at that position, or 1 if the average value at that position is >=0.5.
    """
    # SANITY CHECKS
    assert featvec_type in ['ECFP', 'ECFPL', 'ECFPLi', 'cECFP', 'cECFPL', 'cECFPLi', 'FCFP', 'FCFPL', 'FCFPLi',
                            'cFCFP', 'cFCFPL', 'cFCFPLi', 'RDK5', 'RDK5L', 'ErgFP', 'AP', 'cAP', 'TT', 'cTT',
                            'MACCS', '2Dpp', '3Dpp', 'physchem', 'csv', 'CSFP', 'tCSFP', 'iCSFP', 'fCSFP', 'pCSFP', 'gCSFP',
                            'CSFPL', 'tCSFPL', 'iCSFPL', 'fCSFPL', 'pCSFPL', 'gCSFPL',
                            'CSFPLi', 'tCSFPLi', 'iCSFPLi', 'fCSFPLi', 'gCSFPLi', 'AvalonFP', 'AvalonFPL',
                            'MAP4'] \
            or featvec_type.startswith("mol2vec") or featvec_type.startswith("NGF"), \
        Debuginfo("ERROR: wrong feature vector name %s !" % featvec_type, fail=True)

    ColorPrint("Generating " + featvec_type + " fingeprints of all compounds... ", "OKGREEN")

    if featvec_type.endswith('L') or featvec_type.endswith('Li'): # applies to ECFPL, ECFPLi, FCFPL, FCFPLi, RDK5L, *CSFPL[i]
        nBits = 8192    # with radius >= 4 performance drops!
        maxPath = 7     # applies only to RDK5L
    if featvec_type.startswith('c'):    # applies to cECFP*, cFCFP*, cAP, cTT
        useChirality = True
    else:
        useChirality = False

    # molname_fingerprint_dict format: molname (without _iso[0-9] suffix)->fingerprint
    # fingerprints are independent of protonation state, hence remove _iso[0-9] suffix for clarity
    molname_fingerprint_dict = fingerprint_functions_dict[featvec_type](molname_SMILES_conformersMol_mdict,
                                                           featvec_type=featvec_type,
                                                           as_array=as_array,
                                                           nBits=nBits,
                                                           radius=radius,
                                                           maxPath=maxPath,
                                                           featvec_average_mode=featvec_average_mode,
                                                           useChirality=useChirality)

    if remove_nan_cols: # remove columns that contain at least one nan value (useful for 'physchem')
        ColorPrint("Removing uniform columns or columns with nan value.", "BOLDBLUE")
        molnames = list(molname_fingerprint_dict.keys())
        featvec_mat = remove_uniform_columns(list(molname_fingerprint_dict.values()), noredundant_cols=True)
        molname_fingerprint_dict = {m: fv for m,fv in zip(molnames, featvec_mat)}

    return molname_fingerprint_dict