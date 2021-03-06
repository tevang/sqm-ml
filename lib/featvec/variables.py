from lib.featvec.fingerprints_functions import comp_ECFP, comp_FCFP, comp_RDK5, comp_CSFP, comp_AP, comp_TT, comp_AvalonFP, \
    comp_MACCS, comp_ErgFP, comp_2Dpp, comp_3Dpp, comp_mol2vec, comp_physchem, comp_MAP4

fingerprint_functions_dict = {
                                 'ECFP': comp_ECFP,
                                 'ECFPL': comp_ECFP,
                                 'ECFPLi': comp_ECFP,
                                 'cECFP': comp_ECFP,
                                 'cECFPL': comp_ECFP,
                                 'cECFPLi': comp_ECFP,
                                 'FCFP': comp_FCFP,
                                 'FCFPL': comp_FCFP,
                                 'FCFPLi': comp_FCFP,
                                 'cFCFP': comp_FCFP,
                                 'cFCFPL': comp_FCFP,
                                 'cFCFPLi': comp_FCFP,
                                 'RDK5': comp_RDK5,
                                 'RDK5L': comp_RDK5,
                                 'CSFP': comp_CSFP,
                                 'tCSFP': comp_CSFP,
                                 'iCSFP': comp_CSFP,
                                 'fCSFP': comp_CSFP,
                                 'pCSFP': comp_CSFP,
                                 'gCSFP': comp_CSFP,
                                 'CSFPL': comp_CSFP,
                                 'tCSFPL': comp_CSFP,
                                 'iCSFPL': comp_CSFP,
                                 'fCSFPL': comp_CSFP,
                                 'pCSFPL': comp_CSFP,
                                 'gCSFPL': comp_CSFP,
                                 'CSFPLi': comp_CSFP,
                                 'tCSFPLi': comp_CSFP,
                                 'iCSFPLi': comp_CSFP,
                                 'fCSFPLi': comp_CSFP,
                                 'gCSFPLi': comp_CSFP,
                                 'AP': comp_AP,
                                 'cAP': comp_AP,
                                 'TT': comp_TT,
                                 'cTT': comp_TT,
                                 'AvalonFP': comp_AvalonFP,
                                 'AvalonFPL': comp_AvalonFP,
                                 'MACCS': comp_MACCS,
                                 'ErgFP': comp_ErgFP,
                                 '2Dpp': comp_2Dpp,
                                 '3Dpp': comp_3Dpp,
                                 'mol2vec': comp_mol2vec,
                                 'physchem': comp_physchem,
                                 'MAP4': comp_MAP4
}