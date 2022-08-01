VALID_FEATURE_VECTOR_TYPES = ["moments", "mmgbsa_fp", "sift", "binsift", "physchem", "ECFP", "ECFPL", "ECFPLi",
                                 "cECFP", "cECFPL", "cECFPLi", "FCFP", "FCFPL", "cFCFP", "cFCFPL", "cFCFPLi", "FCFPLi",
                                 "ECFP_sift", "moments_sift", "mmgbsaXsift", "binnedmmgbsa", "binnedmmgbsaXsift", "moments_sift_physchem",
                                 "moments_ECFP", "moments_physchem", "moments_mmgbsa", "mmgbsa_physchem", "bfactors", "RMSF",
                                 "chiRMSF", "phipsiRMSF", "chiphipsiRMSF", "mmgbsa_RMSF", "mmgbsa_bfactors", "mmgbsa_chiRMSF",
                                 "2Dpp", "RDK5", "RDK5L", "3Dpp", "MACCS", "moments_logP", "moments_sol", "shape", "docking",
                                 "mmgbsa_sift", "ErgFP", "plec", "moments_plec",
                                 "mol2vec1000_1", "mol2vec1000_2", "mol2vec1000_3", "mol2vec2000_1", "mol2vec2000_2",
                                 "mol2vec2000_3", "mol2vec3000_1", "mol2vec3000_2", "mol2vec3000_3", "mol2vec400_1",
                                 "mol2vec500_1", "mol2vec500_2", "mol2vec500_3", "mol2vec600_1", "mol2vec900_1",
                                 "mol2vec300_1", "csv", "AP", "cAP", "TT", "cTT", "AvalonFP", "AvalonFPL",
                                 "NGF7936", "NGF3968", "NGF1984", "NGF992", "NGF496", "NGF248", "NGF124", "NGF62",
                                 "CSFP", "tCSFP", "iCSFP", "fCSFP", "pCSFP", "gCSFP",
                                 "CSFPL", "tCSFPL", "iCSFPL", "fCSFPL", "pCSFPL", "gCSFPL",
                                 "CSFPLi", "tCSFPLi", "iCSFPLi", "fCSFPLi", "gCSFPLi", "MAP4", "AttentiveFP", "ChempropFP"]

# The feature vectors I use the most
FREQUENT_FEATURE_VECTOR_TYPES = ["CSFPL", "tCSFPL", "iCSFPL", "fCSFPL" , "pCSFPL", "gCSFPL", "AP", "cAP", "ErgFP",
                       "AvalonFPL", "TT", "cTT", "NGF3968", "ECFPL", "cECFPL", "FCFPL", "cFCFPL", "2Dpp", "CSFPLi", "tCSFPLi",
                       "iCSFPLi", "fCSFPLi", "gCSFPLi", "ECFPLi", "cECFPLi", "FCFPLi", "cFCFPLi", "MAP4"]

BOOLEAN_FEATVEC_TYPES = ["ECFP", "ECFPL", "ECFPLi", "cECFP", "cECFPL", "cECFPLi", "FCFP", "FCFPL", "FCFPLi",
                         "cFCFP", "cFCFPL", "cFCFPLi", "2Dpp", "RDK5", "RDK5L", "3Dpp",
                         "MACCS", "AP", "cAP", "TT", "cTT", "AvalonFP", "AvalonFPL", "CSFP", "tCSFP", "iCSFP", "fCSFP", "pCSFP",
                         "gCSFP", "CSFPL", "tCSFPL", "iCSFPL", "fCSFPL", "pCSFPL", "gCSFPL",
                         "CSFPLi", "tCSFPLi", "iCSFPLi", "fCSFPLi", "gCSFPLi"]

SCALAR_FEATVEC_TYPES = [fv for fv in VALID_FEATURE_VECTOR_TYPES if fv not in BOOLEAN_FEATVEC_TYPES]

# All the following fingerprints will be yield MLPs with the same dimensions
FINGERPRINT_GROUP = ['ECFP', 'ECFPL', 'ECFPLi', 'cECFP', 'cECFPL', 'cECFPLi', 'FCFP', 'FCFPL', 'FCFPLi',
                     'cFCFP', 'cFCFPL', 'cFCFPLi', 'RDK5', 'RDK5L', 'AP', 'cAP', 'TT', 'cTT',
                     'AvalonFP', 'AvalonFPL', 'CSFP', 'tCSFP', 'iCSFP', 'fCSFP', 'pCSFP', 'gCSFP',
                     'CSFPL', 'tCSFPL', 'iCSFPL', 'fCSFPL', 'pCSFPL', 'gCSFPL',
                     'CSFPLi', 'tCSFPLi', 'iCSFPLi', 'fCSFPLi', 'gCSFPLi', 'MAP4']

import os

CONSSCORTK_LIB_DIR = os.path.dirname(os.path.realpath(__file__))
CONSSCORTK_HOME_DIR = CONSSCORTK_LIB_DIR[:-3]
CONSSCORTK_BIN_DIR = CONSSCORTK_LIB_DIR[:-3] + "general_tools"
CONSSCORTK_UTILITIES_DIR = CONSSCORTK_LIB_DIR[:-3] + "utilities"
CONSSCORTK_THIRDPARTY_DIR = CONSSCORTK_LIB_DIR[:-3] + "thirdparty"
