from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys

from library.featvec import distance


def calc_Fingerprint_sim(mol1, mol2, featvec_type="Morgan3", nBits=2048):
    """
        ATTENTION: these functions take also lists of fingerprints as an input. Also check the AllChem.GetMorganFingerprintAsBitVect(dist_matrix, 2, nBits=1024),
        DataStructs.BulkTanimotoSimilarity(fp, train_fps_morgan2), getNumpyArrays().
    """
    if featvec_type == "Morgan2":
        fp1 = AllChem.GetMorganFingerprint(mol1,2)
        fp2 = AllChem.GetMorganFingerprint(mol2,2)
    elif featvec_type == "Morgan3":
        fp1 = AllChem.GetMorganFingerprint(mol1,3)
        fp2 = AllChem.GetMorganFingerprint(mol2,3)
    elif featvec_type == "RDK5":
        fp1 = Chem.RDKFingerprint(mol1, maxPath=5) # max. path length = 5
        fp2 = Chem.RDKFingerprint(mol2, maxPath=5)
    elif featvec_type == "RDK5L":
        fp1 = Chem.RDKFingerprint(mol1, maxPath=7) # max. path length = 7
        fp2 = Chem.RDKFingerprint(mol2, maxPath=7)
    elif featvec_type == "AP":
        fp1 = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol1, nBits=nBits)
        fp2 = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol2, nBits=nBits)
    elif featvec_type == "MACCS":
        fp1 = MACCSkeys.GenMACCSKeys(mol1)
        fp2 = MACCSkeys.GenMACCSKeys(mol2)

    return DataStructs.TanimotoSimilarity(fp1,fp2)


def calc_Fingerprint_sim_fromArrays(query, test):
    """
    Method to calculate binary fingerprint similarity. (I am not sure what happens if 2Dpp and 3Dpp are ensemble averaged).
    test can be a multidimensional array in which case the maximum similarity is returned.
    :param query:
    :param test:
    :return:
    """

    if len(test.shape) == 1:
        # sim = DataStructs.TanimotoSimilarity(query, test)
        # sim = DataStructs.FingerprintSimilarity(query, test)
        # try:
        #     sim = 1.0/distance.euclidean(query, test)
        # except ZeroDivisionError:
        #     sim = 1.0
        sim = 1.0 - distance.jaccard(query, test)    # Jaccard is 1.0 - Tanimoto similarity
        return sim
    elif len(test.shape) == 2: # keep the highest similarity (lowest distance inverted)
        sim_list = []
        for row in range(test.shape[0]):
            # print("DEBUG: query=", query))
            # print("DEBUG: test[row]=", test[row]))
            # sim = DataStructs.TanimotoSimilarity(query, test[row])
            # sim = DataStructs.FingerprintSimilarity(query, test[row])
            # try:
            #     sim = 1.0/distance.euclidean(query, test[row])
            # except ZeroDivisionError:
            #     sim = 1.0
            sim = 1.0 - distance.jaccard(query, test[row])    # Jaccard is Tanimoto similarity
            sim_list.append(sim)
        return np.max(sim_list) # return the maximum similarity between query and all of test rows