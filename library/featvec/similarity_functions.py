import numpy
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys

from library.featvec import distance
from library.global_vars import BOOLEAN_FEATVEC_TYPES


def scalar_vector_similarity(v1, v2, sim_type='ratio'):
    if sim_type == 'ratio':
        return 1.0 / (1.0 + distance.euclidean(v1, v2))
    elif sim_type == 'exp':
        # Alternatively use an exponential but beware that it decays faster as the euclidian distance increases
        # print("DEBUG: euclidean dist=", distance.euclidean(v1, v2))
        return np.exp(-1*distance.euclidean(v1, v2)/10) # reduce the euclidean dist otherwise sim << 0
    # TODO: Normalize the dot product.
    # elif sim_type == 'dotprod':
    #     return np.dot(v1,v2)


def calc_featvec_sim(featvec1, featvec2, M=1, featvec_type=None, sim_type='exp'):
    """
    Method to calculate the distance of two feature vectors of arbitrary size and any type (binary or not)!
    featvec2 can be multidimensional, in which case the highest similarity is returned.

    :param featvec1:
    :param featvec2:
    :param M: if >1, instead of the maximum similarity, the average similarity if the M most similar compounds is
              returned in case featvec2 has multiple rows.
    :param featvec_type:
    :param sim_type:
    :return:
    """

    if len(featvec2.shape) == 1:
        if featvec_type and featvec_type in BOOLEAN_FEATVEC_TYPES:    # FASTEST!!!   # if binary arrays use the Tanimoto distance
            sim = 1.0 - distance.jaccard(featvec1, featvec2)
        elif featvec_type:
            sim = scalar_vector_similarity(featvec1, featvec2, sim_type=sim_type)
        elif ((featvec1 == 0) | (featvec1 == 1)).all() and ((featvec2 == 0) | (featvec2==1)).all():
            sim = 1.0 - distance.jaccard(featvec1, featvec2)
        else:
            sim = scalar_vector_similarity(featvec1, featvec2, sim_type=sim_type)
        return sim
    elif len(featvec2.shape) == 2: # keep the highest similarity (lowest distance inverted)
        sim_list = []
        for row in range(featvec2.shape[0]):
            if featvec_type and featvec_type in BOOLEAN_FEATVEC_TYPES:
                sim = 1.0 - distance.jaccard(featvec1, featvec2[row])
            elif featvec_type:
                sim = scalar_vector_similarity(featvec1, featvec2[row], sim_type=sim_type)
            if set(featvec1) == {0, 1} and set(featvec2[row]) == {0, 1}:   # if binary arrays use the Tanimoto distance
                sim = 1.0 - distance.jaccard(featvec1, featvec2[row])
            else:
                sim = scalar_vector_similarity(featvec1, featvec2[row], sim_type=sim_type)
            sim_list.append(sim)
            sim_list.sort(reverse=True)
        return np.mean(sim_list[:M])  # return the maximum similarity between sift1 and all sift2 rows


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