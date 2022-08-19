from operator import itemgetter
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from sklearn.preprocessing import normalize, StandardScaler
from rdkit.DataStructs import *
import numpy as np


# TODO: check for new possible metrics at https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
from library.utils.print_functions import ColorPrint, Debuginfo
from library.featvec.featvec_array_functions import numpy_to_bitvector

"""
Valid values for metric are:

From scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]. These metrics support sparse matrix inputs. [‘nan_euclidean’] but it does not yet support sparse matrices.

From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’] See the documentation for scipy.spatial.distance for details on these metrics. These metrics do not support sparse matrix inputs.
"""

# Scalar vector distance types (without silly ones, namely 'chebyshev'):
SCIPY_SCALAR_DISTANCE_TYPES = ['braycurtis', 'canberra', 'cityblock', 'correlation', 'cosine', 'euclidean',
'jensenshannon', 'minkowski', 'seuclidean', 'sqeuclidean'] # 'mahalanobis' is very slow
# Boolean vector distance types:
SCIPY_BOOLEAN_DISTANCE_TYPES = ['yule', 'kulsinski', 'hamming', 'sokalsneath']
RDKIT_BOOLEAN_SIMILARITY_TYPES = ['Tanimoto', 'Dice', 'Cosine', 'Sokal', 'Russel', 'RogotGoldberg', 'AllBit',
                                'Kulczynski', 'McConnaughey', 'Asymmetric', 'BraunBlanquet', 'Tversky']

"""
SciPy's similariy functions:
'yule', 'kulsinski', 'rogerstanimoto',
                          'jaccard', 'hamming'

RDKit's similarity functions:
                      
"""
def TverskySimilarity_(vec1, vec2):
    return TverskySimilarity(vec1, vec2, 0.5, 1.0)  # give more weight to the xtest molecule

SIMILARITY_FUNCTIONS = {
'Tanimoto': TanimotoSimilarity,
'Dice': DiceSimilarity,
'Cosine': CosineSimilarity,
'Sokal': SokalSimilarity,
'Russel': RusselSimilarity,
'RogotGoldberg': RogotGoldbergSimilarity,
'AllBit': AllBitSimilarity,
'Kulczynski': KulczynskiSimilarity,
'McConnaughey': McConnaugheySimilarity,
'Asymmetric': AsymmetricSimilarity,
'BraunBlanquet': BraunBlanquetSimilarity,
'Tversky': TverskySimilarity_
}

def calc_norm_dist_matrix(mat1, mat2, disttype):
    if disttype in RDKIT_BOOLEAN_SIMILARITY_TYPES:
        mat1 = [numpy_to_bitvector(v) for v in mat1]    # convert arrays to bit vectors
        mat2 = [numpy_to_bitvector(v) for v in mat2]
        simmat = np.zeros([len(mat1), len(mat2)])
        simmat.fill(np.nan)    # in case a similarity couldn't be calculated, an error will be raised
        for r, vec1 in enumerate(mat1):
            for c, vec2 in enumerate(mat2):
                sim = SIMILARITY_FUNCTIONS[disttype](vec1, vec2)
                simmat[r, c] = sim     # it's not a symmetric matrix!
        distmat = 1.0 - simmat  # to convert it to distance matrix
    elif disttype in SCIPY_BOOLEAN_DISTANCE_TYPES+SCIPY_SCALAR_DISTANCE_TYPES:
        distmat = cdist(mat1, mat2, disttype)  # N x K distance matrix

    ndistmat = normalize(distmat)   # scale between [0,1]
    return ndistmat


def ensemble_maxvariance_sim_matrix(mat1, mat2, is_distance=False, percent_unique_values_threshold=50.0):
    """
    Method that takes two 2D matrices, calculates multiple types of pairwise scalar vector distances,
    scales them to [0,1], finds the variance of each distance type, and returns the -1*z-scores
    of the distance type with the highest variance.

    RECOMMENDED FOR FINDING WHICH TRAINING COMPOUNDS ARE SIMILAR TO THE TEST COMPOUND (~DOMAIN OF APPLICABILITY).

    :param mat1: N compounds x M features (by default the crossval set). Can be a list of a numpy array.
    :param mat2: K compounds x M features (by default the xtest set). Can be a list of a numpy array.
    :param is_distance: whether the returned matrix will contain distances of similarities.
    :param percent_unique_values_threshold: At least that % of the distance/similarity values must be unique,
                                            otherwise something is wrong with this metric
    :return scaled_distmat: N x K z-score matrix of distances or similarities of the distance type with
                            the highest variance
    """
    disttype_ndistmat_dict = {}
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    assert mat1.shape[1] == mat2.shape[1], \
        Debuginfo("ERROR: mat1 and mat2 do not have the same number of columns (%i vs %i)!" %
                    (mat1.shape[1], mat2.shape[1]), fail=True)
    # Find out if the matrices are scalar or boolean
    if ((mat1 == 0) | (mat1 == 1)).all() and ((mat2 == 0) | (mat2 == 1)).all():
        DISTANCE_TYPES = SCIPY_BOOLEAN_DISTANCE_TYPES+RDKIT_BOOLEAN_SIMILARITY_TYPES
    else:
        DISTANCE_TYPES = SCIPY_SCALAR_DISTANCE_TYPES

    # Create the N x K distance matrix
    disttype_variancePercentUnique_list = []
    for disttype in DISTANCE_TYPES:
        ColorPrint("Calculating %s distances." % disttype, "OKBLUE")
        try:
            ndistmat = calc_norm_dist_matrix(mat1, mat2, disttype)  # N x K normalized distance matrix
        except Exception as e:
            print(repr(e))
            ColorPrint("Distance type %s won't be considered." % disttype, "OKRED")
            continue
        except np.linalg.LinAlgError:
            ColorPrint("Singular matrix. %s distances won't be calculated." % disttype, "OKRED")
            continue
        variance = ndistmat.var()
        percent_unique_values = 100*len(set(ndistmat.flatten()))/ndistmat.flatten().shape[0]
        ColorPrint("\t\t\t\tvariance=%f , percent of unique dist/sim values=%f %%." % (variance, percent_unique_values), "OKBLUE")
        disttype_ndistmat_dict[disttype] = ndistmat
        disttype_variancePercentUnique_list.append( (disttype, variance, percent_unique_values) )    # the overall variance

    disttype_variancePercentUnique_list.sort(key=itemgetter(1), reverse=True)    # sort in descending order
    assert np.any([t[2]>percent_unique_values_threshold for t in disttype_variancePercentUnique_list]), \
        Debuginfo("Fail: none of the distance types returned unique distance/similarity values above"
                    "%f %%. Try to reduce the percent_unique_values_threshold." % percent_unique_values_threshold,
                    fail=True)
    for i in range(len(disttype_variancePercentUnique_list)):
        best_disttype = disttype_variancePercentUnique_list[i][0]    # keep the distance type with the maximum variance
        max_variance = disttype_variancePercentUnique_list[i][1]
        percent_unique_values = disttype_variancePercentUnique_list[i][2]
        # At least 50% of the distance/similarity values must be unique, otherwise something is wrong with this metric
        if percent_unique_values > percent_unique_values_threshold:
            break
    ColorPrint("The distance type with the max variance (%f) was %s" % (max_variance, best_disttype), "BOLDBLUE")

    scaler = StandardScaler()
    matrix_shape = disttype_ndistmat_dict[best_disttype].shape # the shape of the original distance matrix
    scaler.fit(disttype_ndistmat_dict[best_disttype].flatten().reshape(-1, 1))

    if is_distance:
        scaled_distmat = scaler.transform(disttype_ndistmat_dict[best_disttype].flatten().reshape(-1, 1))  # return z-score distances
    else:
        scaled_distmat = -1 * scaler.transform(disttype_ndistmat_dict[best_disttype].flatten().reshape(-1, 1))  # return similarities (-1*z-score distances)
    scaled_distmat = scaled_distmat.reshape(matrix_shape)  # recover the distance matrix with scaled values

    return scaled_distmat

