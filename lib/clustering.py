from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.cluster.hierarchy import average, fcluster
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau
from lib.global_fun import *
from sklearn.preprocessing import minmax_scale

def comp_mutual_info_matrix(X):
    """

    :param X: a NxM array.
    :return: NxN distance matrix
    """
    # condensed_vector = []  # contains absolutely no redundant values
    # X = X.transpose() ??? UNTESTED     # in order to get NxN distance matrix with the following 2 AVENUES
    # # # AVENUE 1: slow
    # # for i in range(X.shape[1]):
    # #     for j in range(i + 1, X.shape[1]):
    # #         mi = -1 * mutual_info_regression(X[:,i].reshape(-1,1), X[:,j].reshape(-1,1))[0]
    # #         condensed_vector.append(mi)
    # #AVENUE 2: faster
    # for i in range(X.shape[1]):
    #     if i+1 == X.shape[1]:
    #         break
    #     mi_array = -1* mutual_info_regression(X[:,i+1:], X[:,i].reshape(-1,1))  # -1* to make it distance
    #     for mi in mi_array:
    #         condensed_vector.append(mi)
    # condensed_vector = np.array(condensed_vector)
    # condensed_vector -= condensed_vector.min()  # shift minimum to 0
    # square_matrix = squareform(condensed_vector)
    # return square_matrix

    # Faster way
    def distance(x,y):
        return -1 * mutual_info_regression(x.reshape(-1, 1), y.ravel())
    dist_matrix = pairwise_distances(X, X, metric=distance)
    # IMPORTANT: mutual_info_regression() does not produce the same correlation value of every feature vector
    # IMPORTANT: when compared with itself, which btw is not 0 but low negative (e.g. -2.95601907).
    # IMPORTANT: deviation from the minimum correlation can be up to 0.06. Consequently, in order to
    # IMPORTANT: to obtain a zero-diagonal distance matrix by dist_matrix -= dist_matrix.min(), I would need
    # IMPORTANT: to round(dist_matrix, 0), but that will lead to high loss of precision. Therefore we just
    # IMPORTANT: force all diagonal elements to be 0, pass the matrix to AgglomerativeClustering() without errors.
    dist_matrix -= dist_matrix.min()  # shift the minimum to 0 before passing the matrix to AgglomerativeClustering
    np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix

def comp_kendalls_tau_matrix(X):
    # condensed_vector = []  # contains absolutely no redundant values
    # X = X.transpose() ??? UNTESTED     # in order to get NxN distance matrix with the following 2 AVENUES
    # for i in range(X.shape[1]):
    #     for j in range(i + 1, X.shape[1]):
    #         dist = -1 * kendalltau(X[:,i].reshape(-1,1), X[:,j].reshape(-1,1))[0]   # -1* to make it distance
    #         condensed_vector.append(dist)
    # condensed_vector = np.array(condensed_vector)
    # condensed_vector -= condensed_vector.min()  # shift minimum to 0
    # square_matrix = squareform(condensed_vector)
    # return square_matrix

    # Faster way
    def distance(x,y):
        return -1 * kendalltau(x, y)[0]   # -1* to make it distance
    dist_matrix = pairwise_distances(X, X, metric=distance)
    dist_matrix -= dist_matrix.min()  # shift the minimum to 0 before passing the matrix to AgglomerativeClustering
    return dist_matrix

def sim_to_dist(sim_vector):
    """
    Method to convert the similarities to distances in the range [0,1]
    :param sim_vector:
    :return:
    """
    sim_vector = -1* np.array(sim_vector)   # convert the similarities to distances
    return minmax_scale(sim_vector)     # scale the distances to [0,1]

def silhuette_clustering(square_matrix):
    print("Saving the square RMSD matrix for future usage into file 'square_rmsd_matrix.pickle'.")
    save_pickle("square_rmsd_matrix.pickle", square_matrix)
    print(
        "*** Launching Hierarchical Clustering with automatic cluster number selection based on Silhouette score. ***")
    silhouette_scores = []
    for n_clusters in range(2, min(30, square_matrix.shape[1])):  # max n_cluster<= number of structures
        agg = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed',
                                      linkage='average')
        cluster_labels = agg.fit_predict(square_matrix) # must be distance matrix!
        silhouette_avg = silhouette_score(square_matrix, cluster_labels, metric='precomputed')
        silhouette_scores.append(silhouette_avg)
        print("Cluster number %i has silhouette score %f" % (n_clusters, silhouette_avg))
        # TODO: individual clusters must not have negative silhouette average scores
    # Find the point where silhouette score change starts to drop
    # TODO: check how skipping the 1st score works out, not sure that is valid.
    silhouette_scores = np.array(silhouette_scores)
    ratio = silhouette_scores[:-1] / silhouette_scores[1:]
    # AVENUE 1: using the gradient of the ratio
    best_n_clusters = np.gradient(ratio).argmax() + 2
    # # AVENUE 2: using the differences of ratios
    # best_n_clusters = np.diff(ratio).argmax() + 2  # +2 because we started estimating silhuouette from clustN=2
    #                                                   # +1 because we skipped the 1st score
    print("Employing Hierarchical Clustering method with %i number of clusters to cluster MLPs based on their " \
          "prediction scores on the xtest set." % best_n_clusters)
    agg = AgglomerativeClustering(n_clusters=best_n_clusters, affinity='precomputed',
                                  linkage='average')
    clusters = agg.fit_predict(square_matrix)   # must be distance matrix
    return clusters

def distance_clustering(self, condensed_matrix, cutoff_dist=None):
    print("*** Launching Hierarchical Clustering using an RMSD cutoff of %f Angstroms. ***" % cutoff_dist)
    Z = average(condensed_matrix)
    clusters = fcluster(Z, cutoff_dist, criterion='distance')  # cluster radius in TMaling rmsd units
    return clusters