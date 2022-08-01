from lib.featvec.distance import calc_norm_dist_matrix, ensemble_maxvariance_sim_matrix

def no_weights(features_df, selected_features, xtest_proteins):
    return None


def compute_featvec_similarity_weights(features_df, selected_features, xtest_proteins):
    print("Computing feature vector similarity weights.")
    norm_distances = calc_norm_dist_matrix(features_df.loc[~features_df['protein'].isin(xtest_proteins), selected_features].values,
                                           features_df.loc[features_df['protein'].isin(xtest_proteins), selected_features].values,
                                           'correlation')
    return 1.0 - norm_distances.min(axis=1)


def compute_maxvariance_featvec_similarity_weights(features_df, selected_features, xtest_proteins):
    print("Computing maximum variance feature vector similarity weights.")
    return ensemble_maxvariance_sim_matrix(features_df.loc[features_df['protein'].isin(xtest_proteins), selected_features].values,
                                    features_df.loc[~features_df['protein'].isin(xtest_proteins), selected_features].values,
                                    is_distance=False, percent_unique_values_threshold=50.0)


def compute_activity_ratio_weights(features_df, selected_features, xtest_proteins):
    """ UNTESTED """
    def _assign_ratio_weight(g):
        sample_weight_dict = (g["is_active"].value_counts() / g["is_active"].size).to_dict()
        return g.apply(lambda r: sample_weight_dict[r["is_active"]], axis=1)

    return features_df.loc[features_df["protein"] \
                                        .isin(xtest_proteins), :] \
        .groupby("protein", as_index=False) \
        .apply(_assign_ratio_weight).to_numpy()
