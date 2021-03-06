from sklearn.model_selection import LeaveOneOut

from sqmnn.library.evaluate_model import evaluate_learning_model
from sqmnn.library.features.dimensionality_reduction.PCA import pca_compress_fingerprint
from sqmnn.library.features.dimensionality_reduction.UMAP import umap_compress_fingerprint
from sqmnn.library.features.dimensionality_reduction.feature_selection import best_feature_selector
from sqmnn.library.train_model import *
from sqmnn.library.weights import compute_activity_ratio_weights, compute_featvec_similarity_weights, \
    compute_maxvariance_featvec_similarity_weights, no_weights


def leave_one_out(ALL_PROTEINS):
    if len(ALL_PROTEINS) == 1:
        yield [], ALL_PROTEINS
        return

    loo = LeaveOneOut()
    loo.get_n_splits(ALL_PROTEINS)

    for train_index, test_index in loo.split(ALL_PROTEINS):
        yield [ALL_PROTEINS[i] for i in train_index], [ALL_PROTEINS[i] for i in test_index]


def EXEC_crossval_leave_one_out(features_df, selected_features, CROSSVAL_PROTEINS, XTEST_PROTEINS,
                                learning_model_type="Logistic Regression", sample_weight_type=None,
                                compress_PLEC=False, compress_UMP=False, compress_PMAPPER=True,
                                PLEC_pca_variance_explained_cutoff=0.8, PMAPPER_pca_variance_explained_cutoff=0.8,
                                select_best_features=False, max_best_features=31):

    SAMPLE_WEIGHT_FUNCTIONS = {
        None: no_weights,
        'activity_ratio': compute_activity_ratio_weights,
        'featvec_similarity': compute_featvec_similarity_weights,
        'maxvariance_featvec_similarity': compute_maxvariance_featvec_similarity_weights
    }

    evaluation_dfs = []
    predictor = train_learning_model(learning_model_type=learning_model_type)

    for crossval_proteins, xtest_proteins in \
            leave_one_out(XTEST_PROTEINS):
        mut_features_df = features_df.copy()
        print(crossval_proteins, xtest_proteins)
        print("XTEST: %s" % xtest_proteins[0])  # always one protein in the list
        crossval_proteins += CROSSVAL_PROTEINS

        # UMAP part
        if 'plec1' in mut_features_df.columns and compress_UMP:

            mut_features_df = umap_compress_fingerprint(mut_features_df, crossval_proteins, xtest_proteins,
                                                   fingerprint_type='plec')

        if 'plec1' in mut_features_df.columns and compress_PLEC:
            mut_features_df = pca_compress_fingerprint(mut_features_df, crossval_proteins, xtest_proteins,
                                                   fingerprint_type='plec',
                                                   pca_variance_explained_cutoff=PLEC_pca_variance_explained_cutoff)


        if 'pmap1' in mut_features_df.columns and compress_PMAPPER:
            mut_features_df = pca_compress_fingerprint(mut_features_df, crossval_proteins, xtest_proteins,
                                                   fingerprint_type='pmap',
                                                   pca_variance_explained_cutoff=PMAPPER_pca_variance_explained_cutoff)

        print("FEATURES:", selected_features + mut_features_df.filter(regex='^[plecmarcu_]+[0-9]+$').columns.tolist())


        model, importances_df = predictor(features_df=mut_features_df.loc[mut_features_df["protein"].isin(crossval_proteins), :],
                                          sel_columns=selected_features + mut_features_df.filter(regex='^[plecmarcu_]+[0-9]+$').columns.tolist(),
                                          sample_weight=SAMPLE_WEIGHT_FUNCTIONS[sample_weight_type](mut_features_df, selected_features, xtest_proteins))

        sel_columns = selected_features + mut_features_df.filter(regex='^[plecmarcu_]+[0-9]+$').columns.tolist()

        if select_best_features:
            model, mut_features_df, sel_columns = \
                best_feature_selector(model=model,
                                      features_df=mut_features_df,
                                      train_rows=mut_features_df["protein"].isin(crossval_proteins),
                                      sel_columns=sel_columns,
                                      max_features=max_best_features)


        evaluation_dfs.append( evaluate_learning_model(
            model=model,
            features_df=mut_features_df.loc[mut_features_df["protein"].isin(xtest_proteins), :],
            sel_columns=sel_columns).join(importances_df)
        )
        print("\n")

        if select_best_features:
            mut_features_df = features_df.copy()

    average_stats_df = pd.concat(evaluation_dfs).apply('mean').sort_values(ascending=False)
    print("AVERAGE AUC-ROC OF %s = %f DOR = %f MK = %f" %
          tuple([learning_model_type] + average_stats_df[['AUC-ROC', 'DOR', 'MK']].to_list()))
    print("AVERAGE FEATURE IMPORTANCES:")
    print(average_stats_df.drop(labels=['AUC-ROC', 'DOR', 'MK']).to_string())