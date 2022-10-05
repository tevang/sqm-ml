import os

import pandas as pd
from sklearn.model_selection import LeaveOneOut

from library.evaluate_model import evaluate_learning_model
from library.features.dimensionality_reduction.PCA import pca_compress_fingerprint
from library.features.dimensionality_reduction.UMAP import umap_compress_fingerprint
from library.features.dimensionality_reduction.feature_selection import best_feature_selector
from library.train_model import train_learning_model
from library.utils.print_functions import ColorPrint
from library.weights import compute_activity_ratio_weights, compute_featvec_similarity_weights, \
    compute_maxvariance_featvec_similarity_weights, no_weights


def leave_one_out(ALL_PROTEINS):
    if len(ALL_PROTEINS) == 1:
        yield [], ALL_PROTEINS
        return

    loo = LeaveOneOut()
    loo.get_n_splits(ALL_PROTEINS)

    for train_index, test_index in loo.split(ALL_PROTEINS):
        yield [ALL_PROTEINS[i] for i in train_index], [ALL_PROTEINS[i] for i in test_index]


def EXEC_crossval_leave_one_out(features_df, selected_features, CROSSVAL_PROTEINS, XTEST_PROTEINS, n_neighbors,
                                min_dist, n_components, metric, learning_model_type="Logistic Regression",
                                sample_weight_type=None, compress_PCA=False, compress_UMP=False,
                                PLEC_pca_variance_explained_cutoff=0.8, select_best_features=False,
                                max_best_features=31, perm_n_repeats=10, plot_SHAPLEY=False,
                                write_SHAPLEY=False, csv_path_SHAPLEY=None, plots_dir=None,
                                features_for_training=[]):

    SAMPLE_WEIGHT_FUNCTIONS = {
        None: no_weights,
        'activity_ratio': compute_activity_ratio_weights,
        'featvec_similarity': compute_featvec_similarity_weights,
        'maxvariance_featvec_similarity': compute_maxvariance_featvec_similarity_weights
    }

    evaluation_dfs = []
    predictor = train_learning_model(learning_model_type=learning_model_type, perm_n_repeats=perm_n_repeats,
                                     plot_SHAPLEY=plot_SHAPLEY, write_SHAPLEY=write_SHAPLEY)

    if 'plec' in features_for_training and 'plec1' in features_df.columns and compress_UMP:
        features_df = umap_compress_fingerprint(features_df, n_neighbors, min_dist, n_components, metric,
                                                    fingerprint_type='plec')

    if 'plec' in features_for_training and 'plec1' in features_df.columns and compress_PCA:
        features_df = pca_compress_fingerprint(features_df, fingerprint_type='plec',
                                                   pca_variance_explained_cutoff=PLEC_pca_variance_explained_cutoff)

    # Solely for the Publication
    if 'plec' not in features_for_training: features_df.drop(labels=features_df.filter(regex='^[plecmarcu_]+[0-9]+$') \
                                                             .columns.tolist(), axis=1, inplace=True)
    else:   features_for_training.remove('plec')
    selected_features = [sf for sf in selected_features if sf in features_for_training]

    for crossval_proteins, xtest_proteins in leave_one_out(XTEST_PROTEINS):
        mut_features_df = features_df.copy()
        print(crossval_proteins, xtest_proteins)
        ColorPrint("XTEST: %s" % xtest_proteins[0], "BOLDGREEN")  # always one protein in the list
        crossval_proteins += CROSSVAL_PROTEINS

        print("FEATURES:", selected_features + mut_features_df.filter(regex='^[plecmarcu_]+[0-9]+$').columns.tolist())
        ColorPrint("The xtest set containts {} actives and {} inactives.".format(
            mut_features_df.loc[mut_features_df["protein"].isin(xtest_proteins) &
                                (mut_features_df['is_active']==1)].shape[0],
            mut_features_df.loc[mut_features_df["protein"].isin(xtest_proteins) &
                                (mut_features_df['is_active']==0)].shape[0]), "OKBLUE")
        model, importances_df = predictor(features_df=mut_features_df.loc[mut_features_df["protein"].isin(crossval_proteins), :],
                                          sel_columns=selected_features + mut_features_df.filter(regex='^[plecmarcu_]+[0-9]+$').columns.tolist(),
                                          sample_weight=SAMPLE_WEIGHT_FUNCTIONS[sample_weight_type](
                                              mut_features_df, selected_features + mut_features_df.filter(
                                                  regex='^[plecmarcu_]+[0-9]+$').columns.tolist(), xtest_proteins),
                                          csv_path_SHAPLEY=os.path.join(plots_dir, xtest_proteins[0] + '_SHAPLEY_importances.csv'))

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