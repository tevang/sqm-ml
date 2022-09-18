import pandas as pd
from EXEC_functions.cross_validation.leave_one_out import leave_one_out
from library.utils.plot_activities import plot_ump_with_label_3d, plot_ump_with_protein
from library.features.dimensionality_reduction.UMAP import _ump_trans
from SQMNN_pipeline_settings import Settings



def EXEC_crossval_plots(features_df, CROSSVAL_PROTEINS, XTEST_PROTEINS):
    settings = Settings()
    for crossval_proteins, xtest_proteins in \
            leave_one_out(XTEST_PROTEINS):
        mut_features_df = features_df.loc[features_df["protein"].isin(crossval_proteins), :].filter(regex='plec')
        print('=============================')
        print(crossval_proteins, xtest_proteins)
        print("XTEST: %s" % xtest_proteins[0])  # always one protein in the list
        crossval_proteins += CROSSVAL_PROTEINS


        #Option1 plotting with activity
        # features_for_umap = mut_features_df.join(features_df['is_active'])
        # plot_ump_with_label_3d(features_for_umap,
        #                        xtest_proteins[0]+'_n50_mdist_01_mcorr',
        #                        settings.HYPER_RESULTS_DIR,
        #                        'is_active')

        #Option2 plotting with protein cluster
        features_for_umap = mut_features_df.join(features_df[['is_active', 'protein']])
        plot_ump_with_protein(features_for_umap,
                               xtest_proteins[0]+'_n50_mdist_01_mcorr',
                               settings.HYPER_RESULTS_DIR,
                               'is_active',
                              'protein')




