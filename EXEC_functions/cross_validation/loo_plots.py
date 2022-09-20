from EXEC_functions.cross_validation.leave_one_out import leave_one_out
from SQM_ML_pipeline_settings import settings
from library.utils.plot_activities import plot_ump_with_protein


def EXEC_crossval_plots(features_df, CROSSVAL_PROTEINS, XTEST_PROTEINS, settings):
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
                              figure_title='{}_n{}_mdist{}_m{}'.format(xtest_proteins[0], settings.N_NEIGHBORS,
                                                                       settings.MIN_DIST, settings.METRIC),
                              execution_dir=settings.HYPER_PLOTS_DIR,
                              label_column='is_active',
                              protein_col='protein',
                              n_neighbors=settings.N_NEIGHBORS,
                              min_dist=settings.MIN_DIST,
                              n_components=settings.N_COMPONENTS,
                              metric=settings.METRIC)




