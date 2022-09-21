from EXEC_functions.cross_validation.leave_one_out import leave_one_out
from library.features.dimensionality_reduction.UMAP import _ump_trans
from library.utils.plot_activities import plot_ump_with_label_3d


def EXEC_crossval_plots(features_df, CROSSVAL_PROTEINS, XTEST_PROTEINS, settings):

    ump_df = _ump_trans(features_df.filter(regex='plec'),
                        n_neighbors=settings.N_NEIGHBORS,
                        min_dist=settings.MIN_DIST,
                        n_components=settings.N_COMPONENTS,
                        metric=settings.METRIC) \
        .join(features_df[['is_active', 'protein', 'basemolname']]) \
        .reset_index(drop=True)

    for crossval_proteins, xtest_proteins in leave_one_out(XTEST_PROTEINS):
        print('=============================')
        print(crossval_proteins, xtest_proteins)
        print("XTEST: %s" % xtest_proteins[0])  # always one protein in the list
        crossval_proteins += CROSSVAL_PROTEINS  # added the fixed CROSSVAL proteins

        # Plot activity of xtest protein set
        plot_ump_with_label_3d(ump_df[ump_df['protein']==xtest_proteins[0]],
                               figure_title='{}_n{}_mdist{}_m{}'.format(xtest_proteins[0], settings.N_NEIGHBORS,
                                                                        settings.MIN_DIST, settings.METRIC),
                               execution_dir=settings.HYPER_PLOTS_DIR,
                               label_column='is_active')