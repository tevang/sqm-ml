import os

from EXEC_functions.cross_validation.leave_one_out import leave_one_out
from library.features.dimensionality_reduction.UMAP import _ump_trans
from library.utils.plot_activities import plot_ump_with_label_3d
import pandas as pd

def EXEC_crossval_plots(features_df, CROSSVAL_PROTEINS, XTEST_PROTEINS, Settings):

    ump_file = os.path.join( Settings.HYPER_SQM_ML_ROOT_DIR, Settings.HYPER_EXECUTION_DIR_NAME,
                             "%i_proteins_PLEC-%iD_UMAP" % (len(XTEST_PROTEINS), Settings.N_COMPONENTS) +
                             Settings.create_feature_csv_name())
    if os.path.exists(ump_file):
        ump_df = pd.read_csv(ump_file)
    else:
        ump_df = _ump_trans(features_df.filter(regex='plec'),
                            n_neighbors=Settings.N_NEIGHBORS,
                            min_dist=Settings.MIN_DIST,
                            n_components=Settings.N_COMPONENTS,
                            metric=Settings.METRIC) \
            .join(features_df[['is_active', 'protein', 'basemolname']]) \
            .reset_index(drop=True)
        ump_df.to_csv(ump_file, index=False)

    for crossval_proteins, xtest_proteins in leave_one_out(XTEST_PROTEINS):
        print('=============================')
        print(crossval_proteins, xtest_proteins)
        print("XTEST: %s" % xtest_proteins[0])  # always one protein in the list
        crossval_proteins += CROSSVAL_PROTEINS  # added the fixed CROSSVAL proteins

        # Plot activity of xtest protein set
        plot_ump_with_label_3d(ump_df[ump_df['protein']==xtest_proteins[0]],
                               figure_title='{}_n{}_mdist{}_m{}'.format(xtest_proteins[0], Settings.N_NEIGHBORS,
                                                                        Settings.MIN_DIST, Settings.METRIC),
                               execution_dir=Settings.HYPER_PLOTS_DIR,
                               label_column='is_active')