from EXEC_functions.cross_validation.leave_one_out import leave_one_out
from library.plot_activities import _plot_ump_custom
from library.features.dimensionality_reduction.UMAP import _ump_trans
from old_no_commit.nocmmit_VAE import *

def EXEC_crossval_plots(features_df, CROSSVAL_PROTEINS, XTEST_PROTEINS):


    for crossval_proteins, xtest_proteins in \
            leave_one_out(XTEST_PROTEINS):
        mut_features_df = features_df.copy()
        print('=============================')
        print(crossval_proteins, xtest_proteins)
        print("XTEST: %s" % xtest_proteins[0])  # always one protein in the list
        crossval_proteins += CROSSVAL_PROTEINS

        features_here = mut_features_df.loc[mut_features_df["protein"].isin(crossval_proteins), :].filter(regex='plec')
        print(features_here.shape)


        ump_df = _ump_trans(features_here)
        ump_df = pd.concat([ump_df, features_df['is_active']], axis=1)
        ump_df.columns = ['ump1','ump2','y']
        # plot activites in a condensed space
        _plot_ump_custom(ump_df, xtest_proteins[0]+'_n500_mdist_01_mcorr')



