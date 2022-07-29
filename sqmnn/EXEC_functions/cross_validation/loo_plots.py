from sqmnn.EXEC_functions.cross_validation.leave_one_out import leave_one_out
from sqmnn.library.plot_activities import _plot_ump_custom, _plot_vae_custom
from sqmnn.library.features.dimensionality_reduction.UMAP import _ump_trans
from sqmnn.library.features.dimensionality_reduction.VAE import *

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
        _plot_ump_custom(ump_df)



        mm = get_vae_embeddings(np.array(features_here.astype('float32')))
        vae_df = pd.DataFrame(mm.train_vae().numpy(), columns=['vae1', 'vae2'])
        vae_df = pd.concat([vae_df, features_df['is_active']], axis=1)
        vae_df.columns = ['vae1', 'vae2', 'y']
        # plot activites in a condensed space
        _plot_vae_custom(vae_df)

