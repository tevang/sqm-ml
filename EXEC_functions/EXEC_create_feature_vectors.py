import logging

import pandas as pd

from library.global_fun import get_poseID, get_frameID
from commons.EXEC_caching import EXEC_caching_decorator
from library.EXEC_join_functions import EXEC_merge_dataframes_on_columns
from library.features.ligand_descriptors.pmapper_3D_pharmacophore import load_PMAPPER
from library.features.protein_ligand_complex_descriptors.PLEC import load_PLEC, load_Glide_PLEC
from library.net_charges import EXEC_load_all_net_charges

lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, "Preparing all features.", "_all",
                        full_csv_name=True, append_signature=True, prepend_all_proteins=True)
def EXEC_create_feature_vectors(CROSSVAL_PROTEINS, XTEST_PROTEINS, Settings):
    """
        Works for both PM6/COSMO and Glide. Extracts from precomputed csv.gz files selected descriptors/features.
    """

    # Extract selected SQM energy terms or/and/only Glide descriptors
    features_df = pd.concat([pd.read_csv(Settings.generated_file(Settings.create_feature_csv_name(), protein))
                             for protein in CROSSVAL_PROTEINS + XTEST_PROTEINS], ignore_index=True)\
        [set(['protein', 'basemolname', 'structvar', 'pose', 'frame', 'is_active'] + Settings.HYPER_SQM_FEATURES +
             Settings.HYPER_GLIDE_DESCRIPTORS)]

    # Extract selected 2D descriptors
    if len(Settings.HYPER_2D_DESCRIPTORS) > 0:
        features_df = pd.merge(features_df,
                               pd.concat([pd.read_csv(Settings.raw_input_file('_2Dfeature_vectors.csv.gz', protein)) \
                                                      .assign(protein=protein)
                                          for protein in CROSSVAL_PROTEINS + XTEST_PROTEINS], ignore_index=True) \
                                   [['protein', 'structvar'] + Settings.HYPER_2D_DESCRIPTORS],
                               on=['protein', 'structvar'])

    # Extract selected 3D complex descriptors
    if len(Settings.HYPER_3D_COMPLEX_DESCRIPTORS) > 0:
        features_df = pd.merge(features_df,
                               pd.concat([pd.read_csv(Settings.raw_input_file('_protein_ligand_complex_descriptors.csv.gz', protein)) \
                                         .assign(protein=protein)
                                          for protein in CROSSVAL_PROTEINS + XTEST_PROTEINS], ignore_index=True) \
                               .assign(pose=lambda df: df['complex_name'].apply(get_poseID),
                                       frame=lambda df: df['complex_name'].apply(get_frameID),
                                       structvar=lambda df: df['structvar'].str.lower()) \
                          [['protein', 'structvar', 'pose', 'frame'] + Settings.HYPER_3D_COMPLEX_DESCRIPTORS],
                               on=['protein', 'structvar', 'pose', 'frame'])
    # LOAD NET CHARGES
    if 'net_charge' not in features_df.columns:
        charges_df = EXEC_load_all_net_charges(CROSSVAL_PROTEINS+XTEST_PROTEINS, Settings=Settings)
        features_df = EXEC_merge_dataframes_on_columns(["protein", "structvar"], features_df, charges_df)

    # LOAD PLEC FINGERPRINTS
    if Settings.HYPER_SQM_FEATURES and Settings.HYPER_PLEC:
        features_df = load_PLEC(features_df, CROSSVAL_PROTEINS + XTEST_PROTEINS, Settings)

    elif Settings.HYPER_PLEC:
        features_df = load_Glide_PLEC(features_df, CROSSVAL_PROTEINS + XTEST_PROTEINS, Settings)

    # LOAD PMAPPER FINGERPRINTS
    if Settings.HYPER_PMAPPER:
        features_df = load_PMAPPER(features_df, CROSSVAL_PROTEINS + XTEST_PROTEINS, Settings)

    # LOAD 3D LIGAND DESCRIPTORS OF LOWEST ENERGY FREE STATE CONFORMERS
    if len(Settings.HYPER_3D_LIGAND_DESCRIPTORS) > 0:
        features_df = pd.merge(features_df,
                               pd.concat([pd.read_csv(Settings.raw_input_file('_free_conformers.PM7_COSMO.lowestE.csv.gz', protein)) \
                                         .assign(protein=protein)
                                          for protein in CROSSVAL_PROTEINS + XTEST_PROTEINS], ignore_index=True) \
                               .assign(structvar=lambda df: df['structvar'].str.lower())
                                   [['protein', 'structvar'] + Settings.HYPER_3D_LIGAND_DESCRIPTORS],
                               on=['protein', 'structvar'])

    # LOAD LIGAND Sconf DESCRIPTORS AND CREATE THEIR DIVISIONS WITH valid_pose_num
    if len(Settings.HYPER_LIGAND_Sconf_DESCRIPTORS) > 0:
        features_df = pd.merge(features_df,
                               pd.concat([pd.read_csv(Settings.raw_input_file('_Sconf_descriptors.csv.gz', protein)) \
                                         .assign(protein=protein)
                                          for protein in CROSSVAL_PROTEINS + XTEST_PROTEINS], ignore_index=True),
                               on=['protein', 'structvar']) \
            .pipe(lambda df: df.join(df.filter(regex='^DeltaG_[0-9]to[0-9]$') \
                                     .divide(df['valid_pose_num'], axis=0) \
                                     .add_suffix('_ratio')))


    return features_df[['protein', 'basemolname', 'structvar', 'pose', 'frame', 'is_active'] +
                       Settings.HYPER_SQM_FEATURES + Settings.HYPER_GLIDE_DESCRIPTORS +
                       Settings.HYPER_2D_DESCRIPTORS + Settings.HYPER_3D_COMPLEX_DESCRIPTORS +
                       Settings.HYPER_3D_LIGAND_DESCRIPTORS + Settings.HYPER_LIGAND_Sconf_DESCRIPTORS +
                       features_df.filter(regex='^plec[0-9]+$').columns.values.tolist() +
                       features_df.filter(regex='^pmap[0-9]+$').columns.values.tolist()]
