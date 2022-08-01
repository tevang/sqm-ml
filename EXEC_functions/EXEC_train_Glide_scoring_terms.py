"""
* Select best structvar by Glide score.
* Use the following terms as features to train a learning model.

r_epik_State_Penalty                                                                 -0
i_i_glide_lignum                                                                     56
i_i_glide_rotatable_bonds                                                             6
r_i_glide_lipo                                                                 -3.33318
r_i_glide_hbond                                                                -0.26052
r_i_glide_metal                                                                       0
r_i_glide_rewards                                                              -1.66173
r_i_glide_evdw                                                                 -33.6949
r_i_glide_ecoul                                                               -0.391161
r_i_glide_erotb                                                                0.664148
r_i_glide_esite                                                                       0
r_i_glide_emodel                                                               -43.2796
r_i_glide_energy                                                               -34.0861
r_i_glide_einternal                                                             5.08912
i_i_glide_confnum                                                                    35
r_i_glide_eff_state_penalty                                                           0

* minmax_scale without descritization.
* Compare with Glide on DUD-E.
* Repeat training by including PM6/COSMO terms to see if scoring is improved. Don't forget to descritize
  the PM6/COSMO terms. ==> it makes no difference

"""
from EXEC_functions.cross_validation.leave_one_out import EXEC_crossval_leave_one_out
from library.EXEC_join_functions import EXEC_merge_dataframes_on_columns
from library.net_charges import EXEC_load_all_net_charges
from library.scale_features import EXEC_scale_globaly
import numpy as np
import pandas as pd

# OBSOLETE FUNCTION !!!
def EXEC_train_Glide_scoring_terms(CROSSVAL_PROTEINS, XTEST_PROTEINS, Settings):

    # The scoring terms of Glide SP
    Glide_terms = ['i_i_glide_rotatable_bonds',
          "r_epik_State_Penalty", "r_i_glide_lipo", 'r_i_glide_hbond', 'r_i_glide_metal', 'r_i_glide_rewards',
          'r_i_glide_evdw', 'r_i_glide_ecoul', 'r_i_glide_erotb', 'r_i_glide_esite', 'r_i_glide_emodel',
          'r_i_glide_energy', 'r_i_glide_einternal', 'i_i_glide_confnum', 'r_i_glide_eff_state_penalty']

    # TRAIN AND OPTIMIZE LEARNING MODEL ON CROSSVAL PROTEINS SET AND EVALUATE ON XTEST PROTEINS
    features_df = pd.concat([pd.read_csv(Settings.generated_file("_best_%s_scores.csv.gz" % Settings.HYPER_SELECT_BEST_TAUTOMER_BY, protein))
                             for protein in CROSSVAL_PROTEINS+XTEST_PROTEINS])\
        [['protein', 'basemolname', 'structvar', 'is_active',  "r_i_glide_ligand_efficiency",
        "r_i_docking_score", "r_i_glide_ligand_efficiency_sa"] + Glide_terms ]
    features_df["num_heavy_atoms"] = np.log(features_df["r_i_docking_score"]/features_df["r_i_glide_ligand_efficiency"] + 1)
    features_df["sa"] = features_df["r_i_docking_score"]/features_df["r_i_glide_ligand_efficiency_sa"]
    features_df['i_i_glide_rotatable_bonds'] = np.log(features_df['i_i_glide_rotatable_bonds'] + 1)

    # LOAD NET CHARGES
    charges_df = EXEC_load_all_net_charges(CROSSVAL_PROTEINS+XTEST_PROTEINS, Settings=Settings)
    features_df = EXEC_merge_dataframes_on_columns(["protein", "structvar"], features_df, charges_df)

    # SCALE FEATURES
    features_df.to_csv(Settings.generated_file(Settings.create_feature_csv_name("_features.csv.gz"), protein="all"))  # temporary fix

    features_df, feature_columns = EXEC_scale_globaly(features_df, feature_columns=Glide_terms)
    # features_df, feature_columns = EXEC_discretize_globaly(features_df, feature_columns=["P6C_Eint"])   # it's not the lowest per structvar and basemolname
    # features_df, feature_columns = EXEC_scale_by_protein(features_df)

    selected_features = Glide_terms
    EXEC_crossval_leave_one_out(features_df, selected_features, CROSSVAL_PROTEINS, XTEST_PROTEINS,
                                learning_model_type="Logistic Regression")