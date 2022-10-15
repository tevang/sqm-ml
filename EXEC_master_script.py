#!/usr/bin/env python

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
import pretty_errors
import pandas as pd

from EXEC_functions.EXEC_keep_best_Glide_score_per_basemolname import EXEC_keep_best_Glide_score_per_basemolname
from EXEC_functions.EXEC_prepare_all_features import EXEC_prepare_all_features
from EXEC_functions.EXEC_remove_uniform_features import EXEC_remove_uniform_features
from EXEC_functions.EXEC_sanity_checks import sanity_checks
from EXEC_functions.EXEC_scale_features import EXEC_scale_features
from EXEC_functions.EXEC_create_feature_vectors import EXEC_create_feature_vectors
from EXEC_functions.cross_validation.leave_one_out import EXEC_crossval_leave_one_out
from SQM_ML_pipeline_settings import settings
from library.evaluate_model import evaluate_without_learning
from library.extract_nonuniform_columns import extract_nonuniform_columns


def cmdlineparse():
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description="""
DESCRIPTION:
    This script calculates 
                            """,
                            epilog="""
    EXAMPLE:
    ./EXEC_master_script.py -xtp 'A2A,ACHE,AR,CATL,DHFR,EPHB4,GBA,GR,HIV1RT,JNK2,MDM2,MK2,PARP-1,PPARG,SARS-HCoV,SIRT2,TPA,TP'    
    
    """)
    parser.add_argument("-xtp", "--xtest-proteins", dest="XTEST_PROTEINS", required=True, type=str, default="",
                        help="Protein names of the external test set separated by ','. E.g. -tp 'MARK4,PARP-1,JNK2'")
    parser.add_argument("-cvp", "--crossval-proteins", dest="CROSSVAL_PROTEINS", required=False, type=str, default="",
                        help="Protein names of the cross-validation set (training and validation of SQM-ML)"
                             " separated by ','. E.g. -tp 'MARK4,PARP-1,JNK2'")
    parser.add_argument("-exec_dir", dest="EXECUTION_DIR_NAME", required=False, type=str, default=None,
                        help="Name of the execution directory where all output files will be saved. "
                             "Default: %(default)s")
    parser.add_argument("-n_comp", dest="N_COMPONENTS", required=False, type=int, default=None,
                        help="Number of components for UMAP (if activated). "
                             "Default: %(default)s")
    parser.add_argument("-ml_alg", dest="ML_ALGORITHM", required=False, type=str, default=None,
                        help="The Machine Learning Algorithm. "
                             "Default: %(default)s")
    parser.add_argument("-f", dest="FORCE_COMPUTATION", required=False, default=False, action='store_true',
                        help="Force computation of scores even if the respective CSV files exist. Default: %(default)s")
    parser.add_argument("-max_depth", dest="max_depth", required=False, type=int, default=None,
                        help="Default: %(default)s")
    parser.add_argument("-max_features", dest="max_features", required=False, type=str, default=None,
                        help="Default: %(default)s")
    parser.add_argument("-min_samples_leaf", dest="min_samples_leaf", required=False, type=int, default=None,
                        help="Default: %(default)s")
    parser.add_argument("-min_samples_split", dest="min_samples_split", required=False, type=int, default=None,
                        help="Default: %(default)s")

    args = parser.parse_args()
    return args

def launch_pipeline(CROSSVAL_PROTEINS_STRING, XTEST_PROTEINS_STRING, EXECUTION_DIR_NAME, FORCE_COMPUTATION,
                    N_COMPONENTS, ML_ALGORITHM, max_depth, max_features, min_samples_leaf, min_samples_split,
                    Settings=None):

    if not Settings:
        Settings = settings()

    Settings.HYPER_FORCE_COMPUTATION = FORCE_COMPUTATION
    Settings.HYPER_EXECUTION_DIR_NAME = Settings.HYPER_EXECUTION_DIR_NAME if EXECUTION_DIR_NAME is None else EXECUTION_DIR_NAME
    Settings.N_COMPONENTS = Settings.N_COMPONENTS if N_COMPONENTS is None else N_COMPONENTS
    Settings.HYPER_LEARNING_MODEL_TYPE = Settings.HYPER_LEARNING_MODEL_TYPE if ML_ALGORITHM is None else ML_ALGORITHM
    Settings.max_depth = Settings.max_depth if max_depth is None else max_depth
    Settings.max_features = Settings.max_features if max_features is None else max_features
    Settings.min_samples_leaf = Settings.min_samples_leaf if min_samples_leaf is None else min_samples_leaf
    Settings.min_samples_split = Settings.min_samples_split if min_samples_split is None else min_samples_split

    CROSSVAL_PROTEINS = [p for p in CROSSVAL_PROTEINS_STRING.split(",") if len(p) > 0]
    XTEST_PROTEINS = [p for p in XTEST_PROTEINS_STRING.split(",") if len(p) > 0]

    # CREATE EXECUTION DIR AND SAVE HYPER-PARAMETER VALUES
    Path(Settings.HYPER_SQM_ML_ROOT_DIR + "/" + Settings.HYPER_EXECUTION_DIR_NAME + "/") \
        .mkdir(parents=True, exist_ok=True)
    hyper_params = {hp: Settings.__getattribute__(hp) for hp in Settings.__dir__() if hp.startswith("HYPER")}
    pd.DataFrame([hyper_params]).T.to_csv(Settings.generated_file("_parameters.list", "hyper"),
                                          sep="\t")

    # GENERATE FEATURES FOR ALL PROTEINS
    for protein in CROSSVAL_PROTEINS + XTEST_PROTEINS:

        Settings.HYPER_PROTEIN = protein
        print("Generating features for %s" % protein)
        sanity_checks(protein, Settings=Settings)

        # best_Eint_scores_df, best_meanrank_Eint_scores_df, best_DeltaH_scores_df, best_DeltaHstar_scores_df, best_Glide_scores_df = \
        best_scores_df = EXEC_prepare_all_features(Settings=Settings)
        eval_data = []
        eval_data.append(
            evaluate_without_learning(best_scores_df,
                                      column="%s_%s+Sconf" % (Settings.HYPER_FUSION_METHOD,
                                                              Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY)
                                      if Settings.HYPER_USE_SCONF else
                                      "%s_%s" % (Settings.HYPER_FUSION_METHOD,
                                                 Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY)))

        best_glide_df = EXEC_keep_best_Glide_score_per_basemolname(pd.read_csv(
            Settings.generated_file("_scores_activities.csv.gz")), Settings=Settings,
            basemolname_score_sel_column="r_i_docking_score",
            structvar_pose_sel_column="r_i_docking_score")

        eval_data.append(evaluate_without_learning(best_glide_df, "r_i_docking_score", best_scores_df))

        pd.DataFrame(data=eval_data,
                     columns=["score_name", "num_of_actives", "num_of_inactives", "AUC-ROC", "AUC-CROC",
                              "AUC-BEDROC", "minimum", "mean", "max", "stdev", "range", "scaled_stdev"])\
            .to_csv(Settings.generated_file("_evaluation.csv.gz", protein), index=False)

    if len(CROSSVAL_PROTEINS + XTEST_PROTEINS) > 1:
        Settings.ALL_PROTEINS = sorted(CROSSVAL_PROTEINS + XTEST_PROTEINS)

        # TRAIN AND OPTIMIZE LEARNING MODEL ON CROSSVAL PROTEINS SET AND EVALUATE ON XTEST PROTEINS
        features_df = EXEC_create_feature_vectors(CROSSVAL_PROTEINS, XTEST_PROTEINS, Settings=Settings)

        # TODO added to extract PLEC features, maybe there is a more elegant way
        plec_list = features_df.filter(
            regex='^plec[0-9]+$').columns.values.tolist() if Settings.HYPER_PLEC else []

        nonuniform_features_df = EXEC_remove_uniform_features(features_df, Settings=Settings)

        # added PLEC
        nonuniform_features_list = extract_nonuniform_columns(
            nonuniform_features_df, selected_columns=Settings.HYPER_SQM_FEATURES + Settings.HYPER_GLIDE_DESCRIPTORS +
                                                     Settings.HYPER_2D_DESCRIPTORS + Settings.HYPER_3D_COMPLEX_DESCRIPTORS +
                                                     plec_list)

        # SCALE FEATURES (ADJUST IT INTERNALLY)
        scaled_features_df = EXEC_scale_features(
            nonuniform_features_df, selected_features=nonuniform_features_list, Settings=Settings)

        nonuniform_features_list = [i for i in nonuniform_features_list if not i.startswith('plec')]
        EXEC_crossval_leave_one_out(scaled_features_df, selected_features=nonuniform_features_list,
                                    CROSSVAL_PROTEINS=CROSSVAL_PROTEINS, XTEST_PROTEINS=XTEST_PROTEINS,
                                    n_neighbors=Settings.N_NEIGHBORS, min_dist=Settings.MIN_DIST,
                                    n_components=Settings.N_COMPONENTS, metric=Settings.METRIC,
                                    learning_model_type=Settings.HYPER_LEARNING_MODEL_TYPE,
                                    sample_weight_type=Settings.SAMPLE_WEIGHTS_TYPE,
                                    compress_PCA=Settings.HYPER_COMPRESS_PLEC_PCA,
                                    compress_UMP=Settings.HYPER_COMPRESS_PLEC_UMAP,
                                    PLEC_pca_variance_explained_cutoff=Settings.HYPER_PLEC_PCA_VARIANCE_EXPLAINED_CUTOFF,
                                    perm_n_repeats=Settings.PERM_N_REPEATS,
                                    plot_SHAP=Settings.PLOT_SHAP,
                                    write_SHAP=Settings.WRITE_SHAP,
                                    plots_dir=Settings.HYPER_PLOTS_DIR,
                                    features_for_training=Settings.FEATURES_FOR_TRAINING,
                                    max_depth=Settings.max_depth, max_features=Settings.max_features,
                                    min_samples_leaf=Settings.min_samples_leaf,
                                    min_samples_split=Settings.min_samples_split
                                    )

if __name__ == "__main__":

    args = cmdlineparse()
    launch_pipeline(args.CROSSVAL_PROTEINS, args.XTEST_PROTEINS, args.EXECUTION_DIR_NAME,
                    args.FORCE_COMPUTATION, args.N_COMPONENTS, args.ML_ALGORITHM,
                    args.max_depth, args.max_features, args.min_samples_leaf, args.min_samples_split)