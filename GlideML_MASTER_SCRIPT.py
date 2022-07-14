#!/usr/bin/env python

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
import pretty_errors
import pandas as pd

from lib.global_fun import save_pickle
from sqmnn.EXEC_functions.EXEC_keep_best_Glide_score_per_basemolname import EXEC_keep_best_Glide_score_per_basemolname
from sqmnn.EXEC_functions.EXEC_prepare_Glide_features import EXEC_prepare_Glide_features
from sqmnn.EXEC_functions.EXEC_remove_uniform_features import EXEC_remove_uniform_features
from sqmnn.EXEC_functions.EXEC_sanity_checks import sanity_checks
from sqmnn.EXEC_functions.EXEC_scale_features import EXEC_scale_features
from sqmnn.EXEC_functions.EXEC_create_feature_vectors import EXEC_create_feature_vectors
from sqmnn.EXEC_functions.cross_validation.leave_one_out import EXEC_crossval_leave_one_out
from sqmnn.GlideML_pipeline_settings import Settings
from sqmnn.library.evaluate_model import evaluate_without_learning
## Parse command line arguments
from sqmnn.library.extract_nonuniform_columns import extract_nonuniform_columns


def cmdlineparse():
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description="""
DESCRIPTION:
    This script calculates 
                            """,
                            epilog="""
EXAMPLE:
./GlideML_MASTER_SCRIPT.py -xtp 'MARK4,PARP-1,MDM2,ACHE,MK2,EPHB4,PPARG,JNK2,SARS-HCoV,TP,TPA,AR,SIRT2,PYGL-in,DHFR,GBA,HSP90,CTSK,GR,SRC,A2A,TIE2,CATL,HIV1RT,PR,HIV1PR,PNP,PI3Kg,JAK3,PRKCQ,PIM-1,AURKB,PIM-2,P38-alpha,ERBB2,VEGFR1,BRAF,JNK3,LCK,ITK,IGF1R,COX2,HMGR,INHA,HDAC8,ER-beta,EGFR,AKT1,CDK2,uPA,GSK3B,FXA,TK,PDK1,VEGFR2,FGFR1,JNK1,AURKA,MMP2,PDE5,ADAM17,ALR2,CYP2A6,ACE2,HDAC2,ACE,KIF11'

PLEC fingeprints for MARK4, PNP, CYP2A6 are not available yet.
    """)
    parser.add_argument("-xtp", "--xtest-proteins", dest="XTEST_PROTEINS", required=False, type=str, default="",
                        help="Protein names of the external test set separated by ','. E.g. -tp 'MARK4,PARP-1,JNK2'")
    parser.add_argument("-cvp", "--crossval-proteins", dest="CROSSVAL_PROTEINS", required=False, type=str, default="",
                        help="Protein names of the cross-validation set (training and validation of SQM-ML)"
                             " separated by ','. E.g. -tp 'MARK4,PARP-1,JNK2'")
    parser.add_argument("-exec_dir", dest="EXECUTION_DIR_NAME", required=False, type=str, default="GlideML_execution_dir",
                        help="Name of the execution directory where all output files will be saved. "
                             "Default: %(default)s")
    parser.add_argument("-f", dest="FORCE_COMPUTATION", required=False, default=False, action='store_true',
                        help="Force computation of scores even if the respective CSV files exist. Default: %(default)s")

    args = parser.parse_args()
    return args

def launch_pipeline(CROSSVAL_PROTEINS_STRING, XTEST_PROTEINS_STRING, EXECUTION_DIR_NAME, FORCE_COMPUTATION,
                    settings=None):

    if not settings:
        settings = Settings()

    settings.HYPER_FORCE_COMPUTATION = FORCE_COMPUTATION
    settings.HYPER_EXECUTION_DIR_NAME = EXECUTION_DIR_NAME
    CROSSVAL_PROTEINS = [p for p in CROSSVAL_PROTEINS_STRING.split(",") if len(p) > 0]
    XTEST_PROTEINS = [p for p in XTEST_PROTEINS_STRING.split(",") if len(p) > 0]

    # CREATE EXECUTION DIR AND SAVE HYPER-PARAMETER VALUES
    Path(settings.HYPER_SQMNN_ROOT_DIR + "/" + settings.HYPER_EXECUTION_DIR_NAME + "/") \
        .mkdir(parents=True, exist_ok=True)
    hyper_params = {hp: settings.__getattribute__(hp) for hp in settings.__dir__() if hp.startswith("HYPER")}
    pd.DataFrame([hyper_params]).T.to_csv(settings.generated_file("_parameters.list", "hyper"),
                                          sep="\t")

    # GENERATE FEATURES FOR ALL PROTEINS
    for protein in CROSSVAL_PROTEINS + XTEST_PROTEINS:

        settings.HYPER_PROTEIN = protein
        print("Generating features for %s" % protein)
        sanity_checks(protein, Settings=settings)

        # best_Eint_scores_df, best_meanrank_Eint_scores_df, best_DeltaH_scores_df, best_DeltaHstar_scores_df, best_Glide_scores_df = \
        best_scores_df = EXEC_prepare_Glide_features(Settings=settings)
        eval_data = []
        eval_data.append(
            evaluate_without_learning(best_scores_df,
                                      column="%s_%s+Sconf" % (settings.HYPER_FUSION_METHOD,
                                                              settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY)
                                      if settings.HYPER_USE_SCONF else
                                      "%s_%s" % (settings.HYPER_FUSION_METHOD,
                                                 settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY)))

        pd.DataFrame(data=eval_data,
                     columns=["score_name", "num_of_actives", "num_of_inactives", "AUC-ROC", "AUC-CROC",
                              "AUC-BEDROC", "minimum", "mean", "max", "stdev", "range", "scaled_stdev"])\
            .to_csv(settings.generated_file("_evaluation.csv.gz", protein), index=False)

    if len(CROSSVAL_PROTEINS + XTEST_PROTEINS) > 1:
        settings.ALL_PROTEINS = sorted(CROSSVAL_PROTEINS + XTEST_PROTEINS)
        # TRAIN AND OPTIMIZE LEARNING MODEL ON CROSSVAL PROTEINS SET AND EVALUATE ON XTEST PROTEINS
        features_df = EXEC_create_feature_vectors(CROSSVAL_PROTEINS, XTEST_PROTEINS, Settings=settings)

        nonuniform_features_df = EXEC_remove_uniform_features(features_df, Settings=settings)
        nonuniform_features_list = extract_nonuniform_columns(
            nonuniform_features_df, selected_columns=settings.HYPER_GLIDE_DESCRIPTORS +
                                                     settings.HYPER_2D_DESCRIPTORS +
                                                     settings.HYPER_3D_COMPLEX_DESCRIPTORS +
                                                     settings.HYPER_3D_LIGAND_DESCRIPTORS +
                                                     settings.HYPER_LIGAND_Sconf_DESCRIPTORS)

        # SCALE FEATURES (ADJUST IT INTERNALLY)
        scaled_features_df = EXEC_scale_features(
            nonuniform_features_df, selected_features=nonuniform_features_list, Settings=settings)

        EXEC_crossval_leave_one_out(scaled_features_df,
                                    selected_features=nonuniform_features_list,
                                    CROSSVAL_PROTEINS=CROSSVAL_PROTEINS,
                                    XTEST_PROTEINS=XTEST_PROTEINS,
                                    learning_model_type=settings.HYPER_LEARNING_MODEL_TYPE,
                                    sample_weight_type=settings.SAMPLE_WEIGHTS_TYPE,
                                    compress_PLEC=settings.HYPER_COMPRESS_PLEC,
                                    compress_PMAPPER=settings.HYPER_COMPRESS_PMAPPER,
                                    PLEC_pca_variance_explained_cutoff=settings.HYPER_PLEC_PCA_VARIANCE_EXPLAINED_CUTOFF,
                                    PMAPPER_pca_variance_explained_cutoff=settings.HYPER_PMAPPER_PCA_VARIANCE_EXPLAINED_CUTOFF,
                                    select_best_features=settings.HYPER_SELECT_BEST_FEATURES,
                                    max_best_features=settings.HYPER_MAX_BEST_FEATURES)

if __name__ == "__main__":

    args = cmdlineparse()
    launch_pipeline(args.CROSSVAL_PROTEINS,
                    args.XTEST_PROTEINS,
                    args.EXECUTION_DIR_NAME,
                    args.FORCE_COMPUTATION)