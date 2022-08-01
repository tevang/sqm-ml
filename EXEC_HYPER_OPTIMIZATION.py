#!/usr/bin/env python
import os
from sklearn.model_selection import ParameterGrid

from lib.global_fun import list_files
from SQMNN_MASTER_SCRIPT import launch_pipeline
from SQMNN_pipeline_settings import Settings
from lib.multithreading.parallel_processing_tools import apply_function_to_list_of_args_and_concat_resulting_dfs

CPUs = 16

# TODO: check if r_i_docking_score and r_i_glide_emodel produce indeed same pose rankings
hyper_params_dict = {
    "HYPER_SQMNN_ROOT_DIR": {"/home2/thomas/Documents/QM_Scoring/SQM-ML"},
    "SQM_FOLDER_SUFFIX": {"_SQM_MM"},
    "HYPER_RATIO_SCORED_POSES": {0.8},
    "HYPER_OUTLIER_MAD_THRESHOLD": {2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0},
    "HYPER_KEEP_MAX_DeltaG_POSES": {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0},
    "HYPER_KEEP_POSE_COLUMN": {"r_i_docking_score"},
    "HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY": {"Eint"},
    "HYPER_SELECT_BEST_BASEMOLNAME_POSE_BY": {"Eint"},
    "HYPER_SELECT_BEST_STRUCTVAR_POSE_BY": {"complexE"},
    "CROSSVAL_PROTEINS_STRING": {'MARK4', 'PARP-1', 'MDM2', 'ACHE', 'MK2', 'EPHB4', 'PPARG', 'JNK2', 'SARS-HCoV',
                                 'TP', 'TPA', 'AR', 'SIRT2'},
    "XTEST_PROTEINS_STRING": {""},
    "HYPER_2D_DESCRIPTORS": {""},
    "HYPER_3D_COMPLEX_DESCRIPTORS": {""},
    "HYPER_GLIDE_DESCRIPTORS": {""}
}

args_list = [{**hyper_comb, **{"EXECUTION_DIR_NAME": "hyper_opt_dir/comb%i" % (i+1)}}
             for i, hyper_comb in enumerate(ParameterGrid(hyper_params_dict))]

def _launch_pipeline(**hyper_params):

    settings = Settings()
    for hyper_param, hyper_value in hyper_params.items():
        if hyper_param in settings.__dir__():
            settings.__setattr__(hyper_param, hyper_value)

    launch_pipeline(CROSSVAL_PROTEINS_STRING=hyper_params["CROSSVAL_PROTEINS_STRING"],
                    XTEST_PROTEINS_STRING=hyper_params["XTEST_PROTEINS_STRING"],
                    EXECUTION_DIR_NAME=hyper_params["EXECUTION_DIR_NAME"], FORCE_COMPUTATION=True, settings=settings)

    # Clean intermediate files to release disk space
    for fname in list_files(settings.HYPER_SQMNN_ROOT_DIR + "/" + settings.HYPER_EXECUTION_DIR_NAME,
                            pattern=".*",
                            full_path=True):
        if not (fname.endswith("_evaluation.csv.gz") or fname.endswith("hyper_parameters.list")):
            os.remove(fname)


apply_function_to_list_of_args_and_concat_resulting_dfs(_launch_pipeline,
                                                        args_list=args_list,
                                                        number_of_processors=CPUs)



"""

# ANALYZE HYPER-PARAMETER OPTIMIZATION RESULTS
from sqmnn.SQMNN_pipeline_settings import Settings
from sqmnn.EXEC_functions.EXEC_hyper_opt_analysis import EXEC_collect_hyper_results, EXEC_report_hyper_results
settings = Settings()
settings.HYPER_EXECUTION_DIR_NAME = "hyper_opt_dir"
settings.HYPER_SQMNN_ROOT_DIR = '/home/thomas/Documents/QM_Scoring/SQM-ML'
hyper_opt_df = EXEC_collect_hyper_results(hyper_dir=".", Settings=settings)

# Do the same on every PC you run HYPER-OPT and collect all such files. Then concatenate the dataframes.

EXEC_report_hyper_results(hyper_dir=".",
                         out_csv="hyper_opt_results.txt",
                         VARIABLE_HYPER_PARAMETERS=["HYPER_RATIO_SCORED_POSES", "HYPER_OUTLIER_MAD_THRESHOLD",
                                                    "HYPER_KEEP_MAX_DeltaG_POSES"],
                         DROP_COLUMNS=["HYPER_KEEP_POSE_COLUMN", "HYPER_EXECUTION_DIR_NAME"])

"""
