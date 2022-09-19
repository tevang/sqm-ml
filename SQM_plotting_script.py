#!/usr/bin/env python
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import pretty_errors
import pandas as pd

from EXEC_functions.cross_validation.loo_plots import EXEC_crossval_plots
from SQMNN_pipeline_settings import Settings


def cmdlineparse():
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description="""
DESCRIPTION:
    This script calculates 
                            """,
                            epilog="""
    EXAMPLE:
    ./SQM_plotting_script.py -xtp 'A2A,ACHE,AR,CATL,DHFR,EPHB4,GBA,GR,HIV1RT,JNK2,MDM2,MK2,PARP-1,PPARG,SARS-HCoV,SIRT2,TPA,TP'
    
    """)
    parser.add_argument("-xtp", "--xtest-proteins", dest="XTEST_PROTEINS", required=False, type=str, default="",
                        help="Protein names of the external test set separated by ','. E.g. -tp 'MARK4,PARP-1,JNK2'")
    parser.add_argument("-cvp", "--crossval-proteins", dest="CROSSVAL_PROTEINS", required=False, type=str, default="",
                        help="Protein names of the cross-validation set (training and validation of SQM-ML)"
                             " separated by ','. E.g. -tp 'MARK4,PARP-1,JNK2'")
    parser.add_argument("-exec_dir", dest="EXECUTION_DIR_NAME", required=False, type=str, default="execution_dir",
                        help="Name of the execution directory where all output files will be saved. "
                             "Default: %(default)s")
    parser.add_argument("-f", dest="FORCE_COMPUTATION", required=False, default=False, action='store_true',
                        help="Force computation of scores even if the respective CSV files exist. Default: %(default)s")

    args = parser.parse_args()
    return args

def launch_plotting(CROSSVAL_PROTEINS_STRING, XTEST_PROTEINS_STRING, EXECUTION_DIR_NAME, FORCE_COMPUTATION,
                    settings=None):

    if not settings:
        settings = Settings()

    settings.HYPER_EXECUTION_DIR_NAME = EXECUTION_DIR_NAME
    CROSSVAL_PROTEINS = [p for p in CROSSVAL_PROTEINS_STRING.split(",") if len(p) > 0]
    XTEST_PROTEINS = [p for p in XTEST_PROTEINS_STRING.split(",") if len(p) > 0]

    # Load scaled features and plot
    scaled_features_df = pd.read_csv(os.path.join(
        Settings.HYPER_SQMNN_ROOT_DIR, Settings.HYPER_EXECUTION_DIR_NAME,
        "%i_proteins" % len(Settings.ALL_PROTEINS) + "_scaled_nonuniform_all" + Settings.create_feature_csv_name()))
    EXEC_crossval_plots(scaled_features_df, CROSSVAL_PROTEINS=CROSSVAL_PROTEINS, XTEST_PROTEINS=XTEST_PROTEINS,)


if __name__ == "__main__":

    args = cmdlineparse()
    launch_plotting(args.CROSSVAL_PROTEINS, args.XTEST_PROTEINS, args.EXECUTION_DIR_NAME, args.FORCE_COMPUTATION)