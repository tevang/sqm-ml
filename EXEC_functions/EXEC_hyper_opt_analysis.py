import os
from operator import itemgetter

import pandas as pd
import logging

from library.dataframe_functions import remove_uniform_columns_from_dataframe
from library.global_fun import list_files, save_pickle
from commons.EXEC_caching import EXEC_caching_decorator
from scipy.stats import zscore

lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, "Preparing all SQM features.", "hyper_opt_df.csv.gz",
                        full_csv_name=True)
def EXEC_collect_hyper_results(Settings, hyper_dir="."):

    hyper_df = pd.concat([pd.DataFrame([{kv[0]: kv[1].strip() for kv in [line.split('\t')
                                                    for line in open(hyper_file)][1:] }])
     for hyper_file in ["%s/hyper_parameters.list" % comb_folder
                        for comb_folder in list_files(folder=hyper_dir, pattern="comb[0-9]+", rel_path=True)]],
              ignore_index=True)


    def _join_evaluation(r):
        hyper_folder = r["HYPER_SQMNN_ROOT_DIR"] + "/" + r["HYPER_EXECUTION_DIR_NAME"]
        eval_df = None
        for eval_file in list_files(hyper_folder, pattern=".*_evaluation.csv.gz", full_path=True):
            protein = os.path.basename(eval_file).split('_')[0]
            eval_df = pd.read_csv(eval_file)
            eval_df = eval_df.reset_index(drop=True) \
                .join(pd.concat([r.to_frame().T] * eval_df.shape[0]) \
                      .reset_index(drop=True)).assign(HYPER_PROTEIN=protein)
        return eval_df

    return pd.concat([_join_evaluation(r) for i, r in hyper_df.iterrows()], ignore_index=True)


def EXEC_report_hyper_results(hyper_dir=".",
                              out_csv="hyper_opt_results.csv.gz",
                              VARIABLE_HYPER_PARAMETERS=("HYPER_RATIO_SCORED_POSES",
                                                         "HYPER_OUTLIER_MAD_THRESHOLD", "HYPER_KEEP_MAX_DeltaG_POSES"),
                              DROP_COLUMNS=("HYPER_KEEP_POSE_COLUMN", "HYPER_EXECUTION_DIR_NAME"),
                              sort_by=['global_rank', 'rzscore_AUC_ROC_rounded']):
    """
    :param sort_by: 'global_rank' or 'within_protein_rank'
    """

    save_pickle('input.pkl', hyper_dir, out_csv, VARIABLE_HYPER_PARAMETERS, DROP_COLUMNS, sort_by)

    def _read_hyper_opt_csv_files():
        df_list = []
        for csv in list_files(hyper_dir, pattern=".*hyper_opt_df.csv.gz"):
            print("Reading hyper-optimization results from file %s" % csv)
            df_list.append(pd.read_csv(csv))
        if len(df_list) == 1:
            return df_list[0]
        return pd.concat(df_list, ignore_index=True)

    hyper_opt_df = _read_hyper_opt_csv_files() \
        .drop(columns=DROP_COLUMNS) \
        .pipe(remove_uniform_columns_from_dataframe) \
        .drop_duplicates() \
        .pipe(lambda df: df[df["score_name"] != "r_i_docking_score"]) \
        .assign(AUC_ROC_rounded=lambda df: df["AUC-ROC"].round(2),
                global_rank=lambda df: df["AUC_ROC_rounded"].rank(ascending=False, method="dense"),
                within_protein_rank=lambda df: df.groupby("HYPER_PROTEIN", as_index=False)["AUC_ROC_rounded"] \
                .transform(lambda s: s.rank(ascending=False, method="dense")),
                rzscore_AUC_ROC_rounded=lambda df: -df.groupby("HYPER_PROTEIN", as_index=False)["AUC_ROC_rounded"] \
                .transform(zscore)) \
        .reset_index(drop=True)

    VARIABLE_HYPER_PARAMETERS = hyper_opt_df.columns[hyper_opt_df.columns.isin(VARIABLE_HYPER_PARAMETERS)].to_list()

    sorted_hyper_combs = sorted([list(n) + g[sort_by].mean().to_list() for n, g in hyper_opt_df \
                                 .groupby(by=["score_name"] + VARIABLE_HYPER_PARAMETERS,
                                          as_index=False)],
                                key=itemgetter(*reversed(range(-len(VARIABLE_HYPER_PARAMETERS)-len(sort_by), 0))))

    with open(out_csv, 'w') as f:
        for i, hyper_comb in enumerate(sorted_hyper_combs):
            f.write("\nCombination %i: %s average AUC-ROC %s %.2f\n" %
                    (i+1, hyper_comb[:len(VARIABLE_HYPER_PARAMETERS)+1], sort_by, hyper_comb[-1]))
            f.write(hyper_opt_df[(hyper_opt_df[["score_name"] + VARIABLE_HYPER_PARAMETERS] ==
                          hyper_comb[:len(VARIABLE_HYPER_PARAMETERS)+1]).all(axis=1)] \
                  .sort_values(by="HYPER_PROTEIN", ascending=True)
                      [["score_name", "HYPER_PROTEIN"] + sort_by + ["AUC_ROC_rounded"] + VARIABLE_HYPER_PARAMETERS] \
                    .to_string() + "\n")
