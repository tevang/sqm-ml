import logging

from sqmnn.commons.EXEC_caching import EXEC_caching_decorator
from sqmnn.library.EXEC_join_functions import EXEC_inner_merge_dataframes
from sqmnn.library.outliers import remove_outliers

lg = logging.getLogger(__name__)


@EXEC_caching_decorator(lg, "Removing outlier docking poses.", "_sqm_no_outliers.csv.gz")
def EXEC_remove_outliers(scores_onlyscored_df, Settings):

    if Settings.HYPER_REMOVE_OUTLIER_WRT_WHOLE_SET:
        return EXEC_inner_merge_dataframes(
            *[remove_outliers(scores_onlyscored_df,
                              column=Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY if Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY.startswith("r_i_")
                              else SF + "_" + Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY,
                              thresh=Settings.HYPER_OUTLIER_MAD_THRESHOLD,
                         min_scored_pose_num=Settings.HYPER_OUTLIER_MIN_SCORED_POSE_NUM,
                         outliers_csv=Settings.generated_file("_pose_outliers.csv"))
              for SF in Settings.HYPER_SFs])
    else:
        return EXEC_inner_merge_dataframes(
            *[scores_onlyscored_df.groupby(by=["basemolname", 'stereoisomer', 'ionstate', 'tautomer', 'frame'],
                             as_index=False) \
                  .apply(remove_outliers,
                         column=Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY if Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY.startswith("r_i_")
                         else SF + "_" + Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY,
                         thresh=Settings.HYPER_OUTLIER_MAD_THRESHOLD,
                         min_scored_pose_num=Settings.HYPER_OUTLIER_MIN_SCORED_POSE_NUM,
                         outliers_csv=Settings.generated_file("_pose_outliers.csv"))
              for SF in Settings.HYPER_SFs])