from sqmnn.commons.EXEC_caching import EXEC_caching_decorator
from sqmnn.library.EXEC_join_functions import EXEC_inner_merge_dataframes
from sqmnn.library.pose_count import get_total_pose_num, get_scored_pose_num

import logging

lg = logging.getLogger(__name__)


@EXEC_caching_decorator(lg, "Removing structural variants that were not adequately scored.",
                        "_sqm_onlyscored.csv.gz")
def EXEC_remove_underscored_structvars(count_scored_poses_from_df, remove_structvars_from_df, glide_df,
                                       Settings):

    count_scored_poses_from_df = count_scored_poses_from_df \
        .assign(total_pose_num=lambda df: df.apply(get_total_pose_num(df, glide_df), axis=1),
                scored_pose_num=lambda df: df.apply(get_scored_pose_num(count_scored_poses_from_df), axis=1))

    sqm_glide_df = EXEC_inner_merge_dataframes(count_scored_poses_from_df, remove_structvars_from_df)

    sqm_glide_df["scored_pose_ratio"] = sqm_glide_df["scored_pose_num"] / sqm_glide_df["total_pose_num"]
    sqm_glide_df[sqm_glide_df["scored_pose_ratio"] <= Settings.HYPER_RATIO_SCORED_POSES] \
        [["structvar", "scored_pose_ratio"]].drop_duplicates() \
        .to_csv(Settings.generated_file("_scored_pose_ratio_outliers.csv"), index=False)

    return sqm_glide_df[sqm_glide_df["scored_pose_num"] / sqm_glide_df["total_pose_num"] > Settings.HYPER_RATIO_SCORED_POSES]