from lib.global_fun import get_structvar
from commons.EXEC_caching import EXEC_caching_decorator
import logging

lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, "Keeping only the top Glide poses for SQM scoring.",
                        "_glide_top_poses.csv.gz")
def EXEC_select_top_Glide_poses(df, Settings, structvar_pose_sel_column="r_i_docking_score", DeltaG=5.0, N_poses=100,
                                is_absolute_DeltaG=False):
    if is_absolute_DeltaG:
        # add a column with the number of poses per structvar within the DeltaG window
        df['is_valid_pose'] = df[structvar_pose_sel_column] - df[structvar_pose_sel_column].min() <= DeltaG

    else:
        # ATTENTION: without g.assign() the correct order is not saved!
        df = df \
            .groupby(by=['basemolname', 'stereoisomer', 'ionstate', 'tautomer'], as_index=False) \
            .apply(lambda g: g.assign(
            is_valid_pose=(g[structvar_pose_sel_column] - g[structvar_pose_sel_column].min() <= DeltaG) &
                          (g[structvar_pose_sel_column].rank(method="min") <= N_poses)))

    # add a column with the number of poses per structvar within the DeltaG window
    return df[df['is_valid_pose']] \
        .assign(structvar=lambda df: df['molname'].apply(get_structvar)) \
        .groupby('structvar', as_index=False) \
        .apply(lambda g: g.assign(valid_pose_num=g.shape[0]))