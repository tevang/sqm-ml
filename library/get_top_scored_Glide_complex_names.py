import pandas as pd

from lib.utils.print_functions import Debuginfo
from EXEC_functions.EXEC_decompose_molnames import EXEC_decompose_molname


def get_top_scored_Glide_complex_names(Glide_properties_csv,
                                       structvar_pose_sel_column="r_i_docking_score",
                                       DeltaG=1.0,
                                       N_poses=100):
    """
    This method is essentially a copy of EXEC_load_Glide() and EXEC_select_top_Glide_poses() but without the
    need to provide Settings object.
    """
    # LOAD AND PROCESS GLIDE PROPERTIES
    glide_df = EXEC_decompose_molname(pd.read_csv(Glide_properties_csv))

    invalid_df = glide_df.loc[~glide_df.apply(lambda r: r["basemolname"] in r["molname"], axis=1), :]
    assert invalid_df.size == 0, \
        Debuginfo("Dataframe merging has failed due to the following records:\n" + invalid_df.to_string(), "FAIL")

    # KEEP ONLY TOP GLIDE POSES FOR SQM SCORING
    glide_df = glide_df \
        .groupby(by=['basemolname', 'stereoisomer', 'ionstate', 'tautomer'], as_index=False) \
        .apply(lambda g: g.assign(
        is_valid_pose=(g[structvar_pose_sel_column] - g[structvar_pose_sel_column].min() <= DeltaG) &
                      (g[structvar_pose_sel_column].rank(method="min") <= N_poses)))

    # RETURN THE VALID COMPLEX NAMES
    return glide_df.loc[glide_df['is_valid_pose'], 'molname'] + '_frm1'





