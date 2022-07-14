import logging

from sqmnn.commons.EXEC_caching import EXEC_caching_decorator
from sqmnn.library.EXEC_join_functions import EXEC_inner_merge_dataframes

lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, "Merging SQM with Glide scores.", "_sqm_glide.csv.gz")
def EXEC_merge_SQM_with_Glide(sqm_df, glide_top_poses_df, Settings):

    return EXEC_inner_merge_dataframes(sqm_df, glide_top_poses_df) \
        .assign(num_heavy_atoms=lambda df: df["r_i_docking_score"].div(df["r_i_glide_ligand_efficiency"]))