import logging
import sys

from commons.EXEC_caching import EXEC_caching_decorator
from library.EXEC_join_functions import EXEC_inner_merge_dataframes
from library.utils.print_functions import ColorPrint

lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, "Merging SQM with Glide scores.", "_sqm_glide.csv.gz")
def EXEC_merge_SQM_with_Glide(sqm_df, glide_top_poses_df, Settings):

    try:
        return EXEC_inner_merge_dataframes(sqm_df, glide_top_poses_df) \
            .assign(num_heavy_atoms=lambda df: df["r_i_docking_score"].div(df["r_i_glide_ligand_efficiency"]))
    except ValueError:
        ColorPrint("WARNING: if you got the following error, then just re-execute EXEC_master_script.py"
                   " and it will disappear. I couldn't guess the source of the error.", "WARNING")
        ColorPrint("ValueError: You are trying to merge on object and int64 columns. If you wish to "
                   "proceed you should use pd.concat", "FAIL")
        sys.exit(0)