from sqmnn.commons.EXEC_caching import EXEC_caching_decorator
import logging

from sqmnn.library.EXEC_join_functions import EXEC_inner_merge_dataframes, EXEC_outer_merge_dataframes
from sqmnn.library.deltaH import calc_overall_DeltaH

lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, "Computing DeltaH of all available scoring functions.", "_min_DeltaH.csv.gz")
def EXEC_compute_DeltaH(scores_Sconf_df, min_ligandE_column_name, DeltaH_column_name, Settings):

    if 'DeltaH' in Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY or \
            'DeltaH' in Settings.HYPER_SELECT_BEST_STRUCTVAR_POSE_BY:
        merge_function = EXEC_inner_merge_dataframes
    else:
        merge_function = EXEC_outer_merge_dataframes

    return merge_function(*[calc_overall_DeltaH(Settings.HYPER_PROTEIN, scores_Sconf_df, scoring_function=SF,
                                                min_ligandE_column_name=min_ligandE_column_name,
                                                DeltaH_column_name=DeltaH_column_name,
                                                Settings=Settings)
                            for SF in [c.replace("_Eint", "") for c in scores_Sconf_df.columns
                                       if c.endswith("_Eint")]
                            ] + [scores_Sconf_df])