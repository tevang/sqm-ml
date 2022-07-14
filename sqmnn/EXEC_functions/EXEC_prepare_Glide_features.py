from sqmnn.EXEC_functions.EXEC_add_activities import EXEC_add_activities
from sqmnn.EXEC_functions.EXEC_best_structvar import EXEC_keep_structvar_with_extreme_col_value
from sqmnn.EXEC_functions.EXEC_keep_best_Glide_score_per_basemolname import EXEC_keep_best_Glide_score_per_basemolname
from sqmnn.EXEC_functions.EXEC_load_Glide import EXEC_load_Glide
from sqmnn.EXEC_functions.EXEC_remove_outliers import EXEC_remove_outliers
from sqmnn.EXEC_functions.EXEC_remove_unscored_structvars import EXEC_remove_underscored_structvars
from sqmnn.EXEC_functions.EXEC_select_top_Glide_poses import EXEC_select_top_Glide_poses
from sqmnn.commons.EXEC_caching import EXEC_caching_decorator
from sqmnn.library.EXEC_script_utils import *
from sqmnn.library.add_protein_structvar_frame_columns import add_protein_structvar_frame_columns

lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, 'Preparing all Glide features.', '', full_csv_name=True,
                        append_signature=True, prepend_protein=True)
def EXEC_prepare_Glide_features(Settings):

    # LOAD AND PROCESS GLIDE PROPERTIES
    glide_df = EXEC_load_Glide(Settings=Settings)

    # KEEP ONLY TOP GLIDE POSES FOR SQM SCORING
    glide_top_poses_df = EXEC_select_top_Glide_poses(glide_df, Settings=Settings,
                                                     structvar_pose_sel_column='r_i_docking_score',
                                                     DeltaG=Settings.HYPER_KEEP_MAX_DeltaG_POSES,
                                                     N_poses=Settings.HYPER_KEEP_MAX_N_POSES)

    # ADD EXPERIMENTAL ACTIVITIES
    glide_top_poses_activities_df = add_protein_structvar_frame_columns(
        EXEC_add_activities(glide_top_poses_df, Settings=Settings), Settings=Settings)


    # REMOVE STRUCTVARS THAT WHERE NOT ADEQUATELY SCORED
    scores_df = EXEC_remove_underscored_structvars(count_scored_poses_from_df=glide_top_poses_activities_df,
                                                   remove_structvars_from_df=glide_top_poses_activities_df,
                                                   glide_df=glide_top_poses_activities_df,
                                                   Settings=Settings)

    # REMOVE OUTLIERS IN THE CONTEXT OF THE MOLECULE (structvar)
    scores_df = EXEC_remove_outliers(scores_df, Settings=Settings)

    # FOR EACH MOLECULE KEEP ONLY THE STRUCTVAR WITH THE LOWEST Eint
    # best_scores_df = EXEC_keep_structvar_with_extreme_col_value(scores_df,
    #                                                             relation=min,
    #                                                             Settings=Settings)
    best_scores_df = EXEC_keep_best_Glide_score_per_basemolname(scores_df, Settings=Settings)

    return best_scores_df