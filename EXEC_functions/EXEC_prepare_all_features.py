from EXEC_functions.EXEC_add_activities import EXEC_add_activities
from EXEC_functions.EXEC_compute_DeltaH import EXEC_compute_DeltaH
from EXEC_functions.EXEC_compute_SQM_energy_fluctuations import EXEC_compute_all_sqm_energy_fluctuations
from EXEC_functions.EXEC_compute_ligandE import EXEC_compute_min_structvar_ligandE_free, \
    EXEC_compute_min_ligandE_bound
from EXEC_functions.EXEC_load_Glide import EXEC_load_Glide
from EXEC_functions.EXEC_load_Sconf import EXEC_add_Sconf_to_scores
from EXEC_functions.EXEC_load_sqm_scores import EXEC_load_sqm_scores
from EXEC_functions.EXEC_merge_SQM_with_Glide import EXEC_merge_SQM_with_Glide
from EXEC_functions.EXEC_remove_outliers import EXEC_remove_outliers
from EXEC_functions.EXEC_remove_unscored_structvars import EXEC_remove_underscored_structvars
from EXEC_functions.EXEC_select_top_Glide_poses import EXEC_select_top_Glide_poses
from commons.EXEC_caching import EXEC_caching_decorator
from library.EXEC_join_functions import EXEC_inner_merge_dataframes
from library.EXEC_script_utils import *
from EXEC_functions.EXEC_best_structvar import EXEC_keep_structvar_with_extreme_col_value

lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, 'Preparing all features.', '', full_csv_name=True,
                        append_signature=True, prepend_protein=True)
def EXEC_prepare_all_features(Settings):

    # LOAD AND PROCESS GLIDE PROPERTIES
    glide_df = EXEC_load_Glide(Settings=Settings)

    # KEEP ONLY TOP GLIDE POSES FOR SQM SCORING
    glide_top_poses_df = EXEC_select_top_Glide_poses(glide_df, Settings=Settings,
                                                     structvar_pose_sel_column='r_i_docking_score',
                                                     DeltaG=Settings.HYPER_KEEP_MAX_DeltaG_POSES,
                                                     N_poses=Settings.HYPER_KEEP_MAX_N_POSES,
                                                     is_absolute_DeltaG=Settings.HYPER_ABSOLUTE_MAX_DeltaG)

    # ADD EXPERIMENTAL ACTIVITIES
    glide_top_poses_activities_df = EXEC_add_activities(glide_top_poses_df, Settings=Settings)

    # READ IN SQM SCORES
    sqm_df = EXEC_load_sqm_scores(how=Settings.HYPER_HOW_COMBINE, Settings=Settings)

    # Merge SQM and Glide properties
    scores_df = EXEC_merge_SQM_with_Glide(sqm_df, glide_top_poses_activities_df, Settings=Settings)

    if Settings.HYPER_USE_SCONF or 'DeltaH' == Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY or \
            'DeltaH' == Settings.HYPER_SELECT_BEST_STRUCTVAR_POSE_BY:
        # LOAD CONFORMATIONAL ENTROPY
        scores_df = EXEC_add_Sconf_to_scores(scores_df, Settings=Settings)

    if 'DeltaH' in Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY or \
            'DeltaH' in Settings.HYPER_SELECT_BEST_STRUCTVAR_POSE_BY or \
            any(['DeltaH' in f or '_min_ligandE_bound' in f for f in Settings.HYPER_SQM_FEATURES]):

        if Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY.endswith('DeltaH') or \
                Settings.HYPER_SELECT_BEST_STRUCTVAR_POSE_BY.endswith('DeltaH'):

            min_ligandE_free_df = EXEC_compute_min_structvar_ligandE_free(Settings=Settings)

            # MERGE min_ligandE_free TO THE SCORES
            scores_df = EXEC_inner_merge_dataframes(scores_df, min_ligandE_free_df)
            # COMPUTE THE DeltaH
            scores_df = EXEC_compute_DeltaH(scores_df, min_ligandE_column_name='min_ligandE_free',
                                                DeltaH_column_name='DeltaH', Settings=Settings)

        elif Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY.endswith('DeltaHstar') or \
                Settings.HYPER_SELECT_BEST_STRUCTVAR_POSE_BY.endswith('DeltaHstar') or \
                any([f.endswith('DeltaHstar') or f.endswith('_min_ligandE_bound') for f in Settings.HYPER_SQM_FEATURES]):
            # MERGE min_ligandE_bound TO THE SCORES
            scores_df = EXEC_compute_min_ligandE_bound(scores_df, Settings=Settings)

            # COMPUTE THE DeltaH USING THE min_ligandE_bound INSTEAD OF min_ligandE_free
            scores_df = EXEC_compute_DeltaH(scores_df, min_ligandE_column_name='min_ligandE_bound',
                                                    DeltaH_column_name='DeltaHstar', Settings=Settings)
            os.rename(Settings.generated_file('_min_DeltaH.csv.gz'), Settings.generated_file('_min_DeltaHstar.csv.gz'))

    # REMOVE STRUCTVARS THAT WHERE NOT ADEQUATELY SCORED
    scores_df = EXEC_remove_underscored_structvars(count_scored_poses_from_df=sqm_df,
                                                   remove_structvars_from_df=scores_df,
                                                   glide_df=glide_top_poses_activities_df,
                                                   Settings=Settings)

    # REMOVE OUTLIERS IN THE CONTEXT OF THE MOLECULE (structvar)
    scores_df = EXEC_remove_outliers(scores_df, Settings=Settings)

    # FOR EACH MOLECULE KEEP ONLY THE STRUCTVAR WITH THE LOWEST Eint
    best_scores_df = EXEC_keep_structvar_with_extreme_col_value(scores_df,
                                                                relation=min,
                                                                Settings=Settings)

    # CALCULATE ENERGY FLUNCTUATIONS
    best_scores_Efluct_df = EXEC_compute_all_sqm_energy_fluctuations(best_scores_df, scores_df, Settings=Settings)

    if Settings.HYPER_USE_SCONF:
        for SF in Settings.HYPER_SFs:
            best_scores_Efluct_df['%s_%s+Sconf' % (Settings.HYPER_FUSION_METHOD,
                                                   Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY)] = \
                best_scores_Efluct_df[['%s_%s' % (Settings.HYPER_FUSION_METHOD,
                                                  Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY), 'schrodinger_Sconf']].sum(axis=1)

    return best_scores_Efluct_df