from commons.EXEC_caching import EXEC_caching_decorator
import logging
lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, "Keeping for each molecule only the structural variant with the best score.",
                        "_scores_best_structvar.csv.gz")
def EXEC_keep_structvar_with_extreme_col_value(df, Settings, relation=min):
    """
    'extreme' can be either maximum or minimum. E.g.

    keep_structvar_with_extreme_col_value(scores_df, col_name="Eint", relation=min)

    keep_structvar_with_extreme_col_value(scores_df, col_name="r_epik_State_Penalty", relation=min)

    :param df:
    :param col_name:
    :param relation: 'max' or 'min'
    :return:
    """
    def _structvar_column(settings):
        return settings.HYPER_SELECT_BEST_STRUCTVAR_POSE_BY \
            if settings.HYPER_SELECT_BEST_STRUCTVAR_POSE_BY.startswith("r_i_") \
            else settings.HYPER_SFs[0] + "_" + settings.HYPER_SELECT_BEST_STRUCTVAR_POSE_BY

    def _basemolname_columns(settings):
        return settings.HYPER_SELECT_BEST_BASEMOLNAME_POSE_BY \
            if settings.HYPER_SELECT_BEST_BASEMOLNAME_POSE_BY.startswith("r_i_") \
            else settings.HYPER_SFs[0] + "_" + settings.HYPER_SELECT_BEST_BASEMOLNAME_POSE_BY

    def _best_row_indices(s, relation):
        if relation == min:
            return s.idxmin()
        elif relation == max:
            return s.idxmax()

    if len(Settings.HYPER_SFs) == 1: # keep the original values - don't convert to rank
        best_df = df.loc[_best_row_indices(df.groupby(by=["basemolname", "stereoisomer", "ionstate", "tautomer"])\
                                        [_structvar_column(Settings)], relation)] \
            .pipe(lambda df2: df2.loc[_best_row_indices(df2.groupby(by="basemolname") \
                                                        [_basemolname_columns(Settings)], relation)]) \

        best_df["%s_%s" % (Settings.HYPER_FUSION_METHOD,
                                            Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY)] = \
            best_df[Settings.HYPER_SFs[0] + "_" + Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY]

        return best_df

    # FIXME: make it work also for multiple scoring functions.
    # else:
    #     # Convert the selected column to ranks and combine all scoring functions to select the best pose and
    #     # structvar per compound
    #     for SF in Settings.HYPER_SFs:
    #         df["rank_%s_%s" % (SF, col_name)] = df["%s_%s" % (SF, col_name)].rank()
    #     df[Settings.HYPER_FUSION_METHOD+"_"+col_name] = fuse_columns(df=df,
    #                                               columns=["rank_%s_%s" % (SF, col_name) for SF in Settings.HYPER_SFs],
    #                                               method=Settings.HYPER_FUSION_METHOD)
    #     return df[df.groupby(by="basemolname")[Settings.HYPER_FUSION_METHOD+"_"+col_name] \
    #                   .transform(relation) == df[Settings.HYPER_FUSION_METHOD+"_"+col_name]]
