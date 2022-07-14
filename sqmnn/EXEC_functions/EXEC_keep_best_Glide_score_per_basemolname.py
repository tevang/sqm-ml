from sqmnn.commons.EXEC_caching import EXEC_caching_decorator
import logging

lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, "Keeping for each molecule only the structural variant with the best Glide score.",
                        "_glide_best_structvar.csv.gz")
def EXEC_keep_best_Glide_score_per_basemolname(df, Settings,
                                               basemolname_score_sel_column="r_i_docking_score",
                                               structvar_pose_sel_column="r_i_docking_score"):

    return df.loc[df.groupby(by=["basemolname", "stereoisomer", "ionstate", "tautomer"])[structvar_pose_sel_column] \
        .idxmin()] \
        .pipe(lambda df2: df2.loc[df2.groupby(by="basemolname")[basemolname_score_sel_column] \
              .idxmin()]) \
        .assign(nofusion_r_i_docking_score=lambda df: df['r_i_docking_score'])