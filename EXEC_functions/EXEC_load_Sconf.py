import pandas as pd
from commons.EXEC_caching import EXEC_caching_decorator
import logging

lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, "Loading conformational Entropy to the scores_df.",
                        "_scores_Sconf.csv.gz")
def EXEC_add_Sconf_to_scores(scores_df, Settings):

    Sconf_df = pd.read_csv("%s/%s/%s.schrodinger_confS_rmscutoff%.1f_ecutoff%i.csv.gz" %
                           (Settings.HYPER_SQMNN_ROOT_DIR, Settings.HYPER_PROTEIN, Settings.HYPER_PROTEIN, Settings.HYPER_SCONF_RMSD,
                            Settings.HYPER_SCONF_ECUTOFF)) \
        .rename(columns={"molname": "structvar", "schrodinger_confS": "schrodinger_Sconf"}) \
        .astype({"schrodinger_Sconf": float}) \
        .assign(structvar=lambda df: df["structvar"].str.lower())

    # MERGE CONFORMATIONAL ENTROPY TO THE SCORES
    # scores_df["structvar"] = scores_df["molname"].str \
    #     .extract("^(.*_stereo[0-9]+_ion[0-9]+_tau[0-9]+)_pose[0-9]+$") \
    #     .rename({0: 'structvar'}, axis=1)["structvar"] \
    #     .str.lower()

    return pd.merge(scores_df, Sconf_df[["structvar", "schrodinger_Sconf"]], on="structvar") \
        .assign(basemolname=lambda df: df["basemolname"].str.lower())