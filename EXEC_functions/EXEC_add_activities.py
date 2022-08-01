import logging
import pandas as pd
from commons.EXEC_caching import EXEC_caching_decorator
lg = logging.getLogger(__name__)


@EXEC_caching_decorator(lg, "Merging SQM with Glide scores.", "_scores_activities.csv.gz")
def EXEC_add_activities(scores_df, Settings):

    return pd.merge(scores_df,
                    pd.read_csv("%s/%s/%s_activities.txt" % (Settings.HYPER_SQMNN_ROOT_DIR, Settings.HYPER_PROTEIN, Settings.HYPER_PROTEIN),
                                delim_whitespace=True,
                                comment="#",
                                names=["basemolname", "is_active"]) \
                    .astype({'basemolname': str, 'is_active': int}) \
                    .assign(basemolname=lambda df: df["basemolname"].str.lower()),
                    on="basemolname")