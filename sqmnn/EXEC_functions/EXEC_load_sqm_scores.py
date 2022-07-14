from functools import reduce

import pandas as pd
import logging

from sqmnn.commons.EXEC_caching import EXEC_caching_decorator
from sqmnn.library.EXEC_join_functions import EXEC_inner_merge_dataframes, EXEC_outer_merge_dataframes

lg = logging.getLogger(__name__)

def _read_sqm_score_file(sqm_score_file, SF, Settings):

    if sqm_score_file.endswith(".csv"):
        return pd.read_csv(sqm_score_file) \
            .astype({'basemolname': str, 'stereoisomer': str, 'ionstate': str, 'tautomer': str, 'pose': str}) \
            .rename(columns={"Eint": SF + "_Eint", "complexE": SF + "_complexE",
                             "ligandE_bound": SF + "_ligandE_bound", "proteinE_bound": SF + "_proteinE_bound"}) \
            .assign(basemolname=lambda df: df["basemolname"].str.lower(),
                    protein=Settings.HYPER_PROTEIN,
                    structvar=lambda df: df \
                    .apply(lambda r: "%s_stereo%s_ion%s_tau%s" %
                                     (r.basemolname, r.stereoisomer, r.ionstate, r.tautomer), axis=1))
    else:
        return pd.read_csv(sqm_score_file,
                           delim_whitespace=True,
                           comment="#") \
            .astype({'molname': str, 'stereoisomer': str, 'ionstate': str, 'tautomer': str, 'pose': str}) \
            .rename(columns={"molname": "basemolname", "Eint": SF+"_Eint", "complexE": SF+"_complexE",
                             "ligandE_bound": SF+"_ligandE_bound", "proteinE_bound": SF+"_proteinE_bound"}) \
            .assign(basemolname=lambda df: df["basemolname"].str.lower(),
                    protein=Settings.HYPER_PROTEIN,
                    structvar=lambda df: df \
                    .apply(lambda r: "%s_stereo%s_ion%s_tau%s" %
                                     (r.basemolname, r.stereoisomer, r.ionstate, r.tautomer) , axis=1))


@EXEC_caching_decorator(lg, "Loading PM6_COSMO, PM6_COSMO2 and PM7_COSMO scores.", '_sqm.csv.gz')
def EXEC_load_sqm_scores(how, Settings):

    if Settings.HYPER_SQM_FOLDER_SUFFIX:
        ALL_SCORES_dict = {
            "P6C": "%s/%s/PM6_COSMO%s/ALL_RESULTS.csv" % (Settings.HYPER_SQMNN_ROOT_DIR, Settings.HYPER_PROTEIN,
                                                      Settings.HYPER_SQM_FOLDER_SUFFIX),
            # "P6C2": "%s/%s/PM6_COSMO2%s/ALL_RESULTS" % (Settings.HYPER_SQMNN_ROOT_DIR, Settings.HYPER_PROTEIN,
            # Settings.HYPER_SQM_FOLDER_SUFFIX),
            # "P7C": "%s/%s/PM7_COSMO%s/ALL_RESULTS" % (Settings.HYPER_SQMNN_ROOT_DIR, Settings.HYPER_PROTEIN,
            # Settings.HYPER_SQM_FOLDER_SUFFIX)
        }
    else:
        ALL_SCORES_dict = {
            "P6C": "%s/%s/PM6_COSMO%s/ALL_RESULTS" % (Settings.HYPER_SQMNN_ROOT_DIR, Settings.HYPER_PROTEIN,
                                                          Settings.HYPER_SQM_FOLDER_SUFFIX),
            # "P6C2": "%s/%s/PM6_COSMO2%s/ALL_RESULTS" % (Settings.HYPER_SQMNN_ROOT_DIR, Settings.HYPER_PROTEIN,
            # Settings.HYPER_SQM_FOLDER_SUFFIX),
            # "P7C": "%s/%s/PM7_COSMO%s/ALL_RESULTS" % (Settings.HYPER_SQMNN_ROOT_DIR, Settings.HYPER_PROTEIN,
            # Settings.HYPER_SQM_FOLDER_SUFFIX)
        }
    df_list = [_read_sqm_score_file(ALL_SCORES_dict[SF], SF, Settings=Settings)
               for SF in Settings.HYPER_SFs]

    if how == 'inner':
        return EXEC_inner_merge_dataframes(*df_list)
    elif how == 'outer':
        return EXEC_outer_merge_dataframes(*df_list)