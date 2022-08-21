import logging

import pandas as pd

from library.utils.print_functions import Debuginfo
from EXEC_functions.EXEC_decompose_molnames import EXEC_decompose_molname
from commons.EXEC_caching import EXEC_caching_decorator

lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, "Loading Glide scores.", "_glide.csv.gz")
def EXEC_load_Glide(Settings):

    glide_df = EXEC_decompose_molname(pd.read_csv("%s/%s/%s_Glide_properties.csv.gz" % \
                       (Settings.HYPER_SQMNN_ROOT_DIR, Settings.HYPER_PROTEIN, Settings.HYPER_PROTEIN)))

    invalid_df = glide_df.loc[~glide_df.apply(lambda r: r["basemolname"] in r["molname"], axis=1), :]
    assert invalid_df.size == 0, \
        Debuginfo("Dataframe merging has failed due to the following records:\n" + invalid_df.to_string(), "FAIL")

    return glide_df