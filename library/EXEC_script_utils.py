import os
import pandas as pd

import logging
lg = logging.getLogger(__name__)

def read_df_from_output_file(file_name, logger):
    logger.warning("Reading the resulting dataframe from " + file_name)
    return pd.read_csv(file_name)

def read_df_from_output_file_condition(file_name, force_computation):
    return os.path.exists(file_name) and (not force_computation)