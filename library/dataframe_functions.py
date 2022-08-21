from multiprocessing import Pool
import numpy as np
import pandas as pd

def remove_uniform_columns_from_dataframe(dataframe):
    """
    Drops constant value columns of pandas dataframe.
    """
    return dataframe.loc[:, (dataframe != dataframe.iloc[0]).any()]


def print_whole_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
