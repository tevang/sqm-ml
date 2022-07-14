from multiprocessing import Pool
import numpy as np
import pandas as pd

def parallelize_dataframe(df, func, n_cores=4):
    """
     It breaks the dataframe into n_cores parts, and spawns n_cores processes which apply the
     function to all the pieces. Once it applies the function to all the split dataframes, it
     just concatenates the split dataframe and returns the full dataframe back.

    :param df:  dataframe
    :param func:    function to be applied
    :param n_cores:
    :return:
    """
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def remove_uniform_columns_from_dataframe(dataframe):
    """
    Drops constant value columns of pandas dataframe.
    """
    return dataframe.loc[:, (dataframe != dataframe.iloc[0]).any()]


def print_whole_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
