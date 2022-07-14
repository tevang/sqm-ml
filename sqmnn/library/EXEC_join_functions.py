import pandas as pd
import numpy as np
from functools import reduce

def EXEC_merge_dataframes_on_columns(common_columns, *mol_features_dfs):
    """
    :param common_column: can be single string e.g. 'basemolname', 'structvar', or list of strings.
    :param mol_features_dfs: many dataframes pass as individual arguments or a list of dataframes passes as *df_list
    :return:
    """
    return reduce(lambda df1, df2: pd.merge(df1, df2, on=common_columns), mol_features_dfs)

def EXEC_inner_merge_dataframes(*mol_features_dfs):
    """
    :param mol_features_dfs: many dataframes pass as individual arguments or a list of dataframes passes as *df_list
    :return:
    """
    return reduce(lambda df1, df2:
                  pd.merge(df1, df2, on=np.intersect1d(df1.columns, df2.columns).tolist(), how='inner'),
                  mol_features_dfs)

def EXEC_outer_merge_dataframes(*mol_features_dfs):
    """
    :param mol_features_dfs: many dataframes pass as individual arguments or a list of dataframes passes as *df_list
    :return:
    """
    return reduce(lambda df1, df2:
                  pd.merge(df1, df2, on=np.intersect1d(df1.columns, df2.columns).tolist(), how='outer'),
                  mol_features_dfs)