import logging
from commons.EXEC_caching import EXEC_caching_decorator

lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, "Removing uniform features.", "_nonuniform_all",
                        full_csv_name=True, append_signature=True, prepend_all_proteins=True)
def EXEC_remove_uniform_features(features_df, Settings):
    uniform_columns = features_df.loc[:, (features_df == features_df.iloc[0]).all()].columns
    print("Removing uniform features:", uniform_columns.to_list(), "\n")
    nonuniform_columns = features_df.columns[~features_df.columns.isin(uniform_columns)]
    return features_df.loc[:, nonuniform_columns]