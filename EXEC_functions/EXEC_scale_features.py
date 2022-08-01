import logging
from library.scale_features import EXEC_standardize_by_protein, EXEC_standardize_globaly
from commons.EXEC_caching import EXEC_caching_decorator

lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, "Scaling features.", "_scaled_nonuniform_all",
                        full_csv_name=True, append_signature=True, prepend_all_proteins=True)
def EXEC_scale_features(features_df, selected_features, Settings):
    print("Scaling features and saving them to CSV file (slow).")
    # features_df[selected_features] = features_df[selected_features].rank()
    # features_df, feature_columns = EXEC_scale_globaly(features_df, feature_columns=selected_features)
    features_df, feature_columns = EXEC_standardize_globaly(features_df, feature_columns=selected_features)
    # features_df, feature_columns = EXEC_discretize_globaly_automatically(features_df, feature_columns=selected_features)
    # features_df, feature_columns = EXEC_scale_by_protein(features_df, feature_columns=["P6C_Eint"])
    features_df, feature_columns = EXEC_standardize_by_protein(
        features_df, feature_columns=[f for f in selected_features if f.endswith("_Eint")
                                      or 'protein' in f or 'complex' in f])
    return features_df.dropna(axis=0)