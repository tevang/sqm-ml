from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import minmax_scale, KBinsDiscretizer, StandardScaler

# SCALE FEATURES BY PROTEIN OR SCALE ALL PROTEINS TOGETHER
from lib.featvec.invariants import Discretize
from lib.global_fun import save_pickle

def _get_all_valid_feature_columns(features_df):
    return [c for c in features_df.columns if "name" not in c and c != "structvar"
            and "protein" != c and c != "is_active"]

def EXEC_scale_globaly(features_df, feature_columns=[]):

    if not feature_columns:
        feature_columns = _get_all_valid_feature_columns(features_df)

    features_df[feature_columns] = features_df[feature_columns].apply(minmax_scale, axis=0)

    return features_df, feature_columns

def EXEC_standardize_globaly(features_df, feature_columns=[]):

    if not feature_columns:
        feature_columns = _get_all_valid_feature_columns(features_df)

    def _scaler(X):
        return StandardScaler().fit_transform(X.to_numpy().reshape(-1, 1))[:,0]
    features_df[feature_columns] = features_df[feature_columns].apply(_scaler, axis=0)[feature_columns]

    return features_df, feature_columns

def EXEC_discretize_globaly_automatically(features_df, feature_columns):

    if not feature_columns:
        feature_columns = _get_all_valid_feature_columns(features_df)

    features_df[feature_columns] = features_df[feature_columns].apply(Discretize().fit_transform, axis=0)

    return features_df, feature_columns

def EXEC_scale_by_protein(features_df, feature_columns):

    if not feature_columns:
        feature_columns = _get_all_valid_feature_columns(features_df)

    features_df[feature_columns] = features_df.groupby("protein")[feature_columns].transform(minmax_scale)

    return features_df, feature_columns


def EXEC_standardize_by_protein(features_df, feature_columns):

    if not feature_columns:
        feature_columns = _get_all_valid_feature_columns(features_df)

    def _scaler(X):
        return StandardScaler().fit_transform(X.to_numpy().reshape(-1, 1))[:,0]

    features_df[feature_columns] = features_df.groupby("protein")[feature_columns] \
        .transform(_scaler)

    return features_df, feature_columns