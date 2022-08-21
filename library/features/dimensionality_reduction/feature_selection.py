import numpy as np
from sklearn.feature_selection import SelectFromModel


def best_feature_selector(model, features_df, train_rows, sel_columns, max_features):
    selector = SelectFromModel(estimator=model, prefit=True, threshold='mean', max_features=max_features)
    best_features = np.array(sel_columns)[selector.get_support()]
    print("FEATURES that passed the feature selection process:", best_features)
    unsel_columns = [c for c in features_df.columns if c not in sel_columns]
    new_features_df = features_df[unsel_columns + best_features.tolist()]
    selector.estimator.fit(X=features_df.loc[train_rows, best_features.tolist()],
                           y=features_df.loc[train_rows, 'is_active'])
    return selector.estimator, new_features_df, best_features.tolist()
