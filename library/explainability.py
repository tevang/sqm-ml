from operator import itemgetter

import pandas as pd
from sklearn.inspection import permutation_importance
import shap
from scipy.special import softmax
import numpy as np

from library.global_fun import save_pickle
from library.utils.print_functions import ColorPrint


def _return_perm_imp(model, x: pd.DataFrame, y: pd.Series, n_repeats: int):
    ColorPrint("Computing Permutation Feature Importances with {} repeats.".format(n_repeats), "OKBLUE")
    r = permutation_importance(model, x, y, n_repeats=n_repeats, random_state=0, n_jobs=-1)
    return pd.Series(r.importances_mean, index=x.columns)

def _write_Shapley_values(shap_df, csv_path):
    ''' Calculates the feature importance (mean absolute shap value) for each feature. '''
    print(f'Wrote Shapley importances to {csv_path}')
    shap_df.to_csv(csv_path, index=False)
    print(shap_df.to_string())

# TODO currently only for tree models
def _compute_shap(model, x, plot, write, csv_path):
    ColorPrint("Computing Shapley values", "OKBLUE")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)[0]
    df = pd.DataFrame([[np.mean(np.abs(shap_values[:, i])) for i in range(shap_values.shape[1])]],
                      columns=x.columns, index=['importance'])
    shap_df = pd.concat([df, pd.DataFrame(softmax(df), columns=df.columns, index=['norm_importance'])]) \
        .T.sort_values(by='importance') \
        .reset_index().rename(columns={'index': 'feature'})
    if write:
        _write_Shapley_values(shap_df, csv_path)
    if plot:
        print(shap.summary_plot([shap_values], x.values, plot_type="bar", class_names=[0, 1], feature_names=x.columns))
    return shap_df

