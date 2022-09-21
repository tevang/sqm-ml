import pandas as pd
from sklearn.inspection import permutation_importance
import shap

from library.utils.print_functions import ColorPrint


def _return_perm_imp(model, x: pd.DataFrame, y: pd.Series, n_repeats: int):
    ColorPrint("Computing Permutation Feature Importances with {} repeats.".format(n_repeats), "OKBLUE")
    r = permutation_importance(model, x, y, n_repeats=n_repeats, random_state=0, n_jobs=-1)
    return pd.Series(r.importances_mean, index=x.columns)


# TODO currently only for tree models
def _plot_shap(model, x):
    ColorPrint("Computing Shapley values", "OKBLUE")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    print(shap.summary_plot(shap_values, x.values,
                            plot_type="bar", class_names=[0, 1], feature_names=x.columns))