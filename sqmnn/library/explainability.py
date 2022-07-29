import pandas as pd


def _return_perm_imp(model, x: pd.DataFrame, y: pd.Series):
    """

    Parameters
    ----------
    model
    X
    y

    Returns
    -------

    """
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, x, y, n_repeats=1, random_state=0)

    imp_mean = pd.Series(r.importances_mean, index=x.columns)
    return imp_mean


# TODO currently only for tree models
def _plot_shap(model, x):
    """

    Parameters
    ----------
    model
    x

    Returns
    -------

    """
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)

    p = shap.summary_plot(shap_values, x.values,
                            plot_type="bar", class_names=[0, 1], feature_names=x.columns)
    return p
