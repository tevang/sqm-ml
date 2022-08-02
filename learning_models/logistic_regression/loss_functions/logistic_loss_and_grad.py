import numpy as np
from scipy import optimize
from sklearn.linear_model._logistic import _intercept_dot
from sklearn.utils.extmath import log_logistic, safe_sparse_dot
from scipy.special import expit
import pretty_errors
from library.global_fun import save_pickle, load_pickle


def _logistic_loss_and_grad(w, X, y, alpha, sample_weight=None):
    """Computes the logistic loss and gradient.

    Parameters
    ----------
    w : ndarray of shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : ndarray of shape (n_samples,)
        Array of labels.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like of shape (n_samples,), default=None
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    out : float
        Logistic loss.

    grad : ndarray of shape (n_features,) or (n_features + 1,)
        Logistic gradient.
    """
    n_samples, n_features = X.shape
    grad = np.empty_like(w)

    w, c, yz = _intercept_dot(w, X, y)

    if sample_weight is None:
        sample_weight = np.ones(n_samples)

    # Logistic loss is the negative of the log of the logistic function.
    out = -np.sum(sample_weight * log_logistic(yz)) + .5 * alpha * np.dot(w, w)

    z = expit(yz)
    z0 = sample_weight * (z - 1) * y

    grad[:n_features] = safe_sparse_dot(X.T, z0) + alpha * w

    # Case where we fit the intercept.
    if grad.shape[0] > n_features:
        grad[-1] = z0.sum()

    return out, grad


def _logistic_loss_and_grad_grouped_samples(w, X, y, alpha, X_class, sample_weight=None):
    """

    :param w:
    :param X:
    :param y:
    :param alpha:
    :param sample_weight:
    :param X_class: same dimension as y, indicates in which sample group every row of X and y belong to.
    :return:
    """
    outs, grads = [], []
    _classes, _class_counts = np.unique(X_class, return_counts=True)
    _class_weights = (_class_counts.sum()/_class_counts)/(_class_counts.sum()/_class_counts).sum()
    for xc in _classes:
        out, grad = _logistic_loss_and_grad(w, X[X_class==xc], y[X_class==xc], alpha,
                                           sample_weight=sample_weight[X_class==xc])
        outs.append(out)
        grads.append(grad)
    return np.average(outs, weights=_class_weights), np.average(grads, axis=0, weights=_class_weights)



w0, X, target, C = load_pickle("/home/thomas/Documents/consscortk/sqmnn/input_args.pkl")
features_df, sel_columns = load_pickle("/home/thomas/Documents/consscortk/sqmnn/features_df.pkl")
X = features_df[sel_columns].to_numpy()
target = features_df["is_active"].replace({0:-1})
X_class, levels = features_df["protein"].factorize()

def _assign_sample_weight(g):
    sample_weight_dict = (g["is_active"].value_counts() / g["is_active"].size).to_dict()
    return g.apply(lambda r: sample_weight_dict[r["is_active"]], axis=1)
sample_weight = features_df.groupby("protein", as_index=False).apply(_assign_sample_weight).to_numpy()

verbose=0
iprint = [-1, 50, 1, 100, 101][
                np.searchsorted(np.array([0, 1, 2, 3]), verbose)]
tol=1e-4
max_iter=100
opt_res = optimize.minimize(
                _logistic_loss_and_grad_grouped_samples, w0, method="L-BFGS-B", jac=True,
                args=(X, target, 1. / C, X_class, sample_weight),
                options={"iprint": iprint, "gtol": tol, "maxiter": max_iter}
            )
print("OPTIMIZATION RESULTS:", opt_res)