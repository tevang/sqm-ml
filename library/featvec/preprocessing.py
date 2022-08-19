import numpy as np
from sklearn.preprocessing import minmax_scale

class MinMax_Scaler():
    """
    Same as sklearn.preprocessing.minmax_scale but with memory. E.g. You can use the same scaler object
    to scale anytime in the future new vectors in the same manner but of course there will not be in the [0,1]
    anymore.
    """

    def __init__(self):
        self.Xmin = None
        self.Xmax = None

    def fit(self, X):
        X = np.array(X)
        self.Xmin = X.min(axis=0)
        self.Xmax = X.max(axis=0)

    def transform(self, X, feature_range=(0, 1)):
        X = np.array(X)
        fmin, fmax = feature_range
        X_std = (X - self.Xmin) / (self.Xmax - self.Xmin)
        X_scaled = X_std * (fmax - fmin) + fmin
        return X_scaled

    def fit_transform(self, X, feature_range=(0, 1)):
        self.fit(X)
        return self.transform(X, feature_range=feature_range)

def remove_uniform_columns(mat, no0=True, no1=False, noredundant_cols=False):
    """
        NEW FASTER method to remove all identical or all zero or all one columns from a bit matrix.
    """
    mat = np.array(mat)
    mat = mat[:, ~np.isnan(mat).any(axis=0)]    # remove all columns that contain 'nan'
    rownum, colnum = mat.shape
    if noredundant_cols:
        cols2remove = []
        for c in range(colnum):
            val = mat[0, c]
            remove = True
            for v in mat[:, c]:
                if v != val:
                    remove = False
                    break
            if remove:
                cols2remove.append(c)
        mat = np.delete(mat, cols2remove, axis=1)
        return mat

    # TODO: rewrite the following case in the same efficient manner!
    if no0:
        mat = mat[:, ~np.all(mat==0, axis=0)]
    if no1:
        mat = mat[:, ~np.all(mat==1, axis=0)]
    return mat
