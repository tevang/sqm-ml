import numpy as np
from scipy.stats import zscore

from lib.utils.print_functions import ColorPrint


def is_outlier_MAD(points, thresh=3.5):
    """
    TODO: check if it works on multiple dimensions. I don't think it works.

    Returns a boolean array with True if points are outliers and False
    otherwise, according to median-absolute-deviation (MAD).

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def remove_outliers_2D(data, method='zscore', zmin=-2.0, zmax=2.0, axis=1):
    """

    :param data:
    :param method: 'zscore', 'DBSCAN'
    :param zmin:
    :param zmax:
    :param axis: has the reverse meaning as in np.mean()
    :return:
    """
    # print("DEBUG: data=", list(data))
    data = np.array(data)
    mean_data = []
    # TODO: add is_outlier_MAD() but first check that it works on multiple dimensions.
    if method == 'zscore':
        ColorPrint("Employing z-score method to remove outliers from the predictions.", 'OKBLUE')
        for c in range(data.shape[axis]):
            if axis==0:
                values = data[c,:]  # extract the row values
            elif axis == 1:
                values = data[:,c]  # extract the column values
            zvalues = zscore(values)
            valid_values = []
            for v,z in zip(values, zvalues):
                if z > zmin and z < zmax:
                    valid_values.append(v)
            mean_data.append( np.mean(valid_values) )  # save the mean value without the outliers
    elif method == 'DBSCAN':
        from sklearn.cluster import DBSCAN
        ColorPrint("Employing DBSCAN method to remove outliers from the predictions.", 'OKBLUE')
        for c in range(data.shape[axis]):
            if axis==0:
                values = data[c,:]  # extract the row values
            elif axis == 1:
                values = data[:,c]  # extract the column values
            db = DBSCAN().fit(values.reshape(-1, 1))
            valid_values = []
            for v,l in zip(values, db.labels_):
                if l == 0:
                    valid_values.append(v)
            mean_data.append( np.mean(valid_values) )  # save the mean value without the outliers
        if np.isnan(mean_data).any():
            ColorPrint("WARNING: too few alternative prediction values for DBSCAN method! It produced nan values.",
                       "OKRED")

    return np.array(mean_data)


def remove_outliers_1D(data, method='mad', thresh=3.5, zmin=-2.0, zmax=2.0, get_outlier_indices=False):
    """

    :param data:
    :param method: 'zscore', 'DBSCAN'
    :param zmin:
    :param zmax:
    :return:
    """
    data = np.array(data)
    mean_data, outlier_indices = [], []
    i = 0
    if method == "mad":
        # Keep only the "good" points
        # "~" operates as a logical not operator on boolean numpy arrays
        is_outlier = is_outlier_MAD(data, thresh=thresh)
        valid_values = data[~is_outlier]
        outlier_indices = np.where(is_outlier==True)[0]
    elif method == 'zscore':
        # ColorPrint("Employing z-score method to remove outliers from the predictions.", 'OKBLUE')
        zvalues = zscore(data)
        valid_values = []
        for v,z in zip(data, zvalues):
            if z > zmin and z < zmax:
                valid_values.append(v)
            else:
                outlier_indices.append(i)
            i += 1
    elif method == 'DBSCAN':
        from sklearn.cluster import DBSCAN
        # ColorPrint("Employing DBSCAN method to remove outliers from the predictions.", 'OKBLUE')
        db = DBSCAN().fit(data.reshape(-1, 1))
        valid_values = []
        for v,l in zip(data, db.labels_):
            if l == 0:
                valid_values.append(v)
            else:
                outlier_indices.append(i)
            i += 1
        # if np.isnan(mean_data).any():
        #     ColorPrint("WARNING: too few alternative prediction values for DBSCAN method! It produced nan values.",
        #                "OKRED")

    if get_outlier_indices:
        return np.array(valid_values), outlier_indices
    else:
        return np.array(valid_values)