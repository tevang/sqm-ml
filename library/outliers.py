import numpy as np
from scipy.stats import zscore

from lib.dataframe_functions import print_whole_df


def is_outlier_MAD(points, thresh=3.5):
    """
    THE INPUT MUST BE VERTICAL! samples x features

    TODO: check if it works on multiple dimensions. According to the parameters it must but each dimension
        treated individually.

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
    assert points.shape[0] > 1, "FAIL: input data to is_outlier_MAD() must be vertical! You entered array " \
                              "with dimensions %s" % points.shape[0]

    if len(points.shape) == 1:
        points = points.to_numpy()[:, None]

    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def remove_outliers(g, column, thresh=3.5, min_scored_pose_num=30):
    outlier_mask = is_outlier_MAD(g[column], thresh=thresh)
    min_scored_mask = g["scored_pose_num"] >= min_scored_pose_num
    outliers = g.loc[outlier_mask & min_scored_mask, :]
    if outliers.size > 0:
        print("Outliers removed:", outliers)
    return g.loc[~(outlier_mask & min_scored_mask), :]



def remove_outliers(g, column, thresh=3.5, min_scored_pose_num=0, outliers_csv=None,
                    log10_transform=False, outliers_columns_to_save=[]):

    if g.shape[0] == 1:
        return g
    elif log10_transform:
        outlier_mask = is_outlier_MAD(np.log10(g[column]), thresh=thresh)
    else:
        outlier_mask = is_outlier_MAD(g[column], thresh=thresh)

    min_scored_mask = (g["scored_pose_num"] < min_scored_pose_num).values
    outliers = g.loc[outlier_mask | min_scored_mask, :]

    if outliers.size > 0:

        if not outliers_columns_to_save:
            outliers_columns_to_save = [column]

        print("Outliers removed (thresh=%f):" % thresh)
        print_whole_df(outliers.sort_values(by=column)[outliers_columns_to_save])

        if outliers_csv:
            outliers.sort_values(by=column)[outliers_columns_to_save] \
                .to_csv(outliers_csv, mode='a', header=False, index=False)

    return g.loc[~outlier_mask, :]