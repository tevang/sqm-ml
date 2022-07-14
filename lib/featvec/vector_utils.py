import numpy as np


def fuse_featvecs(*args):
    """
        Method that takes an arbitrary number of lists of feature vectors and returns the list of their fusions.
    """
    fv1_list = args[0]
    for fv2_list in args[:1]:
        for i in range(len(fv1_list)):
            fv1_list[i] = np.append(fv1_list[i], fv2_list[i])

    return fv1_list