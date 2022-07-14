import numpy as np
from rdkit import DataStructs

from lib.utils.print_functions import Debuginfo


def getNumpyArray_list(fplist):
    nplist = []
    for fp in fplist:
        arr = np.zeros((1,), np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        nplist.append(arr)
    return nplist


def getNumpyArray(fp):
    arr = np.zeros((1,), np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def numpy_to_bitvector(vector):
    """
    Helper method to convert a 1D numpy array to a RDKit BitVector in order to calculate similarities with RDKdit functions.
    :param vector:  must be a numpy array
    :return:
    """
    assert type(vector) == np.ndarray, Debuginfo("ERROR: the input vector must be a numpy.ndarray!", fail=True)
    bitvec = DataStructs.ExplicitBitVect(len(vector))
    bitvec.SetBitsFromList(np.where(vector)[0].tolist())
    return bitvec