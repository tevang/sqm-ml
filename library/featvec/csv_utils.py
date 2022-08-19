import csv
import numpy as np
from library.featvec.invariants import ints_to_bitvector
from library.global_fun import get_basemolname


def load_featvec_from_csv(csv_fname, to_bitvec=False):
    """
    Method to load the feature vectors from a csv file. If replicate molnames are present, only the feature vector of
    the 1st occurrence will be saved. The same applies to alternative stereo/ion/tau structural variants.

    :param csv_fname:
    :param to_bitvec:
    :return molname_fingerprint_dict:
    """
    csv_file = open(csv_fname, 'r')
    csv_reader = csv.reader(csv_file, delimiter=',')
    header = next(csv_reader)
    molname_fingerprint_dict = {}
    for featvec in csv_reader:
        try:
            float(featvec[1])
        except ValueError:  # skip empty lines
            continue
        molname = get_basemolname(
            featvec[0].lower())  # 1st column is the molname, I assume the all molnames are lowercase
        if molname in molname_fingerprint_dict.keys():
            continue
        if to_bitvec:
            molname_fingerprint_dict[molname] = ints_to_bitvector(np.array(featvec[1:], dtype=np.int))
        else:
            molname_fingerprint_dict[molname] = np.array(featvec[1:], dtype=np.float)
    return molname_fingerprint_dict
