import csv
import numpy as np
from library.featvec.invariants import ints_to_bitvector
from library.global_fun import get_basemolname


def load_featvec_from_csv_using_serialnum(csv_fname, molname_SMILES_conformersMol_mdict, is_serial_num=False):
    # TODO: write a method to load csv that is independent of the serial number
    """
    Method to load the feature vectors from a csv file. The last column is the serial number (unique number identifier),
    which is used to match feature vectors to molecular variants (stereoisomers, ionization & tautomerization states).
    :param csv_fname:
    :param molname_SMILES_conformersMol_mdict:  instead of SMILES, it has the serial number
    :return:
    """
    csv_file = open(csv_fname, 'r')
    csv_reader = csv.reader(csv_file, delimiter=',')
    header = next(csv_reader)
    csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    # TODO: currently only the 1st molecular variant is used (stereo1_ion1_tau1).
    serialnum2molname_dict = {}
    for molname in molname_SMILES_conformersMol_mdict.keys():
        serial_nums = list(molname_SMILES_conformersMol_mdict[molname].keys())
        serial_nums.sort()
        serialnum2molname_dict[serial_nums[0]] = molname  # keep the lowest serial number (dominant molecular variant)
    molname_fingerprint_dict = {}
    for featvec in csv_reader:
        try:
            float(featvec[0])
        except ValueError:  # skip empty lines
            continue
        serial_num = featvec[-1]
        if serial_num in serialnum2molname_dict.keys():
            molname = serialnum2molname_dict[featvec[-1]]
            molname_fingerprint_dict[molname] = featvec[:-1]
    return molname_fingerprint_dict


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


def append_csv_file_features_to_featvec(csv_file, to_bitvec, *molname_fingerprint_dict_args):
    molname_csvfeatvec_dict = load_featvec_from_csv(csv_file, to_bitvec=to_bitvec)
    extended_molname_fingerprint_dict_args = []
    for molname_fingerprint_dict in molname_fingerprint_dict_args:
        for molname in molname_fingerprint_dict.keys():
            if molname not in molname_csvfeatvec_dict.keys():
                continue
            extended_featvec = np.append(molname_fingerprint_dict[molname], molname_csvfeatvec_dict[molname])
            molname_fingerprint_dict[molname] = extended_featvec
        extended_molname_fingerprint_dict_args.append(molname_fingerprint_dict)

    if len(extended_molname_fingerprint_dict_args) == 1:
        return extended_molname_fingerprint_dict_args[0]
    else:
        return extended_molname_fingerprint_dict_args