
import bz2
import collections
import glob
import gzip
import os
import pickle
import re
import sys
from functools import partial
from subprocess import call

import numpy as np
from scipy.stats import zscore

from .utils.print_functions import ColorPrint, Debuginfo

CONSSCORTK_LIB_DIR = os.path.dirname(os.path.realpath(__file__))
CONSSCORTK_BIN_DIR = CONSSCORTK_LIB_DIR[:-3] + "general_tools"

from collections import OrderedDict
class tree(OrderedDict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


def save_pickle(fname, *kargs):
    """
        FUNCTION to save a variable number of predictors (single or composite) into a pickled file.
    """
    with bz2.BZ2File(fname, 'wb') as f:
        pickler = pickle.Pickler(f)
        for item in kargs:
            pickler.dump(item)
        # newData = cPickle.dumps(kargs, 1)
        # f.write(newData)

def load_pickle(fname, pred_num=1000000):
    """
        FUNCTION to load a variable number of predictors (single or composite) from a pickled file.
        It returns a list with the objects loaded from the pickle file.
    """
    object_list = []
    with bz2.BZ2File(fname, "rb") as pickled_file:
        unpickler = pickle.Unpickler(pickled_file)
        for i in range(pred_num):
            try:
                object_list.append(unpickler.load())
            except EOFError:
                break
        # pickle_object = cPickle.load(pickled_file)
    return object_list

def chunkIt(seq, chunk_num, weights=[]):
    """
    Method to split a list into a specified number of approximately equal sublists.
    :param seq: input iterable
    :param chunk_num: number of chunks to split seq
    :param weights: must be a list of floats of the length num
    :return:
    """
    if chunk_num == 1:
        return [seq]
    elif chunk_num > len(seq):
        return [[s] for s in seq]
    if not weights:
        weights = [1] * chunk_num

    assert len(weights) == chunk_num, Debuginfo("ERROR: weights must be a list of floats of length num.", fail=True)

    quantum = len(seq) / np.sum(weights) ;
    out = []
    last = 0.0

    for i in range(chunk_num):
        out.append(seq[int(last):int(last + quantum*weights[i])])
        last += quantum*weights[i]
    out[i].extend(seq[int(last):])    # append the remaining (if any) to the last chunk

    return out

def flatten(l):
    """
    FUNCTION to flatten any Iterable object.
    """
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, str):
            for sub in flatten(el):
                yield sub
        else:
            yield el


def list_files(folder, pattern, full_path=False, rel_path=False):
    """
    Method to list the files in 'folder' that match the 'pattern'.
    :param folder: this can take Unix shell regex patterns (e.g. '*/*' but not '.*/.*')
    :param pattern: this must take a Python regex pattern
    :param full_path:
    :param rel_path: get the relative path or just the filename
    :return:
    """
    if not folder:
        folder = "."
    abspath_folder = os.path.abspath(folder)
    if folder.endswith("/"):    # fix to abspath which removes trailing '/'
        folder = abspath_folder + "/"
    else:
        folder = abspath_folder
    if folder.endswith("/*") == False:
        folder += "/*"   # if not regex, without '/*' it won't return anything!
    fpaths = glob.glob(folder)
    fpattern = re.compile(pattern)
    file_list = list(filter(fpattern.search, fpaths))
    if full_path:
        file_list = [os.path.abspath(f) for f in file_list]
    elif rel_path:
        file_list = [os.path.relpath(f) for f in file_list]
    else:
        file_list = [os.path.basename(f) for f in file_list]
    return file_list

def replace_alt(text, alt_txtlist, new_txt):
    "Method to do replace multiple alternative texts with one specific text."
    for txt in alt_txtlist:
        text = text.replace(txt, new_txt)
    return text

def sub_alt(text, alt_patterns, new_txt):
    "Method to do replace multipel alternative texts with one specific text."
    for pattern in alt_patterns:
        text = re.sub(pattern, new_txt, text)
    return text

def replace_alt(text, alt_txtlist, new_txt):
    "Method to do replace multiple alternative texts with one specific text."
    for txt in alt_txtlist:
        text = text.replace(txt, new_txt)
    return text

def get_basemolname(structvar):
    """
    Method to remove the structural variant suffix from the molname including the pose suffix.
    :param structvar: 
    :return: 
    """
    return sub_alt(structvar, ["_stereo[0-9]+_ion[0-9]+_tau[0-9]+", "_pose[0-9]+", "_frm[0-9]+", "_iso[0-9]+", "_noWAT", "_WAT"], "")

def get_structvar(molname):
    """
    Method the get the structural variant without the poseID.
    :param molname:
    :return:
    """
    molname = re.sub("_pose[0-9]+", "", molname)
    return re.sub("_frm[0-9]+", "", molname)

def get_structvar_suffix(structvar, as_numbers=False):
    """
    Method the get the structural variant suffix information from the full molname.
    :param structvar:
    :param as_numbers:  return just the stereo, ion and tau indices as strings (not as integers!)
    :return:
    """
    if as_numbers:
        m = re.search(".*stereo([0-9]+)_ion([0-9]+)_tau([0-9]+)[^0-9]*", structvar)
        if m:
            return m.groups()
        else:
            return None
    else:
        m = re.search(".*_(stereo[0-9]+_ion[0-9]+_tau[0-9]+)[^0-9]*", structvar)
        if m:
            return m.group(1)
        else:
            return None

def get_poseID(molname):
    m = re.search(".*_pose([0-9]+)[^0-9]*", molname)
    if m:
        return int(m.group(1))
    else:
        return None

def get_frameID(molname):
    m = re.search(".*frm([0-9]+)$", molname)
    if m:
        return int(m.group(1))
    else:
        return None

class GetOutOfLoop( Exception ):
    """
    Use this exception to exist nested for loops. E.g.
    try:
        for ...
            for ...
                if ..
                    raise GetOutOfLoop
    except GetOutOfLoop:
        pass
    """
    pass

def ligfile_open(filename, mode='r'):
    """
    General file opener. Works for both gunzipped and non-ganzipped files.
    :param filename:
    :param mode:
    :return:
    """
    return gzip.open(filename, mode+"t") if filename.endswith(".gz") else open(filename, mode)

def get_line_count(filename):
    """
    Method to count the number of lines of a very large file.
    :param filename:
    :return:
    """
    f = open(filename, 'rb')
    # bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))    # original line
    bufgen = iter(partial(f.raw.read, 1024 * 1024), b'')
    return sum( buf.count(b'\n') for buf in bufgen )
