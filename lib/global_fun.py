
import bz2
import collections
import glob
import gzip
import os
import pickle
import re
import sys
import traceback
from functools import partial
from subprocess import call
from decimal import *

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


class RegexDict(dict):
    """
    Dictionary that can be invoked using wildcards in the keys.
    E.g.
    dtypes_dict = RegexDict({"mol2vec1000_1": 1000, "mol2vec3000_2": 3000})
    print(list(dtypes_dict.get_matching("mol2vec.*")))
    """
    def get_matching(self, pattern):
        return (self[key] for key in self if re.match(pattern=pattern,string=key))

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

def scale_manually(dataset, Xmin, Xmax):

    scaled_dataset = (np.array(dataset) - Xmin) / float(Xmax - Xmin)
    return list(scaled_dataset)

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


def list_files_recursively(directory: str, pattern: str, full_path=False, rel_path=False) -> list:
    """
    Method to list recursively all the files in 'dir' that match the 'pattern'.
    :param dir: this can take Unix shell regex patterns (e.g. '*/*' but not '.*/.*')
    :param pattern: this must take a Python regex pattern
    :param full_path:
    :param rel_path: get the relative path or just the filename
    :return:
    """
    all_files = []
    for root, dirs, files in os.walk(directory):
        all_files += list_files(folder=root, pattern=pattern, full_path=full_path, rel_path=rel_path)
    return all_files


def multimatch_string_list(List, pattern, pindex=1, unique=True):
    """
    Method to search for a pattern in every string of a list and return all matches of just one specified subgroup.
    :param List:
    :param pattern:
    :param pindex:
    :return:
    """
    matches = []
    for l in List:
        m = re.search(pattern, l)
        if m:
            matches.append(m.group(pindex))
    if unique:
        return list(set(matches))
    else:
        return matches

def match_string_list(List, pattern):
    """
    Method to search for a pattern in every string of a list and return all matching subgroups in the 1st line
    that matches. Useful to scan a file for an occurrence of a string that contains important numbers.

    :param List:    list of strings
    :param pattern:
    :return:
    """
    matches = []
    for l in List:
        m = re.search(pattern, l)
        if m:
            return m.groups()

def extract_vars_from_string(string, pattern):
    """
    Helper method to extract variables that match a given pattern in a string.
    :param string:
    :param pattern:
    :return:
    """
    m = re.search(pattern, string)
    return m.groups()

def which(program):
    """
        FUNCTION to find the full path of the executable 'program'. If it is not found, it returns None.
    """
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def write2file(string, fname):
    with open(fname, 'w') as f:
        f.write(string)


def run_commandline(commandline, logname="log", append=False, return_out=False, error_keywords=[],
                    error_messages=[], skip_fail=False, verbose=True):
    """
        FUNCTION to run a single command on the UNIX shell. The worker will only receive an index from network.
    """
    if append:
        fout = open(logname, 'a')
    else:
        fout = open(logname, 'w')
    if verbose:
        print("Running commandline:", commandline)
    return_code = call(commandline, stdout=fout, stderr=fout, shell=True, executable='/bin/bash')

    if (return_code != 0):
        print(ColorPrint("ERROR, THE FOLLOWING COMMAND FAILED TO RUN:", "FAIL"))
        print(commandline)
        print("return_code=", return_code)
        fout.close()
        print("Output:")
        with open(logname, 'r') as f:
            contents = f.readlines()
            for line in contents:
                print(line)
        if not skip_fail:
            raise Exception()
    fout.close()

    if len(error_keywords) > 0:
        with open(logname, 'r') as f:
            contents = f.readlines()
            for line in contents:
                for i, word in enumerate(error_keywords):
                    if word in line:
                        ColorPrint("ERROR, THE FOLLOWING COMMAND FAILED TO RUN:", "FAIL")
                        print(commandline)
                        ColorPrint("COMMAND OUTPUT:", "WARNING")
                        for line in contents:
                            print(line)
                        if len(error_messages) >= i + 1:
                            raise Exception(error_messages[i])
                        else:
                            raise Exception()

    if return_out:
        with open(logname, 'r') as f:
            contents = f.readlines()
            return contents


def writelist2log(L, f):
    """
        FUNCTION to write a list of strings and numbers in file f as well as print them.
    """
    string = str(L[0])
    for l in L[1:]:
        string += " " + str(l)
    print(string)
    f.write(string + "\n")
    f.flush


def writelist2file(List, file, header=None, append=False):
    """
    Method to write the contents of a list into a file, each element to a different line.
    :param List:
    :param file: can be a filename or a file handler
    :param header:
    :param append:
    :return:
    """
    if type(file) == str:
        if append:
            mode = 'a'
        else:
            mode = 'w'
        f = open(file, mode)
    else:
        f = file

    if header:
        f.write(header+"\n")
    for l in List:
        if type(l) == str:
            if l[-1] != '\n':
                l += '\n'
            f.write(l)
        elif type(l) in [int, float]:
            f.write(str(l) + "\n")
        else:
            l = [str(i) for i in l] # convert all elements to string
            f.write(" ".join(l) + "\n")

def writelists2file(fname, *lists):
    """
        FUNCTION to write the contents of multiple lists of strings and numbers in filename fname.
        E.g. if l1 = ["a", "b", "c"], l2 = [25.6, 50.5, 100.3], then fname will contain:
        a 25.6
        b 50.5
        c 100.3
    """
    N = len(lists[0])  # asume that all lists have the same size
    with open(fname, 'w') as f:
        for i in range(N):
            line = lists[0][i] + " "
            for l in lists[1:]:
                line += " " + str(l[i])
            f.write(line + "\n")
            f.flush


def concatenate_files(filename_list, outfilename, clean=False):
    with open(outfilename.replace("\\'", "'"), 'w') as outfile:
        for filename in filename_list:
            with open(filename.replace("\\'", "'"), 'r') as infile:
                for line in infile:
                    outfile.write(line)
            if clean:
                os.remove(filename.replace("\\'", "'"))

def print_dict_contents(dictionary):
    """
    Print the contents of a dictionary. Can work with multi-dimensional dictionaries, too.
    :param dictionary: a dict or tree() object
    :return:
    """
    for k,v in list(dictionary.items()):
        sys.stdout.write(str(k) + " --> ")
        try:
            print_dict_contents(v)
        except AttributeError:
            print(v)

def write_dict_contents(dictionary, f):
    """
    Print the contents of a dictionary. Can work with multi-dimensional dictionaries, too.
    :param dictionary: a dict or tree() object
    :return:
    """
    for k,v in list(dictionary.items()):
        f.write(str(k) + " --> ")
        f.flush()
        try:
            write_dict_contents(v, f)
        except AttributeError:
            f.write(" ".join([str(e) for e in v]) + "\n")
            f.flush()

def do_files_exist(files_list=[]):
    for fname in files_list:
        if fname and not os.path.exists(fname):
            raise IOError(ColorPrint("ERROR: file %s does not exist" % fname, "FAIL"))

def core_zscore(values, zmin=-3.0, zmax=3.0):
    """
    Calculates the z-score by ignoring the outliers that are zmin stdevs below and zmax stdevs above the mean. At the end all values
    are converted to z-scores and returned.
    :param values:
    :param zmin:
    :param zmax:
    :return:
    """

    zvalues = zscore(values)
    core_values = [v for z,v, in zip(zvalues, values) if zmin < z < zmax]   # values without outliers
    core_mean = np.mean(core_values)
    core_std = float(np.std(core_values))
    core_zvalues = np.array([(v-core_mean)/core_std for v in values])   # convert all values to zscores
    return core_zvalues

def multitype_zscore(values, types, ignore_outliers=False):
    """
    Given a list of values of mixed type, this method calculates the zscores individually for each type
    and returnes them in the order they occured in the input list.
    :param values:
    :param types:
    :return:
    """
    types_set = set(types)
    type_values_dict = {t:[] for t in types_set}
    for v,t in zip(values,types):
        type_values_dict[t].append(v)
    if ignore_outliers:
        type_zscores_dict = {k:core_zscore(v, zmin=-3.0, zmax=3.0) for k,v in list(type_values_dict.items())}
    else:
        type_zscores_dict = {k:zscore(v, zmin=-3.0, zmax=3.0) for k,v in list(type_values_dict.items())}
    zvalues = []
    type_index_dict = {t:0 for t in types_set}
    for v,t in zip(values, types):
        i = type_index_dict[t]
        zvalues.append(type_zscores_dict[t][i])
        type_index_dict[t] += 1
    return zvalues

def replace_multi(text, dict):
    "Method to do multiple replacements in a string."
    for i, j in dict.items():
        text = text.replace(i, j)
    return text

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

def get_relative_path(full_path2file):
    """
    The the relative path to the file from your current directory.
    :param full_path2file:
    :return:
    """
    cwd = os.getcwd()
    relpath2file = full_path2file.replace(cwd, "")
    if relpath2file[0] == '/':
        return relpath2file[1:]
    else:
        return relpath2file

def grep(pattern, *filenames):
    pattern = ".*%s.*" % pattern    # for re.match to work pattern must have trailing '.*'
    matches = []
    for fname in filenames:
        with open(fname, 'r') as f:
            for line in f:
                if re.match(pattern, line):
                    matches.append("%s:%s" % (fname, line))
    return matches

def is_pattern_in_file(pattern, filename):
    """
    Method to tell if a specific keyword pattern is in a file. E.g. "ERROR".

    :param pattern:
    :param filename:
    :return:
    """
    return len(grep(pattern, filename)) > 0

def add_suffix_to_filename(fname, suffix):
    s,e = os.path.splitext(fname)
    return s + suffix + e

def replace_multi(text, dict):
    "Method to do multiple replacements in a string."
    for i, j in dict.items():
        text = text.replace(i, j)
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

def strip_poseID(structvar):
    return get_structvar(structvar)

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

def is_structvar(molname):
    return re.match(".*_(stereo[0-9]+_ion[0-9]+_tau[0-9]+)[^0-9]*", molname)

def decompose_structvar(structvar):
    basename = get_basemolname(structvar)
    m = re.search(".*(stereo[0-9]+)_(ion[0-9]+)_(tau[0-9]+)[^0-9]*", structvar)
    return basename, m.group(1), m.group(2), m.group(3)

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


def Dec(number, decpoints=3):
    """
    Method to create a read fixed decimal point float number. Beware that every time you do operations between 2 such numbers
    then the decimal points will change (e.g. v1*v2 will have 8 decimal points if v1 & v2 had 4 decimal points).
    :param number: float, int, Decimal or string
    :param decpoints:
    :return:
    """
    try:
        # NOTE: ...if the length of the coefficient after the quantize operation would be greater than precision,
        # NOTE: then an InvalidOperation is signaled. SOLUTION: increase getcontext().prec
        return Decimal(number).quantize(Decimal('1.' + '0'*decpoints))
    except InvalidOperation:
        Debuginfo("FAIL: number " + str(number) + " %s return decimal.InvalidOperation. "
                                                    "Try a smaller 'decpoints' input parameters." % type(number),
                    fail=True)

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

def exception_traceback(func):
    def wrap(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            type, value, tb = sys.exc_info()
            lines = traceback.format_exception(type, value, tb)
            print((''.join(lines)))
            raise
    return wrap