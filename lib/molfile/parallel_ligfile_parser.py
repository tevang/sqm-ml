from lib.molfile.ligfile_parser import *
from lib.molfile.sdf_parser import *
import ray


class Parallel_Ligfile_Operations():

    def __init__(self, INPUTFILE, FUNCTION, CHUNK_MOLS, POOLSIZE):

        ray.init(num_cpus=POOLSIZE)

        with ligfile_open(INPUTFILE, 'r') as inp:
            if INPUTFILE.endswith(".sdf") or INPUTFILE.endswith(".sdf.gz"):
                miter = chunked_sdf_iterator(inp, CHUNK_MOLS)
            elif INPUTFILE.endswith(".mol2") or INPUTFILE.endswith(".mol2.gz"):
                miter = chunked_mol2_iterator(inp, CHUNK_MOLS)

            function_calls = []
            for i, text in enumerate(miter):
                function_calls.append( _get_mol2_molnames.remote(ligfile_text=text) )
        results = ray.get(function_calls)

@ray.remote
def _get_mol2_molnames(ligfile_text):

    delimiter = "@<TRIPOS>MOLECULE"
    molname_list = []
    for line in ligfile_text:
        if line.startswith(delimiter):
            molname = next(ligfile_text).rstrip()
            molname_list.append(molname)

    return set(molname_list), len(molname_list)

"""
if __name__ == "__main__":

    INPUTFILE = "/home2/thomas/Documents/QM_Scoring/DEKOIS2.0_library/PARP-1/PARP-1_all_compouds-ligprep.renstereo_ion_tau-3000confs_part5.mol2"

    CHUNK_MOLS = 250
    POOLSIZE = 1  # the number of CPUs to use = number of parts to split the molfile

    Parallel_Ligfile_Operations(INPUTFILE, "split_file", CHUNK_MOLS, POOLSIZE)
"""