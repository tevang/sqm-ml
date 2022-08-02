import multiprocessing
from library.molfile.sdf_parser import *
from library.utils.print_functions import ColorPrint


class Parallel_Ligfile_Operations():

    def __init__(self, CHUNK_MOLS=250, POOLSIZE=multiprocessing.cpu_count(), CHUNKSIZE=1):

        self.INPUTFILE, self.CHUNK_MOLS, self.POOLSIZE, self.CHUNKSIZE = \
            INPUTFILE, CHUNK_MOLS, POOLSIZE, CHUNKSIZE
        self.pool = multiprocessing.Pool(POOLSIZE)

        # with ligfile_open(INPUTFILE) as inp:
        #     if INPUTFILE.endswith(".sdf"):
        #         miter = chunked_sdf_iterator(inp, CHUNK_MOLS)
        #     elif INPUTFILE.endswith(".mol2"):
        #         miter = chunked_mol2_iterator(inp, CHUNK_MOLS)
        #
        #     for data in self.pool.imap_unordered(_get_unique_molnames, miter, CHUNKSIZE):
        #         molname_set, molnum = data


    def has_replicate_molnames(self, mol2_file):
        ColorPrint("Checking for the presence of replicate molnames.", "BOLDGREEN")
        molname_list = []
        with open(mol2_file, 'r') as f:
            for line in f:
                if line.startswith("@<TRIPOS>MOLECULE"):
                    molname = next(f).strip()
                    molname_list.append(molname)
        total_molnum = len(molname_list)
        if self.CHUNK_MOLS < total_molnum:
            chunk_num = total_molnum//self.CHUNK_MOLS
        else:
            chunk_num = 1

        all_molname_set = set()
        all_molnum = 0
        for data in self.pool.imap_unordered(_get_unique_molnames, chunkIt(molname_list, chunk_num=chunk_num)):
            molname_set, molnum = data
            if len(molname_set) < molnum:
                print(len(molname_set), molnum)
                return True
            all_molname_set = all_molname_set.union(molname_set)
            all_molnum += molnum
            if len(all_molname_set) < all_molnum:
                print(len(all_molname_set), all_molnum)
                return True
        return False

def _get_unique_molnames(molnames):
    return [set(molnames), len(molnames)]

"""
if __name__ == "__main__":

        CHUNKSIZE = 1
        INPUTFILE = "/home2/thomas/Documents/QM_Scoring/DEKOIS2.0_library/PARP-1/" \
                    "PARP-1_all_compouds-ligprep.renstereo_ion_tau-3000confs.renpose.mol2"
        CHUNK_MOLS = 100000
        POOLSIZE = 8  # the number of CPUs to use = number of parts to split the molfile
        print(ReplicateMols.has_replicates(INPUTFILE))  # no speed gain!
        # print(Parallel_Ligfile_Operations(POOLSIZE=POOLSIZE, CHUNK_MOLS=CHUNK_MOLS).has_replicate_molnames(INPUTFILE))

"""