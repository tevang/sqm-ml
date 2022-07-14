#!/usr/bin/env python

from collections import defaultdict

from pymol import cmd, CmdException

# execfile('~/.pymolrc')
from lib.ConsScoreTK_Statistics import *
from lib.clustering import *
from lib.utils.print_functions import ColorPrint


class PocketClust():

    def __init__(self, pdb_list, ref_pdb="", CHAIN='A', radius=16.0, TMALIGN=False, PYMOL_SCRIPT_REPO="", pymolf=None):
        self.pdb_list = pdb_list
        if not ref_pdb:
            ref_pdb = pdb_list[0]
        self.ref_pdb = ref_pdb
        self.CHAIN = CHAIN
        self.radius = radius    # the radius of the pocket from the COM
        self.TMALIGN = TMALIGN
        self.pymolf = pymolf

        # TODO: in Python 3 the following raises "SyntaxError: import * only allowed at module level"
        try:
            from tmalign import *
        except ImportError:
            assert not TMALIGN or os.path.exists(PYMOL_SCRIPT_REPO + "/tmalign.py"), "ERROR: please set correctly -PYMOL_SCRIPT_REPO and make " \
                                                                         "sure that the directory contains 'tmalign.py' script."
            sys.path.append(PYMOL_SCRIPT_REPO)
            # importing tmalign here will not make it visible within focus_alignment()

    def load_structure(self, pdb):
        print("Loading structure file %s to PyMOL." % pdb)
        cmd.load(pdb)
        cmd.remove("not alt +A")
        model = os.path.basename(pdb.replace(".pdb", ""))
        resid, chain = self.get_largest_ligand_resid(model, CHAIN=self.CHAIN, get_chain=True)
        if resid == None:
            print("Deleting model %s which has no ligand." % model)
            cmd.delete(model)
            return None, chain
        # print("DEBUG: pdb=%s largest resid=%i chain=%s" %(pdb, resid, chain))
        if chain != self.CHAIN:
            print("WARNING: file %s does not contain chain %s! I will rename %s to %s!" % \
                  (pdb, self.CHAIN, chain, self.CHAIN))
            cmd.alter("(chain " + chain + ")", "chain='" + self.CHAIN + "'")  # rename the chain and overwrite the file
            cmd.save(pdb, model)    # IMPORTANT: alternative coordinates confuse the alignment!!!
            return resid, self.CHAIN
        else:
            cmd.save(pdb, model)    # IMPORTANT: alternative coordinates confuse the alignment!!!
            return resid, chain

    def load_structures(self, CHAIN='A'):
        """
        Loads all structures, aligns them to the pocket of the reference structure, and creates a pseudoatom at the Center of Mass
        of all ligands. You can use this pseudoatom to select the pocket residues that will be used for RMSD calculation.
        :param CHAIN:
        :return:
        """
        model_ligresidChain_dict = {}
        # First load the reference structure
        ref_model = os.path.basename(self.ref_pdb).replace(".pdb", "")
        print("Using %s as REFERENCE MODEL and aligning all other structures on it." % ref_model)
        reflig_resid, reflig_chain = self.load_structure(self.ref_pdb)
        if self.pymolf:  self.pymolf.write("from tmalign import *\ncmd.load('%s')\n" % self.ref_pdb)
        refsel = "( ( byres (%s within %f of (%s and resi %g and chain %s) ) ) and polymer )" % \
                 (ref_model, self.radius, ref_model, reflig_resid, reflig_chain)
        model_ligresidChain_dict[ref_model] = [reflig_resid, reflig_chain]

        # Then load all others and align them to the reference
        remaining_pdbs = self.pdb_list
        remaining_pdbs.remove(self.ref_pdb)
        pdbs_without_ligand = []    # just for information (never used)
        for pdb in remaining_pdbs:
            lig_resid, lig_chain = self.load_structure(pdb)
            if lig_resid == None:   # ignore pdb without ligand (no need to remove it because remaining_pdbs is never used again)
                pdbs_without_ligand.append(pdb)
                continue
            if self.pymolf: self.pymolf.write("cmd.load('%s')\n" % pdb)
            model = os.path.basename(pdb).replace(".pdb", "")
            self.focus_alignment(model, ref_model, sel=refsel, fit=True, get_rms=False, atomsel="n. CA")
            model_ligresidChain_dict[model] = [lig_resid, lig_chain]

        # Finally find the Center Of Mass of all ligands and create a pseudo atom
        ColorPrint("Creating a pseudoatom at the COM of all crystallized ligands.", "BOLDGREEN")
        allligsel = ""
        for model in list(model_ligresidChain_dict.keys()):
            lig_resid, lig_chain = model_ligresidChain_dict[model]
            allligsel += "(%s and resi %i and chain %s) " % (model, lig_resid, lig_chain)
        COM = cmd.centerofmass(allligsel)
        if self.pymolf:  self.pymolf.write("cmd.centerofmass('%s')\n" % allligsel)
        cmd.pseudoatom("COM", pos=COM)  # create the pseudoatom at the COM
        if self.pymolf:  self.pymolf.write("cmd.pseudoatom('COM', pos=%s)\n" % COM)

    def get_largest_ligand_resid(self, model, CHAIN='A', get_chain=False):

        hetero_resids, all_chains, chains = [], [], []
        space = {   "hetero_resids": hetero_resids,
                    "all_chains": all_chains,
                    "chains": chains
                 }
        cmd.iterate('%s' % (model) ,'hetero_resids.append(resi)', space=space)
        hetero_resids = set(hetero_resids)
        cmd.iterate('hetatm', 'all_chains.append(chain)', space=space)
        all_chains = set(all_chains)
        largest_ligand_resid = None  # the index of the largest hetero molecule in the list
        max_num = 0
        # print("DEBUG: model", model, "hetero_resids=", hetero_resids)
        for resid in hetero_resids:
            for chain in all_chains:
                num = cmd.count_atoms("%s and chain %s and resi %s and not name H*" %
                (model, chain, resid))  # count the heavy atoms of this residue
                # print("DEBUG: resid=%s chain=%s atom count= %i" % (resid, chain, num))
                if num > max_num:
                    max_num = num
                    largest_ligand_resid = resid
        # one more round to save all chains and resids with the max_num atom count
        resid_chainList_dict = {}  # resid of the ligands with max_num atom count and the chains where they belong
        for resid in hetero_resids:
            for chain in all_chains:
                num = cmd.count_atoms("%s and chain %s and resi %s and not name H*" % (
                model, chain, resid))  # count the heavy atoms of this residue
                # print("DEBUG chain=%s num=%i" % (chain, num))
                if num == max_num:
                    try:
                        resid_chainList_dict[resid].append(chain)
                    except KeyError:
                        resid_chainList_dict[resid] = [chain]
        # Chech if CHAIN is in the values and if yes, keep that resid as the largest
        # print("DEBUG: resid_chainList_dict=", resid_chainList_dict)
        for resid, chains in list(resid_chainList_dict.items()):
            if CHAIN in chains:
                largest_ligand_resid = resid
                break

        # print("DEBUG: largest_ligand_resid=", largest_ligand_resid)
        if largest_ligand_resid == None:
            print("ERROR: model", model, "has no ligand!")
            if get_chain:
                return None, 'A'
            else:
                return None
        else:
            largest_ligand_resid = int(largest_ligand_resid)

        if not get_chain:
            return largest_ligand_resid

        # find the chain of the largest ligand (if not chain 'A', return the closest)
        cmd.iterate('%s and resi %i' % (model, largest_ligand_resid), 'chains.append(chain)', space=space)
        chains = list(set(chains))
        # print("DEBUG: chains=", chains)
        chains.sort()
        if CHAIN in chains:
            return largest_ligand_resid, CHAIN
        else:
            return largest_ligand_resid, chains[0]


    def focus_alignment(self, obj1, obj2, sel, fit=True, get_rms=False, atomsel="n. CA", debug=False):
        """
        PARAMS
          obj1
              mobile structure

          obj2
              reference structure

          sel
              the selection from either obj1 or
              obj2 to focus the pair_fitting on.
              When providing this selection, please
              make sure you also specify selected
              atoms from ONE object.

        NOTES

        This function will first align obj1 and obj2 using a sequence alignment. This creates a mapping of
        residues from obj1 to obj2. Next, the selection, sel, is used to find only those atoms in the alignment and
        in sel. These atoms are paired with their mapped atoms from the alignment in the other object. These
        two subsets of atoms are then pair_fit to give an optimal sub-alignment.
        """

        aln = "aln_%s_%s" % (obj1, obj2)
        _sel = "__sel"
        ssel_model = ""
        a1, a2, a_target, modelA, modelB, sel_model = [], [], [], [], [], []
        obj1, obj2 = "polymer and " + obj1, "polymer and " + obj2

        if fit and self.TMALIGN and not get_rms:
            print("Superimposing %s onto %s with TMAlign." % (obj1, obj2))
            from tmalign import *
            tmalign('%s' % obj1, '%s' % obj2, quiet=True)
            if self.pymolf:  self.pymolf.write('tmalign("%s", "%s")\n' % (obj1, obj2))
            return

        # space dictionary for iterate
        space = {'a1': a1,
                 'a2': a2,
                 'a_target': a_target,
                 'sel_model': sel_model,
                 'modelA': modelA,
                 'modelB': modelB}

        # initial unfocused alignment
        cmd.align(obj1, obj2, cutoff=20, cycles=0, object=aln)
        if self.pymolf and fit: self.pymolf.write("cmd.align('%s', '%s', cutoff=20, cycles=0, object='%s')\n" % (obj1, obj2, aln))

        # record the initial indices, include only the specified atoms of the matched residues
        s = atomsel + " and (%s and %s)"
        cmd.iterate(s % (obj1, aln), "a1.append(index)", space=space)
        cmd.iterate(s % (obj2, aln), "a2.append(index)", space=space)

        assert len(a1) == len(a2), "ERROR in focus_alignment(): num atoms in aln1 (%g, %s) != num atoms in aln2 (%g, %s)" % \
                                   (len(a1), obj1, len(a2), obj2)

        # determine who owns the focused selection and
        # get canonical object names
        cmd.iterate("first %s" % sel, "sel_model.append(model)", space=space)
        cmd.iterate("first %s" % obj1, "modelA.append(model)", space=space)
        cmd.iterate("first %s" % obj2, "modelB.append(model)", space=space)
        ssel_model = sel_model[0]

        if debug:
            print("# [debug] selection is in object %s" % ssel_model)

        # focus the target selection
        cmd.iterate(sel + " and " + atomsel, "a_target.append(index)", space=space)

        if debug:
            print("# [debug] a_target has %d members." % len(a_target))

        # select the correct object from which to index
        target_list = None
        if ssel_model == modelA[0]:
            target_list = a1
        elif ssel_model == modelB[0]:
            target_list = a2
        else:
            print("# error: selection on which to focus was not found")
            print("# error: in either object passed in.")
            print(sel_model)
            print(modelA)
            print(modelB)
            return False

        id1, id2 = [], []
        for x in a_target:
            try:
                idx = target_list.index(x)
                if debug:
                    print("Current index: %d" % idx)
                id1.append(str(a1[idx]))
                id2.append(str(a2[idx]))
            except:
                pass

        if debug:
            print("# [debug] id1 = %s" % id1)
            print("# [debug] id2 = %s" % id2)

        # sel1 = "+".join(id1)
        # sel2 = "+".join(id2)
        id1 = [int(i) for i in id1]
        id2 = [int(i) for i in id2]
        cmd.select_list('sel1', obj1, id1, mode='index')
        cmd.select_list('sel2', obj2, id2, mode='index')
        if self.pymolf and fit:
            self.pymolf.write("id1 = %s\n" % id1)
            self.pymolf.write("id2 = %s\n" % id2)
            self.pymolf.write("cmd.select_list('sel1', '%s', [int(i) for i in id1], mode='index')\n" % (obj1))
            self.pymolf.write("cmd.select_list('sel2', '%s', [int(i) for i in id2], mode='index')\n" % (obj2))

        # if debug:
        #     print(")# [debug] sel1 = %s" % sel1)

        if fit:
            cmd.pair_fit("sel1 and %s" % aln, "sel2 and %s" % aln)
            if self.pymolf:  self.pymolf.write('cmd.pair_fit("sel1 and %s", "sel2 and %s")\n' % (aln, aln))
        if get_rms:
            rmsd = cmd.rms_cur("sel1 and %s" % aln, "sel2 and %s" % aln)
            cmd.delete('%s' % aln)
            cmd.delete('ref_pocket')
            cmd.delete('sel1')
            cmd.delete('sel2')
            return rmsd
        cmd.delete('%s' % aln)
        cmd.delete('ref_pocket')
        cmd.delete('sel1')
        cmd.delete('sel2')

    def get_pocket_rmsd(self, model1, model2):
            """
            I assume that the models have identical sequence. Therefore I use 'align' and 'rms_cur' instead of 'tmalign'.
            :param model1:
            :param model2:
            :return:
            """
            print("\n Calculating RMSD between pocket of model %s and %s." % (model1, model2))
            radius = self.radius
            while True:
                try:  # DO NOT ALIGN, MODELS ARE ALREADY ALIGNED PROPERLY TO THE REFERENCE STRUCTURE
                    rmsd = self.focus_alignment(model1, model2,
                                            sel="byres(%s within %i of COM) and chain %s and polymer" % (model1, radius, self.CHAIN),
                                            fit=False, get_rms=True, atomsel="n. N+C+CA+CB+CG+CD")
                    break
                except CmdException:  # sometimes alignment fails with this error
                    print("Repeating RMSD calculation with radius = %f" % radius)
                    radius -= 1.0
            print("RMSD = %f" % rmsd)
            return rmsd

    def find_cluster_representatives(self, cluster2models_dict, rmsd_mdict):
        """
        Find cluster representative models by averaging the distances between each model and all of its cluster's members.
        :param cluster2models_dict:
        :param rmsd_mdict:
        :return:
        """

        cluster_reprModel_dist = {}
        for cluster, models in list(cluster2models_dict.items()):
            modelMeandist_list = []
            for model in models:
                mean_dist = np.mean([rmsd_mdict[model][m] for m in models if m!=model])
                if np.isnan(mean_dist): # this means that the cluster contains only one stucture
                    mean_dist = 0.0
                modelMeandist_list.append((model, mean_dist))
            modelMeandist_list.sort(key=itemgetter(1))  # sort by ascending rmsd
            cluster_reprModel_dist[cluster] = modelMeandist_list[0][0] # keep the one with the lowest average rmsd
        return cluster_reprModel_dist

    def cluster_models(self, cutoff_dist=None, CHAIN='A', RMSD_MATRIX=None, print_clusters=False):
        """

        :param pdb_list:
        :param cutoff_dist:
        :param CHAIN:
        :return:
        """
        self.load_structures(CHAIN)
        models = cmd.get_names()  # get all loaded structures
        models.remove("COM")  # COM is not a model for RMSD calculation ! However, the Pymol selection COM will remain
        rmsd_mdict = tree()
        condensed_vector = []  # contains absolutely no redundant values
        if not RMSD_MATRIX:
            for i in range(len(models)):
                rmsd_mdict[models[i]][models[i]] = 0.0
                for j in range(i+1, len(models)):
                    rmsd = self.get_pocket_rmsd(models[i], models[j])
                    condensed_vector.append(rmsd)
                    rmsd_mdict[models[i]][models[j]] = rmsd
                    rmsd_mdict[models[j]][models[i]] = rmsd
            square_matrix = squareform(condensed_vector)
        else:
            print("Loading Square RMSD Matrix from file %s." % RMSD_MATRIX)
            [ square_matrix ] = load_pickle(RMSD_MATRIX)

        if cutoff_dist:
            condensed_vector = []
            for i in range(square_matrix.shape[1]):
                for j in range(i+1, square_matrix.shape[1]):
                    condensed_vector.append(square_matrix[i][j])
                    rmsd_mdict[models[i]][models[j]] = square_matrix[i][j]
                    rmsd_mdict[models[j]][models[i]] = square_matrix[i][j]
            clusters = distance_clustering(condensed_vector, cutoff_dist)  # clustering by pre-defined distance
        else:
            clusters = silhuette_clustering(square_matrix) # automatic cluster number selection

        cluster2models_dict = defaultdict(list)
        for c,m in zip(clusters, models):
            cluster2models_dict[c].append(m)
        write_dict_contents(cluster2models_dict, open("receptor_clusters.txt", 'w'))
        if print_clusters:
            print("The receptor structures were grouped into the following clusters:")
            print_dict_contents(cluster2models_dict)
            print("rmsd_mdict=", rmsd_mdict)
        return cluster2models_dict, self.find_cluster_representatives(cluster2models_dict, rmsd_mdict)
