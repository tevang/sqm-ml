from lib.pdb.pdbligands import *
from lib.pymol.pocket_clustering import *
from lib.utils.print_functions import ColorPrint


class SelReceptor():

    def __init__(self, SDF, PDB_FOLDER):
        self.SDF = SDF
        self.PDB_FOLDER = PDB_FOLDER
        self.cluster2models_dict = defaultdict(list)
        self.cluster_reprModel_dist = {}

    def find_receptor_with_max_ligsim(self, refpdbSim_list):
        bestpdb = refpdbSim_list[0][0]  # the receptor co-crystalized with the most similar ligand
        bestmodel = os.path.basename(bestpdb).replace(".pdb", "")
        best_cluster = [c for c,s in list(self.cluster2models_dict.items()) if bestmodel in s][0]
        return self.cluster_reprModel_dist[best_cluster]

    def select_best_receptor(self, cutoff_dist=None, radius=16.0, TMALIGN=False, PYMOL_SCRIPT_REPO="", RMSD_MATRIX=None, pymolf=None):
        # TODO: select the most suitable receptor for each compound based on shape similarity. Create embeded conformers of each
        # TODO: compound on the crystal ligand and measure the shape similarity of the lowest energy conformer. ==> SLOW!
        ColorPrint("Associating compounds for docking to receptor structures.", "BOLDGREEN")
        # Load compounds for docking
        molname_SMILES_conformersMol_mdict = load_multiconf_sdf(self.SDF, get_conformers=True, get_molnames=False,
                                                                get_isomolnames=False, keep_structvar=True,
                                                                get_SMILES=False)

        pdbligs = PDBLigands(list_files(self.PDB_FOLDER, pattern=".*A\.pdb", full_path=True))
        pdbligs.extract_ligands()
        pdbligs.load_native_ligands(addHs=True)  # load crystal ligands
        pclust = PocketClust(pdbligs.pdb_list, ref_pdb="", CHAIN='A', radius=radius, TMALIGN=TMALIGN,
                             PYMOL_SCRIPT_REPO=PYMOL_SCRIPT_REPO, pymolf=pymolf)
        self.cluster2models_dict, self.cluster_reprModel_dist = \
            pclust.cluster_models(print_clusters=True, cutoff_dist=cutoff_dist, RMSD_MATRIX=RMSD_MATRIX)  # cluster all valid pdbs by their pocket similarity
        docked_molname_refpdbSim_dict = {}  # associates each compound to be docked to a reference crystal complex or homology model
        for molname in list(molname_SMILES_conformersMol_mdict.keys()):
            qMOL = molname_SMILES_conformersMol_mdict[molname]['SMI']
            refpdbSim_list = pdbligs.get_pdbs_above_simthreshold_MOL(qMOL, sim_threshold=0.0, strip_filechars=False, get_sim=True)
            if len(refpdbSim_list) > 0:
                docked_molname_refpdbSim_dict[molname] = (self.find_receptor_with_max_ligsim(refpdbSim_list), refpdbSim_list[0][1])
                ColorPrint("Found a crystal structure (%s, ligand similarity %.3f) for docking of compound %s." %
                           (refpdbSim_list[0][0], refpdbSim_list[0][1], molname), "OKBLUE")
        assert len(docked_molname_refpdbSim_dict) > 0, ColorPrint(
            "FAIL: no docked molecule was associated with crystal ligand in order to "
            "calculate RMSD!", "FAIL")

        # Write the results to a file
        association_list = [[m, d[0], d[1]] for m,d in list(docked_molname_refpdbSim_dict.items())]
        association_list.sort(key=itemgetter(2), reverse=True)    # sort by descending similarity to the crystal ligand
        writelist2file(association_list, "compound_to_receptor.list")