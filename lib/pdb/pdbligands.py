from Bio import pairwise2
from Bio.PDB import Superimposer, PDBIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Polypeptide import three_to_one

from lib.USRCAT_functions import *
from lib.lig_rmsd import *
from lib.molfile.ligfile_parser import load_structure_file
from lib.molfile.mol2_parser import mol2_to_sdf
# if "SCHRODINGER" in os.environ and os.path.exists(os.environ.get('SCHRODINGER') + "/utilities/structconvert"):
from lib.utils.print_functions import ColorPrint


class PDBLigands():
    """
    This class contains functions to extract the crystal ligands from a list of pdb files. Two possible uses:

    2. We have many crystal structures and homology models of a protein receptor, e.g. hs90a. We have done Virtual Screening of
    a chemical library against this receptor. Now we want to find which of the crystal ligands were docked and measure the RMSD
    of each docking pose.

    >>> pdbligs = PDBLigands(crystal_struct_list)
    >>> pdbligs.extract_ligands()
    >>> pdbligs.load_ligands()
    >>> docked_molname_SMILE_dict = {...}  # molnames and SMILES of the docked compounds
    >>> molname_complexList_dict = {}   # chemlib molname -> [pose1 complex, pose2 complex, ...]
    >>> docked_molname_refpdb_dict = {}     # associates each docked compound to a reference crystal complex or homology model
    >>> for molname,SMILES in docked_molname_SMILE_dict.items():
    >>>     docked_molname_refpdb_dict[molname] = pdbligs.get_pdbs_above_simthreshold_SMILES(SMILES,sim_threshold=1.0)[0]  # keep the most similar
    >>> complex_rmsd_dict = {}  # pdb filename of a docking pose complex -> RMSD of its compound pose with the selected reference crystal ligand
    >>> for molname,refpdb in docked_molname_refpdb_dict.items():
    >>>     refpdblig = PDBLigands(refpdb)
    >>>     for complex_pdb in molname_complexList_dict[molname]:
    >>>         aln_qpdb = refpdblig.superimpose_pdbs(complex_pdb)[0]    # the superimpose pdb file
    >>>         rmsd = refpdblig.rmsd(aln_qpdb)[0]
    >>>         complex_rmsd_dict[complex_pdb] = rmsd
    >>>


    """

    def __init__(self, pdb_list):
        if type(pdb_list) == str:
            pdb_list = [pdb_list]
        else:
            pass

        self.pdb_list = []
        for pdb in pdb_list:
            if '_' in os.path.basename(pdb):
                dirname = os.path.dirname(pdb)
                if not dirname:
                    dirname = "."
                new_pdb = dirname + "/" + os.path.basename(pdb).replace("_", "")
                ColorPrint("WARNING: underscore '_' characters are not allowed in pdb filenames. Renaming "
                           "file %s to %s" % (pdb, new_pdb), "WARNING")
                if not os.path.exists(new_pdb): # if it hasn't been renamed already from a previous round
                    os.rename(pdb, new_pdb)
                pdb = new_pdb
            self.pdb_list.append(pdb)

        self.pdb2ligMOL_dict = {}   # this dict will cary the MOL objects of every ligand in each pdb in self.pdb_list

    def extract_ligands(self, complex_pdb_list=[]):
        """
        I recommend first to extract ligands files and then attempt to load them from multiple threads, otherwise every thread will try to
        read/write the same files and some of them will fail.
        :return:
        """
        global CONSSCORTK_THIRDPARTY_DIR

        if not complex_pdb_list:
            complex_pdb_list = self.pdb_list
        for pdb in complex_pdb_list:
            # print("DEBUG: pdb=%s" % pdb)
            ligmol2 = pdb.replace(".pdb","") + "_LIG.mol2"
            written_ligfiles = list_files(os.path.dirname(pdb),
                                          pattern=os.path.basename(ligmol2)+".*",
                                          full_path=True) # check if pdb contained multiple ligands
            if len(written_ligfiles) == 0:  # if the ligands were not extracted already (otherwise multiple threads try to write/read and some fail)
                run_commandline("%s/fconv -l %s --t=%s" % (CONSSCORTK_THIRDPARTY_DIR, pdb, ligmol2))
                written_ligfiles = list_files(os.path.dirname(pdb),
                                              pattern=os.path.basename(ligmol2) + ".*",
                                              full_path=True)  # check if pdb contained multiple ligands
            for wf in written_ligfiles:
                mol2_to_sdf(wf)

    def load_ligands_from_pdbs(self, pdb_list, addHs=True):
        """
        Generic method to extract (if not already extracted) all ligands from each pdb file in pdb_list, and return a
        a dict with keys the pdb names and values the MOL objects of the largest ligand in each pdb file.
        :param pdb_list:
        :return:
        """
        pdb2ligMOL_dict = {}
        for pdb in pdb_list:
            ligsdf = pdb.replace(".pdb", "") + "_LIG"
            written_ligfiles = list_files(os.path.dirname(pdb),
                                          pattern=os.path.basename(ligsdf) + ".*\.mol2",
                                          full_path=True)  # check if pdb contained multiple ligands
            if len(written_ligfiles) == 0:  # if the ligands were not extracted already (otherwise multiple threads try to write/read and some fail)
                run_commandline("%s/fconv -l %s --t=%s" % (CONSSCORTK_THIRDPARTY_DIR, pdb, ligsdf))
                written_ligfiles = list_files(os.path.dirname(pdb),
                                              pattern=os.path.basename(ligsdf) + ".*\.mol2",
                                              full_path=True)  # check if pdb contained multiple ligands
            # Check if pdb contains multiple ligands
            if len(written_ligfiles) == 1:
                ligfile = written_ligfiles[0]
                molname_SMILES_conformersMol_mdict = load_structure_file(ligfile, get_SMILES=False, addHs=addHs)
                # TODO: use SCHRODINGER'S tools if available for file conversion.
                if len(molname_SMILES_conformersMol_mdict) == 0:  # if ligand failed to be loaded, continue
                    continue
                molname = list(molname_SMILES_conformersMol_mdict.keys())[0]
                ligMOL = molname_SMILES_conformersMol_mdict[molname]['SMI']
                pdb2ligMOL_dict[pdb] = ligMOL
            elif len(written_ligfiles) > 1:
                ligMolAtomNum_list = []
                for ligfile in written_ligfiles:
                    molname_SMILES_conformersMol_mdict = load_structure_file(ligfile, get_SMILES=False, addHs=addHs)
                    # TODO: use SCHRODINGER'S tools if available for file conversion.
                    if len(molname_SMILES_conformersMol_mdict) == 0:  # if ligand failed to be loaded, continue
                        continue
                    molname = list(molname_SMILES_conformersMol_mdict.keys())[0]
                    ligMOL = molname_SMILES_conformersMol_mdict[molname]['SMI']
                    ligMOL.SetProp("pdb_filename", pdb)
                    ligMOL.SetProp("mol2_filename", ligfile)
                    ligMolAtomNum_list.append((ligMOL, ligMOL.GetNumAtoms()))
                ligMolAtomNum_list.sort(key=itemgetter(1), reverse=True)
                try:
                    pdb2ligMOL_dict[pdb] = ligMolAtomNum_list[0][0]  # save the largest ligand only!
                except IndexError:
                    ColorPrint("FAIL: largest ligand from pdb file %s could not be loaded." % pdb, "OKRED")
            else:
                ColorPrint("WARNING: pdb file %s contains no ligand!" % pdb, "WARNING")

        if addHs:   # Add hydrogens to all ligands
            pdb2ligMOL_dict = {pdb: Chem.AddHs(ligMOL) for pdb, ligMOL in list(pdb2ligMOL_dict.items())}

        return pdb2ligMOL_dict

    def load_native_ligands(self, addHs=True):
        """
        Method to extract the ligands from all pdbs in self.pdb_list and save the MOL objects into a dict.
        :return:
        """
        # Store the ligand MOL objects
        self.pdb2ligMOL_dict = self.load_ligands_from_pdbs(self.pdb_list, addHs=addHs)

    def calc_ligand_similarities(self, qMOL, removeHs=True):
        """
        This methods is called internally by get_pdbs_above_simthreshold().
        Measures the fingerprints similarity between the reference ligand in SMILES format and all the crystal ligands in self.pdb_list.
        :param qMOL:    RDKit MOL object
        :param removeH: remove the hydrogens before measuring fingerprint(similarity. Similarities with hydrogens can be elusive.
        :return:
        """
        if removeHs:
            qMOL = AllChem.RemoveHs(qMOL)
        else:
            qMOL = Chem.AddHs(qMOL)
        pdbSim_list = []
        for pdb, ligMOL in list(self.pdb2ligMOL_dict.items()):
            if removeHs:
                ligMOL = AllChem.RemoveHs(ligMOL)
            sim = calc_Fingerprint_sim(qMOL, ligMOL, featvec_type="Morgan2") # the default for PDB querying
            pdbSim_list.append((pdb,sim))
        pdbSim_list.sort(key=itemgetter(1), reverse=True)
        return pdbSim_list

    def get_pdbs_above_simthreshold_SMILES(self, SMILES, sim_threshold, strip_filechars=True, get_sim=False):
        """
        Given a SMILES string of a reference compound, measure its fingerprint(similarity with all ligands in self.pdb_list files
        and return those pdbs that have ligands above the specified sim_threshold. The returned pdbs will be sorted by their similarity.
        :param SMILES:
        :param sim_threshold:
        :return:
        """
        qMOL = Chem.MolFromSmiles(SMILES)
        return self.get_pdbs_above_simthreshold_MOL(qMOL, sim_threshold, strip_filechars=strip_filechars, get_sim=get_sim)

    def get_pdbs_above_simthreshold_MOL(self, qMOL, sim_threshold, strip_filechars=True, get_sim=False):
        """
        Given a SMILES string of a reference compound, measure its fingerprint(similarity with all ligands in self.pdb_list files
        and return those pdbs that have ligands above the specified sim_threshold. The returned pdbs will be sorted by their similarity.
        :param SMILES:
        :param sim_threshold:
        :return:
        """
        pdbSim_list = self.calc_ligand_similarities(qMOL, removeHs=True)
        valid_pdbs = []
        for pdb, sim in pdbSim_list:
            if sim < sim_threshold:
                break
            else:
                if strip_filechars:
                    pdb = os.path.basename(pdb).replace(".pdb", "")
                if get_sim:
                    pdb = (pdb, sim)
                valid_pdbs.append(pdb)
        return valid_pdbs

    def get_fasta_from_pdb(self, pdb):
        """
        Get fasta sequence from any pdb file along with the structure object. Unknown amino acids and hetero atoms will me designated as X.
        :param pdb:
        :return:
        """
        p = PDBParser(PERMISSIVE=1)
        structure = p.get_structure(pdb, pdb)
        ## Now go through the hierarchy of the PDB file
        ##
        ## 1- Structure
        ##      2- Model
        ##          3- Chains
        ##              4- Residues
        ##

        model_chain_seq_mdict = tree()
        for model in structure:
            for chain in model:
                seq = list()
                chainID = chain.get_id()

                for residue in chain:
                    ## The test below checks if the amino acid is one of the 20 standard amino acids
                    ## Some proteins have "UNK" or "XXX", or other symbols for missing or unknown residues
                    if is_aa(residue.get_resname(), standard=True):
                        seq.append(three_to_one(residue.get_resname()))
                    else:
                        seq.append("X")
                ## This line is used to display the sequence from each chain
                # print(">Chain_" + chainID + "\n" + str("".join(seq)))
                model_chain_seq_mdict[model.get_id()][chain.get_id()] = str("".join(seq))
        return model_chain_seq_mdict, structure

    def pdb_seq_align(self, seq1, seq2, modelNum1=0, modelNum2=0):
        """
        Helper method to align seq multidicts, NOT fasta sequences. It returns the
        :param seq1:
        :param seq2:
        :param modelNum1:
        :param modelNum2:
        :return:
        """
        # print("seq1=", seq1)
        # print("seq2=", seq2)

        # TODO: currently presumes that each seq multidict contains only one chain.
        for chain1, fasta1 in list(seq1[modelNum1].items()):
            for chain2, fasta2 in list(seq2[modelNum2].items()):
                alignments = pairwise2.align.globalxx(fasta1, fasta2)
                aln_seq1 = alignments[0][0]
                aln_seq2 = alignments[0][1]
                return aln_seq1, aln_seq2


    def superimpose_pdbs(self, mobile_pdb):
        """
        Superimpose a mobile_pdb structure to all pdbs in self.pdb_list and save the new coordinates in a new pdb file.
        It returns the names of the new pdb files with superimposed coordinates of the mobile_pdb.
        :param mobile_pdb:
        :return:
        """
        seq1, structure1 = self.get_fasta_from_pdb(mobile_pdb)  # get the FASTA sequence and the BioPython structure object with coordinates
        model1 = structure1[0]
        pdb_out_filenames = []
        # print("DEBUG: self.pdb_list=", self.pdb_list, "\n\n\n\n\n")
        for i, pdb2 in enumerate(self.pdb_list):    # exceptionally for "docking_pose_rmsd.py" this list contains a single pdb file
            # Align the two sequences and find the common/matched residues
            seq2, structure2 = self.get_fasta_from_pdb(pdb2)
            model2 = structure2[0]     # I presume that each pdb file contains only one model
            # alignments = pairwise2.align.globalxx(seq1, seq2)
            # aln_seq1 = alignments[0][0]
            # aln_seq2 = alignments[0][1]
            aln_seq1, aln_seq2 = self.pdb_seq_align(seq1, seq2)
            use = [not c1 in ['-', 'X'] and not c2 in ['-', 'X'] for c1, c2 in zip(aln_seq1, aln_seq2)]   # which characters in the two aligned sequences to compare (no gaps!)

            ref_atoms = []
            mob_atoms = []
            for (ref_chain, mob_chain) in zip(model2, model1):
                ref_chain = iter(ref_chain)
                mob_chain = iter(mob_chain)
                for amino1, amino2, allow in zip(aln_seq1, aln_seq2, use):
                    if amino2 != '-': ref_res = next(ref_chain)
                    if amino1 != '-': mob_res = next(mob_chain)
                    if not allow:
                        continue
                    ref_res.resname = replace_alt(ref_res.resname, ["HIP", "HIE", "HID"], "HIS")
                    mob_res.resname = replace_alt(mob_res.resname, ["HIP", "HIE", "HID"], "HIS")
                    assert ref_res.resname == mob_res.resname, ColorPrint("%s, %s, %s, %s\n" % (ref_res.resname, amino1,
                                                                                                mob_res.resname, amino2), "FAIL")
                    # CA = alpha carbon
                    ref_atoms.append(ref_res['CA'])
                    mob_atoms.append(mob_res['CA'])

            # Align these paired atom lists:
            super_imposer = Superimposer()
            super_imposer.set_atoms(ref_atoms, mob_atoms)
            # Update the structure by moving all the atoms in
            # this model (not just the ones used for the alignment)
            super_imposer.apply(model1.get_atoms())
            ColorPrint("RMS(%s, %s) = %0.2f A." % (mobile_pdb, pdb2, super_imposer.rms), "OKBLUE")

            pdb_out_filename = "%s/%s" % (os.path.abspath(os.path.dirname(mobile_pdb)) ,os.path.basename(mobile_pdb).replace(".pdb", "_aln%i.pdb" % i))
            print("Saving aligned structure as PDB file %s" % pdb_out_filename)
            io = PDBIO()
            io.set_structure(structure1)
            io.save(pdb_out_filename)
            pdb_out_filenames.append(pdb_out_filename)

        return pdb_out_filenames

    def dist_2(self, atoma_xyz, atomb_xyz):
        """
        Sum of squared differences between x,y,z coordinates of two atoms.
        :param atoma_xyz:
        :param atomb_xyz:
        :return:
        """
        dis2 = 0.0
        for i, j in zip(atoma_xyz, atomb_xyz):
            dis2 += (i - j) ** 2
        return dis2

    def RMSD(self, mob_mol, ref_mol, amap):
        rmsd = 0.0
        atomNum = ref_mol.GetNumAtoms() + 0.0
        # print("RMSD between molecules %s and %s." % (mob_mol.GetProp('_Name'), ref_mol.GetProp('_Name')))
        for (pi, ri) in amap:
            posp = mob_mol.GetConformer().GetAtomPosition(pi)
            posf = ref_mol.GetConformer().GetAtomPosition(ri)
            rmsd += self.dist_2(posp, posf)
        rmsd = math.sqrt(rmsd / atomNum)
        return rmsd

    def calc_ligand_rmsd(self, mob_mol, ref_mol):

        mcs = rdFMCS.FindMCS([mob_mol, ref_mol], timeout=5, matchValences=False, ringMatchesRingOnly=True, completeRingsOnly=False)
        qmatched_atomIdx_list = mob_mol.GetSubstructMatch(MolFromSmarts(mcs.smartsString))
        cmatched_atomIdx_list = ref_mol.GetSubstructMatch(MolFromSmarts(mcs.smartsString))
        assert len(qmatched_atomIdx_list) == len(cmatched_atomIdx_list), \
            ColorPrint("ERROR in RMSD: mobile %s and reference ligand %s do not contain the same number of atoms!" %
                                       (mob_mol.GetProp("mol2_filename"), ref_mol.GetProp("mol2_filename")), "FAIL")
        atomMap = []
        qcoords2charge_dict = {}
        for q, c in zip(qmatched_atomIdx_list, cmatched_atomIdx_list):
            qatom = mob_mol.GetAtomWithIdx(q)
            catom = ref_mol.GetAtomWithIdx(c)
            # NOTE: the following two conditions are probably redundant since the 2 molecules are identical and all atoms are included
            # in the MCS.
            # assert qatom.GetAtomicNum() == catom.GetAtomicNum() and qatom.GetProp('_TriposAtomType') == catom.GetProp('_TriposAtomType'), \
            assert qatom.GetAtomicNum() == catom.GetAtomicNum(), \
                ColorPrint("ERROR IN RMSD: mobile %s and reference ligand %s do not have identical atom types "
                           "(%i,%s vs %i,%s)!" % (mob_mol.GetProp("mol2_filename"), ref_mol.GetProp("mol2_filename"),
                                                  qatom.GetAtomicNum(), qatom.GetProp('_TriposAtomType'),
                                                  catom.GetAtomicNum(), catom.GetProp('_TriposAtomType')), "FAIL")
            atomMap.append([q, c])
        return self.RMSD(mob_mol, ref_mol, atomMap)

    def calc_multi_ligand_rmsd(self, mob_pdb_list, superimpose=False, addHs=False):
        """
        Calculate the RMSD between the ligands in pdbs under mob_pdb_list and the ligands in pdbs under self.pdb_list.
        :param query_pdb:
        :return:
        """
        if type(mob_pdb_list) == str:
            mob_pdb_list = [mob_pdb_list]

        self.load_native_ligands()     # populate self.pdb2ligMOL_dict with the ligand coordinates from the self.pdb_list
        if superimpose:     # if asked, superimpose query_pdb to all self.pdb_list before RMSD calculation
            aln_mob_pdb_list = self.superimpose_pdbs(mob_pdb_list)
        else:   # otherwise keep the original pdb files
            aln_mob_pdb_list = mob_pdb_list
        self.extract_ligands(aln_mob_pdb_list)    # extract the ligands and write them to mol2 files
        aln_query_pdb2ligMOL_dict = self.load_ligands_from_pdbs(aln_mob_pdb_list, addHs=addHs) # load the extracted ligand mol2 file (keep only the largest)
        rmsd_list = []
        for mob_pdb, mob_mol in list(aln_query_pdb2ligMOL_dict.items()):
            for ref_pdb, ref_mol in list(self.pdb2ligMOL_dict.items()):   # usually only one ref_pdb
                ColorPrint("Calculating RMSD between mobile ligand from file %s and reference ligand from file %s." %
                           (mob_pdb, ref_pdb), "OKBLUE")
                # rmsd = self.calc_ligand_rmsd(mob_mol, ref_mol)
                rmsd = Ligand_RMSD(mob_mol, ref_mol).get_rmsd()
                ColorPrint("RMSD=%.2f" % rmsd, "OKBLUE")
                rmsd_list.append(rmsd)

        return rmsd_list