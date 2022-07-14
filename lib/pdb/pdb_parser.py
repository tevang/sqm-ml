import shutil
import string
import uuid
from operator import itemgetter

from Bio.PDB import Select, PDBParser, PDBIO
from Bio.PDB.Polypeptide import three_to_one
from Bio import pairwise2
from Bio.SubsMat.MatrixInfo import blosum90
from lib.global_fun import *
from biopandas.pdb import PandasPdb
import pandas as pd
import numpy as np

from lib.utils.print_functions import ColorPrint, Debuginfo


class RejectWAT(Select):

    def __init__(self, valid_residues=[]):
        """

        :param valid_residues: non-water residues to keep, must be in the format "resname_resid".
        """
        self.valid_residues = [(r.split('_')[0], int(r.split('_')[1])) for r in valid_residues]

    def is_aminoacid(self, residue):
        """
        Returns True if this residues belongs to the protein.
        :param residue:
        :return:
        """
        if len([a for a in residue.get_atoms() if a.get_name() in ['C', 'N', 'CA', 'O']]) == 4:
            return True
        else:
            return False

    def accept_residue(self, residue):
        if residue.get_resname() in ['H2O', 'HH0', 'OHH', 'HOH', 'OH2', 'SOL', 'WAT', 'TIP', 'TIP2', 'TIP3' 'TIP4']:
            return 0

        if self.valid_residues:
            # TODO: non-protein residues (e.g. LIG) may have different resid in each pdb file due to the presence of
            # TODO: varying number of WAT and ions. I am not sure if I should neglect the resid in these cases or not.
            # TODO: (look method 'is_aminoacid()'.
            for resname, resid in self.valid_residues:
                if residue.get_resname() == resname and residue.get_id()[1] == resid:
                    return 1
            return 0
        else:
            return 1


def strip_waters(inpdb, outpdb, valid_residues=[]):
    """
    Method to stript waters from a pdb file and keep all the rest. If valid_residues is specified, then only
    these non-water residues will be retained. Charges of the receptor and the ligand will be also prepended
    as comments to the outpdb.

    :param inpdb: input pdb with waters
    :param outpdb: output pdb without waters
    :param valid_residues: non-water residues to keep, must be in the format "resname_resid". If empty then all non-water
                            residues will be kept.
    :return:
    """

    parser = PDBParser()
    try:
        structure = parser.get_structure("complex_forscoring", inpdb)
    except IndexError:
        assert False, Debuginfo("FAIL: pdb file %s has wrong format!" % inpdb, fail=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(outpdb, RejectWAT(valid_residues=valid_residues))
    # Add TER records after protein, ion and ligand ends.
    prepare_pdb_for_amber(outpdb, outpdb.replace(".pdb", "_tmp.pdb"), overwritte=True, clean_files=True)

def check_pdb_integrity(pdb):
    """
    I think that this method it redundant. pdb4amber also finds gaps by default.
    :param pdb:
    :return:
    """
    parser = PDBParser(PERMISSIVE=False, QUIET=False)
    structure = parser.get_structure("complex_forscoring", pdb)
    from Bio.PDB.Polypeptide import PPBuilder
    ppb = PPBuilder()
    peptide_termini = []
    for pp in ppb.build_peptides(structure, aa_only=0): # consider also non-standard residues
        pp.sort()
        chain = pp[0].full_id[2]
        first_resid = pp[0].id[1]
        last_resid = pp[-1].id[1]
        peptide_termini.append( (chain, first_resid, last_resid) )
    peptide_termini.sort(key=itemgetter(0,1))
    is_pdb_good = True
    for i in range(len(peptide_termini)-1):
        pt1 = peptide_termini[i]
        pt2 = peptide_termini[i+1]
        if pt1[0] == pt2[0] and pt1[2] + 1 != pt2[1]:   # if same chain but incontinuous resids
            ColorPrint("WARNING: missing standard residues between %i and %i in chain %s." %
                       (pt1[2], pt2[1], pt1[0]), "OKRED")
            is_pdb_good = False
    return is_pdb_good

def has_chainIDs(pdb):
    ppdb = PandasPdb().read_pdb(pdb)
    chainIDs = list(set(ppdb.df['ATOM'].chain_id))
    return not (len(chainIDs) == 1 and len(chainIDs[0].strip()) == 0)

def rename_duplicate_protons(inpdb, outpdb=None):
    """
    Renames duplicate protons, like in GLH renames H->HE2, but removes the HEADER, CONECT, END, TER, and other records. Keeps
    only ATOM and HETATM.
    :param pdb:
    :return:
    """
    ppdb = PandasPdb().read_pdb(inpdb)
    adf = ppdb.df['ATOM']
    updated_adf = pd.DataFrame()
    for chainID, chain_df in adf.groupby(["chain_id"]):
        for resid, residue_df in chain_df.groupby(["residue_number"]):
            atoms = [a[1] for a in residue_df.iterrows()]
            renamed_atoms = __rename_duplicate_protons(atoms)
            updated_adf = updated_adf.append(renamed_atoms, ignore_index=True)

    ppdb.df['ATOM'] = pd.DataFrame()  # reset ATOM dataframe
    for chainID, chain_df in updated_adf.groupby(["chain_id"]):
        chain_df.sort_values(["residue_number"], ascending=[True], inplace=True)
        ppdb.df['ATOM'] = ppdb.df['ATOM'].append(chain_df, ignore_index=True)
    # renumber atoms
    ordered_unique_elements = list(OrderedDict.fromkeys(ppdb.df['ATOM']['atom_number']))
    mapping_dict = {ordered_unique_elements[i]: i + 1 for i in range(0, len(ordered_unique_elements))}
    ppdb.df['ATOM']['atom_number'] = ppdb.df['ATOM']['atom_number'].map(mapping_dict)
    # renumber 'line_idx' because Biopandas writes the rows to PDB according to this field.
    ordered_unique_elements = list(OrderedDict.fromkeys(ppdb.df['ATOM']['line_idx']))
    mapping_dict = {ordered_unique_elements[i]: i + 1 for i in range(0, len(ordered_unique_elements))}
    ppdb.df['ATOM']['line_idx'] = ppdb.df['ATOM']['line_idx'].map(mapping_dict)

    if outpdb:
        ppdb.to_pdb(path=outpdb, records=['ATOM', 'HETATM'], gz=False)  # discard CONECT recons, headers and comments
    else:
        outpdb = "tmp.%s.pdb" % str(uuid.uuid4())
        ppdb.to_pdb(path=outpdb, records=['ATOM', 'HETATM'], gz=False)  # discard CONECT recons, headers and comments
        shutil.move(outpdb, inpdb)    # overwrite the original PDB

def __atom_rmsd(a1, a2):
    return np.sqrt((a2.x_coord - a1.x_coord) ** 2 + (a2.y_coord - a1.y_coord) ** 2 + (a2.z_coord - a1.z_coord) ** 2)

def __rename_duplicate_protons(atoms):
    """
    Only for internal usage.
    :param atoms: list of Dataframe rows
    :return:
    """
    special_resname_proton_dict = {("GLU","HE2"): "GLH", ("ASP","HD2") : "ASH"}  # rename protonated GLU, ASP, fir pdb4amber doesn't!
    proton_names = [a.atom_name for a in atoms if a.atom_name.startswith("H")]
    if len(proton_names) == len(set(proton_names)):     # nothing to rename
        return atoms
    heavy_atoms = [a for a in atoms if not a.atom_name.startswith("H")]
    duplicate_proton_names = {p for p in proton_names if proton_names.count(p)>1}
    correct_atoms = [a for a in atoms if a.atom_name not in duplicate_proton_names]
    corrected_protons = []
    change_resname_to = None
    for pname in duplicate_proton_names:
        protons = [a for a in atoms if a.atom_name==pname]
        # Find the closest heavy atom to each duplicate proton, and rename the proton accordingly
        for p in protons:
            mindist = 10000
            closest_heavy_atom = None
            for ha in heavy_atoms:
                dist = __atom_rmsd(p, ha)
                if dist < mindist:
                    mindist = dist
                    closest_heavy_atom = ha
            correct_proton_name = 'H' + closest_heavy_atom.atom_name[1:]
            print("Renaming proton %s of residue %s%i to %s" % (p.atom_name, p.residue_name, p.residue_number, correct_proton_name))
            p.atom_name = correct_proton_name
            corrected_protons.append(p)
            if (p.residue_name, p.atom_name) in special_resname_proton_dict.keys():
                change_resname_to = special_resname_proton_dict[(p.residue_name, p.atom_name)]
    all_correct_atoms =  correct_atoms + corrected_protons
    if change_resname_to:
        print("Renaming residue %s%i to %s%i" % (p.residue_name, p.residue_number, change_resname_to, p.residue_number))
        for a in all_correct_atoms:
            a.residue_name = change_resname_to
    return all_correct_atoms

def assign_chainIDs(inpdb, outpdb):
    """
    Complicated case. This PDB file contains some residues twice but has no chain IDs. This method assigns chain ID to
    each replica. Only 2 chain assignment is supported. The method can also handle missing residues, either in one chain or in
    both chains.

    inpdb='/home2/thomas/Documents/Fragments/receptors/P07900/easy/ligand_3B25_1_1_protein_3K98_3_3_entry_00001_pose_10.pdb'

    :param inpdb:
    :return:
    """

    def __group_residue_atoms(resid):
        """
        Group atoms to individual residues. It also renames duplicate protons, like in GLH renames H->HE2.
        :param resid:
        :return:
        """
        atoms = adf.loc[adf['residue_number'] == resid]
        if 'N' in atoms.atom_name.to_list():
            root_atom_name = 'N'
        elif 'C' in atoms.atom_name.to_list():
            root_atom_name = 'C'
        else:
            root_atom_name = atoms.atom_name.iloc[0]
        chain_atoms_dict = {i: [a[1]] for i,a in enumerate(atoms[atoms.atom_name==root_atom_name].iterrows())}
        nonN_atoms = atoms[atoms.atom_name != root_atom_name]    # group the non-N atoms now
        nonN_atoms.sort_values(["atom_name"], ascending=[True], inplace=True)   # necessary to find the closest atom
        for i, a in nonN_atoms.iterrows():    # pool of unassigned atoms
            mindist = 10000
            closest_chainID = None
            for chainID in chain_atoms_dict.keys():
                for ra in chain_atoms_dict[chainID]:
                    dist = __atom_rmsd(a, ra)
                    if dist < mindist:
                        mindist = dist
                        closest_chainID = chainID
            chain_atoms_dict[closest_chainID].append(a)   # save this atom to the closest chain
        return {chainID: __rename_duplicate_protons(atoms) for chainID, atoms in chain_atoms_dict.items()}  # rename duplicate protons

    def __is_peptide_bond(atoms_iminus1, atoms_i, get_dist=False):
        """
        Assuming that these groups represent successive residues, we check if the O-N bond length is < 1.5 Angstroms.
        :param atoms_i: list of atoms of residue i
        :param atoms_iplus: list of atoms of residue i+1
        :return:
        """
        if [a.atom_name in ["N", "CA", "C"] for a in atoms_i].count(True) < 3:  # if not amino acid
            return False

        a_iminus1 = [a for a in atoms_iminus1 if a.atom_name == 'C'][0]
        a_i = [a for a in atoms_i if a.atom_name == 'N'][0]
        if get_dist:
            return __atom_rmsd(a_iminus1, a_i)
        else:
            return __atom_rmsd(a_iminus1, a_i) < 1.4

    def __set_chainID_to(atoms, chainID):
        new_atoms = []
        for a in atoms:
            a.chain_id = chainID
            new_atoms.append(a)
        return new_atoms

    ppdb = PandasPdb().read_pdb(inpdb)
    adf = ppdb.df['ATOM']
    adf.sort_values(["residue_number"], ascending=[True], inplace=True)
    updated_adf = pd.DataFrame()    # this is where we store the updated atom records with the correct chain_id
    all_resids = sorted(list(set(adf.residue_number)))
    resid_chain_atoms_mdict = tree()    # has the real chainID
    # Assign the very first resid
    chain_resatoms_dict = __group_residue_atoms(all_resids[0])  # chain index -> list of atoms belonging to one residue
    for cindex in chain_resatoms_dict.keys():
        chainID = string.ascii_uppercase[cindex]
        updated_atoms = __set_chainID_to(chain_resatoms_dict[cindex], chainID)
        resid_chain_atoms_mdict[all_resids[0]][chainID] = updated_atoms
        updated_adf = updated_adf.append(updated_atoms, ignore_index=True)
    all_chainIDs = set(resid_chain_atoms_mdict[all_resids[0]].keys())    # keep all found chainIDs
    # Now assign progressively the remaining
    for ridx in range(1, len(all_resids)):
        resid_i = all_resids[ridx]
        resid_im1 = all_resids[ridx-1]
        chain_resatoms_dict = __group_residue_atoms(resid_i)
        unassigned_cindices = list(chain_resatoms_dict.keys())
        assert len(unassigned_cindices) <= 2, Debuginfo(
            "FAIL: file %s contains %i x copies of the atoms of residue %i."
            " More that 2 chain-assignment is not supported!" %
            (inpdb, len(unassigned_cindices), resid_i), fail=True)
        for cindex in chain_resatoms_dict.keys():
            unassigned_chainIDs = list(all_chainIDs)   # to keep track which chainIDs were used
            if resid_i == resid_im1+1:  # successive residues
                for chainID in resid_chain_atoms_mdict[resid_im1].keys():
                    if __is_peptide_bond(resid_chain_atoms_mdict[resid_im1][chainID], chain_resatoms_dict[cindex]):
                        # These two residues are successive, thus set the chainID of their atoms accordingly.
                        updated_atoms = __set_chainID_to(chain_resatoms_dict[cindex], chainID)
                        resid_chain_atoms_mdict[resid_i][chainID] = updated_atoms
                        updated_adf = updated_adf.append(updated_atoms, ignore_index=True)
                        unassigned_cindices.remove(cindex)
                        unassigned_chainIDs.remove(chainID)
                        all_chainIDs.add(chainID)
                        break
            elif resid_i != resid_im1+1:    # missing intermediate residues
                # TODO: untested.
                # print("DEBUG: resid_im1=",resid_im1, "resid_i=", resid_i )
                chainIDDist_list = []
                for chainID in resid_chain_atoms_mdict[resid_im1].keys():
                    chainIDDist_list.append( (chainID,
                                              __is_peptide_bond(resid_chain_atoms_mdict[resid_im1][chainID],
                                                                chain_resatoms_dict[cindex], get_dist=True)) )
                chainIDDist_list.sort(key=itemgetter(1))
                closest_chainID = chainIDDist_list[0][0]
                updated_atoms = __set_chainID_to(chain_resatoms_dict[cindex], closest_chainID)
                resid_chain_atoms_mdict[resid_i][closest_chainID] = updated_atoms
                updated_adf = updated_adf.append(updated_atoms, ignore_index=True)
                unassigned_cindices.remove(cindex)
                unassigned_chainIDs.remove(chainID)
                all_chainIDs.add(chainID)

        if len(unassigned_cindices) == 1 and len(unassigned_chainIDs):
            cindex = unassigned_cindices[0]
            chainID = unassigned_chainIDs[0]
            updated_atoms = __set_chainID_to(chain_resatoms_dict[cindex], chainID)
            resid_chain_atoms_mdict[resid_i][chainID] = updated_atoms
            updated_adf = updated_adf.append(updated_atoms, ignore_index=True)
            unassigned_cindices.remove(cindex)
            unassigned_chainIDs.remove(chainID)

    ppdb.df['ATOM'] = pd.DataFrame()    # reset ATOM dataframe
    for chainID, chain_df in updated_adf.groupby(["chain_id"]):
        chain_df.sort_values(["residue_number"], ascending=[True], inplace=True)
        ppdb.df['ATOM'] = ppdb.df['ATOM'].append( chain_df, ignore_index=True )
    # renumber atoms
    ordered_unique_elements = list(OrderedDict.fromkeys(ppdb.df['ATOM']['atom_number']))
    mapping_dict = {ordered_unique_elements[i]: i+1 for i in range(0, len(ordered_unique_elements))}
    ppdb.df['ATOM']['atom_number'] = ppdb.df['ATOM']['atom_number'].map(mapping_dict)
    # renumber 'line_idx' because Biopandas writes the rows to PDB according to this field.
    ordered_unique_elements = list(OrderedDict.fromkeys(ppdb.df['ATOM']['line_idx']))
    mapping_dict = {ordered_unique_elements[i]: i+1 for i in range(0, len(ordered_unique_elements))}
    ppdb.df['ATOM']['line_idx'] = ppdb.df['ATOM']['line_idx'].map(mapping_dict)
    ppdb.to_pdb(path=outpdb, records=['ATOM', 'HETATM'], gz=False)  # discard CONECT recons, headers and comments

def __align_chain_dataframes(tchain_df, rchain_df):
    t_resnameResid_list = list({(three_to_one(n), i) for n, i in zip(tchain_df.residue_name, tchain_df.residue_number)})
    t_resnameResid_list.sort(key=itemgetter(1))
    t_seq = "".join([rr[0] for rr in t_resnameResid_list])

    r_resnameResid_list = list({(three_to_one(n), i) for n, i in zip(rchain_df.residue_name, rchain_df.residue_number)})
    r_resnameResid_list.sort(key=itemgetter(1))
    r_seq = "".join([rr[0] for rr in r_resnameResid_list])

    # Align the two sequences
    alignments = pairwise2.align.globalds(t_seq, r_seq, blosum90, -10, -0.5)
    aln_t_seq, aln_r_seq = list(alignments[0][0]), list(alignments[0][1])   # aligned sequences
    tresid_to_rrresid_dict = OrderedDict()     # association between resids in target and in reference chains
    tindex, rindex = 0, 0
    for taa, raa in zip(aln_t_seq, aln_r_seq):
        if "-" in [taa, raa]:
            print("found gap!", t_resnameResid_list[tindex][1], r_resnameResid_list[rindex][1])
            if taa != "-":
                tindex += 1
            if raa != "-":
                rindex += 1
            continue
        tresid = t_resnameResid_list[tindex][1]
        rresid = r_resnameResid_list[rindex][1]
        tresid_to_rrresid_dict[tresid] = rresid     # save correspondence
        tindex += 1
        rindex += 1
    return tresid_to_rrresid_dict

def copy_missing_structure(target_pdb, ref_pdb, outpdb=None):
    """
    TODO: incomplete. I modified openmm_em_GB.py to do energy minimization to structures with missing loops by fixing
    TODO: the C- and N-terms.
    Method that, given an incomplete PDB file and a full-length PDB structure (e.g. after loop modeling), finds
    automatically the missing sections and copies the coordinates of the missing atoms.

    :param target_pdb: the incomplete PDB files
    :param ref_pdb: the full-length structure from where to copy the coordinates.
    :return:
    """
    def __is_atom_in_residue(a, residue_df):
        """

        :param a: Dataframe row
        :param residue: DataFrame
        :return:
        """
        # residue_name
        # residue_number
        return ((residue_df['atom_name'] == a.atom_name) &
                (residue_df['x_coord'] == a.x_coord) &
                (residue_df['y_coord'] == a.y_coord) &
                (residue_df['z_coord'] == a.z_coord)).any()

    tppdb = PandasPdb().read_pdb(target_pdb)
    rppdb = PandasPdb().read_pdb(ref_pdb)
    tadf = tppdb.df['ATOM']
    radf = rppdb.df['ATOM']
    tadf.sort_values(["residue_number"], ascending=[True], inplace=True)
    radf.sort_values(["residue_number"], ascending=[True], inplace=True)
    updated_tadf = pd.DataFrame()    # this is where we store the updated atom records with the correct chain_id
    all_tresids = sorted(list(set(tadf.residue_number)))
    all_rresids = sorted(list(set(radf.residue_number)))
    assert np.all([tr in all_rresids for tr in all_tresids]), \
        Debuginfo("FAIL: residue number is probably different between the target PDB %s and the reference PDB %s." %
                    (target_pdb, ref_pdb), fail=True)

    for tchainID, tchain_df in tadf.groupby(["chain_id"]):
        rchain_df = radf.loc[radf["chain_id"]==tchainID]
        for rresid, rresidue_df in rchain_df.groupby(["residue_number"]):
            if rresid in tchain_df.residue_number:

                # NOTE: only whole-missing residues are copied.

                # tresidue_df = tchain_df.loc[tchain_df["residue_number"] == rresid]
                # for i, ra in rresidue_df.iterrows():
                #     if __is_atom_in_residue(ra, tresidue_df) == False:
                #         tresidue_df = tresidue_df.append(ra, ignore_index=True)
                # updated_tadf = updated_tadf.append(tresidue_df, ignore_index=True)  # add the original coordinates
                pass
            else:   # add the whole rresidue_df as it is
                updated_tadf = updated_tadf.append( rresidue_df, ignore_index=True )

    tppdb.df['ATOM'] = pd.DataFrame()    # reset ATOM dataframe
    for chainID, chain_df in updated_tadf.groupby(["chain_id"]):
        chain_df.sort_values(["residue_number"], ascending=[True], inplace=True)
        tppdb.df['ATOM'] = tppdb.df['ATOM'].append( chain_df, ignore_index=True )
    # renumber atoms
    ordered_unique_elements = list(OrderedDict.fromkeys(tppdb.df['ATOM']['atom_number']))
    mapping_dict = {ordered_unique_elements[i]: i+1 for i in range(0, len(ordered_unique_elements))}
    tppdb.df['ATOM']['atom_number'] = tppdb.df['ATOM']['atom_number'].map(mapping_dict)
    # renumber 'line_idx' because Biopandas writes the rows to PDB according to this field.
    ordered_unique_elements = list(OrderedDict.fromkeys(tppdb.df['ATOM']['line_idx']))
    mapping_dict = {ordered_unique_elements[i]: i+1 for i in range(0, len(ordered_unique_elements))}
    tppdb.df['ATOM']['line_idx'] = tppdb.df['ATOM']['line_idx'].map(mapping_dict)

    if outpdb:
        tppdb.to_pdb(path=outpdb, records=['ATOM', 'HETATM'], gz=False)  # discard CONECT recons, headers and comments
    else:
        outpdb = "tmp.%s.pdb" % str(uuid.uuid4())
        tppdb.to_pdb(path=outpdb, records=['ATOM', 'HETATM'], gz=False)  # discard CONECT recons, headers and comments
        shutil.move(outpdb, target_pdb)    # overwrite the original PDB

def find_gaps_in_pdb_AMBER(pdb):
    """
    This method runs pdb4amber script just to find the gap ends in the protein structure.
    :param pdb:
    :return:
    """
    # ATTENTION: pdb4amber -l option appends the log output to an existing file and hence you can get spurious results.
    # ATTENTION: Use run_commandline's logname argument to save the log.
    run_commandline("pdb4amber -i %s -o %s --add-missing-atoms" %
                    (pdb, "tmp.pdb"),
                    logname="pdb4amber.log",
                    error_keywords=["Could not open file leaprc.dna.bsc1: not found"],
                    error_messages=["This error means that on your PC it is sensitive to the case. To correct it, "
                                    "simply copy the file: cp /home2/thomas/Programs/amber18/dat/leap/cmd/leaprc.DNA.bsc1 "
                                    "/home2/thomas/Programs/amber18/dat/leap/cmd/leaprc.dna.bsc1"])
    gaps_list = []
    with open("pdb4amber.log", 'r') as f:
        for line in f:
            if line.startswith("gap of"):
                words = line.split()
                Cresid, Nresid = words[6], words[9]
                gaps_list.append( (Cresid, Nresid) )
    # os.remove("tmp.pdb")

    return gaps_list


def erase_columns(column_range, inpdb, outpdb=None):
    """

    :param columns_range: starting and ending index to be erased in each line of the input PDB file
    :param pdb:
    :return:
    """
    ColorPrint("Removing columns 71-73 (secondary structure) in file %s added by Preparation Wizard "
               "cause they confuse AMBER." % inpdb,
               "OKBLUE")

    if not outpdb:
        outpdb = inpdb

    with open(inpdb) as f:
        contents = f.readlines()
    writelist2file(List=[l[:column_range[0]] + " "*(column_range[1]-column_range[0]+1) + l[column_range[1]+1:]
                         for l in contents],
                   file=outpdb)

def prepare_pdb_for_amber(inpdb, outpdb, pdb_args="", overwritte=False, clean_files=False):
    """
    This method runs pdb4amber script just to find the gap ends in the protein structure. If there are gaps, it adds TER
    records so that tleap will add N- and C-term caps to the topology file.

    Other possible usage is when you have capped protein without TER likes and you want to add them. Works also
    if the PDB file contains ions of ligands.

    :param pdb:
    :return:
    """
    # ATTENTION: pdb4amber -l option appends the log output to an existing file and hence you can get spurious results.
    # ATTENTION: Use run_commandline's logname argument to save the log.


    logfile = outpdb.replace(".pdb", ".log")
    run_commandline("pdb4amber -i %s -o %s --add-missing-atoms -l %s %s" %
                    (inpdb, outpdb, logfile, pdb_args),
                    logname="pdb4amber.log",
                    error_keywords=["Could not open file leaprc.dna.bsc1: not found"],
                    error_messages=["This error means that on your PC it is sensitive to the case. To correct it, "
                                    "simply copy the file: cp /home2/thomas/Programs/amber18/dat/leap/cmd/leaprc.DNA.bsc1 "
                                    "/home2/thomas/Programs/amber18/dat/leap/cmd/leaprc.dna.bsc1"])
    gaps_list = []
    with open(logfile, 'r') as f:
        for line in f:
            if line.startswith("gap of"):
                ColorPrint(line.strip() + ". Added TER card and removed the N-term H.", "OKRED")
                words = line.split()
                Cresid, Nresid = words[6], words[9]
                gaps_list.append( (Cresid, Nresid) )
   # Add TER records to outpdb if gaps were present and remove the H from the N-ends
    if len(gaps_list) > 0:
        Cends = [g[0] for g in gaps_list]
        Nends = [g[1] for g in gaps_list]
        tmp_pdb = "tmp.%s.pdb" % str(uuid.uuid4())
        with open(outpdb, 'r') as fin, open(tmp_pdb, 'w') as fout:
            prev_resid = None
            for line in fin:
                resid = "".join(line[22:26]).strip()
                if prev_resid == None and line.startswith("ATOM"):
                    Nends.insert(0, resid)  # to remove the H from the N-terminus
                atom_name = "".join(line[12:16]).strip()
                # print("DEBUG: resid=", resid, "atom_name=", atom_name)
                if line.startswith("ATOM") and resid in Nends and prev_resid in Cends:
                    fout.write("TER\n")
                if line.startswith("ATOM") and resid in Nends and atom_name == "H":
                    continue
                fout.write(line)
                prev_resid = resid
        os.rename(tmp_pdb, outpdb)

    if overwritte:
        os.rename(outpdb, inpdb)

    # remove intermediate files
    if clean_files:
        for fname in [logfile, outpdb.replace(".pdb", "_nonprot.pdb"), outpdb.replace(".pdb", "_sslink"),
                      outpdb.replace(".pdb", "_renum.txt")]:
            os.remove(fname)
