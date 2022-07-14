import multiprocessing
import os

from Bio import AlignIO
from modeller.automodel import automodel, assess

from lib.utils.print_functions import ColorPrint

# TODO: chech the license aggreement and see if commercial usage of MODELLER is allowed!
from lib.global_fun import tree, list_files

os.environ['KEY_MODELLER'] = 'MODELIRANJE'
from modeller import *
from modeller.parallel import *
import modeller.salign

def create_pairwise_seq_alignment(fasta_file, pdb_file, CHAINS=['A']):

    env = environ()  # or environ(rand_seed=-12312)  # To get different models from another script
    env.io.atom_files_directory = [os.path.dirname(pdb_file)]
    env.io.hetatm = True  # Read in HETATM records from template PDBs

    pdb_name = pdb_file.replace(".pdb", "")

    aln = alignment(env)
    ColorPrint("Appending for alignment the template %s" % (pdb_file), "BOLDBLUE")
    CHAINS.sort()   # sort by alphabetical order in order to capture the whole 'model_segment'
    m = model(env, file=pdb_file, model_segment=('FIRST:' + CHAINS[0], 'LAST:' + CHAINS[-1]))
    aln.append_model(m, atom_files=pdb_file, align_codes=pdb_name+"".join(CHAINS))
    aln.append(file=fasta_file, alignment_format='FASTA')
    aln.salign(output='ALIGNMENT')
    aln.write(file="%s_chain%s.ali" % (pdb_name, "".join(CHAINS)), alignment_format='PIR')
    return "%s_chain%s.ali" % (pdb_name, "".join(CHAINS)), env

def create_models(env,
                  ref_pdb_list,
                  QUERY_NAME,
                  MSA_FILE,
                  CPUs=multiprocessing.cpu_count(),
                  CHAIN="A",
                  EXTRA_PDB=None,
                  DELETE=False,
                  MODEL_NAME=None):
    """
        The method that creates Homology models.

    :param ref_pdb_list:
    :param QUERY_NAME:
    :param MSA_FILE:
    :param CHAIN:
    :param EXTRA_PDB:
    :param DELETE:
    :return:
    """
    # Use many CPUs in a parallel job on this machine
    j = job()
    for i in range(CPUs):
        j.append(local_slave())
    # log.verbose()

    # TODO: make chain ID structure depended
    # start = ref_pdb_list.index('3zsg')
    start = 0
    template_round_dict = {pdb:1 for pdb in ref_pdb_list[start:]}    # round 1 is just to check if the sequences in pdb files
                                                                     # match with the alignment. Round 2 is modelling.
    for refpdb in ref_pdb_list[start:] + ref_pdb_list[start:]:
    # for pdb in ['1ywn', '2oh4', '3c7q']:
        if template_round_dict[refpdb] == 1:
            ColorPrint("Checking sequence consistency of template %s between alignment and .pdb file." % refpdb, "BOLDGREEN")
        else:
            ColorPrint("Generating homology models from template %s" % refpdb, "BOLDGREEN")

        # Create a new pairwise alignment by adding the HETATMS from the template to the query sequence
        print("DEBUG: MSA_FILE= %s" % MSA_FILE)
        alignment = AlignIO.read(open(MSA_FILE), "pir")
        try:
            template_record = [record for record in alignment if record.name == refpdb + CHAIN][0]
        except IndexError:  # if this template is not in the alignment, skip it
            ColorPrint("WARNING: template %s is not in the alignment file %s. I will skip it!" % (refpdb + CHAIN, MSA_FILE), "WARNING")
            continue
        template_seq = list(template_record.seq)
        query_record = [record for record in alignment if record.name==QUERY_NAME][0]
        query_seq = list(query_record.seq)
        new_query_seq = []
        new_template_seq = []
        if EXTRA_PDB != None:
            HOMO_NAME = EXTRA_PDB+CHAIN
            homo_record = [record for record in alignment if record.name==HOMO_NAME][0]
            homo_seq = list(homo_record.seq)
            # homo_seq = [a.replace('.', '-') for a in homo_seq]
            new_homo_seq = []   # homologous structure to steer the homology modeling
        for i in range(len(template_seq)):
            taa,qaa = template_seq[i], query_seq[i]
            # print("DEBUG: taa,qaa=", taa,qaa)
            if EXTRA_PDB != None:
                haa = homo_seq[i]
            if taa == '.' and qaa == '-':   # add any ligand in the template to the query sequence
                qaa = '.'
            elif taa == '.' and qaa != '-': # probably a modified residue that does not exist in the equivalent position of the query protein
                if set(query_seq[i+1:]) == {'-'}:   # if this is the ligand (last bulk residue in the sequence followed by gaps!!)
                    new_template_seq.append('-')
                    new_template_seq.append('.')
                    new_query_seq.append(qaa)
                    new_query_seq.append('.')
                    if EXTRA_PDB != None:
                        new_homo_seq.append('-')
                        new_homo_seq.append('-')
                    continue
                else:   # if not the ligand, ignore it
                    pass
            new_query_seq.append(qaa)
            new_template_seq.append(taa)
            if EXTRA_PDB != None:
                new_homo_seq.append(haa)
        # For safety, move the BULK residue '.' at the very end of the alignment, because it may interfere and break the
        # contigency of the alignment between the query sequence and the homologous guide-structure.
        lig_index = [i for i,c in enumerate(new_template_seq) if c=='.'][-1] # keep the index of the last BULK residue as the ligand
        new_template_seq[lig_index] = '-'
        new_template_seq.append('.')
        if new_query_seq[lig_index] == '.':  # wrong without this if, look case 2x2lA, 4x3jA, TIE2
            new_query_seq[lig_index] = '-'
        new_query_seq.append('.')
        if EXTRA_PDB != None:
            new_homo_seq.append('-')

        with open(QUERY_NAME + '_' + refpdb + CHAIN + '.ali', 'w') as f:
            f.write(">P1;" + template_record.name + "\n")
            f.write(template_record.description + "\n")
            f.write("".join(new_template_seq) + "*\n\n")
            if EXTRA_PDB != None:
                f.write(">P1;" + homo_record.name + "\n")
                f.write(homo_record.description + "\n")
                f.write("".join(new_homo_seq) + "*\n\n")
            f.write(">P1;" + query_record.name + "\n")
            f.write(query_record.description + "\n")
            f.write("".join(new_query_seq) + "*\n\n")


        # # Create a new class based on 'loopmodel' so that we can redefine
        # # select_loop_atoms
        # # ATTENTION: you must import this class from another file in order to run it in parallel!
        # class MyLoop(loopmodel):
        #     # This routine picks the residues to be refined by loop modeling
        #     def select_loop_atoms(self):
        #         # Two residue ranges (both will be refined simultaneously)
        #         return selection(self.residue_range('163:', '184:'))

        if EXTRA_PDB != None:
            templates_tuple = (refpdb + CHAIN, HOMO_NAME)
        else:
            templates_tuple = (refpdb + CHAIN)

        a = automodel(env,
                      alnfile=QUERY_NAME + '_' + refpdb + CHAIN + '.ali',  # alignment filename
                      knowns=templates_tuple,  # codes of the templates
                      sequence=QUERY_NAME,  # code of the target
                      assess_methods=assess.DOPE  # request DOPE assessment
                      )
        if template_round_dict[refpdb] == 1:   # Check for alignment's sequences correctness
            aln = a.read_alignment()
            a.check_alignment(aln)
            template_round_dict[refpdb] = 2
            continue
        else:
            a.initial_malign3d = True           # do structural alignment of all templates
            a.starting_model= 1                 # index of the first model
            a.ending_model  = 50                 # index of the last model
                                                # (determines how many models to calculate)
            a.deviation = 4.0                   # has to be >0 if more than 1 model
            a.md_level = None                   # No refinement of model

            # # LOOP MODELING IS DEACTIVATED BECAUSE IT DOES NOT INCLUDE THE LIGANDS IN THE MODELS!
            # a.loop.starting_model = 1           # First loop model
            # a.loop.ending_model   = 1          # Last loop model
            # a.loop.md_level       = None        # Loop model refinement level

            a.use_parallel_job(j)               # Use the job for model building
            a.make()                            # do homology modelling

            # Get a list of all successfully built models from a.outputs
            ok_models = [x for x in a.outputs if x['failure'] is None]

            # Rank the models by DOPE score
            key = 'DOPE score'
            ok_models.sort(key=lambda a: a[key])

            # Get top model
            m = ok_models[0]
            ColorPrint("Top model: %s (DOPE score %.3f)" % (m['name'], m[key]), "BOLDBLUE")
            os.rename(m['name'], QUERY_NAME + "_from_" + refpdb + ".pdb")
            # print("DEBUG: model=%s" % QUERY_NAME + "_from_" + refpdb + ".pdb", "template=%s" % refpdb + "_" + CHAIN + ".pdb")
            if refpdb.startswith(QUERY_NAME) and refpdb.endswith("_" + CHAIN):
                rename_hetmols(QUERY_NAME + "_from_" + refpdb + ".pdb",
                               refpdb + ".pdb")  # rename the resnames of the ligands in the homology model
            elif refpdb.startswith(QUERY_NAME):
                rename_hetmols(QUERY_NAME + "_from_" + refpdb + ".pdb",
                               refpdb + "_" + CHAIN + ".pdb")  # rename the resnames of the ligands in the homology model
            if MODEL_NAME != None:
                os.rename(QUERY_NAME + "_from_" + refpdb + ".pdb", MODEL_NAME)
                rename_hetmols(MODEL_NAME, refpdb + ".pdb")  # rename the resnames of the ligands in the homology model

            if DELETE:
                # Delete the rest
                for m in ok_models[1:]:
                    os.remove(m['name'])
                for fname in list_files(folder=".", pattern="%s\.[VD][0-9]{8}" % QUERY_NAME, full_path=True):
                    os.remove(fname)
                for fname in list_files(folder=".", pattern="model_missing_loops.slave[0-9]+", full_path=True):
                    os.remove(fname)
                os.remove(QUERY_NAME + ".sch")
                os.remove(QUERY_NAME + ".rsr")
                os.remove(QUERY_NAME + ".ini")

def load_reference_PDB_IDs(fname, ident_threshold=80):
    ref_pdb_list = []
    with open(fname, 'r') as f:
        for line in f:
            words = line.split()
            if len(words) == 1:
                ColorPrint("WARNING: missing the sequence identity of %s to the target receptor. I am assuming"
                           " that it must be used as template for homology modeling." % words[0], "WARNING")
                ref_pdb_list.append(words[0].replace(".pdb", ""))   # OLD: words[0].lower().replace(".pdb", "")
            elif int(words[1]) >= ident_threshold:
                ref_pdb_list.append(words[0].replace(".pdb", ""))   # OLD: words[0].lower().replace(".pdb", "")
        return ref_pdb_list


def hetmols_from_pdb(pdb):
    resname_resid_atoms_dict = tree()
    with open(pdb, 'r') as f:
        for line in f:
            if line[:6] == 'HETATM':
                atom = line[12:16].strip()
                if atom[0] == 'H':  # exclude hydrogens
                    continue
                resname = line[17:20].strip()
                if resname == 'HOH':    # exclude waters
                    continue
                resid = line[22:26].strip()
                try:
                    resname_resid_atoms_dict[resname][resid].add(atom)
                except (AttributeError, KeyError):
                    resname_resid_atoms_dict[resname][resid] = set([atom])

    return resname_resid_atoms_dict


def rename_hetmols(model, template):

    template_resname_resid_atoms_dict = hetmols_from_pdb(template)
    resid2resname_dict = {} # the resid in the model pointing to the correct resname of this hetero mol
    model_resname_resid_atoms_dict = hetmols_from_pdb(model)

    for tresname in list(template_resname_resid_atoms_dict.keys()):
        for tresid in list(template_resname_resid_atoms_dict[tresname].keys()):
            for mresname in list(model_resname_resid_atoms_dict.keys()):
                for mresid in list(model_resname_resid_atoms_dict[mresname].keys()):
                    if template_resname_resid_atoms_dict[tresname][tresid] == model_resname_resid_atoms_dict[mresname][mresid]:
                        resid2resname_dict[mresid] = tresname

    fout = open("new_"+model, 'w')
    with open(model, 'r') as fin:
        for line in fin:
            if line[:6] == 'HETATM':
                atom = line[12:16].strip()
                resname = line[17:20].strip()
                resid = line[22:26].strip()
                if resid in list(resid2resname_dict.keys()):
                    resname = "%3s" % (resid2resname_dict[resid])
                    line = list(line)
                    line[17:20] = list(resname)
                    line = "".join(line)
            fout.write(line)
    fout.close()
    os.remove(model)
    os.rename("new_"+model, model)


def clean():
    for fname in list_files(".", "^.*\.[0-9]+$"):
        os.remove(fname)
    for fname in list_files(".", "^.*\.sch$"):
        os.remove(fname)
    for fname in list_files(".", "^homology_model_generator.slave[0-9]+$"):
        os.remove(fname)


def homomodel_to_template():
    """
    Method to associate each homology model to the source template
    :return:
    """
    pass