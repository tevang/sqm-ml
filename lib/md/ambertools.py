from lib.molfile.mol2_parser import *
from lib.MD_Analysis_functions import *
from lib.pdb.pdb_parser import *
import parmed as pmd
from collections import defaultdict
import shutil

from lib.utils.print_functions import ColorPrint


def create_prmtop(frcmod, ligfile, ligand_ff="gaff2"):

    if os.path.exists("tmp/"):
        shutil.rmtree("tmp/")
    os.mkdir("tmp/")
    run_commandline("ln -s %s %s/frcmod.ligand" % (os.path.abspath(frcmod), os.path.abspath("tmp/")))

    # convert with antechamber to mol2 with GAFF2 atom types
    # NOTE: -at gaff2 writes some unknown atom types that are not in the frcmod file (e.g. nh->nu, n->ns, n3->n7).
    run_commandline("antechamber -i %s -fi %s -o tmp/ligand.%s.mol2 -fo mol2 -rn LIG -at %s -dr n"
                        % (ligfile, ligfile.split('.')[-1], ligand_ff, ligand_ff))

    ligand_leap = """
source leaprc.%s
loadAmberParams tmp/frcmod.ligand
LIG = loadMol2 tmp/ligand.%s.mol2
saveAmberParm LIG tmp/ligand.prmtop tmp/ligand.inpcrd
quit
    """ % (ligand_ff, ligand_ff)

    with open("tmp/ligand_leap.in", 'w') as f:
        f.write(ligand_leap)
    leap_out = run_commandline("tleap -s -f tmp/ligand_leap.in", return_out=True, error_keywords=['FATAL:'])


def write_corrected_frcmod(ligfile, frcmod, out_frcmod, ligand_ff="gaff2", verbose=False):
    """
    This method takes the equilibrium bond lengths and angles from the ligfile and writes a new
    frcmod file with corrected GAFF2 ligand parameters for MD.

    :param ligfile: mol2 or sdf file with optimized ligand geometry from where to copy bond lengths and angles.
    :param frcmod: the frcmod file that needs corrections.
    :param out_frcmod: the name of the output frcmod file that carries the corrections.
    :return:
    """
    global args

    # create the prmtop and inpcrd file within a 'tmp/' folder
    create_prmtop(frcmod, ligfile, ligand_ff=ligand_ff)

    # load them to PARMED
    mol = pmd.load_file("tmp/ligand.prmtop", xyz="tmp/ligand.inpcrd", structure=True)
    bond_dict = defaultdict(list)
    for bond in mol.bonds:
        # print("%s-%s XXX %f" % (bond.atom1.type, bond.atom2.type, bond.measure()))
        bond_dict["%s-%s" % (bond.atom1.type, bond.atom2.type)].append(bond.measure())
        bond_dict["%s-%s" % (bond.atom2.type, bond.atom1.type)].append(bond.measure())    # add the reverse bond, too

    if verbose:
        print("\nBond = mean value += stdev, min-max")
        for bondname, distlist in bond_dict.items():
            print("%s = %f +- %f, %f" % (bondname, np.mean(distlist), np.std(distlist), np.ptp(distlist)))

    angle_dict = defaultdict(list)
    for angle in mol.angles:
        angle_dict["%s-%s-%s" % (angle.atom1.type, angle.atom2.type, angle.atom3.type)].append(angle.measure())
        angle_dict["%s-%s-%s" % (angle.atom3.type, angle.atom2.type, angle.atom1.type)].append(angle.measure())   # add the reverse angle, too

    if verbose:
        print("\nAngle = mean value += stdev, min-max")
        for anglename, anglelist in angle_dict.items():
            print("%s = %f +- %f, %f" % (anglename, np.mean(anglelist), np.std(anglelist), np.ptp(anglelist)))

    # par = pmd.load_file(frcmod)

    for bond in mol.bonds:
        bondname = "%s-%s" % (bond.atom1.type, bond.atom2.type)
        assert bondname in bond_dict.keys(), "ERROR: bond %s does not exist in the mol2 file with " \
                                               "the optimized geometry!" % bondname
        idx = bond.type.idx
        bond.type.req = round(np.mean(bond_dict[bondname]), 3)    # replace with the mean bond value
        mol.bond_types[idx].req = round(np.mean(bond_dict[bondname]), 3)    # replace with the mean bond value

    for angle in mol.angles:
        anglename = "%s-%s-%s" % (angle.atom1.type, angle.atom2.type, angle.atom3.type)
        assert anglename in angle_dict.keys(), "ERROR: angle %s does not exist in the mol2 file with " \
                                               "the optimized geometry!" % anglename
        idx = angle.type.idx
        angle.type.theteq = round(np.mean(angle_dict[anglename]), 3)    # replace with the mean angle value
        mol.angle_types[idx].theteq = round(np.mean(angle_dict[anglename]), 3)    # replace with the mean angle value

    # par.write('edited_'+frcmod, title="Created by mod_frcmod.py script.", style='frcmod')
    pmd.tools.writeFrcmod(mol, out_frcmod).execute()

    # clean intermediate files
    shutil.rmtree("tmp/")

def fix_failed_charge_calculations(command_file, args):
    """
    Method to find those antechamber jobs that failed due to wrong net charge definition and write a new command file
    named "commands_altcharge.list", with the alternative charges of each molecule.
    :param command_file:
    :param args:
    :return:
    """

    fout = open(add_suffix_to_filename(command_file, "_altcharge"), 'w')
    molnum2fix = 0
    with open(command_file, 'r') as f:
        for line in f:
            m = re.search("^mkdir ([^;]+); cd .*$", line)
            if m:
                molname = m.group(1)
                if os.path.exists(args.CHARGE_TYPE+"/"+molname+".bcc.mol2"):
                    continue
                elif not os.path.exists("%s/sqm.out" % molname):
                    ColorPrint("WARNING: molname %s failed due to wrong MOL2 format! Nothing I can do until you fix it." %
                               molname, "WARNING")
                    continue
                else:
                    if grep(".*You most likely have the charge of.*", "%s/sqm.out" % molname):
                        formal_charge, alt_charge, mol2 = get_formal_charge("%s.mol2" % molname, get_alternative_charge=True)
                        line = line.replace("-nc %i -pf" % formal_charge, "-nc %i -pf" % alt_charge)
                        fout.write(line + "\n")
                        molnum2fix += 1
    ColorPrint("Found %i molecules whose charge calculation has failed due to wrong net charge definition." % molnum2fix,
               "BOLDGREEN")

    fout.close()
    return fout.name

def calc_pl_interactions():
    
    "pairwise PL_int '(!:LIG<6 | :LIG & !:WAT)' out interaction_energies.dat avgout avg_interaction_energies.dat cuteelec 0 cutevdw 0"

def find_gaps_in_capped_prmtop(prmtop, coord):
    """
    Deletes atoms OXT and H1,H2,H3 from intermediate gaps in the protein. Useful for pdb4amber to identify
    subsequently
    :param prmtop:
    :param coord:
    :return:
    """
    def __is_aminoacid(residue):
        """
        :param residue:  ParmED residue object
        :return:
        """
        return [a.name in ["N", "CA", "C"] for a in residue.atoms].count(True) == 3

    mol = pmd.load_file("test_TER.protein.prmtop", xyz="test_TER.protein.inpcrd", structure=True)
    protein_indices = [i for i,r in enumerate(mol.residues) if __is_aminoacid(r)] # only protein resids
    gap_list = []
    for i in protein_indices[1:-1]: # exclude the beginning and the end of the polypeptide
        res = mol.residues[i]
        for a in res.atoms:
            if a.name == "OXT":
                next_res = mol.residues[i + 1]
                gap_list.append((res.number+1, next_res.number+1))     # residue numbering in prmtop begins from 0 but in AMBERMAsK from 1!
                # print("Deleting atom OXT from residue %i" % (res.number + 1))
                # res.delete_atom(a)
                # for na in next_res.atoms:
                #     if na.name in ["H1", "H2", "H3"]:
                #         print("Deleting atom %s from residue %i" % (na.name, next_res.number + 1))
                #         next_res.delete_atom(na)
                break
    return gap_list

def find_gaps_in_uncapped_prmtop(prmtop, coord):
    outpdb = "tmp.%s.pdb" % str(uuid.uuid4())
    MD_Analysis.write_pdb_from_prmtop(prmtop, coord, outpdb)
    gaps_list = find_gaps_in_pdb_AMBER(outpdb)
    os.remove(outpdb)
    return gaps_list

def get_protein_ligand_interaction_surfaces(pdb):
    surface = "trajin %s\nmolsurf ':LIG<:1.0 & ! :LIG' radii parse out %s\ngo\nquit" % \
              (pdb, pdb.replace(".pdb", "_protsurf.dat"))
    write2file(surface, "protsurf.ptj")
    run_commandline("cpptraj -p %s -i protsurf.ptj" % pdb)
    pass

def get_ligand_surface(pdb):
    pass
