#!/usr/bin/env python

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from rdkit.Chem.rdmolfiles import SDMolSupplier, SDWriter
from lib.utils.print_functions import ColorPrint
from lib.global_fun import get_structvar
from lib.molfile.sdf_parser import extract_selected_props_from_sdf
import pretty_errors

from sqmnn.library.features.protein_ligand_complex_descriptors.PLEC import calc_PLEC_from_pose_sdf


def cmdlineparse():
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description="""

First extracts the top-scored pose per compound and thein it computes it's PLEC vector.

""",
                            epilog="""
EXAMPLE: 


""")

    parser.add_argument("-prot_name", dest="PROTEIN_NAME", required=True, type=str, default=None,
                        help="Receptor PDB name.")
    parser.add_argument("-prot_pdb", dest="PROTEIN_PDB", required=True, type=str, default=None,
                        help="Receptor PDB file.")
    parser.add_argument("-posefile", dest="POSE_SDF", required=False, type=str, default=None,
                        help="SDF file of Glide's docking poses with properties.")
    parser.add_argument("-only_best_pose", dest="ONLY_BEST_POSE", required=False, default=False, action='store_true',
                        help="Compute the PLEC of only the top-scored pose per compound.")
    parser.add_argument("-outcsv", dest="OUT_CSV", required=False, type=str, default=None,
                        help="Name of output CSV file with the PLEC vectors.")
    args=parser.parse_args()
    return args


args = cmdlineparse()
ColorPrint("Extracting docking scores from %s" % args.POSE_SDF, "OKBLUE")
prop_df = extract_selected_props_from_sdf(sdf=args.POSE_SDF, csv_prop_file=None,
                                propnames=['r_i_docking_score'], default_propvalue_dict={'r_i_docking_score': 0})

top_poses_df = prop_df.groupby(['molname', 'property'], as_index=False).apply(min)
top_poses_df['molname'] = top_poses_df['molname'].apply(get_structvar).str.lower()
molname_bestscore_dict = top_poses_df[['molname', 'value']].set_index('molname').to_dict()['value']

pose_sdf = args.POSE_SDF
if args.ONLY_BEST_POSE:
    pose_sdf = args.POSE_SDF.replace('.sdf', '_best_poses.sdf')
    ColorPrint("Extracting top-scored poses from %s" % args.POSE_SDF, "OKBLUE")
    suppl = SDMolSupplier(args.POSE_SDF, sanitize=False, removeHs=False)
    writer = SDWriter(pose_sdf)
    for mol in suppl:
        if mol == None or mol.GetNumAtoms() == 0:
            continue  # skip empty molecules

        assert mol.GetNumConformers() == 1, ColorPrint("ERROR: mol has more that 1 conformers!!!", "FAIL")

        molname = get_structvar(mol.GetProp('_Name').lower())
        score = float(mol.GetProp('r_i_docking_score'))
        if molname_bestscore_dict[molname] == score:
            writer.write(mol)
            writer.flush()
    writer.close()

ColorPrint("Computing PLEC fingerprints of the top-scored docking poses.", "OKBLUE")
calc_PLEC_from_pose_sdf(args.PROTEIN_NAME, args.PROTEIN_PDB, pose_sdf=pose_sdf,
                        depth_ligand=1, depth_protein=5, distance_cutoff=4.5,
                        size=8192, count_bits=True, sparse=False, ignore_hoh=True,
                        remove_poseID=True) \
    .to_csv(args.OUT_CSV, index=False)