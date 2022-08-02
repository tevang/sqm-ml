#!/usr/bin/env python
import multiprocessing
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import pandas as pd

from library.utils.print_functions import ColorPrint
from library.global_fun import list_files, get_structvar, replace_alt

try:
    from library.features.ligand_descriptors.pmapper_3D_pharmacophore import write_PMAPPER_to_csv, \
        gather_all_PMAPPER_to_one_csv
except ImportError:
    print("WARNING: PMAPPER module could not be found.")

try:
    from library.features.protein_ligand_complex_descriptors.PLEC import write_PLEC_to_csv, \
        gather_all_PLEC_to_one_csv
except ImportError:
    print("WARNING: PLEC module could not be found.")

from library.features.protein_ligand_complex_descriptors.descriptors_3D import calc_3D_complex_descriptors, \
    calc_3D_ligand_descriptors
from library.get_top_scored_Glide_complex_names import get_top_scored_Glide_complex_names
from library.multithreading.parallel_processing_tools import apply_function_to_list_of_args_and_concat_resulting_dfs


def cmdlineparse():
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description="""
This scripts is useful to create feature vectors from a folder with PDB files that contain the protein-ligand complexes.
Usually, these are the docking poses of a VS library. Various properties can be calculates. For example, from the complex's
coordinates the script can calculate the protein and ligand interface surfaces. From the ligand topology the script can calculate
the full range of QSAR descriptors provided by 

""",
                            epilog="""
EXAMPLE: 
python -m scoop -n 16 `which calc_PL_complex_features.py` \\
-structdir onlyvalid_PARP-1_min_complexes_forscoring \\
-receptor PARP-1 \\
-ligfile PARP-1_all_compouds-ligprep.renstereo_ion_tau.sdf \\
-ocsv PARP-1_feature_vectors.csv
        
""")

    parser.add_argument("-structdir", dest="STRUCTURE_DIR", required=False, type=str, default=None,
                        help="the folder with all the PDB files to be analyzed.")
    parser.add_argument("-receptor", dest="RECEPTOR", required=True, type=str, default=None,
                        help="The name of the receptor protein.")
    parser.add_argument("-ligfile", dest="LIG_FILE", required=False, type=str, default=None,
                        help="A SDF or MOL2 and SMILES file that contains all the structural variants that were docked"
                             " only once. Normally it is the output of LigPrep, not of Glide.")
    # TODO: implement the following option
    # parser.add_argument("-structvars", dest="STRUCTVARS", required=True, type=str, default=None,
    #                     help="Optionally, the structural variants to be analyzed in the -structvars folder.")
    parser.add_argument("-ocsv", dest="OUT_CSV", required=True, type=str, default=None,
                        help="The name of the CSV files where the feature vectors will be writen.")
    parser.add_argument("-ligresname", dest="LIG_RESNAME", required=False, type=str, default="LIG",
                        help="The resname of the ligand in the PDB file. Default: %(default)s")
    parser.add_argument("-d", "-descriptor", dest="DESCRIPTORS", required=False, action='append', default=[],
                        choices=['prot_interface_surf', 'prot_interface_SASA', 'lig_interface_surf', 'lig_interface_SASA',
                                 'mean_interface_surf', 'mean_interface_SASA', 'net_charge', 'MW', 'AMW', 'SLogP',
                                 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5',
                                 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10',
                                 'SlogP_VSA11', 'deepFl_logP', 'num_rotbonds', 'contiguous_rotbonds',
                                 'rotor_count', 'terminal_CH3_count', 'function_group_count', 'ring_flexibility',
                                 'Hbond_foldability', 'pipi_stacking_foldability', 'Asphericity', 'Eccentricity',
                                 'InertialShapeFactor', 'NPR1', 'NPR2', 'PMI1', 'PMI2', 'PMI3', 'RadiusOfGyration',
                                 'SpherocityIndex'],
                        help="The QSAR descriptor names to be computed.")
    parser.add_argument("-cpus", dest="CPUs", type=int, required=False, default=multiprocessing.cpu_count(),
                        help="the number of CPUs to use",
                        metavar="<number of CPUs>")
    parser.add_argument("-only_ligand", dest="ONLY_LIGAND_DESCRIPTORS", required=False, action='store_true', default=False,
                        help="Compute only ligand descriptors.")
    parser.add_argument("-only_3D_ligand", dest="ONLY_3D_LIGAND_DESCRIPTORS", required=False, action='store_true', default=False,
                        help="Compute only 3D ligand descriptors.")
    parser.add_argument("-only_complex", dest="ONLY_COMPLEX_DESCRIPTORS", required=False, action='store_true', default=False,
                        help="Compute only protein-ligand complex descriptors.")
    parser.add_argument("-only_plec", dest="ONLY_PLEC", required=False, action='store_true', default=False,
                        help="Compute only protein-ligand extended connectivity fingerprints.")
    parser.add_argument("-only_pmapper", dest="ONLY_PMAPPER", required=False, action='store_true', default=False,
                        help="Compute only PMAPPER 3D pharmacophore fingerprints.")
    parser.add_argument("-glide_csv", dest="GLIDE_CSV", required=False, type=str, default=None,
                        help="A CSV file with Glide properties including 'r_i_docking_score'.")
    parser.add_argument("-glide_deltag", dest="GLIDE_DeltaG", required=False, type=float, default=1.0,
                        help="Only complexes originating Glide poses within this DeltaG window will be retained and"
                             "converted to fingerprints.")
    args=parser.parse_args()
    return args

####################################### FUNCTION DEFINITIONS #####################################


# NOT USED!
# def extract_ligand_from_pdb(pdb_file, LIG_RESNAME="LIG"):
#     """
#     Method to extract the ligand from a PDB and write it to an SDF using PyMOL.
#     :param pdb_file:
#     :param LIG_RESNAME:
#     :return:
#     """
#     import pymol as pml
#
#     pml.cmd.load(pdb_file)
#     ligand_sdf = pdb_file.replace(".pdb", "") + "_LIG.sdf"
#     pml.cmd.save(ligand_sdf, "resn %s" % LIG_RESNAME)  # save the ligand into an SDF to calculate QSAR descriptors

########################################################################################################


def launch_pipeline(args):

    #################################### SANITY CHECKS & PREPROCESSING ################################

    complex_descriptors = ['prot_interface_surf', 'prot_interface_SASA', 'lig_interface_surf', 'lig_interface_SASA',
                           'mean_interface_surf', 'mean_interface_SASA', 'net_charge']
    physchem_descriptors = ['MW', 'AMW', 'SLogP', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4',
                             'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9',
                             'SlogP_VSA10', 'SlogP_VSA11']
    rotbond_descriptors = ['num_rotbonds', 'contiguous_rotbonds']
    bond_type_descriptors = ['bondType_THREECENTER', 'bondType_UNSPECIFIED', 'bondType_OTHER', 'bondType_HYDROGEN',
                             'bondType_DATIVER', 'bondType_QUADRUPLE', 'bondType_TWOANDAHALF', 'bondType_DATIVEL',
                             'bondType_HEXTUPLE', 'bondType_ONEANDAHALF', 'bondType_DATIVE', 'bondType_ZERO',
                             'bondType_FOURANDAHALF', 'bondType_AROMATIC', 'bondType_QUINTUPLE', 'bondType_TRIPLE',
                             'bondType_DATIVEONE', 'bondType_IONIC', 'bondType_FIVEANDAHALF', 'bondType_DOUBLE',
                             'bondType_SINGLE', 'bondType_THREEANDAHALF']
    Chan_entropy_descriptors = ['rotor_count', 'terminal_CH3_count', 'function_group_count', 'ring_flexibility',
                                'Hbond_foldability', 'pipi_stacking_foldability']
    rdkit_3D_ligand_descriptors = ['Asphericity', 'Eccentricity', 'InertialShapeFactor', 'NPR1', 'NPR2', 'PMI1',
                                  'PMI2', 'PMI3', 'RadiusOfGyration', 'SpherocityIndex']
    logP_descriptors = ['deepFl_logP']

    if len(args.DESCRIPTORS) > 0:
        sel_complex_descriptors = [descr for descr in complex_descriptors if descr in args.DESCRIPTORS]
        sel_physchem_descriptors = [descr for descr in physchem_descriptors if descr in args.DESCRIPTORS]
        sel_rotbond_descriptors = [descr for descr in rotbond_descriptors if descr in args.DESCRIPTORS]
        sel_bond_type_descriptors = [descr for descr in bond_type_descriptors if descr in args.DESCRIPTORS]
        sel_Chan_entropy_descriptors = [descr for descr in Chan_entropy_descriptors if descr in args.DESCRIPTORS]
        sel_rdkit_3D_ligand_descriptors = [descr for descr in rdkit_3D_ligand_descriptors if descr in args.DESCRIPTORS]
        sel_logP_descriptors = [descr for descr in logP_descriptors if descr in args.DESCRIPTORS]
    else:
        sel_complex_descriptors = complex_descriptors
        sel_physchem_descriptors = physchem_descriptors
        sel_rotbond_descriptors = rotbond_descriptors
        sel_bond_type_descriptors = bond_type_descriptors
        sel_Chan_entropy_descriptors = Chan_entropy_descriptors
        sel_rdkit_3D_ligand_descriptors = rdkit_3D_ligand_descriptors
        sel_logP_descriptors = logP_descriptors
        if args.ONLY_LIGAND_DESCRIPTORS:
            sel_complex_descriptors = []
        elif args.ONLY_COMPLEX_DESCRIPTORS:
            sel_physchem_descriptors, sel_rotbond_descriptors, sel_bond_type_descriptors, sel_Chan_entropy_descriptors, \
            sel_rdkit_3D_ligand_descriptors, sel_logP_descriptors = [[]]*6
        elif args.ONLY_PLEC or args.ONLY_PMAPPER:
            sel_physchem_descriptors, sel_rotbond_descriptors, sel_bond_type_descriptors, sel_Chan_entropy_descriptors, \
            sel_rdkit_3D_ligand_descriptors, sel_logP_descriptors, sel_complex_descriptors = [[]] * 7
        elif args.ONLY_3D_LIGAND_DESCRIPTORS:
            sel_rdkit_3D_ligand_descriptors = rdkit_3D_ligand_descriptors
            sel_physchem_descriptors, sel_rotbond_descriptors, sel_bond_type_descriptors, sel_Chan_entropy_descriptors, \
            sel_logP_descriptors, sel_complex_descriptors = [[]] * 6

    all_sel_descriptor_names = sel_complex_descriptors + sel_physchem_descriptors + sel_rotbond_descriptors +\
                               sel_Chan_entropy_descriptors + sel_rdkit_3D_ligand_descriptors + \
                               sel_logP_descriptors + sel_bond_type_descriptors

    if 'contiguous_rotbonds' in sel_rotbond_descriptors:
        all_sel_descriptor_names.remove('contiguous_rotbonds')
        for crb in range(50+1): # +1 because we count also contiguous rotbonds 0
            all_sel_descriptor_names.append( str(crb) + '_contiguous_rotbonds' )
    ###################################################################################################
    descriptors_df = pd.DataFrame(columns=['structvar'])
    ##
    ## Calculate the 2D descriptors
    ##
    if len(sel_logP_descriptors + sel_rotbond_descriptors + sel_bond_type_descriptors + sel_Chan_entropy_descriptors +
           sel_logP_descriptors) > 0:
        from library.features.ligand_descriptors.descriptors_2D import calc_2D_descriptors
        descriptors_df = calc_2D_descriptors(args.LIG_FILE, sel_physchem_descriptors, sel_rotbond_descriptors, args.CPUs)

    if len(sel_rdkit_3D_ligand_descriptors) > 0:
        if descriptors_df.shape[0] > 0:
            descriptors_df = pd.merge(descriptors_df, calc_3D_ligand_descriptors(args.LIG_FILE), on='structvar')
        else:
            descriptors_df = calc_3D_ligand_descriptors(args.LIG_FILE)

    if len(sel_complex_descriptors) > 0 or args.ONLY_PLEC or args.ONLY_PMAPPER:
        pdb_files = list_files(args.STRUCTURE_DIR, pattern=".*\.pdb", full_path=True)
        if args.GLIDE_CSV:
            valid_complex_names = get_top_scored_Glide_complex_names(args.GLIDE_CSV,
                                                                     structvar_pose_sel_column="r_i_docking_score",
                                                                     DeltaG=args.GLIDE_DeltaG,
                                                                     N_poses=100)
            pdb_file_args = [[pdb_file] for pdb_file in pdb_files if not pdb_file.endswith("_LIG.pdb") and
                             replace_alt(os.path.basename(pdb_file), ["_noWAT.pdb", ".pdb"], "").lower() in
                             valid_complex_names.values]  # ignore saved ligands
        else:
            pdb_file_args = [[pdb_file] for pdb_file in pdb_files if not pdb_file.endswith("_LIG.pdb")]
    if len(sel_complex_descriptors) > 0:
        ##
        print('Computing 3D protein-ligand complex descriptors of %i complexes in %s and writing '
              'them to csv.gz files.' % (len(pdb_file_args), args.STRUCTURE_DIR))
        ##
        # # Serial execution
        # descr_cmplx3D_df = pd.concat([calc_3D_descriptors(arg[0]) for arg in pdb_file_args], ignore_index=True)

        # Parallel execution
        descr_cmplx3D_df = apply_function_to_list_of_args_and_concat_resulting_dfs(
            calc_3D_complex_descriptors, args_list=pdb_file_args, number_of_processors=args.CPUs, concat_axis=0) \
            .reset_index(drop=True)

        failed_molfiles = descr_cmplx3D_df.loc[descr_cmplx3D_df.isna().any(axis=1), 'complex_name'].to_list()
        descr_cmplx3D_df.dropna(inplace=True)
        descr_cmplx3D_df['structvar'] = descr_cmplx3D_df['complex_name'].apply(get_structvar).str.lower()
        all_sel_descriptor_names.insert(0, 'complex_name')

        if descriptors_df.shape[0] > 0:
            descriptors_df = pd.merge(descriptors_df, descr_cmplx3D_df, on='structvar')
        else:
            descriptors_df = descr_cmplx3D_df

        ColorPrint("THE FOLLOWING PDB FILES COULD NOT BE LOADED:", "FAIL")
        print(failed_molfiles)

    if args.ONLY_PLEC:
        # Parallel execution
        print('Computing PLEC vectors of %i complexes in %s and writing them to csv.gz files.' %
              (len(pdb_file_args), args.STRUCTURE_DIR))
        apply_function_to_list_of_args_and_concat_resulting_dfs(
            write_PLEC_to_csv, args_list=pdb_file_args, number_of_processors=args.CPUs, concat_axis=None)

        gather_all_PLEC_to_one_csv(args.STRUCTURE_DIR, pdb_file_args, args.OUT_CSV)
        return

    if args.ONLY_PMAPPER:
        # Parallel execution
        print('Computing PMAPPER vectors of %i complexes in %s and writing them to csv.gz files.' %
              (len(pdb_file_args), args.STRUCTURE_DIR))
        apply_function_to_list_of_args_and_concat_resulting_dfs(
            write_PMAPPER_to_csv, args_list=pdb_file_args, number_of_processors=args.CPUs, concat_axis=None)

        gather_all_PMAPPER_to_one_csv(args.STRUCTURE_DIR, pdb_file_args, args.OUT_CSV)
        return

    descriptors_df[['structvar'] + all_sel_descriptor_names].assign(protein=args.RECEPTOR).to_csv(args.OUT_CSV, index=False)

if __name__ == "__main__":
    launch_pipeline(cmdlineparse())