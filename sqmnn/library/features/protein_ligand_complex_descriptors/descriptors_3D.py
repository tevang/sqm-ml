import os

import pandas as pd

from lib.utils.print_functions import ColorPrint
from lib.global_fun import replace_alt
from lib.molfile.ligfile_parser import load_structure_file
from sqmnn.library.features.ligand_descriptors.rdkit_3D_shape_descriptors import compute_rdkit_3d_shape_descriptors
from sqmnn.library.features.protein_ligand_complex_descriptors.interface_surfaces import get_interface_surfaces_from_pdb
from lib.multithreading.parallel_processing_tools import apply_function_to_list_of_args_and_concat_resulting_dfs


def calc_3D_ligand_descriptors(multimol_sdf):

    structvar_SMI_conf_mdict = load_structure_file(multimol_sdf, keep_structvar=True, get_SMILES=False, addHs=True)
    mols = [structvar_SMI_conf_mdict[m]['SMI'] for m in structvar_SMI_conf_mdict.keys()]
    mol_args = [[mol] for mol in mols]

    # RDKit's 3D shape descriptors
    print("Computing RDKit's 3D shape descriptors.")
    return apply_function_to_list_of_args_and_concat_resulting_dfs(
        compute_rdkit_3d_shape_descriptors, args_list=mol_args, number_of_processors=1, concat_axis=0) \
                                    .reset_index(drop=True)  # use 1 processor to avoid unexplained KeyError exceptions


def calc_3D_complex_descriptors(pdb_file):
    out_csv = pdb_file.replace(".pdb", "_descr3D.csv")
    if os.path.exists(out_csv):
        return pd.read_csv(out_csv)

    complex_name = replace_alt(os.path.basename(pdb_file), ["_noWAT.pdb", ".pdb"], "")
    prot_interface_surf, prot_interface_SASA, lig_interface_surf, lig_interface_SASA, \
    mean_interface_surf, mean_interface_SASA = \
        get_interface_surfaces_from_pdb(pdb_file, LIG_RESNAME="LIG")

    if prot_interface_surf == None:  # protein or ligand selection was empty!
        return pd.DataFrame([[pdb_file, None, None, None, None, None, None, None]],
                            columns=['complex_name', 'prot_interface_surf', 'prot_interface_SASA', 'lig_interface_surf',
                                     'lig_interface_SASA', 'mean_interface_surf', 'mean_interface_SASA', 'net_charge'])
    # Get the ligand's net charge
    net_charge = None
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("HEADER ligand net charge") or line.startswith("# ligand net charge"):
                net_charge = int(line.split()[5])
                break
    if net_charge == None:
        ColorPrint("FAIL: pdb file %s does not contain the ligand net charge in the header!" % pdb_file, "FAIL")

    descr3D_df = pd.DataFrame([[complex_name, prot_interface_surf, prot_interface_SASA, lig_interface_surf, lig_interface_SASA,
            mean_interface_surf, mean_interface_SASA, net_charge]],
                        columns=['complex_name', 'prot_interface_surf', 'prot_interface_SASA', 'lig_interface_surf',
                                'lig_interface_SASA', 'mean_interface_surf', 'mean_interface_SASA', 'net_charge'])
    descr3D_df.to_csv(out_csv, index=False)
    return descr3D_df