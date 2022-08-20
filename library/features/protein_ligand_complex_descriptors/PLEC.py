import gzip
import os
import numpy as np
import oddt
from oddt.fingerprints import PLEC

from library.utils.print_functions import ColorPrint
from library.global_fun import replace_alt, get_poseID, get_frameID, get_structvar, list_files, save_pickle

oddt.toolkit = oddt.toolkits.rdk # force ODDT to use RDKit
import pandas as pd

def split_complex_pdb(complex_pdb, ligand_resname='LIG'):
    protein_f = open(complex_pdb.replace('.pdb', '_prot.pdb'), 'w')
    ligand_f = open(complex_pdb.replace('.pdb', '_lig.pdb'), 'w')
    missing_ligand = True
    with open(complex_pdb, 'r') as f:
        for line in f:
            if line[17:20] == ligand_resname:
                ligand_f.write(line)
                missing_ligand = False
            else:
                protein_f.write(line)
    protein_f.close()
    ligand_f.close()
    return (None, None) if missing_ligand else (protein_f.name, ligand_f.name)


def _return_empty_PLEC(complex_name, size):
    print("PLEC fingerprint of %s could not be computed." % complex_name)
    plec_df = pd.DataFrame([[None] * size], dtype='uint8')
    plec_df.insert(0, 'complex_name', complex_name)
    return plec_df


def calc_PLEC_from_complex_pdb(complex_pdb, ligand_resname='LIG', depth_ligand=1, depth_protein=5, distance_cutoff=4.5,
                               size=8192, count_bits=True, sparse=False, ignore_hoh=True):

    complex_name = replace_alt(os.path.basename(complex_pdb), ["_noWAT.pdb", ".pdb"], "")
    protein_pdb, ligand_pdb = split_complex_pdb(complex_pdb, ligand_resname)
    if ligand_pdb is None:
        return _return_empty_PLEC(complex_name, size)
    rdkit_protein = list(oddt.toolkit.readfile('pdb', protein_pdb, sanitize=False))[0]
    rdkit_ligand = list(oddt.toolkit.readfile('pdb', ligand_pdb, sanitize=False))[0]
    if rdkit_ligand is None:
        return _return_empty_PLEC(complex_name, size)
    try:
        oddt_protein = oddt.toolkit.Molecule(rdkit_protein)
        oddt_ligand = oddt.toolkit.Molecule(rdkit_ligand)
    except AttributeError:
        return _return_empty_PLEC(complex_name, size)
    os.remove(protein_pdb)
    os.remove(ligand_pdb)
    plec_vector = PLEC(oddt_ligand, oddt_protein, depth_ligand=depth_ligand, depth_protein=depth_protein,
                    distance_cutoff=distance_cutoff,
                    size=size, count_bits=count_bits, sparse=sparse, ignore_hoh=ignore_hoh)
    plec_df = pd.DataFrame([plec_vector.tolist()],
                           columns=list(range(size)), dtype='uint8')
    plec_df.insert(0, 'complex_name', complex_name)
    return plec_df


def calc_PLEC_from_pose_sdf(receptor_name, receptor_pdb, pose_sdf, depth_ligand=1, depth_protein=5,
                            distance_cutoff=4.5, size=8192, count_bits=True, sparse=False, ignore_hoh=True,
                            remove_poseID=True):

    rdkit_protein = list(oddt.toolkit.readfile('pdb', receptor_pdb, sanitize=False))[0]
    oddt_protein = oddt.toolkit.Molecule(rdkit_protein)
    plec_vector_list = []
    for rdkit_ligand in list(oddt.toolkit.readfile('sdf', pose_sdf)):
        if rdkit_ligand is None:
            return _return_empty_PLEC(rdkit_ligand.Mol.GetProp('_Name').lower(), size)
        try:
            oddt_ligand = oddt.toolkit.Molecule(rdkit_ligand)
        except AttributeError:
            return _return_empty_PLEC(rdkit_ligand.Mol.GetProp('_Name').lower(), size)
        if remove_poseID:
            plec_vector_list.append(np.append(get_structvar(rdkit_ligand.Mol.GetProp('_Name').lower()),
                                              PLEC(oddt_ligand, oddt_protein, depth_ligand=depth_ligand,
                                                   depth_protein=depth_protein, distance_cutoff=distance_cutoff,
                                                   size=size, count_bits=count_bits, sparse=sparse, ignore_hoh=ignore_hoh)))
        else:
            plec_vector_list.append(np.append(rdkit_ligand.Mol.GetProp('_Name').lower(),
                                              PLEC(oddt_ligand, oddt_protein, depth_ligand=depth_ligand,
                                                   depth_protein=depth_protein, distance_cutoff=distance_cutoff,
                                                   size=size, count_bits=count_bits, sparse=sparse,
                                                   ignore_hoh=ignore_hoh)))
    plec_df = pd.DataFrame(plec_vector_list,
                           columns=['molname'] + list(range(size)), dtype='uint8')
    plec_df.insert(0, 'receptor_name', receptor_name)
    return plec_df


def write_PLEC_to_csv(complex_pdb, ligand_resname='LIG', depth_ligand=1, depth_protein=5, distance_cutoff=4.5,
                      size=8192, count_bits=True, sparse=False, ignore_hoh=True):
    out_csv = complex_pdb.replace(".pdb", "_PLEC.csv.gz")
    if not os.path.exists(out_csv):
        calc_PLEC_from_complex_pdb(complex_pdb, ligand_resname, depth_ligand, depth_protein, distance_cutoff, size,
                                   count_bits, sparse, ignore_hoh) \
            .to_csv(out_csv, index=False)


def gather_all_PLEC_to_one_csv(PLEC_dir, pdb_file_args, out_csv):
    print('Gathering all PLEC csv.gz files from %s and writing them to %s' % (PLEC_dir, out_csv))
    failed_files = []
    valid_csv_files = {pdb[0].replace(".pdb", "_PLEC.csv.gz") for pdb in pdb_file_args}
    existing_csv_files = set(list_files(folder=PLEC_dir,
                                        pattern='_PLEC.csv.gz',
                                        full_path=True))
    csv_files = list(valid_csv_files.intersection(existing_csv_files))
    with gzip.open(out_csv, 'wt') as f:
        f.write(','.join(pd.read_csv(csv_files[0]).columns.astype('str')) + '\n')
        for csv in csv_files:
            df = pd.read_csv(csv)
            if df.isnull().any(axis=1).any():
                failed_files.append(csv)
                continue
            f.write(df.to_csv(header=None, index=False))
            f.flush()

    ColorPrint("THE FOLLOWING PLEC FILES WERE EMPTY:", "FAIL")
    print(failed_files)

def load_PLEC(features_df, PROTEINS, Settings):

    complex_names = features_df.apply(lambda r: '%s_pose%i_frm%i' %
                                                (r['structvar'], r['pose'], r['frame']), axis=1)
    df_list = []
    for protein in PROTEINS:
        print("Loading %s PLEC fingerprints." % protein)
        df_reader = pd.read_csv(Settings.raw_input_file('_PLEC.csv.gz', protein), chunksize=10000)
        for df in df_reader:
            df.dropna(axis=1)
            df = df.astype({col: 'uint8' for col in df.filter(regex='^[0-9]').columns})
            df['complex_name'] = df['complex_name'].str.lower()
            df_list.append(df.loc[df['complex_name'].isin(complex_names)].assign(protein=protein))

    features_df = pd.merge(features_df,
                           pd.concat(df_list, ignore_index=True) \
                           .assign(pose=lambda df: df['complex_name'].apply(get_poseID),
                                   frame=lambda df: df['complex_name'].apply(get_frameID),
                                   structvar=lambda df: df['complex_name'].apply(get_structvar).str.lower()),
                           on=['protein', 'structvar', 'pose', 'frame'])

    return features_df.rename(columns={c: 'plec%s' % c for c in features_df.filter(regex='^[0-9]+$').columns})

def load_Glide_PLEC(features_df, PROTEINS, Settings):

    df_list = []
    for protein in PROTEINS:
        print("Loading %s Glide PLEC fingerprints." % protein)
        df_reader = pd.read_csv(Settings.raw_input_file('_Glide_PLEC.csv.gz', protein), chunksize=10000)
        for df in df_reader:
            df.dropna(axis=1)
            df = df.astype({col: 'uint8' for col in df.filter(regex='^[0-9]').columns})
            df['molname'] = df['molname'].str.lower()
            df_list.append(df.loc[df['molname'].isin(
                features_df.loc[features_df['protein']==protein, 'structvar'])].assign(protein=protein))

    features_df = pd.merge(features_df,
                           pd.concat(df_list, ignore_index=True) \
                           .rename(columns={'molname': 'structvar'}),
                           on=['protein', 'structvar'])

    return features_df.rename(columns={c: 'plec%s' % c for c in features_df.filter(regex='^[0-9]+$').columns})