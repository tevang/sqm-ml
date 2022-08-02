import gzip
import os
import pandas as pd
from pmapper.pharmacophore import Pharmacophore as P
from rdkit import Chem
from library.utils.print_functions import ColorPrint
from library.global_fun import replace_alt, get_poseID, get_frameID, get_structvar, list_files
from library.features.protein_ligand_complex_descriptors.PLEC import split_complex_pdb


def _return_empty_PMAPPER(complex_name, nbits):
    print("PMAPPER fingerprint of %s could not be computed." % complex_name)
    df = pd.DataFrame([[None] * nbits], dtype='uint8')
    df.insert(0, 'complex_name', complex_name)
    return df


def calc_PMAPPER(complex_pdb, ligand_resname='LIG', min_features=3, max_features=3, tol=0, nbits=8192,
                 activate_bits=3):
    complex_name = replace_alt(os.path.basename(complex_pdb), ["_noWAT.pdb", ".pdb"], "")

    print("Computing PMAPPER fingerprint of complex %s" % complex_pdb)
    protein_pdb, ligand_pdb = split_complex_pdb(complex_pdb, ligand_resname)
    if ligand_pdb is None:
        return _return_empty_PMAPPER(complex_pdb, nbits)
    mol = Chem.rdmolfiles.MolFromPDBFile(ligand_pdb, sanitize=True)
    if mol is None:
        return _return_empty_PMAPPER(complex_pdb, nbits)
    os.remove(protein_pdb)
    os.remove(ligand_pdb)
    p = P()
    p.load_from_mol(mol)
    df = pd.DataFrame([[0]*nbits], dtype='uint8')
    df.loc[:, p.get_fp(min_features, max_features, tol, nbits, activate_bits)] = 1
    df.insert(0, 'complex_name', complex_name)
    return df


def write_PMAPPER_to_csv(complex_pdb, ligand_resname='LIG', min_features=3, max_features=3, tol=0, nbits=8192,
                         activate_bits=3):
    out_csv = complex_pdb.replace(".pdb", "_PMAPPER.csv.gz")
    if not os.path.exists(out_csv):
        calc_PMAPPER(complex_pdb, ligand_resname, min_features, max_features, tol, nbits, activate_bits) \
            .to_csv(out_csv, index=False)


def gather_all_PMAPPER_to_one_csv(PMAPPER_dir, pdb_file_args, out_csv):
    print('Gathering all PMAPPER csv.gz files from %s and writing them to %s' % (PMAPPER_dir, out_csv))
    failed_files = []
    valid_csv_files = {pdb[0].replace(".pdb", "_PMAPPER.csv.gz") for pdb in pdb_file_args}
    existing_csv_files = set(list_files(folder=PMAPPER_dir,
                                        pattern='_PMAPPER.csv.gz',
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

    ColorPrint("THE FOLLOWING PMAPPER FILES WERE EMPTY:", "FAIL")
    print(failed_files)


def load_PMAPPER(features_df, PROTEINS, Settings):
    complex_names = features_df.apply(lambda r: '%s_pose%i_frm%i' %
                                                (r['structvar'], r['pose'], r['frame']), axis=1)
    df_list = []
    for protein in PROTEINS:
        print("Loading %s PMAPPER fingerprints." % protein)
        df_reader = pd.read_csv(Settings.raw_input_file('_PMAPPER.csv.gz', protein), chunksize=10000)
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

    return features_df.rename(columns={c: 'pmap%s' % c for c in features_df.filter(regex='^[0-9]+$').columns})