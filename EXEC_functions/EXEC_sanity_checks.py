import os
import gzip

def _is_file_complete(fname, linenum_threshold=5):
    if fname.endswith('.gz'):
        file_obj = gzip.open(fname, 'rb')
    else:
        file_obj = open(fname, 'r')
    for linenum, line in enumerate(file_obj):
        if linenum >= linenum_threshold-1:
            return True
    return False

def sanity_checks(protein, Settings):
    """
    Checks that all necessary files exist.
    :return:
    """
    fpaths = ["_Glide_properties.csv.gz",
                  "_activities.txt",
                  "_all_compounds-net_charges.csv.gz"]

    if Settings.HYPER_USE_SCONF:
        fpaths += ["_all_compounds-ligprep.renstereo_ion_tau-3000confs.OPLS2005_Energies.csv.gz",
                  "_all_compounds-ligprep.renstereo_ion_tau-3000confs.PM6_COSMO.csv.gz",
                  ".schrodinger_confS_rmscutoff2.0_ecutoff6.csv.gz"]

    if len(Settings.HYPER_2D_DESCRIPTORS) > 0:
        fpaths += ["_2Dfeature_vectors.csv.gz"]

    if len(Settings.HYPER_3D_COMPLEX_DESCRIPTORS) > 0:
        fpaths += ["_protein_ligand_complex_descriptors.csv.gz"]

    if Settings.HYPER_PLEC and Settings.HYPER_SQM_FEATURES:
        fpaths += ["_PLEC.csv.gz"]

    if Settings.HYPER_PLEC and not Settings.HYPER_SQM_FEATURES:
        fpaths += ["_Glide_PLEC.csv.gz"]

    for fpath in fpaths:
        fpath = Settings.raw_input_file(fpath, protein)
        assert os.path.exists(fpath), "FAIL: input file %s doesn't exist!" % fpath
        assert _is_file_complete(fpath, linenum_threshold=5), "FAIL: input file %s seems to be incomplete!" % fpath
