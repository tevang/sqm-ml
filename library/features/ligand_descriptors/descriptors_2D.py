from rdkit.Chem.Lipinski import NumRotatableBonds

from library.features.ligand_descriptors.Chan2021_entropy_descriptors.compute_Chan_entropy_descriptors import \
    compute_Chan_entropy_descriptors
from library.features.ligand_descriptors.bond_types import get_bond_types_count_as_vector
from library.features.ligand_descriptors.deepFl_logP.compute_logP import compute_logP
from library.features.ligand_descriptors.physchem_descriptors import calculate_physchem_descriptors_from_mols
from library.features.ligand_descriptors.rotbonds import create_rotbond_featvec_from_mol
from library.molfile.ligfile_parser import load_structure_file
import pandas as pd

from library.multithreading.parallel_processing_tools import apply_function_to_list_of_args_and_concat_resulting_dfs


def get_NumRotatableBonds_df(mols):

    return pd.DataFrame(data=[ [mol.GetProp("_Name"), NumRotatableBonds(mol)]
                              for mol in mols], columns=["structvar", "num_rotbonds"])

def get_rotbond_featvec_from_mol_df(max_rotbond_count=50, include_numrotbonds=False):

    func = create_rotbond_featvec_from_mol(max_rotbond_count=max_rotbond_count,
                                           include_numrotbonds=include_numrotbonds)

    def _get_rotbond_featvec_from_mol_df(mols):

        return pd.DataFrame(data=[ [mol.GetProp("_Name")] + func(mol) for mol in mols],
                        columns=["structvar"] + list(map(lambda x: "%i_contiguous_rotbonds" % x,
                                                       range(max_rotbond_count+1))))

    return _get_rotbond_featvec_from_mol_df

def calc_2D_descriptors(multimol_sdf, sel_physchem_descriptors, sel_rotbond_descriptors,
                        selected_descriptors_logtransform=[]):

    structvar_SMI_conf_mdict = load_structure_file(multimol_sdf, keep_structvar=True, get_SMILES=False, addHs=True)

    # MORDRED are rubbish descriptors!!
    featvecs_df = \
        calculate_physchem_descriptors_from_mols(mols=[structvar_SMI_conf_mdict[m]['SMI']
                                                       for m in structvar_SMI_conf_mdict.keys()],
                                                 selected_descriptors=sel_physchem_descriptors,
                                                 selected_descriptors_logtransform=selected_descriptors_logtransform)

    descr_function = {
        'num_rotbonds': get_NumRotatableBonds_df,
        'contiguous_rotbonds': get_rotbond_featvec_from_mol_df(max_rotbond_count=50, include_numrotbonds=False)
    }

    mols = [structvar_SMI_conf_mdict[m]['SMI'] for m in structvar_SMI_conf_mdict.keys()]
    for descr in sel_rotbond_descriptors:
        featvecs_df = featvecs_df.merge(descr_function[descr](mols), on="structvar")

    # Bond type count descriptors
    print("Computing bond type count descriptors.")
    mol_args = [[mol] for mol in mols]
    featvecs_df = featvecs_df.merge(apply_function_to_list_of_args_and_concat_resulting_dfs(
        get_bond_types_count_as_vector, args_list=mol_args, number_of_processors=1, concat_axis=0) \
                                    .reset_index(drop=True), on='structvar')    # use 1 processor to avoid unexplained KeyError '_Name'

    # Chan2021 entropy descriptors
    print("Computing Chan2021 entropy descriptors.")
    featvecs_df = featvecs_df.merge(apply_function_to_list_of_args_and_concat_resulting_dfs(
        compute_Chan_entropy_descriptors, args_list=mol_args, number_of_processors=1, concat_axis=0) \
                                    .reset_index(drop=True), on='structvar')    # use 1 processor to avoid unexplained KeyError exceptions

    # deepFl_logP
    print("Computing deepFl_logP.")
    featvecs_df = featvecs_df.merge(compute_logP(mols), on='structvar')

    return featvecs_df