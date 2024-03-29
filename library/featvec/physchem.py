import multiprocessing

import numpy as np
from rdkit import Chem

from library.utils.print_functions import ColorPrint


def calculate_physchem_descriptors_from_mols(mols,
                                             return_molnames=True,
                                             return_descr_names=False,
                                             selected_descriptors=[],
                                             nproc=multiprocessing.cpu_count(),
                                             get_logtransform=False):
    """
        Method to calculate all physicochemical descriptors available in the MORDRED package
        from a Mol object or a SMILES string.

    :param mol: can be a Mol object or a SMILES string
    :param molname: the name of the molecule corresponding to the input SMILES
    :param return_names:
    :param get_logtransform:    fails for -1 values!    # TODO: fix this
    :return molname:    if return_molnames=True then the input molname will be returned first
    :return descriptor_array:   numpy array with the physchem descriptors for each molecule
    :return descriptor_names:   list of descriptor names in the order they occur in descriptor_array.
    """
    ColorPrint("Calculating physicochemical descriptors of %i molecules..." % len(mols), "OKBLUE")

    for i in range(len(mols)):
        if type(mols[i]) == str:  # if mol is a SMILES string
            mols[i] = Chem.AddHs(Chem.MolFromSmiles(mols[i]))

    # NEW WAY USING MORDRED
    from mordred import Calculator, descriptors
    from library.physchem.mordred_descriptors import mordred_dfunc
    if len(
            selected_descriptors) > 0 and 'all' not in selected_descriptors:  # create descriptor calculator with selected descriptors
        calc = Calculator()
        for d in selected_descriptors:
            # To register only a selected descriptor: calc.register(mordred.RotatableBond.RotatableBondsCount)
            calc.register(mordred_dfunc[d])
    else:  # create descriptor calculator with all descriptors
        calc = Calculator(descriptors, ignore_3D=True)

    results = calc.map(mols, nproc=nproc)
    # To access a specific descriptor value by its name: results.name['nRot']
    molnames, featvecs = [], []
    for r in list(results):
        molnames.append(r.mol.GetProp("_Name"))
        featvec = list(r.values())
        if get_logtransform:  # append also the logarithmically transformed values
            # TODO: this np.min is temporary fix. For universal compatibility you must find unique values for each descriptor
            featvec += np.log(np.array(featvec) - np.min(featvec) + 1).tolist()  # +1 to avoid inf values
        featvecs.append(featvec)
    featvecs = np.array(featvecs)

    returned_values = [featvecs]
    if return_molnames:
        returned_values.append(molnames)
    if return_descr_names:
        descriptor_names = [n.to_json()['name'] for n in results.keys()]  # TODO: remove redundancy!
        returned_values.append(descriptor_names)
    return returned_values
