import multiprocessing
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from lib.physchem.mordred_descriptors import mordred_dfunc
from lib.utils.print_functions import ColorPrint
import pandas as pd

def calculate_physchem_descriptors_from_mols(mols,
                                             selected_descriptors=[],
                                             nproc=multiprocessing.cpu_count(),
                                             selected_descriptors_logtransform=[]):
    """
    Method to calculate physicochemical descriptors available in the MORDRED package
    from a Mol object or a SMILES string.

    # TODO: get_logtransform=True fails for -1 values. Fix it!
    """
    ColorPrint("Calculating physicochemical descriptors of %i molecules..." % len(mols), "OKBLUE")

    for i in range(len(mols)):
        if type(mols[i]) == str:  # if mol is a SMILES string
            mols[i] = Chem.AddHs(Chem.MolFromSmiles(mols[i]))

    # NEW WAY USING MORDRED
    if len(selected_descriptors) > 0 and 'all' not in selected_descriptors:  # create descriptor calculator with selected descriptors
        calc = Calculator()
        for d in selected_descriptors:
            # To register only a selected descriptor: calc.register(mordred.RotatableBond.RotatableBondsCount)
            calc.register(mordred_dfunc[d])
    else:  # create descriptor calculator with all descriptors
        calc = Calculator(descriptors, ignore_3D=True)

    # Note: To access a specific descriptor value by its name: results.name['nRot']
    featvec_df = pd.DataFrame(data=[[r.mol.GetProp("_Name")] + list(r.values())
                                    for r in list(calc.map(mols, nproc=nproc))],
                 columns=["structvar"] + selected_descriptors)

    if selected_descriptors_logtransform:  # append also the logarithmically transformed values
        # TODO: this np.min is temporary fix. For universal compatibility you must find unique values for each descriptor
        featvec_df.loc[:, selected_descriptors_logtransform] = featvec_df[selected_descriptors_logtransform] \
            .apply(lambda c: np.log(c - c.min() + 1))   # +1 to avoid inf values

    return featvec_df

