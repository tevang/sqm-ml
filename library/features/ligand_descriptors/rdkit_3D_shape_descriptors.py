import pandas as pd
from rdkit.Chem.Descriptors3D import Asphericity, Eccentricity, InertialShapeFactor, NPR1, NPR2, PMI1, PMI2, PMI3, \
    RadiusOfGyration, SpherocityIndex

def compute_rdkit_3d_shape_descriptors(mol):
    return pd.DataFrame([[mol.GetProp('_Name'), Asphericity(mol), Eccentricity(mol), InertialShapeFactor(mol),
                          NPR1(mol), NPR2(mol), PMI1(mol), PMI2(mol), PMI3(mol), RadiusOfGyration(mol),
                          SpherocityIndex(mol)]], columns=['structvar', 'Asphericity', 'Eccentricity', 'InertialShapeFactor',
                                                           'NPR1', 'NPR2', 'PMI1', 'PMI2', 'PMI3', 'RadiusOfGyration',
                                                           'SpherocityIndex'])

