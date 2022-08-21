import pandas as pd
import tensorflow as tf
import os
from library.features.ligand_descriptors.deepFl_logP.create_feature_vector import create_feature_vector
import numpy as np

model = tf.keras.models.load_model(os.path.dirname(os.path.realpath(__file__)) + "/2020_DNN_v44_epoch_78_r0.892_rms0.359.h5")

def compute_logP(mols):
    logP_df = pd.DataFrame([[mol.GetProp('_Name')] for mol in mols], columns=['structvar'])
    return logP_df.assign(deepFl_logP=model.predict(np.array([create_feature_vector(mol) for mol in mols]), verbose=1))