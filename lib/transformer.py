"""
You can find example usages of DeepChem transformers in file deepchem/trans/tests/test_transformers.py.
"""

from lib.MMGBSA_functions import MMGBSA
import deepchem as dc
from scipy import stats
import numpy as np
from scipy.stats.mstats import zscore

class transformer():

    def __init__(self, y, method='none', original_scale=False, ):
        if method == 'none':
            return y
        elif method == 'deltag':
            return self.pseudoDeltaG(y, original_scale=original_scale)
    
    @staticmethod
    def pseudoDeltaG(y, original_scale=False):
        mmgbsa = MMGBSA()
        if original_scale:
            return mmgbsa.Ki2DeltaG(mmgbsa.DeltaG2Kd(y, original=False), original=True)
        else:
            return mmgbsa.DeltaG2Kd(y, original=False)

    @staticmethod
    def Clipping(featvec_list, y, x_max=5.0, y_max=100.0, transform_X=True, transform_y=False):
        dataset = dc.data.NumpyDataset(X=featvec_list, y=y)
        transformer = dc.trans.ClippingTransformer(transform_X=transform_X, transform_y=transform_y, x_max=5.0, y_max=500.0)
        dataset2 = transformer.transform(dataset)
        return dataset2.X, dataset2.y

    @staticmethod
    def Normalization(featvec_list, y, move_mean=False, transform_X=True, transform_y=False):
        dataset = dc.data.NumpyDataset(X=featvec_list, y=y)
        transformer = dc.trans.NormalizationTransformer(transform_X=transform_X, transform_y=transform_y,
                                                        dataset=dataset, move_mean=move_mean)
        dataset2 = transformer.transform(dataset)
        return dataset2.X, dataset2.y

    # TODO: the following transform functions are untested!
    @staticmethod
    def boxcox(featvec_list, type="boxcox_pearson"):
        return transformer(featvec_list, type=type)

    @staticmethod
    def log(featvec_list):
        return transformer(featvec_list, type="log")

    @staticmethod
    def log10(featvec_list):
        return transformer(featvec_list, type="log10")

    @staticmethod
    def transform(probability_array, type):
        """
        Method to do a mathematical transform on an array of numbers.
        :param probability_array:
        :param type:    the type of the mathematical transform, can be 'None', 'log', 'log10', 'boxcox_pearson',
                        'boxcox_mle'
        :return:
        """
        if type == "None":
            return probability_array
        elif type == "log":
            return np.log(probability_array)
        elif type == "log10":
            return np.log10(probability_array)
        elif type == "boxcox_pearson":
            lmax_pearsonr = stats.boxcox_normmax(probability_array)
            prob_pearson = stats.boxcox(probability_array, lmbda=lmax_pearsonr)
            return prob_pearson
        elif type == "boxcox_mle":
            prob_mle, lmax_mle = stats.boxcox(probability_array)
            return prob_mle