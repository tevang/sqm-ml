from rdkit.Chem import GetPeriodicTable
from library.utils.print_functions import ColorPrint
import numpy as np

def get_mol_property_vals(mol, propname):
    """
    Get this property's values of one molecule in the order of the atoms in the Mol.
    :param mol:
    :return:
    """
    if ',' in mol.GetProp(propname):
        return [float(c) for c in mol.GetProp(propname).split(',')]
    else:
        return [float(mol.GetProp(propname))]

class Discretize():
    """
    This class discretizes an iterable of scalar values and assigns a bin ID to each scalar value.
    """
    def __init__(self):
        self.bins = None

    def fit(self, X, bin_width=None, center_to=None):
        bin_num = 'auto'
        if bin_width != None:
            if center_to != None:
                self.bins = Discretize.centered_bins(X, bin_width=bin_width, center_to=center_to)
                ColorPrint("Discretizing this property to %i number of bins" % len(self.bins), "OKBLUE")
                return
            else:
                bin_num = np.arange(X.min(), X.max(), bin_width).size + 1
        self.bins = np.histogram_bin_edges(X, bins=bin_num)
        ColorPrint("Discretizing this property to %i number of bins" % len(self.bins), "OKBLUE")

    def fit_transform(self, X, bin_width=None, center_to=None):
        bin_num = 'auto'
        if bin_width != None:
            if center_to != None:
                self.bins = Discretize.centered_bins(X, bin_width=bin_width, center_to=center_to)
            else:
                bin_num = np.arange(X.min(), X.max(), bin_width).size + 1
        self.bins = np.histogram_bin_edges(X, bins=bin_num) # X is flattened, therefore it can't take multiple features
        return np.digitize(X, self.bins)

    def transform(self, X):
        return np.digitize(X, self.bins)

    def transform_mol(self, mol, propname):
        """
        Method to discretize the specified property of a Mol object, either molecular property or atomic property.
        :param mol:
        :param propname:
        :return:
        """
        try:
            discrete_vals = self.transform(get_mol_property_vals(mol, propname))
        except ValueError:
            ColorPrint("WARNING: failed to discretize property %s with value %s" %
                       (propname,), "WARNING")
            return mol
        if len(discrete_vals) == 1: # this is a molecular property
            mol.SetProp('discrete ' + propname, str(discrete_vals[0]))
        elif len(discrete_vals) > 1:    # this is an atomic property
            for atom, discrete_val in zip(list(mol.GetAtoms()), discrete_vals):
                atom.SetProp('discrete ' + propname, str(discrete_val))
        return mol

    @staticmethod
    def centered_bins(X, bin_width=None, center_to=None):
        """
        Creates histogram bins for vector X, which center at center_to and width bin_width
        :param X:
        :param bin_width:
        :param center_to:
        :return:
        """
        bins = [center_to]
        edge_max = max(X)
        edge_min = min(X)
        loc = center_to
        while edge_min <= loc:
            loc -= bin_width
            bins.append(loc)
        loc = center_to
        while loc <= edge_max:
            loc += bin_width
            bins.append(loc)
        bins.sort()
        return bins