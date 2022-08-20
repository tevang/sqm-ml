from library.global_fun import *
from collections import defaultdict
import csv
import pandas as pd

try:
    from rdkit.Chem import SDMolSupplier, SDWriter, MolToSmiles
except ImportError:
    print("WARNING: rdkit module could not be found!")
try:
    from library.modlib.pybel import Outputfile, readfile
except ImportError:
    print("WARNING: openbabel module could not be found.")


def extract_selected_props_from_sdf(sdf, propnames, csv_prop_file=None, default_propvalue_dict={}):
    """
    Method to add new property fields into an existing sdf file from a csv file.

    :param sdf:
    :param csv_prop_file:   the output CSV files with the property values. The 1st column will be the
                            'molname' and the 2nd the property value (can be int, float or bool or
                            lists of them separated with ','). The file will contain header.
    :return:
    """

    suppl = SDMolSupplier(sdf, removeHs=False, sanitize=False)
    prop_data = []
    for mol in suppl:

        if mol == None or mol.GetNumAtoms() == 0:
            continue  # skip empty molecules

        props_dict = mol.GetPropsAsDict()   # get all molecular properties
        for p in propnames:
            try:
                propval = str(props_dict[p])
            except KeyError:
                if p in default_propvalue_dict.keys():
                    ColorPrint("Molecule %s does not have property %s. Using default value %s." %
                               (mol.GetProp("_Name"), p, default_propvalue_dict[p]), "FAIL")
                    propval = 'nan'
                else:
                    ColorPrint("Molecule %s does not have property %s, thus 'nan' value will be written." %
                               (mol.GetProp("_Name"), p), "FAIL")
                    propval = 'nan'
            prop_data.append([mol.GetProp("_Name"), p, propval])
    prop_df = pd.DataFrame(data=prop_data,
                           columns=["molname", "property", "value"])
    prop_df = prop_df.astype(dtype={"molname": str, "property": str, "value": float})   # by default columns have dtype "object" and you can't use .nsmallest()
    if csv_prop_file:
        prop_df.to_csv(csv_prop_file, index=False)
    return prop_df


def sdf_text_iterator(sdf_file):
    if isinstance(sdf_file, str):
        f = open(sdf_file, 'r')
    else:
        f = sdf_file

    data = []
    for line in f:
        data.append(line)
        if line.startswith("$$$$"):
            yield "".join(data), data[0].rstrip()
            data = []


def chunked_sdf_iterator(sdf_file, num_mols):

    miter = sdf_text_iterator(sdf_file)
    tmp = []
    i = 0
    for sdf in miter:
        if i==num_mols:
            yield "".join(tmp)
            tmp = []
            i = 0
        i += 1
        tmp.append(sdf)
    yield "".join(tmp) # DON'T FORGET THE LAST MOLECULE!
