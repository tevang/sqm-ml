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


def get_molnames_satisfying_property(sdf, prop_name, min_propval=None, max_propval=None, selected_basenames=[],
                                     keep_mols_without_prop=False):

    if min_propval and max_propval:
        ColorPrint("Looking for molecules in file %s with property %s value between %f and %f." %
                   (sdf, prop_name, min_propval, max_propval), "BOLDGREEN")

    valid_molnames = []
    suppl = SDMolSupplier(sdf, sanitize=False, removeHs=False)
    structvar_maxprop_dict = defaultdict(list)  # applicable only to properties: 'i_i_glide_confnum'
    total_molnum = 0
    for mol in suppl:
        total_molnum += 1
        if mol == None or mol.GetNumAtoms() == 0:
            continue  # skip empty molecules

        assert mol.GetNumConformers() == 1, ColorPrint("ERROR: mol has more that 1 conformers!!!", "FAIL")

        full_molname = mol.GetProp('_Name').lower()
        basename = sub_alt(full_molname, ["_pose[0-9]+", "_iso[0-9]+", "_tau[0-9]+", "_stereo[0-9]+", "_ion[0-9]+",
                                          "_noWAT", "_frm[0-9]+"], "").lower()
        if selected_basenames and basename not in selected_basenames:
            continue

        if not prop_name or len(prop_name) == 0:
            valid_molnames.append(full_molname)
            continue

        try:
            propval = float(mol.GetProp(prop_name))
        except KeyError:
            if keep_mols_without_prop:
                ColorPrint("WARNING: property %s does not exist for molecule %s but it will be included!" %
                           (prop_name, full_molname), "OKRED")
                propval = max_propval   # trick to keep the molname
            else:
                ColorPrint("WARNING: property %s does not exist for molecule %s thus it will be excluded!" %
                           (prop_name, full_molname), "OKRED")
                continue
        except ValueError:
            ColorPrint("WARNING: property %s for molecule %s is not a number (%s)" %
                       (prop_name, full_molname, mol.GetProp(prop_name)))

        if (min_propval == None or min_propval <= propval) and (max_propval == None or propval <= max_propval) \
                and full_molname not in valid_molnames:
            valid_molnames.append(full_molname)

    if len(valid_molnames) < total_molnum:
        ColorPrint("Only %i out of %i molnames were retrieved!" % (len(valid_molnames), total_molnum), "OKRED")

    return valid_molnames

def extract_mols_satisfying_property(sdf, out_sdf, prop_name, min_propval=None, max_propval=None, selected_basenames=[],
                                     keep_mols_without_prop=False):
    """
    Same as above but instead of molnames, it writes into the specified SDF file those molecules that satisfy the
    given property.
    :param sdf:
    :param prop_name:
    :param min_propval:
    :param max_propval:
    :param selected_basenames:
    :param keep_mols_without_prop:
    :return:
    """

    if min_propval and max_propval:
        ColorPrint("Looking for molecules in file %s with property %s value between %f and %f." %
                   (sdf, prop_name, min_propval, max_propval), "BOLDGREEN")

    valid_molnames = []
    suppl = SDMolSupplier(sdf, sanitize=False, removeHs=False)
    writer = SDWriter(out_sdf)
    structvar_maxprop_dict = defaultdict(list)  # applicable only to properties: 'i_i_glide_confnum'
    total_molnum = 0
    for mol in suppl:
        total_molnum += 1
        if mol == None or mol.GetNumAtoms() == 0:
            continue  # skip empty molecules

        assert mol.GetNumConformers() == 1, ColorPrint("ERROR: mol has more that 1 conformers!!!", "FAIL")

        full_molname = mol.GetProp('_Name').lower()
        basemolname = sub_alt(full_molname, ["_pose[0-9]+", "_iso[0-9]+", "_tau[0-9]+", "_stereo[0-9]+", "_ion[0-9]+",
                                          "_noWAT", "_frm[0-9]+"], "").lower()
        if selected_basenames and basemolname not in selected_basenames:
            continue

        if not prop_name or len(prop_name) == 0:
            writer.write(mol)
            writer.flush()
            valid_molnames.append(full_molname)
            continue

        try:
            propval = float(mol.GetProp(prop_name))
        except KeyError:
            if keep_mols_without_prop:
                ColorPrint("WARNING: property %s does not exist for molecule %s but it will be included!" %
                           (prop_name, full_molname), "OKRED")
                propval = max_propval   # trick to keep the molname
            else:
                ColorPrint("WARNING: property %s does not exist for molecule %s thus it will be excluded!" %
                           (prop_name, full_molname), "OKRED")
                continue
        except ValueError:
            ColorPrint("WARNING: property %s for molecule %s is not a number (%s)" %
                       (prop_name, full_molname, mol.GetProp(prop_name)))

        if (min_propval == None or min_propval <= propval) and (max_propval == None or propval <= max_propval) \
                and full_molname not in valid_molnames:
            writer.write(mol)
            writer.flush()
            valid_molnames.append(full_molname)

    if len(valid_molnames) < total_molnum:
        ColorPrint("Only %i out of %i molnames were retrieved!" % (len(valid_molnames), total_molnum), "OKRED")

def add_props_to_sdf(sdf, csv_prop_file):
    """
    Method to add new property fields into an existing sdf file from a csv file.

    :param sdf:
    :param csv_prop_file:   the 1st column should be the 'molname' and the 2nd the property value (can be int, float or bool or
                            lists of them separated with ','). The file must contain header.
    :return:
    """
    # Read is the properties from the csv file
    csv_file = open(csv_prop_file, 'r')
    csv_reader = csv.reader(csv_file, delimiter=',')
    header = next(csv_reader)
    propnames = header[1:]
    csv_reader = csv.reader(csv_file, delimiter=',')
    molname_propname_propval_mdict = tree()
    for line in csv_reader:
        molname = line[0].lower()   # lowercase for safety
        propvals = line[1:]
        for propname, propval in zip(propnames, propvals):
            molname_propname_propval_mdict[molname][propname] = propval
    # TODO: replace OpenBabel function with RDKit.
    largeSDfile = Outputfile(format="sdf", filename="tmp.sdf", overwrite=True)
    for mymol in readfile(format="sdf", filename=sdf):

        # Add all property values in new property fields in the sdf file
        for propname, propval in molname_propname_propval_mdict[mymol.title.lower()].items():
            if len(propval) > 1:
                mymol.data[propname] = str(propval)

        # Write this molecules with the extra property fields into the sdf file
        largeSDfile.write(mymol)

    largeSDfile.close()
    csv_file.close()

    os.rename("tmp.sdf", sdf)   # overwrite the old sdf

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

def extract_all_props_from_sdf(sdf, csv_prop_file, default_propvalue_dict={}):
    """
    Method to add new property fields into an existing sdf file from a csv file.

    :param sdf:
    :param csv_prop_file:   the output CSV files with the property values. The 1st column will be the
                            'molname' and the 2nd the property value (can be int, float or bool or
                            lists of them separated with ','). The file will contain header.
    :return:
    """
    # Find all existing properties in this SDF
    suppl = SDMolSupplier(sdf, removeHs=False, sanitize=False)
    propnames = []
    for mol in suppl:
        if len(mol.GetPropsAsDict().keys()) > len(propnames):
            propnames = list(mol.GetPropsAsDict().keys())

    # Now load the properties
    suppl = SDMolSupplier(sdf, removeHs=False, sanitize=False)
    prop_data = []
    for mol in suppl:

        if mol == None or mol.GetNumAtoms() == 0:
            continue  # skip empty molecules

        props_dict = mol.GetPropsAsDict()   # get all molecular properties
        propvalues = [mol.GetProp("_Name")]
        for propname in propnames:    # get all molecular properties
            try:
                propval = str(props_dict[propname])
            except KeyError:
                if propname in default_propvalue_dict.keys():
                    ColorPrint("Molecule %s does not have property %s. Using default value %s." %
                               (mol.GetProp("_Name"), propname, default_propvalue_dict[propname]), "FAIL")
                    propval = 'nan'
                else:
                    ColorPrint("Molecule %s does not have property %s, thus 'nan' value will be written." %
                               (mol.GetProp("_Name"), propname), "FAIL")
                    propval = 'nan'
            propvalues.append(propval)
        prop_data.append(propvalues)

    prop_df = pd.DataFrame(data=prop_data,
                           columns=["molname"]+propnames)   # Assuming that last mol has all the properties
    prop_df.to_csv(csv_prop_file, index=False)

    del suppl
    del prop_df

def set_molname_from_prop(sdf, propname, out_fname):
    """
    Method to change the molname of a molecule by using instead a property field.

    :param propname:    name of the property to use as molname.
    :param out_sdf:     the name of the output SDF file with the new molnames.
    :return:
    """
    suppl = SDMolSupplier(sdf, removeHs=False, sanitize=False)

    if out_fname.endswith(".sdf"):
        writer = SDWriter(out_fname)

        def writer_function(mol, molname, file):
            file.write(mol)
            file.flush()

    elif out_fname.endswith(".smi"):
        writer = open(out_fname, "w")

        def writer_function(mol, molname, file):
            file.write("%s\t%s\n" % (MolToSmiles(mol), molname))
            file.flush()

    for i, mol in enumerate(suppl):

        if mol == None or mol.GetNumAtoms() == 0:
            continue  # skip empty molecules

        try:
            propval = mol.GetProp(propname)
            mol.SetProp("_Name", propval)
        except KeyError:
            ColorPrint("Molecule %s does not have property %s, thus it will be renamed to %s." %
                       (mol.GetProp("_Name"), propname, "UNK%i" % i), "FAIL")
            mol.SetProp("_Name", "UNK%i" % i)
        writer_function(mol, propval, writer)   # dump the mol into the output file

    writer.close()
    del suppl

    del writer

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


def get_molnames_from_sdf(sdf, get_refmolnames=False, lowercase=False, ignore_refmolname=True,
                          get_unique_molnames=True):
    """
    Method to return a list with all unique molnames in the order they occur in the sdf file.

    :param sdf:
    :param get_refmolnames: the molnames used for reference in Homology Modeling, alignment or something similar.
    :param lowercase:
    :param ignore_refmolname:
    :return:
    """
    molname_list = []
    refmolname_list = []
    with open(sdf, 'r') as f:
        line = next(f)
        molname = line.rstrip()
        if not get_unique_molnames or molname not in molname_list:
            molname_list.append(molname)  # first line in sdf file is molname
        for line in f:
            if line[:14] == ">  <refligand>":
                line = next(f)
                refmolname_list.append(line.rstrip())
            elif line[:4] == "$$$$":
                try:
                    line = next(f)
                    molname = line.rstrip()
                    if not get_unique_molnames or molname not in molname_list:
                        molname_list.append(molname)
                except StopIteration:  # if this is the last line of the file
                    break

    if not ignore_refmolname and len(refmolname_list) != len(molname_list):
        raise Exception(ColorPrint("ERROR: not all molecules in file %s have the '>  <refligand>' field!"%sdf, "FAIL"))

    if lowercase:
        molname_list = [m.lower() for m in molname_list]
        refmolname_list = [m.lower() for m in refmolname_list]

    if get_refmolnames:
        return molname_list, refmolname_list
    else:
        return molname_list