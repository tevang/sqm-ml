import csv
import os

from library.global_fun import ligfile_open
from library.utils.print_functions import ColorPrint

try:
    from library.modlib.pybel import Outputfile, readfile
except ImportError:
    print("WARNING: openbabel module could not be found.")

try:
    from rdkit.Chem import rdFMCS, MolFromSmarts, MolFromMol2File, rdmolops, MolFromMol2Block, SDMolSupplier, SDWriter #, \
        # AtomKekulizeException, KekulizeException
    from rdkit.Chem.PropertyMol import PropertyMol
except ImportError:
    print("WARNING: rdkit module could not be found!")

try:
    import parmed as pmd
except ImportError:
    print("WARNING: parmed module could be found!")


def mol2_text_iterator(mol2_file, delimiter="@<TRIPOS>MOLECULE", property_name=None,
                       get_partial_charges=False, get_molname=True):
    """
    Generator which retrieves one mol2 block at a time from a multi-mol file and returns it as text along with a
    specified property. If no property name is given, then it will return just mol2 text blocks.
    """
    if property_name and get_partial_charges:
        def _return_func(mol2_list): return "".join(mol2), partial_charges, property_value
    elif property_name and not get_partial_charges:
        def _return_func(mol2_list): return "".join(mol2), property_value
    elif not property_name and get_partial_charges:
        def _return_func(mol2_list): return "".join(mol2), partial_charges
    elif not property_name and not get_partial_charges:
        def _return_func(mol2_list): return "".join(mol2)
    elif not property_name and get_molname:
        def _return_func(mol2_list): return "".join(mol2), mol2[1].rstrip()

    if isinstance(mol2_file, str):
        f = ligfile_open(mol2_file)
    else:
        f = mol2_file

    mol2, partial_charges = [], []
    property_value = None
    save_charges = False
    for line in f:
        if line.startswith(delimiter) and mol2:
            yield _return_func(mol2)
            mol2 = []
            property_value = None
        elif property_name and line.startswith(property_name):
            property_value = float(line.split()[1])
        if line.startswith("@<TRIPOS>ATOM"):
            partial_charges = []
            save_charges = True
        elif line.startswith("@<TRIPOS>BOND"):
            save_charges = False

        mol2.append(line)
        if save_charges and not line.startswith("@<TRIPOS>ATOM"):
            partial_charges.append(line.split()[-1])
    if mol2:    # DON'T FORGET THE LAST MOLECULE!
        yield _return_func(mol2)


def chunked_mol2_iterator(mol2_file, num_mols):

    miter = mol2_text_iterator(mol2_file, get_molname=False)
    tmp = []
    i = 0
    for mol2 in miter:
        if i==num_mols:
            yield "".join(tmp)
            tmp = []
            i = 0
        i += 1
        tmp.append(mol2)
    yield "".join(tmp) # DON'T FORGET THE LAST MOLECULE!


def rdkit_mol2_reader(mol2, property_name=None, sanitize=True, removeHs=False, cleanupSubstructures=True):
    """
    Multi-mol MOL2 file reader using RDKit. Works as an iterator.

    :param mol2:
    :param sanitize:
    :param removeHs:
    :param cleanupSubstructures:
    :return:
    """
    with open(mol2, 'r') as f:
        if property_name:
            for molblock, partial_charges, property_value in mol2_text_iterator(f, property_name=property_name, get_partial_charges=True):
                try:
                    Mol = MolFromMol2Block(molblock,
                                           sanitize=sanitize,
                                           removeHs=removeHs,
                                           cleanupSubstructures=cleanupSubstructures)
                    Mol.SetProp(property_name, str(property_value))
                    # Add the Partial Charges of the atoms separated by ',' in a new property field in the sdf file
                    if len(set(partial_charges)) > 1:
                        Mol.SetProp("partial charge", ",".join(partial_charges))
                    yield PropertyMol(Mol)
                except AttributeError:
                    ColorPrint("WARNING: a molecule failed to be sanitized and thus be loaded from MOL2 file to "
                               "RDKit Mol object.", "WARNING")
                    pass
        else:
            for molblock, partial_charges in mol2_text_iterator(f, get_partial_charges=True, get_molname=False):
                try:
                    Mol = MolFromMol2Block(molblock,
                                           sanitize=sanitize,
                                           removeHs=removeHs,
                                           cleanupSubstructures=cleanupSubstructures)
                    # Add the Partial Charges of the atoms separated by ',' in a new property field in the sdf file
                    if len(set(partial_charges)) > 1:
                        Mol.SetProp("partial charge", ",".join(partial_charges))
                    yield PropertyMol(Mol)
                except AttributeError:
                    ColorPrint("WARNING: a molecule failed to be sanitized and thus be loaded from MOL2 file to "
                               "RDKit Mol object.", "WARNING")
                    pass


def get_property_from_mol2(mol2, propname, value_index):
    """
    Method to extract a property from a MOL2 file.

    :param mol2:
    :param propname: the property name, the line must start with this string
    :param value_index: the index of the property value in the string
    :return:
    """
    molnamePropval_list = []
    with open(mol2, 'r') as f:
        for line in f:
            if line.startswith("@<TRIPOS>MOLECULE"):
                molname = next(f).strip()
                propvalue = None
                for l in range(6):  # check at maximum 6 lines from the molnames for the property name
                    line = next(f).strip()
                    if line.startswith(propname):
                        propvalue = float(line.split()[value_index])
                        break
                if propvalue == None:
                    ColorPrint("WARNING: molecule %s doesn't have property '%s' thus 'nan' will be saved!"
                               % (molname, propname), "WARNING")
                    propvalue = 'nan'
                molnamePropval_list.append( (molname, propvalue) )

    return molnamePropval_list


def extract_props_from_mol2(mol2, csv_prop_file, propname, value_index):
    """
    Method to add new property fields into an existing sdf file from a csv file.

    :param mol2:
    :param csv_prop_file:   the 1st column should be the 'molname' and the 2nd the property value (can be int, float or bool or
                            lists of them separated with ','). The file must contain header.
    :return:
    """
    # Write the header of the CSV file
    csv_file = open(csv_prop_file, 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(["molname", propname])

    molnamePropval_list = get_property_from_mol2(mol2, propname, value_index)
    for molname, propvalue in molnamePropval_list:
        csv_writer.writerow([molname, propvalue])    # write them to CSV
    csv_file.close()


def get_ligand_freestate_energies_from_mol2(mol2, lowercase=False):
    """
    If the molnames do not contain "_pose[0-9]+" suffix, then the lowest energy from each molname will be returned.
    :param mol2:
    :param lowercase:
    :return:
    """
    molname_list = []
    ligandE_list = []
    with open(mol2, 'r') as f:
        for line in f:
            if line.startswith("@<TRIPOS>MOLECULE"):
                molname = next(f).strip()
                molname_list.append(molname)
            elif line.startswith("Energy:"):
                ligandE = float(line.split()[1])
                ligandE_list.append(ligandE)
            if len(molname_list) > 2 and len(ligandE_list) == 0:
                # This mol2 file contains no energy information, therefore return empty dict
                return {}

    if lowercase:
        molname_list = [m.lower() for m in molname_list]

    molname_FE_dict = {}
    for molname, ligandE in zip(molname_list, ligandE_list):
        if molname in molname_FE_dict.keys() and molname_FE_dict[molname] < ligandE:
            continue
        molname_FE_dict[molname] = ligandE

    return molname_FE_dict

def mol2_to_sdf(mol2_file, sdf_file=None, property_name=None, kekulize=False):
    """
        Method to convert a multi-mol2 file to sdf format and maintaining all comments and properties & charges,
        e.g. "molecular energy".
    """
    if sdf_file == None:
        sdf_file = os.path.splitext(mol2_file)[0] + ".sdf"

    writer = SDWriter(sdf_file)
    writer.SetKekulize(kekulize)
    for Mol in rdkit_mol2_reader(mol2_file, sanitize=False, removeHs=False, cleanupSubstructures=False,
                                 property_name=property_name):
        # TODO: temporarily deactivated until I upgrade RDKit to > '2019.03.4'
        try:
            writer.write(Mol)
        # except (AtomKekulizeException, KekulizeException):
        except:
            ColorPrint("WARNING: molecule %s failed to be kekulized and thus won't be written to %s file." %
                       (Mol.GetProp("_Name"), sdf_file), "WARNING")