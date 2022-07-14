import csv
import uuid

try:
    from lib.modlib.pybel import Outputfile, readfile
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

from lib.ConsScoreTK_Statistics import *
from lib.global_fun import run_commandline


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


def mol2_list_iterator(mol2_file, delimiter="@<TRIPOS>MOLECULE"):
    """
    Generator which retrieves one mol2 block at a time from a multi-mol file and returns it as list.
    """
    with open(mol2_file, 'r') as f:
        mol2 = []
        for line in f:
            if line.startswith(delimiter) and mol2:
                yield mol2
                mol2 = []
            mol2.append(line)
        if mol2:
            yield mol2

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


def rename_mol2(mol2, lowercase=False, suffix="_pose"):
    with open(mol2, 'r') as f:
        for lineNo, line in enumerate(f):
            if lineNo == 1:
                if lowercase:
                    newfname = os.path.dirname(os.path.realpath(mol2)) + "/" + line.strip().lower() + ".mol2"
                else:
                    newfname = os.path.dirname(os.path.realpath(mol2)) + "/" + line.strip() + ".mol2"
                break
    # Check if such mol2 files already exist
    matches = list_files(os.path.dirname(newfname),
                         pattern=os.path.basename(newfname).replace(".mol2", ".*\.mol2"))
    if len(matches) >= 1:
        newfname2 = newfname.replace(".mol2", "%s%i.mol2" % (suffix, len(matches)+1))
        os.rename(mol2, newfname2)
        if os.path.exists(newfname):    # rename pose 1
            os.rename(newfname, newfname.replace(".mol2", "%s%i.mol2" % (suffix, 1)))
        return newfname2
    elif len(matches) == 0:
        os.rename(mol2, newfname)
        return newfname
    else:   # if this mol2 file doesn't exist
        return None

def rename_all_mol2(dirname=".", pattern=".*\.mol2", lowercase=False, suffix="_pose"):
    """
    Method to rename all mol2 files in a folder to the molname followed by a suffix in case of multiple copies.
    :param dirname:
    :param lowercase:
    :param suffix:
    :return:
    """
    all_mol2 = list_files(dirname, pattern=pattern)
    indices = multimatch_string_list(all_mol2, ".*_([0-9]+)\.mol2", pindex=1, unique=False)
    if len(indices) == len(all_mol2):   # if all the matched files have an index, then sort them by their index
                                        # to do the pose renaming properly
        mol2Index_list = [(mol2, index) for mol2, index in zip(all_mol2, indices)]
        mol2Index_list.sort(key=itemgetter(1))
        all_mol2 = [mol2 for mol2,index in mol2Index_list]

    for mol2 in all_mol2:
        rename_mol2(mol2, lowercase=lowercase, suffix=suffix)

def change_molname_in_mol2(mol2, new_molname):
    with open(mol2, 'r') as f:
        contents = f.readlines()

    for i in range(len(contents)):
        if contents[i][:17] == "@<TRIPOS>MOLECULE":
            contents[i + 1] = new_molname + "\n"
            break

    with open(mol2, 'w') as f:
        f.writelines(contents)

def calc_net_charge(query_mol2, get_alternative_charge=False):
    # use parmed Python API:
    m1 = pmd.load_file(query_mol2)
    partial_charges = [float(a1.charge) for a1 in m1.atoms]
    if set(partial_charges) == {0.0}:
        if get_alternative_charge:
            return None, None
        else:
            return None
    net_charge = sum(partial_charges)
    rounded_net_charge = int(round(net_charge, 0))
    diff = net_charge - rounded_net_charge
    # print("DEBUG: formal and alternative charge difference of %s is %f" % (query_mol2, diff))
    assert abs(diff) < 0.2, \
        Debuginfo("FAIL: large difference between actual net charge (%f) and rounded"
                    " net charge (%i)!" % (net_charge, rounded_net_charge), fail=True)
    if get_alternative_charge:
        # EXAMPLE: if diff=-2e-08, then it returns 0, -1
        try:
            return rounded_net_charge, int(rounded_net_charge + diff/abs(diff))
        except ZeroDivisionError:
            return rounded_net_charge, rounded_net_charge
    else:
        return rounded_net_charge

def transfer_charges_diff_atom_order(target_mol2, ref_mol2, out_mol2=None):
    """
    Method to transfer partial charges from one MOL2 file to another. Both files must contain the same molecule with the same
    atom types (6th column) but the atom order (1st column) and atom names (2nd column) do not need to be the same.

    :param target_mol2: mol2 file with the coordinates you wish to keep
    :param ref_mol2: mol2 file with the charges
    :param out_mol2: name of the output mol2 file
    :return:
    """

    # Find the corresponding atoms between the two mol objects through their MCS
    # NOTE: to avoid getting error like "warning - O.co2 with non C.2 or S.o2 neighbor." from RDKit's primitive
    # NOTE: MOL2 file reader, convert the two files to SDF format before searching for MCS. The atom order remains the same
    # NOTE: as in the MOL2 files.
    target_sdf = "tmp.%s.sdf" % str(uuid.uuid4())
    ref_sdf = "tmp.%s.sdf" % str(uuid.uuid4())
    mol2_to_sdf(target_mol2, sdf_file=target_sdf)
    mol2_to_sdf(ref_mol2, sdf_file=ref_sdf)
    qmol = next(SDMolSupplier(target_sdf, removeHs=False, sanitize=False))
    cmol = next(SDMolSupplier(ref_sdf, removeHs=False, sanitize=False))
    ref_charges = list(map(float, cmol.GetProp('partial charge').split(',')))
    os.remove(ref_sdf)
    os.remove(target_sdf)
    # qmol = MolFromMol2File(target_mol2, sanitize=False, removeHs=False)   # do not touch the hydrogens!
    # cmol = MolFromMol2File(ref_mol2, sanitize=False, removeHs=False)  # do not touch the hydrogens!
    try:
        mcs = rdFMCS.FindMCS([qmol, cmol], timeout=5, matchValences=False, ringMatchesRingOnly=True, completeRingsOnly=False)
    except ValueError:
        Debuginfo("FAIL: something is wrong with the format of (probably atom types) of either %s or %s, probably "
                    "due to GAFF to SYBYL format conversion with antechamber." % (target_mol2, ref_mol2), fail=True)
        sys.exit(0)

    qmatched_atomIdx_list = qmol.GetSubstructMatch(MolFromSmarts(mcs.smartsString))
    cmatched_atomIdx_list = cmol.GetSubstructMatch(MolFromSmarts(mcs.smartsString))
    assert len(qmatched_atomIdx_list) == len(cmatched_atomIdx_list), \
        Debuginfo("ERROR: files %s and %s do not contain the same number of atoms!" %
                    (target_mol2, ref_mol2), fail=True)
    atomMap = []
    qcoords2charge_dict = {}
    for q, c in zip(qmatched_atomIdx_list, cmatched_atomIdx_list):
        qatom = qmol.GetAtomWithIdx(q)
        catom = cmol.GetAtomWithIdx(c)
        # NOTE: the following two conditions are probably redundant since the 2 molecules are identical and all atoms are included
        # in the MCS.
        # assert qatom.GetAtomicNum() == catom.GetAtomicNum() and qatom.GetProp('_TriposAtomType') == catom.GetProp('_TriposAtomType'), \
        #     Debuginfo("ERROR COPYING CHARGES: query file %s and charge file %s do not contain molecules with identical atom types (%i,%s vs %i,%s)!" % \
        #           (target_mol2, ref_mol2, qatom.GetAtomicNum(), qatom.GetProp('_TriposAtomType'), catom.GetAtomicNum(),
        #            catom.GetProp('_TriposAtomType')), fail=True)
        assert qatom.GetAtomicNum() == catom.GetAtomicNum() and qatom.GetExplicitValence() == catom.GetExplicitValence(), \
            Debuginfo(
                "ERROR COPYING CHARGES: query file %s and charge file %s do not contain molecules with identical atom explicit valence (%i,%s vs %i,%s)!" % \
                (target_mol2, ref_mol2, qatom.GetAtomicNum(), qatom.GetExplicitValence(), catom.GetAtomicNum(),
                 catom.GetExplicitValence()), fail=True)
        atomMap.append([q, c])
        # save the patial charge of this query atom
        qcoords2charge_dict[tuple(qmol.GetConformer().GetAtomPosition(q))] = ref_charges[c]

    assert len(qcoords2charge_dict) == len(qmatched_atomIdx_list), \
        Debuginfo("ERROR: file %s contains at least two atoms with identical coordinates!" %
                    target_mol2, fail=True)

    # Finally, copy the charges to the target_mol
    with open(target_mol2, 'r') as f:
        contents = f.readlines()
    start = contents.index('@<TRIPOS>ATOM\n') + 1
    for i in range(start, start+len(qcoords2charge_dict)):
        coords = (float(contents[i].split()[2]), float(contents[i].split()[3]), float(contents[i].split()[4]))
        charge = qcoords2charge_dict[coords]
        contents[i] = re.sub(r'\s+[0-9-.]+\s*$', r'\t%2.6f\n' % charge, contents[i])

    # Save the query mol2 file with charges
    if not out_mol2:
        out_mol2 = os.path.splitext(target_mol2)[0] + ".charges.mol2"
    with open(out_mol2, 'w') as f:
        f.writelines(contents)

def transfer_charges_via_coordinates(ref_mol2, target_mol2, out_mol2=None):
    """
    ref_mol2 and target_mol2 must have exactly the same coordinates.

    :param ref_mol2:
    :param target_mol2:
    :param out_mol2:
    :return:
    """
    ColorPrint("Transfering charges from %s to %s and saving it in %s" % (ref_mol2, target_mol2, out_mol2), "OKBLUE")
    # use parmed Python API:
    m1 = pmd.load_file(ref_mol2)
    m2 = pmd.load_file(target_mol2)
    assert len(m1.atoms) == len(m2.atoms), ColorPrint("ERROR: files %s and %s do not have the same"
          " number of atoms! Please check the inputs. Most frequent cause is wrong protonation,"
          " which can be fixed with $SCHRODINGER/utilities/applyhtreat script." %
          (ref_mol2, target_mol2), "FAIL")
    a1_to_a2_dict = {}  # atom idx in mol1 -> matching atom idx in mol2
    for i in range(len(m1.atoms)):
        a1 = m1.atoms[i]
        for j in range(len(m2.atoms)):
            a2 = m2.atoms[j]
            if round(a1.xx,3) == round(a2.xx,3) and round(a1.xy,3) == round(a2.xy,3) and round(a1.xz,3) == round(a2.xz,3):
                a1_to_a2_dict[a1.idx] = a2.idx
                break
    assert len(m2.atoms) ==len(a1_to_a2_dict), \
        Debuginfo("FAIL: files %s and %s do not contain atoms with the same coordinates!", fail=True)

    # Transfer the charges
    for i,j in a1_to_a2_dict.items():
        m2.atoms[j].charge = m1.atoms[i].charge

    # Write the modified mol to a file
    if out_mol2 == None:    # overwrite the original target file
        out_mol2 = "tmp.%s.mol2" % str(uuid.uuid4())
        m2.save(out_mol2)
        os.rename(out_mol2, target_mol2)
    elif out_mol2 == ref_mol2:  # overwrite the charge mol2 file
        out_mol2 = "tmp.%s.mol2" % str(uuid.uuid4())
        m2.save(out_mol2)
        os.rename(out_mol2, ref_mol2)
    else:
        m2.save(out_mol2)

def transfer_atom_types_via_coordinates(ref_mol2, target_mol2, out_mol2=None):
    """
    NOT USED!
    ref_mol2 and target_mol2 must have exactly the same coordinates!

    :param ref_mol2:
    :param target_mol2:
    :param out_mol2:
    :return:
    """
    ColorPrint("Transfering atom types from %s to %s and saving it in %s" % (ref_mol2, target_mol2, out_mol2), "OKBLUE")
    # use parmed Python API:
    m1 = pmd.load_file(ref_mol2)
    m2 = pmd.load_file(target_mol2)
    assert len(m1.atoms) == len(m2.atoms), ColorPrint("ERROR: files %s and %s do not have the same"
          " number of atoms! Please check the inputs. Most frequent cause is wrong protonation,"
          " which can be fixed with $SCHRODINGER/utilities/applyhtreat script." %
          (ref_mol2, target_mol2), "FAIL")
    a1_to_a2_dict = {}  # atom idx in mol1 -> matching atom idx in mol2
    for i in range(len(m1.atoms)):
        a1 = m1.atoms[i]
        for j in range(len(m2.atoms)):
            a2 = m2.atoms[j]
            if round(a1.xx,3) == round(a2.xx,3) and round(a1.xy,3) == round(a2.xy,3) and round(a1.xz,3) == round(a2.xz,3):
                a1_to_a2_dict[a1.idx] = a2.idx
                break
    assert len(m2.atoms) ==len(a1_to_a2_dict), \
        Debuginfo("FAIL: files %s and %s do not contain atoms with the same coordinates!", fail=True)

    # Transfer the atom types
    for i,j in a1_to_a2_dict.items():
        m2.atoms[j].type = m1.atoms[i].type

    # Transfer also the bonds
    # TODO: if it is to transfer also the bonds, just copy the charges from target_mol2 to ref_mol2

    # Write the modified mol to a file
    if out_mol2 == None:    # overwrite the original target file
        out_mol2 = "tmp.%s.mol2" % str(uuid.uuid4())
        m2.save(out_mol2)
        os.rename(out_mol2, target_mol2)
    else:
        m2.save(out_mol2)

def transfer_atom_types_via_order(ref_mol2, target_mol2, out_mol2=None):
    """
    ref_mol2 and target_mol2 must have exactly the same atom order! The advantage of this functions is that it
    uses PARMED, which can read all file produced by AMBER Tools without problem.

    :param ref_mol2:
    :param target_mol2:
    :param out_mol2:
    :return:
    """
    ColorPrint("Transfering atom types from %s to %s and saving it in %s" % (ref_mol2, target_mol2, out_mol2), "OKBLUE")
    # use parmed Python API:
    m1 = pmd.load_file(ref_mol2)
    m2 = pmd.load_file(target_mol2)
    assert len(m1.atoms) == len(m2.atoms), ColorPrint("ERROR: files %s and %s do not have the same"
          " number of atoms! Please check the inputs. Most frequent cause is wrong protonation,"
          " which can be fixed with $SCHRODINGER/utilities/applyhtreat script." %
          (ref_mol2, target_mol2), "FAIL")
    for i in range(len(m1.atoms)):
        # do as many tests as possible to ensure that your are copying charges between identical atoms
        assert m1.atoms[i].name == m2.atoms[i].name and \
               m1.atoms[i].element == m2.atoms[i].element and \
               [b.name for b in m1.atoms[i].bond_partners] == [b.name for b in m2.atoms[i].bond_partners], \
            Debuginfo("ERROR: %s and %s do not have the same atom order!" % (ref_mol2, target_mol2), fail=True)
        m2.atoms[i].type = m1.atoms[i].type

    if out_mol2 == None:    # ovewrite the original target file
        out_mol2 = "tmp.%s.mol2" % str(uuid.uuid4())
        m2.save(out_mol2)
        os.rename(out_mol2, target_mol2)
    else:
        m2.save(out_mol2)

def transfer_charges_via_atom_order(ref_mol2, target_mol2, out_mol2=None, transfer_atom_types=False):
    """
    Same method but need the atom order between the two files to be the same. It's advantage is that is works
    :param target_mol2:
    :param ref_mol2:
    :param out_mol2:
    :return:
    """
    # use parmed Python API:
    m1 = pmd.load_file(ref_mol2)
    m2 = pmd.load_file(target_mol2)
    assert len(m1.atoms) == len(m2.atoms), ColorPrint("ERROR: pose file %s and charge file %s do not have the same"
          " number of atoms! Please check the inputs, especially the pose file. Most frequent cause is wrong protonation,"
          " which can be fixed with $SCHRODINGER/utilities/applyhtreat script." %
                                                      (target_mol2, target_mol2), "FAIL")
    for a1,a2 in zip(m1.atoms, m2.atoms):
        # do as many tests as possible to ensure that your are copying charges between identical atoms
        assert a1.name == a2.name and \
               a1.element == a2.element and \
               [b.name for b in a1.bond_partners] == [b.name for b in a2.bond_partners], \
            Debuginfo("ERROR: %s and %s do not have the same atom order!" % (ref_mol2, target_mol2), fail=True)
        a2.charge = a1.charge
        if transfer_atom_types:
            a2.type = a1.type

    # Write the modified mol to a file
    if out_mol2 == None:    # overwrite the original target file
        out_mol2 = "tmp.%s.mol2" % str(uuid.uuid4())
        m2.save(out_mol2)
        os.rename(out_mol2, target_mol2)
    elif out_mol2 == target_mol2:  # overwrite the charge mol2 file
        out_mol2 = "tmp.%s.mol2" % str(uuid.uuid4())
        m2.save(out_mol2)
        os.rename(out_mol2, target_mol2)
    else:
        m2.save(out_mol2)

def transfer_charges_gaff_atom_types(target_mol2, ref_mol2, out_mol2=None, ligand_ff="gaff2"):
    """
    Combines the two methods above to transfer charges independent of the atom order or type in the two mol2 files.
    Used only in run_MMGBSA.py as:
                transfer_charges_gaff_atom_types("ligand.mol2",
                                             "charge_file.mol2",
                                             "ligand.antechamber.mol2",
                                             ligand_ff=args.LIGAND_FF)
    we want the right SYBYL atom types which are in "ligand.mol2" in order to transfer charges. "charge_file.mol2"
    has GAFF atom types.

    :param target_mol2:
    :param ref_mol2:
    :param out_mol2:
    :return:
    """
    ## OBSOLETE
    # # reformat the query file to SYBYL to be readable by RDKit and delete charges (if present)
    # run_commandline("antechamber -i %s -fi mol2 -o %s -fo mol2 -c dc -dr n -at sybyl" %
    #                 (target_mol, target_mol.replace(".mol2", ".sybyl.mol2")) )
    # # reformat the file to SYBYL to be readable by RDKit
    # run_commandline("antechamber -i %s -fi mol2 -o %s -fo mol2 -dr n -at sybyl" %
    #                 (ref_mol, ref_mol.replace(".mol2", ".sybyl.mol2")) )
    # # Now transfer charges
    # transfer_charges_diff_atom_order(target_mol.replace(".mol2", ".sybyl.mol2"),
    #                                  ref_mol.replace(".mol2", ".sybyl.mol2"),
    #                                  target_mol.replace(".mol2", ".sybyl.charges.mol2"))

    # Now transfer charges
    transfer_charges_diff_atom_order(target_mol2, ref_mol2, target_mol2.replace(".mol2", ".charges.mol2"))
    # Reformat the query to GAFF2
    run_commandline("antechamber -i %s -fi mol2 -o %s -fo mol2 -c dc -dr n -at %s" %
                    (target_mol2, target_mol2.replace(".mol2", ".%s.mol2" % ligand_ff), ligand_ff))
    # Copy charges to GAFF2 mol2 file
    if not out_mol2:
        out_mol2 = os.path.splitext(target_mol2)[0] + ".%s.charges.mol2" % ligand_ff
    transfer_charges_via_atom_order(ref_mol2=target_mol2.replace(".mol2", ".charges.mol2"),
                                    target_mol2=target_mol2.replace(".mol2", ".%s.mol2" % ligand_ff),
                                    out_mol2=out_mol2)

def get_formal_charge(mol2, skip_fail=False, get_alternative_charge=False):
    """
    Method to return the formal charge of the molecule inside a mol2 file.

    :param mol2:
    :return:    formal charge, mol2 filename. If an error occurs, then it returns None, mol2 filename.
    """
    # AllChem.ComputeGasteigerCharges(mol)  # it doesn't work always!
    # Better calculate Gasteiger charges with Antechamber
    # run_commandline("antechamber -i %s -fi mol2 -o %s -fo mol2 -dr n -at sybyl -c gas" %
    #                 (mol2, mol2.replace(".mol2", ".charges.mol2")))
    try:
        # First try to see if there are partial charges within the mol2
        formal_charge, alt_charge = calc_net_charge(mol2, get_alternative_charge=True)
        if formal_charge == None:
            # If not, generate Gasteiger charges with antechamber
            run_commandline("antechamber -i %s -fi mol2 -o %s -fo mol2 -dr n -at sybyl -c gas" %
                            (mol2, mol2.replace(".mol2", ".charges.mol2")), skip_fail=skip_fail)
            formal_charge, alt_charge = calc_net_charge(mol2.replace(".mol2", ".charges.mol2"),
                                                        get_alternative_charge=True)
            os.remove(mol2.replace(".mol2", ".charges.mol2"))
        else:
            ColorPrint("File %s had existing partial charges. Formal charge is %i. Alternative charge is %i." %
                       (mol2, formal_charge, alt_charge), "OKBLUE")

        if get_alternative_charge:
            return formal_charge, alt_charge, mol2
        else:
            return formal_charge, mol2
    except Exception:
        traceback.print_exc()
        if get_alternative_charge:
            return None, None, mol2
        else:
            return None, mol2

def get_molname_from_mol2(mol2):
    """
        FUNCTION to return the molname of the first molecule in the mol2 file.
    """
    with open(mol2, 'r') as f:
        for line in f:
            if line[:17] == "@<TRIPOS>MOLECULE":
                molname = next(f).strip()
                return molname


def get_molnames_from_mol2(mol2, get_unique_molnames=False, lowercase=False, strip_pose=False):
    """
    Method to return a list with all (optionally unique) molnames in the order they occur in the
    multi-molecule mol2 file.

    :param mol2: multi-molecule mol2 file
    :param get_unique_molnames:
    :param lowercase:
    :param strip_pose:  remove "_pose[0-9]" suffix from molnames
    :return:
    """
    molname_list = []
    with open(mol2, 'r') as f:
        for line in f:
            if line.startswith("@<TRIPOS>MOLECULE"):
                molname = next(f).strip()
                if strip_pose:
                    molname = re.sub("_pose[0-9]+", "", molname)
                if not get_unique_molnames:
                    molname_list.append(molname)
                elif not molname in molname_list:
                    molname_list.append(molname)

    if lowercase:
        molname_list = [m.lower() for m in molname_list]

    return molname_list


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

def get_lowest_ligand_freestate_energies_from_mol2(mol2, lowercase=False):
    """
    Method to read in the mol2 file with the molecular energies and to return a dict with the minimum energy of each
    structural variant of each molecule.
    :param mol2:
    :param lowercase:
    :return stuctvar_minFE_dict:    dict: structural variant -> min molecular Free Energy
    """
    ColorPrint("Searching for the lowest energy structural variant of each molecule.", "OKBLUE")
    molname_FE_dict = get_ligand_freestate_energies_from_mol2(mol2, lowercase=lowercase)
    stuctvar_minFE_dict = {}
    for molname in molname_FE_dict.keys():
        FE = molname_FE_dict[molname]
        structvar = re.sub("_pose[0-9]+", "", molname)
        if structvar not in stuctvar_minFE_dict.keys() or FE < stuctvar_minFE_dict[structvar]:
            stuctvar_minFE_dict[structvar] = FE
    return stuctvar_minFE_dict

def mol2_to_sdf(mol2_file, sdf_file=None, property_name=None, kekulize=False):
    """
        Method to convert a multi-mol2 file to sdf format and maintaining all comments and properties & charges,
        e.g. "molecular energy".
    """

    # try:
    #     if "SCHRODINGER" in os.environ and os.path.exists(os.environ.get('SCHRODINGER') + "/utilities/structconvert"):
    #         run_commandline("%s/utilities/structconvert -imol2 %s -osdf %s" % (os.environ.get('SCHRODINGER'), mol2_file, mol2_file.replace(".mol2", ".sdf")))
    # # TODO: add exception handling.
    # except:
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

    # # OBSOLETE (delete it if you encounter no problems with Entropy computation.
    # largeSDfile = Outputfile("sdf", sdf_file, overwrite=True)
    # for mymol in readfile("mol2", mol2_file):
    #
    #     # Add the Molecular (Free) Energy in a new property field in the sdf file
    #     if 'Comment' in mymol.data.keys() and "Energy:" in mymol.data['Comment']:
    #         mymol.data["molecular energy"] = float(mymol.data['Comment'].split()[1])
    #         del mymol.data['Comment']   # if you keep this the energy will be writen under the molname in the sdf
    #     # Add the Partial Charges of the atoms separated by ',' in a new property field in the sdf file
    #     charges = [str(a.partialcharge) for a in mymol.atoms]
    #     if len(set(charges)) > 1:
    #         mymol.data["partial charge"] = ",".join(charges)
    #
    #     # Write this molecules with the extra property fields into the sdf file
    #     largeSDfile.write(mymol)
    #
    # largeSDfile.close()