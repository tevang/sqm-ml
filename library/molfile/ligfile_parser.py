"""
Methods for both sdf and mol2 file manipulation.
"""

from library.molfile.mol2_parser import *
from library.molfile.sdf_parser import *
# import ray
from library.utils.print_functions import ProgressBar

try:
    from library.modlib.pybel import Outputfile, readfile
except ImportError:
    print("WARNING: openbabel module could not be found!")

try:
    from rdkit import Chem
    from rdkit.Chem.PropertyMol import PropertyMol
except ImportError:
    print("WARNING: rdkit module could not be found!")


def get_molnames_in_sdf(sdf_file, lower_case=False):

    molname_list = []
    with open(sdf_file, 'r') as f:
        line = next(f)
        molname_list.append(line.rstrip())  # first line is molname
        try:
            for line in f:
                if line[:4] == '$$$$':
                    molname_list.append(next(f).rstrip())
        except StopIteration:
            pass

    if lower_case:
        molname_list = [m.lower() for m in molname_list]

    return molname_list

def load_multiconf_sdf(sdf_file, get_conformers=True, get_molnames=False, get_isomolnames=False, keep_structvar=True,
                       lowestE_structvars=[], get_SMILES=False, save_poseid_as_prop=True, get_serial_num=False,
                       addHs=True, removeHs=False, properties_to_store=set()):
    """
        FUNCTION that distinguishes between the different isomers of each compound in an .sdf file, and loads them along with their conformers
        into separate Chem.rdchem.Mol() objects (all the conformers are in the same mol). As of October 2016, RDKit did not support
        multi-conformer file reader, therefore I store each conformer loaded from the sdf file as a separate mol. Then I check if the
        molname and SMILES string already exist in the multidict and if yes, I add a new conformer to the existing molecule.

        EXPERIMENTAL MODIFICATION: because for some molecules their conformers give 2, 3 or more SMILES, I set all the SMILES for each
        molname to 'SMI', in order to calculate moments for all the conformers of each molname. I presume that each molname corresponds to
        a DIFFERENT tautomer/ionization state.

        keep_structvar:   keep the "_stereo[0-9]_ion[0-9]_tau[0-9]" suffix in the molname. Use with caution because
                            if False only the dominant ('stereo1_ion1_tau1') will be saved.
        lowestE_structvars: list with the lowest energy structural variant of each compound. Only those will be saved.
        TODO: if the lowest energy structvar doesn't exist in the sdf, then save the next lowest energy structvar.
        get_SMILES: if True then the real SMILES strings will be saved in molname_SMILES_conformersMol_mdict. If False then all the
                    molecules will have 'SMI' in the SMILES position.
        get_conformers: if True then all the conformers of each molecule in the sdf file are returned.
        save_poseid_as_prop: keep the _pose[0-9]+" prefix in the molname. You must also set properties_to_store=[] !
    """
    ColorPrint("Loading multi-conformer .sdf file %s" % sdf_file, "BOLDGREEN")

    if lowestE_structvars:  # Sanity check to see if some compounds are missing from lowestE_structvars
        lowestE_structvars = [sv.lower() for sv in lowestE_structvars]
        lowestE_basemolnames = set([get_basemolname(sv) for sv in lowestE_structvars])

    molname_SMILES_conformersMol_mdict = tree()    # molname->SMILES->Chem.rdchem.Mol() object containing all the conformers of this compound
    all_molnames = get_molnames_in_sdf(sdf_file, lower_case=True)   # get all molnames from the sdf file and make them lower case
    loaded_molnames = []    # the molnames that were successfully loaded
    if addHs == True and removeHs == False:
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)
    else:
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)
    molnames_list = []
    molname2iso_dict = OrderedDict()
    for mol in suppl:
        if mol == None or mol.GetNumAtoms() == 0:
            continue # skip empty molecules
        mol = PropertyMol(mol)
        assert mol.GetNumConformers() == 1, ColorPrint("ERROR: mol has more that 1 conformers!!!", "FAIL")
        # VERY IMPORTANT: ALWAYS USE THE FULLY PROTONATED FORMS OTHERWISE FINGERPRINTS ARE NOT CONSISTENT BETWEEN PROTONATED
        # AND UNPROTONATED FORMS!!!
        if addHs == True and removeHs == False:
            mol = check_for_all_hydrogens(mol, get_molH=True)   # add hydrogens if missing
        elif removeHs == True:
            mol = Chem.RemoveHs(mol)

        structposevar = mol.GetProp('_Name').lower()   # contains also the "_pose[0-9]+" suffix, if present

        # Remove the poseID from the molname and store it as a property (more elegant)
        # TODO: not working correctly if multiple poses of the same isomolname are present in the file. poseID
        # TODO: should contain a comma separated list of the available poseIDs.
        if "_pose" in structposevar and save_poseid_as_prop:
            isomolname, poseID = structposevar.split("_pose")
            properties_to_store.add("poseID")
            mol.SetProp("poseID", poseID)
        else:
            isomolname = structposevar

        molname = isomolname
        print("reading ", structposevar, "from file", sdf_file)
        try:
            molname2iso_dict[molname].append(isomolname) # save the original molname with the iso extensions (if present)
        except KeyError:
            molname2iso_dict[molname] = [isomolname]

        if lowestE_structvars:
            basemolname = get_basemolname(molname)
            if basemolname not in lowestE_basemolnames:
                ColorPrint("Compound %s is not in the list of lowest energy structural variants per compound."
                           " In this case only the last occurance of this molecules in the structure file will be saved."
                           % basemolname, "WARNING")
            elif molname not in lowestE_structvars: # if not the lowest energy structvar, then don't save it
                continue

        if keep_structvar == False:
            # If this basemolname has been saved and the current structvar is not the No. 1, skip it. Otherwise save it.
            molname = get_basemolname(
                molname)  # THIS IS CORRECT! If we don't want to keep all structvars, then just replace
                          # the full molname with the basemolname.
            if get_structvar_suffix(molname) != "stereo1_ion1_tau1" \
                    and molname in molname_SMILES_conformersMol_mdict.keys():
                loaded_molnames.append(structposevar)   # consider it loaded to prevent the warning at the end.
                continue

        mol.SetProp("_Name", molname)   # save the molecule name as a property
        mol.SetProp("SDF_file", sdf_file)   # save the source SDF file path as a property
        molnames_list.append(molname)
        props = [p for p in mol.GetPropNames(True, True)]
        if get_conformers:
            if get_SMILES:
                if 'SMILES' in props:   # distinguish the isomers and protonation states by the canonical SMILES string
                    SMILES = mol.GetProp('SMILES')  # syntax correct?????
                else:   # if not present in the inpust structure files, compute it
                    SMILES = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True, allBondsExplicit=True)
                    mol.SetProp('SMILES', SMILES)
            else:
                SMILES = 'SMI'

            if get_serial_num:  # get the serial number of this molecule if it exists
                # serial_num helps associate featvecs to specific molecular variants (e.g. tautomers).
                assert 'i_mmod_Serial_Number-OPLS-2005' in props, \
                    Debuginfo("ERROR: the sdf file %s does not contain the field 'i_mmod_Serial_Number-OPLS-2005'!"
                                % sdf_file, fail=True)
                # Instead of the SMILES string, save the serial number of this molecule
                SMILES = int(mol.GetProp('i_mmod_Serial_Number-OPLS-2005'))

            try:
                # If given, add the designated properties for the new conformer that you add
                for name in properties_to_store:
                    prop_name = "conf_"+name    # conformational properties have the prefix "conf_"
                    prev_val = molname_SMILES_conformersMol_mdict[molname][SMILES].GetProp("conf_"+name)
                    extra_val = mol.GetProp(name)
                    confID = molname_SMILES_conformersMol_mdict[molname][SMILES].GetNumConformers()
                    molname_SMILES_conformersMol_mdict[molname][SMILES].\
                        SetProp(prop_name, "%s|%i:%s" % (prev_val, confID, extra_val))
                # Now add the new conformer
                molname_SMILES_conformersMol_mdict[molname][SMILES].AddConformer(mol.GetConformer(), assignId=True)
            except (AttributeError, KeyError):
                for name in properties_to_store:
                    mol.SetProp("conf_"+name, "0:%s" % mol.GetProp(name))
                # TODO: add partial charges as a property of each conformer. Currently, only unique partial charges are saved.
                if 'partial charge' in mol.GetPropNames():
                    formal_charge = int(np.sum( [float(c) for c in mol.GetProp('partial charge').split(',')] ).round())
                    for atom, charge in zip(mol.GetAtoms(), mol.GetProp('partial charge').split(',')):
                        # By default the Atom object does not have a property for its partial charge, therefore add one
                        atom.SetDoubleProp('partial charge', float(charge))
                        atom.SetFormalCharge(formal_charge)
                molname_SMILES_conformersMol_mdict[molname][SMILES] = mol
            except RuntimeError:    # this occurs if keep_iso=False and the second isomers has different number of atoms than the first
                continue
        loaded_molnames.append(structposevar)  # contains also the "_pose[0-9]+" suffix, if present

    # Keep the tautomer with the lowest LigPrep state penalty
    for molname in list(molname2iso_dict.keys()):
        molname2iso_dict[molname].sort()
        molname2iso_dict[molname] = molname2iso_dict[molname][0]

    results = []
    if get_conformers:
        results.append(molname_SMILES_conformersMol_mdict)
    if get_molnames:
        results.append(molnames_list)
    if get_isomolnames:
        results.append(list(molname2iso_dict.values()))

    # print(the molnames that were not loaded successfully)
    all_molnames_set = set(all_molnames)
    loaded_molnames_set = set(loaded_molnames)
    failed_molnames_set = all_molnames_set.difference(loaded_molnames_set)
    # print("DEBUG: all_molnames_set=", all_molnames_set)
    # print("DEBUG: loaded_molnames_set=", loaded_molnames_set)
    # print("DEBUG: failed_molnames_set=", failed_molnames_set)
    failed_molnames_set = [f for f in failed_molnames_set if len(f)>0]  # discard empty names
    if len(failed_molnames_set) > 0:
        ColorPrint("FAIL: the following molecules failed to be loaded from the sdf file " + sdf_file + ":",
                   "OKRED")
        ColorPrint(" ".join(failed_molnames_set), "OKRED")

    if len(results) == 1:
        return results[0]
    else:
        return results


def mol_from_SMILES(SMILES, molname, LIGAND_STRUCTURE_FILE, sanitize, addHs, removeHs, get_SMILES, genNconf=0):
    """
    Convenient method to return the SMILES and a RDKit MOL object.
    :param SMILES:
    param molname:  actually it is the structvar
    :param sanitize:
    :param addHs:
    :param removeHs:
    :param genNconf:
    :return:
    """
    mol = Chem.MolFromSmiles(SMILES, sanitize=sanitize)
    if mol == None:
        ColorPrint("WARNING: skipping invalid SMILES string: %s" % SMILES, "WARNING")
        return None, None, None
    if addHs == True and removeHs == False:
        molH = check_for_all_hydrogens(mol, get_molH=True)
    elif removeHs == True:
        molH = Chem.RemoveHs(mol)
    else:
        molH = mol

    if genNconf > 0:
        Chem.AllChem.EmbedMultipleConfs(molH, numConfs=genNconf, params=Chem.AllChem.ETKDG())
    molH.SetProp("_Name", molname)
    molH.SetProp("SMILES_file", LIGAND_STRUCTURE_FILE)
    if get_SMILES == False:
        SMILES = 'SMI'
    # WARNING: PROPERTIES ARE NOT MAINTAINED DURING PICKLING OR FUNCTION RETURNING BY DEFAULT.
    # You need the PropertyMol function
    return molname, SMILES, PropertyMol(molH)

def load_structure_file(LIGAND_STRUCTURE_FILE,
                        keep_structvar=True,
                        get_SMILES=False,
                        save_poseid_as_prop=True,
                        addHs=True,
                        removeHs=False,
                        genNconf=0,
                        molnames2load=[],
                        sanitize=True,
                        properties_to_store=set()):
    """
    Generic method to load any type of compound file (.sdf, .mol2, .smi) and return conformers.
    The name of the original LIGAND_STRUCTURE_FILE along with the intermediate SDF file (in case of MOL2) will
    be saved in the same multidict along with the loaded structures.

    :param LIGAND_STRUCTURE_FILE:
    :param keep_structvar:  keep the "_stereo[0-9]_ion[0-9]_tau[0-9]" suffix in the molname. Use with caution because
                            if False only the dominant ('stereo1_ion1_tau1') will be saved.
    :param get_SMILES:
    :param addHs:
    :param genNconf:    number of conformers to generate. Applies only to SMILES input.
    :param molnames2load:   list with selected molnames to load from the LIGAND_STRUCTURE_FILE.
    :return:
    """
    ColorPrint("Loading ligand structure file " + LIGAND_STRUCTURE_FILE, "OKGREEN")
    assert os.path.exists(LIGAND_STRUCTURE_FILE), \
        Debuginfo("ERROR: %s IS NOT A VALID COMPOUND STRUCTURE FILE (.sdf or .smi)." % LIGAND_STRUCTURE_FILE,
                    fail=True)

    if LIGAND_STRUCTURE_FILE.endswith(".sdf"):  # if an 3D sdf file create moments
        molname_SMILES_conformersMol_mdict = load_multiconf_sdf(LIGAND_STRUCTURE_FILE,
                                                                keep_structvar=keep_structvar,
                                                                get_SMILES=get_SMILES,
                                                                save_poseid_as_prop=save_poseid_as_prop,
                                                                addHs=addHs,
                                                                removeHs=removeHs,
                                                                properties_to_store=properties_to_store)

    elif LIGAND_STRUCTURE_FILE.endswith(".mol2"): # if an 3D mol2 file convert it to sdf and create moments
        sdf_file = LIGAND_STRUCTURE_FILE.replace(".mol2", ".sdf") # DO NOT DELETE THIS FILE!
        mol2_to_sdf(mol2_file=LIGAND_STRUCTURE_FILE,
                    sdf_file=sdf_file,
                    property_name="Energy:")    # by default save the Energy - if present- to the sdf
        molname_SMILES_conformersMol_mdict = load_multiconf_sdf(sdf_file, keep_structvar=keep_structvar,
                                                                get_SMILES=get_SMILES,
                                                                save_poseid_as_prop=save_poseid_as_prop, addHs=addHs,
                                                                removeHs=removeHs)
        # shutil.rmtree(tmp_folder)   # delete the tmp folder with the sdf file
        if 'r_mmod_Potential_Energy-OPLS-2005' in properties_to_store:   # special case for 'schrodinger_confS' ligand entropy
            molname_ligFE_dict = get_ligand_freestate_energies_from_mol2(LIGAND_STRUCTURE_FILE, lowercase=True)
            molname_SMILES_conformersMol_mdict = \
                add_property_to_mols(property_dict=molname_ligFE_dict,
                                     property_name="conf_r_mmod_Potential_Energy-OPLS-2005",
                                     molname_SMILES_conformersMol_mdict=molname_SMILES_conformersMol_mdict)
    elif LIGAND_STRUCTURE_FILE[-4:] in [".smi", ".ism"]: # if a smiles file create conformers
        # Only for SMILES files (usually very large and slow to be loaded), display a progress bar
        total_lineNum = get_line_count(LIGAND_STRUCTURE_FILE)   # total number of lines in the file
        molname_SMILES_conformersMol_mdict = tree()
        UKN_molindex = 0    # to enumerate molecules without name
        ColorPrint("Reading all lines from file %s." % LIGAND_STRUCTURE_FILE, "OKBLUE")
        with open(LIGAND_STRUCTURE_FILE, 'r') as f:
            progbar = ProgressBar(100)
            for lnum, line in enumerate(f):
                words = line.split()
                SMILES = str(words[0])
                try:
                    structvar = str(words[1].lower())
                    basemolname = get_basemolname(structvar)
                    basemolname = sub_alt(basemolname,
                                      ["_noWAT"], "")
                except IndexError:  # assign molname to unnamed mol
                    UKN_molindex += 1
                    basemolname = "unk%i" % UKN_molindex    # in lowercase for compatibility
                    structvar = basemolname
                progbar.set_progress((lnum+1) / total_lineNum)

                # SERIAL EXECUTION
                try:
                    _, _, molH = mol_from_SMILES(SMILES,
                                                   structvar,
                                                   LIGAND_STRUCTURE_FILE,
                                                   sanitize,
                                                   addHs,
                                                   removeHs,
                                                   get_SMILES,
                                                   genNconf=0)
                except Chem.rdchem.MolSanitizeException:    # sanitization during H removing failed
                    ColorPrint("WARNING: sanitization of molecule %s failed during Hydrogen removal."
                               % structvar, "WARNING")
                    continue

                molname_SMILES_conformersMol_mdict[basemolname][SMILES] = molH
    else:
        assert None, ColorPrint("Unknown structure file format. Accepting only files ending in '.sdf', 'mol2', "
                   "'.smi' or '.ism'", "FAIL")

    if len(molnames2load) > 0:   # if specified, delete the molecules that are not in this list
        loaded_molnames = list(molname_SMILES_conformersMol_mdict.keys())
        molnames2load = [m.lower() for m in molnames2load]
        molnames2remove = [m for m in loaded_molnames if not get_basemolname(m) in molnames2load]
        for molname in molnames2remove:
            del molname_SMILES_conformersMol_mdict[molname]

    return molname_SMILES_conformersMol_mdict


def add_property_to_mols(property_dict, property_name, molname_SMILES_conformersMol_mdict):
    """
    Method to update the Mol objects in the molname_SMILES_conformersMol_mdict with a new property which has different
    value for every conformer, e.g. Free Energy. Currently works only if SMILES='SMI' for every molecule.
    :param property_dict:
    :param propetry_name:
    :param molname_SMILES_conformersMol_mdict:
    :return:
    """
    molname_list = list(molname_SMILES_conformersMol_mdict.keys())
    for molname in molname_list:
        mol = molname_SMILES_conformersMol_mdict[molname]['SMI']
        property_string = ""
        for prop in mol.GetProp('conf_poseID').split('|'):
            confID, poseID = prop.split(':')
            full_molname = "%s_pose%s" % (molname, poseID)
            prop_value = property_dict[full_molname]
            property_string += "|%s:%s" % (confID, prop_value)
        mol.SetProp(property_name, property_string[1:])     # without the first '|'
        molname_SMILES_conformersMol_mdict[molname]['SMI'] = mol    # replace with the new mol object

    return molname_SMILES_conformersMol_mdict

def check_for_all_hydrogens(mol, get_molH=False):
    """
    Returns True if all hydrogens are present in the molecule and False if not.
    :param mol:
    :param get_molH:    if True and not all hydrogens are present, returns the fully protonated mol
    :return:
    """
    molH = Chem.AddHs(mol)  # add hydrogens
    if len([a for a in molH.GetAtoms() if a.GetAtomicNum() == 1]) > len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 1]):
        if get_molH:
            return molH
        else:
            return False
    if get_molH:
        return mol
    else:
        return True
