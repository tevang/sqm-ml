"""
Methods for both sdf and mol2 file manipulations.
"""
import shutil
from distutils.spawn import find_executable

# import ray
from lib.utils.print_functions import ProgressBar
from ligand_tools.lib.conformer import Conformer
from lib.molfile.mol2_parser import *
from lib.molfile.sdf_parser import *
from lib.molfile.sdf_parser import get_molnames_from_sdf

try:
    from lib.modlib.pybel import Outputfile, readfile
except ImportError:
    print("WARNING: openbabel module could not be found!")

try:
    from rdkit import Chem
    from rdkit.Chem.PropertyMol import PropertyMol
except ImportError:
    print("WARNING: rdkit module could not be found!")


def mol2ref(ligfile_list, ref_pdb_file=None, onlytau1=False, maxposes=0, only_molnames=False, ignore_refmolname=False):
    """
    Method to read a list of .sdf or .mol2 files with aligned compounds and associate the compound names with
    the reference ligands that were used to align them using the ">  <refligand>" section of the .sdf file.

    :param ligfile_list:
    :param ref_pdb_file:
    :param onlytau1:
    :param maxposes:
    :param only_molnames:
    :param ignore_refmolname:
    :return:
    """
    molname2refmolname_dict = {}
    for ligfile in ligfile_list:
        ColorPrint("Reading file %s" % ligfile, "BOLDBLUE")
        full_molname_list = []
        full_refmolname_list = []    # the reference ligand for alignment in the ">  <refligand>" section, if present in the file
        # with open(ligfile, 'r') as f:
        if ligfile.endswith(".sdf"):
            if ref_pdb_file:
                full_molname_list, full_refmolname_list = \
                    get_molnames_from_sdf(ligfile,
                                          get_refmolnames=True,
                                          lowercase=True,
                                          ignore_refmolname=True)
            else:
                full_molname_list, full_refmolname_list = \
                    get_molnames_from_sdf(ligfile,
                                          get_refmolnames=True,
                                          lowercase=True,
                                          ignore_refmolname=ignore_refmolname)

        elif ligfile.endswith(".mol2"):
            full_molname_list = get_molnames_from_mol2(ligfile, lowercase=True)
            # TODO: make it read the reference ligand of the alignment from the .mol2
            full_refmolname_list = full_molname_list

        # extend molname2refmolname_dict
        molname_list = []
        for molname, refmolname in zip(full_molname_list, full_refmolname_list):
            if onlytau1:
                m = re.match(".*_tau([0-9]+)[^0-9]*", molname)
                if m and int(m.group(1)) > 1:
                    continue
            if maxposes > 0:  # keep up to the maximum number of docking poses
                m = re.match(".*_pose([0-9]+)[^0-9]*", molname)
                if m and int(m.group(1)) > maxposes:
                    continue
            molname_list.append(molname)
            if not only_molnames:
                molname2refmolname_dict[molname] = refmolname

    if molname2refmolname_dict == {} and ignore_refmolname == False:
        if ligfile.endswith(".mol2"):   # in case of MOL2 it doesn't apply
            molname2refmolname_dict = {m: m for m in molname_list}
        elif ligfile.endswith(".sdf"):
            ColorPrint("No '>  <refligand>' fields were were found in the sdf file(s). If you have not used HomoLigAlign "
                       "but docking to create the poses, write 'yes' + Enter to continue.", "BOLDGREEN")
            answer = input("")
            if answer.lower() == "yes":
                molname2refmolname_dict = {m: m for m in molname_list}
            else:
                raise Exception(ColorPrint("ERROR: invalid option.", "FAIL"))

    if only_molnames:
        return molname_list
    else:
        return molname2refmolname_dict


def slow_split_file(INPUT_FILE,
                    OUT_FILETYPE="mol2",
                    SUFFIX="_pose",
                    FILE_SUFFIX="",
                    tolower=False,
                    get_molnames=False,
                    molnum=1):
    """
    USE THIS METHOD IF THE INPUT_FILE IS SMALL AND CONTAINS REPLICATE MOLNAMES NAMES. OTHERWISE< USE fast_split_file().

    Method to split a multi-mol .sdf or .mol2 file into separate files, one for each molecule. Each output file is named
    according to the molecule name in the original multi-mol file. The output file format can also be controlled, i.e.
    the user may provide a multi-mol .sdf file and split it to multiple .mol2 files. The output files will be named
    according to the containing molname.

    COMMENTS ARE ALSO WRITTEN TO THE OUTPUT FILE.

    :param INPUT_FILE:  a sdf or mol2 file with one or more compounds. The file type will be guessed for the extension.
    :param OUT_FILETYPE:    OBSOLET CAUSE IT NEED BABEL! the output file type. Can be 'mol2', 'sdf', or 'pdb'. (default: 'mol2')
    :param SUFFIX:
    :param tolower: convert all filenames to lowercase
    :param get_molnames:
    :return:
    """
    ColorPrint("Initiating splitting of file %s to several %s files." % (INPUT_FILE, OUT_FILETYPE), "OKBLUE")

    # First get all the molnames from the file
    ColorPrint("Getting molnames from file %s." % INPUT_FILE, "OKBLUE")
    molnames_list = []
    ifbasename, iftype = os.path.splitext(INPUT_FILE)
    if INPUT_FILE.endswith('.mol2'):
        molnames_list = get_molnames_in_mol2(INPUT_FILE, lower_case=tolower)
    elif INPUT_FILE.endswith('.sdf'):
        molnames_list = get_molnames_in_sdf(INPUT_FILE, lower_case=tolower)

    assert molnum < len(molnames_list), ColorPrint("ERROR: -n must be lower than the total number of molecules in"
        " the file! You gave -n %i but the molecules are %i " % (molnum, len(molnames_list)), "FAIL")

    # Now save each compound separately, if the file contains >1 molecules
    ColorPrint("Saving the molecules in new files", "OKBLUE")
    if len(molnames_list) > 1 or INPUT_FILE.endswith(".mol2"):
        prev_filename = None    # used only when molnum>1
        molname_count = defaultdict(int)
        total_mol_num = len(molnames_list)
        progbar = ProgressBar(100)
        molindex = 1
        for mol_list in mol2_list_iterator(INPUT_FILE):
            molname = mol_list[1].rstrip()
            if tolower:
                molname = molname.lower()
            N = molnames_list.count(molname)    # number of copies of this molecule in the input file
            if N == 1 and molnum == 1:
                filename = molname + FILE_SUFFIX + "." + OUT_FILETYPE
            elif N > 1 and molnum == 1:
                molname_count[molname] += 1
                filename = molname + SUFFIX + str(molname_count[molname]) + FILE_SUFFIX + "." + OUT_FILETYPE
                mol_list[1] = molname + SUFFIX + str(molname_count[molname]) + "\n"
            elif 1 < molnum < len(molnames_list):
                start = (molindex//molnum)*molnum
                end = (1+molindex//molnum)*molnum
                if end > len(molnames_list):
                    end = len(molnames_list)
                fileindex = "_%i-%i" % (start, end)
                filename = ifbasename + fileindex + FILE_SUFFIX + "." + OUT_FILETYPE
                if filename != prev_filename:
                    if prev_filename != None:
                        output.close()
                    output = open(filename, 'w')
                writelist2file(mol_list, file=output, append=True)
                prev_filename = filename
                molindex += 1
                continue

            output = open(filename, "w")  # erase existing file, if any
            # output.write(mol.write(OUT_FILETYPE))
            writelist2file(mol_list, file=output, append=True)
            output.close()
            progbar.set_progress(molindex/total_mol_num)
            molindex += 1

        if 1 < molnum < len(molnames_list):
            output.close()

    else:   # TODO: make this work with SDF files too
        # ALWAYS USE LOWER CASE LETTERS WHEN DEALING WITH MOLNAMES!
        shutil.copy2(INPUT_FILE, molnames_list[0].lower() + ".mol2")

    if get_molnames:
        return [n.lower() for n in molnames_list]

def fast_split_file(INPUT_FILE,
                    FILE_SUFFIX="",
                    molnum=1,
                    use_real_molnames=False):
    """
    USE THIS METHOD IF THE INPUT_FILE IS VERY LARGE, molnum IS HIGH AND THE MOLNAMES ARE ALREADY PROOF-READ, REPLICATE ONES WERE
    RENAMED, etc. For molnum==1, use slow_split_file() instead.

    It simply splits the INPUT_FILE to sub-files containing molnum molecules.

    :param INPUT_FILE:
    :param molnum:
    :return:
    """

    ifbasename, iftype = os.path.splitext(INPUT_FILE)
    if INPUT_FILE.endswith(".mol2") or INPUT_FILE.endswith(".mol2.gz"):
        iftype = "mol2"
        reader_function = mol2_text_iterator
    elif INPUT_FILE.endswith(".sdf") or INPUT_FILE.endswith(".sdf.gz"):
        iftype = "sdf"
        reader_function = sdf_text_iterator

    molindex = 1
    prev_part = 1
    prev_molname = ""
    filename = ifbasename + "_part" + str(prev_part) + FILE_SUFFIX + "." + iftype
    output = open(filename, 'w')
    for mol_text, molname in reader_function(INPUT_FILE):
        part = 1+molindex//molnum -1    # -1 to start writing from *_part1.sdf
        if part != prev_part:
            filename = molname + "." + iftype if use_real_molnames else ifbasename + "_part" + str(part) + FILE_SUFFIX + "." + iftype
            output.close()
            if use_real_molnames and prev_part == 1:
                os.rename(ifbasename + "_part1" + FILE_SUFFIX + "." + iftype, prev_molname + "." + iftype)
            output = open(filename, 'w')
        output.write(mol_text)
        prev_part = part
        prev_molname = molname
        molindex += 1
    output.close()


def merge_files(OUT_FILE, INPUT_FILES):
    """
    Method to merge currently only MOL2 files and alter the _pose[0-9]+ suffices in order not to
    :param OUT_FILE:
    :param INPUT_FILES:
    :return:
    """
    ColorPrint("Merging files %s to file %s." % (INPUT_FILES, OUT_FILE), "OKBLUE")

    # First get all the molnames from all files
    molnames_list = []
    maxpose_per_structvar_dict = defaultdict(int)
    of = open(OUT_FILE, 'w')
    for fname in INPUT_FILES:
        ColorPrint("Getting molnames from file %s." % fname, "OKBLUE")
        for mol_list in mol2_list_iterator(fname):
            molname = mol_list[1].rstrip()
            structvar = strip_poseID(molname)
            poseID = get_poseID(molname)
            # TODO: there are more cases. E.g. max poseID to be 13 and the current 34...
            if structvar in maxpose_per_structvar_dict.keys() and poseID <= maxpose_per_structvar_dict[structvar]:
                poseID = maxpose_per_structvar_dict[structvar] + 1
            maxpose_per_structvar_dict[structvar] = poseID
            new_molname = structvar + "_pose%i\n" % poseID
            mol_list[1] = new_molname
            writelist2file(mol_list, file=OUT_FILE, append=True)
    # TODO: make this work with SDF files too

def sdf_to_mol2(sdf_file, outfname="", software='unicon', split=False, tolower=False, get_molnames=True):
    """
        FUNCTION to convert a multi-mol sdf file to mol2 format.
    """
    if get_molnames:
        contents = open(sdf_file, 'r').readlines()
        start = 0
        for i,l in enumerate(contents):
            if l.startswith("$$$$"):
                start = i+1
            else:
                break
        open("for_molnames_list.sdf", 'w').writelines(contents[start:])
        molnames_list = [mymol.title.lower() for mymol in readfile("sdf", "for_molnames_list.sdf")]
        os.remove("for_molnames_list.sdf")

    if not outfname:
        # WARNING: this will not work with Unicon if it contains '.' within the basemolname!. Inocon will split the outfname
        # at the '.' and use only the first substring to name the output file(s). Therefore with software='unicon'
        # it is recommended to give the full outfile name using outfname arguments but without inner '.'.
        outfname = os.path.splitext(sdf_file)[0] + ".mol2"

    if software == 'openbabel':
        largeMOL2file = Outputfile("mol2", outfname, overwrite=True)
        for mymol in readfile("sdf", sdf_file):
            largeMOL2file.write(mymol)
        largeMOL2file.close()
    elif software == 'unicon':
        # NOTE: Unicon does not accept input files with only one molecule!
        contents = open(sdf_file, 'r').readlines()
        contents.insert(0, "$$$$\n")
        open(sdf_file, 'w').writelines(contents)
        unicon_exe = find_executable('Unicon')
        if unicon_exe:
            # ATTENTION: NEVER SPLIT WITH Unicon, BECAUSE EACH FILE WILL CONTAIN THE LAST MOLECULE OF THE PREVIOUS FILE
            # (total 2 molecules if -s1 is used)!
            run_commandline("%s -i %s --inFormat sdf -o %s --outFormat mol2" % \
                            (unicon_exe, sdf_file, outfname) )
            if split:
                slow_split_file(INPUT_FILE=outfname, OUT_FILETYPE="mol2", tolower=tolower)
                # NO NEED TO RENAME THE OUTPUT MOL2 FILES!
                os.remove(outfname) # remove the file that was split.
        else:
            raise IOError(ColorPrint("FAIL: Unicon is not in the PATH!", "FAIL"))

    if get_molnames:
        return molnames_list

def get_molnum_from_file(fname):
    """
    Works for both SDF and MOL2 files.

    :param fname:
    :return:
    """
    molnum = 0
    if fname.endswith(".sdf"):
        with open(fname, 'r') as f:
            for line in f:
                if line[:4] == '$$$$':
                    molnum += 1
    elif fname.endswith(".mol2"):
        with open(fname, 'r') as f:
            for line in f:
                if line.startswith("@<TRIPOS>MOLECULE"):
                    molnum += 1
    return molnum

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


def get_molnames_in_mol2(mol2_file, lower_case=False):
    molnames_list = []
    for mol_list in mol2_list_iterator(mol2_file):
        molname = mol_list[1].rstrip()
        if lower_case:
            molname = molname.lower()
        molnames_list.append(molname)
    return molnames_list


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


def write_mol_to_sdf(mol, outfname):
    """
    Helper method to write one RDKit Mol object (may be multi-conformer) to an sdf file.
    :param mol:
    :param outfname:
    :return:
    """
    writer = Chem.rdmolfiles.SDWriter(outfname)
    for i in range(mol.GetNumConformers()):
        writer.write(mol, confId=i)
    writer.close()


def write_mols2files(sdf_file, get_molnames=False):
    """
        FUNCTION to read a multi-mol, multi-conformer .sdf file and save each molecule along with its conformers
        to a separate .sdf file.
    """

    molname_SMILES_conformersMol_mdict = load_multiconf_sdf(sdf_file, get_conformers=True, get_molnames=False,
                                                            get_isomolnames=False, keep_structvar=True,
                                                            get_SMILES=False)
    for molname in list(molname_SMILES_conformersMol_mdict.keys()):
        mol = molname_SMILES_conformersMol_mdict[molname]['SMI']
        write_mol_to_sdf(mol, molname + ".sdf")

    if get_molnames:
        return list(molname_SMILES_conformersMol_mdict.keys())


def load_structures_from_SMILES(SMILES_input, N=0, keep_SMILES=True):
    """
    KIND OF OBSOLETE.
    Method to load SMILES and generate conformers using RDKits code (not recommended).
    :param SMILES_input:    if a string then the SMILES will be loaded from the respective file. If a dict, then it is assumed that
                            SMILES_input = molname_SMILES_dict .
    :param N:   number of conformers to generate. If 0 then no conformer will be generated.
    :param keep_SMILES: if False, then the SMILES string is replaced by 'SMI'
    :return:
    """
    from scoop import shared, futures

    if N > 0:
        conf = Conformer()  # this is the ConsScorTK Conformer class
    if type(SMILES_input) == str:
        molname_SMILES_dict = {}
        UKN_molindex = 0  # to enumerate molecules without name
        with open(SMILES_input, 'r') as f:
            for line in f:
                try:
                    SMILES, molname = [str(w) for w in line.split()[:2]]
                    if molname in molname_SMILES_dict.keys():   # load only the 1st occurrence of each molname
                        continue
                except IndexError:  # assign molname to unnamed mol
                    UKN_molindex += 1
                    molname = "unk%i" % UKN_molindex    # in lowercase for compatibility
                # print("DEBUG: loaded SMILES=%s molname=%s" % (SMILES, molname))
                molname_SMILES_dict[molname] = SMILES
    elif type(SMILES_input) == dict:
        molname_SMILES_dict = SMILES_input
    molname_SMILES_Mol_dict = tree()
    molname_SMILES_conformersMol_mdict = tree()
    for molname,SMILES in list(molname_SMILES_dict.items()):
        Mol = Chem.MolFromSmiles(SMILES)
        Mol.SetProp("_Name", molname)
        Mol.SetProp("SMILES_file", SMILES_input)    # save the source SMILES file path
        if N >= 3:  # <== CHANGE ME
            # Hydrogens are added inside conf.gen_singlemol_conf() method
            Mol = conf.gen_singlemol_conf(Mol, N)
        elif N == 0:
            Mol = Chem.AddHs(Mol)
        if not keep_SMILES:
            SMILES = 'SMI'
        molname_SMILES_conformersMol_mdict[molname][SMILES] = Mol   # if 1 <= N <3, Mol will contain not conformer

    if 1 <= N <3:   # PARALLEL CONFORMER GENERATION
        try:
            shared.setConst(MOLNAME_SMILES_CONFORMERSMOL_mdict=molname_SMILES_conformersMol_mdict)
        except TypeError:   # if the shared variable already exists, just pass
            pass
        molname_args = list(molname_SMILES_conformersMol_mdict.keys())
        SMILES_args = [list(molname_SMILES_conformersMol_mdict[molname].keys())[0] for molname in molname_args]
        # The gen_singlemol_conf_parallel will update molname_SMILES_conformersMol_mdict
        results = list(futures.map(conf.gen_singlemol_conf_parallel, molname_args, SMILES_args, [1]*len(molname_args)))

    return molname_SMILES_conformersMol_mdict


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

# @ray.remote(num_return_vals=3)
# def _mol_from_SMILES(SMILES, molname, LIGAND_STRUCTURE_FILE, sanitize, addHs, removeHs, get_SMILES, genNconf=0):
#     """
#     Same as mol_from_SMILES, but for RAY parallelized execution
#     :param SMILES:
#     param molname:  actually it is the structvar
#     :param sanitize:
#     :param addHs:
#     :param removeHs:
#     :param genNconf:
#     :return:
#     """
#     # print("DEBUG: creating molecule from %s" %SMILES)
#     mol = Chem.MolFromSmiles(SMILES, sanitize=sanitize)
#     if mol == None:
#         ColorPrint("WARNING: skipping invalid SMILES string: %s" % SMILES, "WARNING")
#         return None, None, None
#     if addHs == True and removeHs == False:
#         molH = check_for_all_hydrogens(mol, get_molH=True)
#     elif removeHs == True:
#         molH = Chem.RemoveHs(mol)
#     else:
#         molH = mol
#
#     if genNconf > 0:
#         Chem.AllChem.EmbedMultipleConfs(molH, numConfs=genNconf, params=Chem.AllChem.ETKDG())
#     molH.SetProp("_Name", molname)
#     molH.SetProp("SMILES_file", LIGAND_STRUCTURE_FILE)
#     if get_SMILES == False:
#         SMILES = 'SMI'
#     # WARNING: PROPERTIES ARE NOT MAINTAINED DURING PICKLING OR FUNCTION RETURNING BY DEFAULT.
#     # You need the PropertyMol function
#     return molname, SMILES, PropertyMol(molH)

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

        # tmp_folder = "tmp.%s/" % str(uuid.uuid4())
        # os.mkdir(tmp_folder)
        # sdf_file = tmp_folder + os.path.basename(LIGAND_STRUCTURE_FILE).replace(".mol2", ".sdf")
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

        ## NOTE: PARALLEL EXECUTION WITH RAY DOES NOT WORK FOR >5,000,000 SMILES IN THE FILE. THE PROGRAM
        ## NOTE: SLOWS DOWN FOR UNEXPLAINED REASON. BETTER RUN THE SERIAL SCRIPT WITH GNU PARALLEL ON MULTIPLE THREADS.
        # # PARALLEL EXECUTION WITH RAY: GLOBAL VARIABLE DEFINITIONS
        # ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, memory=1000000000)
        # _LIGAND_STRUCTURE_FILE = ray.put(LIGAND_STRUCTURE_FILE)  # FOR RAY: get the RAY object ID to use it as function argument
        # _sanitize = ray.put(sanitize)
        # _addHs = ray.put(addHs)
        # _removeHs = ray.put(removeHs)
        # _get_SMILES = ray.put(get_SMILES)
        # _genNconf = ray.put(genNconf)
        # ray_function_calls = []

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

                # # PARALLEL EXECUTION WITH RAY
                # ray_function_calls.append(_mol_from_SMILES.remote(SMILES=SMILES,
                #                                               molname=structvar,
                #                                               LIGAND_STRUCTURE_FILE=_LIGAND_STRUCTURE_FILE,
                #                                               sanitize=_sanitize,
                #                                               addHs=_addHs,
                #                                               removeHs=_removeHs,
                #                                               get_SMILES=_get_SMILES,
                #                                               genNconf=_genNconf))

        # # PARALLEL EXECUTION WITH RAY
        # ColorPrint("Creating RDKit MOL objects from SMILES in parallel.", "OKBLUE")
        # results = ray.get(ray_function_calls)
        # ColorPrint("Storing RDKit MOL objects into a multidict.", "OKBLUE")
        # for structvar, SMILES, molH in results:
        #     basemolname = get_basemolname(structvar) # recover the basemolname
        #     molname_SMILES_conformersMol_mdict[basemolname][SMILES] = molH

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

def get_SMARTS_from_structure_file(LIGAND_STRUCTURE_FILE,
                                keep_structvar=True,
                                get_SMILES=False,
                                addHs=True,
                                genNconf=0,
                                molnames2load=[],
                                get_properties=set()):
    """
    Same as load_structure_file, but instead of RDKit Mol objects, it returns SMARTS strings.

    :param LIGAND_STRUCTURE_FILE:
    :param keep_structvar:
    :param get_SMILES:
    :param addHs:
    :param genNconf:
    :param molnames2load:
    :param get_properties:
    :return:
    """

    molname_SMILES_conformersMol_mdict = load_structure_file(LIGAND_STRUCTURE_FILE, keep_structvar, get_SMILES, addHs,
                                                             genNconf, molnames2load, get_properties)
    molname_SMILES_SMARTS_mdict = tree()
    for molname in list(molname_SMILES_conformersMol_mdict.keys()):
        for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
            mol = molname_SMILES_conformersMol_mdict[molname][SMILES]
            molname_SMILES_SMARTS_mdict[molname][SMILES] = Chem.MolToSmarts(mol, isomericSmiles=True)

    return molname_SMILES_SMARTS_mdict

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

def extract_molecules(QUERY_MOLFILE, OUTFILE, MOLNAME_LIST=None, NAME_PATTERN=None, lEstructvar_FE_dict={}):
    """
    Method to extract all molecules with names that match the given pattern, or contained in the provided file,
    and writes them into a new file. It works both with .sdf and .mol2 files.
    It also works for cases where the MOLNAME_LIST contains the base molnames, e.g LAM00005677 instead of
    LAM00005677_stereo1_ion1_tau2_pose3.

    :param QUERY_MOLFILE:
    :param OUTFILE:
    :param MOLNAME_LIST:    list with basemolnames or structvars to be extracted. In case of basemolnames all the matching
                            structvars within the file will be extracted
    :param NAME_PATTERN:
    :param lEstructvar_FE_dict:   molecular Free Energy of each structural variant (OPTIONAL)
    :return:
    """

    if NAME_PATTERN and NAME_PATTERN.lower() != NAME_PATTERN:
        ColorPrint("WARNING: all molnames are converted to lowercase. However, your -namepattern contains "
                   "capital letters, which will be converted to lowercase to match with the molnames.",
                   "WARNING")
        NAME_PATTERN = NAME_PATTERN.lower()

    # Converl all molenames to lower case
    molname_list = []
    if MOLNAME_LIST and type(MOLNAME_LIST) == str:
        molname_list = [l.split()[0].lower() for l in open(MOLNAME_LIST)]
    elif MOLNAME_LIST and type(MOLNAME_LIST) == list:
        molname_list = [l.lower() for l in MOLNAME_LIST]

    if QUERY_MOLFILE.endswith(".sdf"):  # if this is an .sdf file
        fout = open(OUTFILE, 'w')
        write = False
        print("SDF file detected.")
        with open(QUERY_MOLFILE, 'r') as f:
            contents = f.readlines()
            molname = contents[0].strip().lower()
            basemolname = get_basemolname(molname)
            if (NAME_PATTERN and re.match(NAME_PATTERN, molname)) \
                    or molname in molname_list \
                    or basemolname in molname_list:
                write = True
            for i in range(0, len(contents) - 1):
                if write:
                    fout.write(contents[i])
                if re.match("\$\$\$\$", contents[i]):
                    molname = contents[i + 1].strip().lower()
                    basemolname = sub_alt(molname, ["_stereo[0-9]+_ion[0-9]+_tau[0-9]+", "_pose[0-9]+", "_frm[0-9]+"], "")
                    if (NAME_PATTERN and re.match(NAME_PATTERN, molname)) \
                            or molname in molname_list \
                            or basemolname in molname_list:
                        write = True
                    else:
                        write = False
        fout.close()

    if QUERY_MOLFILE.endswith(".mol2"):  # if this is a .mol2 file
        print("MOL2 file detected.")
        if os.path.exists(OUTFILE):
            os.remove(OUTFILE)
        write = False
        mol_lines = []  # the lines for a molecular entry in them mol2 file
        with open(QUERY_MOLFILE, 'r') as f:

            while True:
                try:
                    line = f.__next__()
                    if line.startswith("@<TRIPOS>MOLECULE"):
                        writelist2file(mol_lines, file=OUTFILE, append=True)  # write the previous molecule
                        mol_lines = []  # reset mol_lines
                        next_line = f.__next__()
                        molname = next_line.strip().lower()
                        basemolname = get_basemolname(molname)
                        structvar = get_structvar(molname)
                        if NAME_PATTERN and re.match(NAME_PATTERN, molname):
                            mol_lines.append(line)  # initialize mol_lines, clean any previous contents
                            mol_lines.append(next_line)
                            write = True
                            continue    # start saving from the next line, we already saved the header and the molname.
                        elif NAME_PATTERN and not re.match(NAME_PATTERN, molname):
                            write = False
                        elif molname in molname_list \
                            or basemolname in molname_list \
                            or structvar in molname_list \
                            or structvar in lEstructvar_FE_dict.keys():
                            mol_lines.append(line)  # initialize mol_lines, clean any previous contents
                            mol_lines.append(next_line)
                            write = True
                            continue  # start saving from the next line, we already saved the header and the molname.
                        else:
                            write = False
                    # TODO: currently only if NAME_PATTERN is given, the Energy will be ignored. Complete the other cases, too.
                    elif not NAME_PATTERN and not MOLNAME_LIST and line.startswith("Energy:"):
                        ligandE = float(line.split()[1])
                        structvar = get_structvar(molname)
                        if structvar in lEstructvar_FE_dict.keys() \
                                and lEstructvar_FE_dict[structvar] != ligandE:   # if the energies were given
                            mol_lines = []  # clean the list, this is not the right structure
                            write = False

                    if write:
                        mol_lines.append(line)
                except StopIteration:
                    break
        # write the last molecule
        writelist2file(mol_lines, file=OUTFILE, append=True)
