from lib.molfile.mol2_parser import *
from collections import defaultdict
from lib.global_fun import replace_alt
from lib.molfile.sdf_parser import get_molnames_from_sdf
from lib.utils.print_functions import ColorPrint


class ReplicateMols():

    def __init__(self):
        pass

    @staticmethod
    def has_replicates(fname):
        """
        Method to check if a file has replicate molnames.
        :param fname:
        :return:
        """
        molnames_list = []
        iftype = fname.split('.')[-1]
        if iftype == "mol2":
            molnames_list = get_molnames_from_mol2(fname, get_unique_molnames=False)
        elif iftype == "sdf":
            molnames_list = get_molnames_from_sdf(fname, get_unique_molnames=False)
        # TODO: to avoid invoking the problem-causing OpenBabel, write functions for the rest file types (e.g. SMILES).
        # else:   # OpenBabel must be the last resort!
        #     for mol in pybel.readfile(iftype, fname):
        #         molname = mol.OBMol.GetTitle()
        #         molnames_list.append(molname)
        if len(molnames_list) < 10000:  # IF THE FILE IS SMALL
            for molname in molnames_list:
                if molnames_list.count(molname) > 1:
                    ColorPrint("Found replicate molname! %s exists %i times." %
                               (molname, molnames_list.count(molname)), "OKRED")
                    return True
            return False
        else:   # IF THE FILE IS BIG, USE THIS FOR FASTER SPEED BUT YOU WILL NOT GET WHICH MOLECULE WAS FOUND
            # WITH REPLICATES
            return len(set(molnames_list)) < len(molnames_list)

    @staticmethod
    def rename_isomers(fname):
        return ReplicateMols.rename_replicates(fname, suffix="_iso")

    @staticmethod
    def rename_stereo(fname):
        return ReplicateMols.rename_replicates(fname, suffix="_stereo")

    @staticmethod
    def rename_poses(fname, get_molnum=False):
        return ReplicateMols.rename_replicates(fname, suffix="_pose", get_molnum=get_molnum)

    @staticmethod
    def rename_ligprep(fname):
        """
        Currently only SDF format is supported. It presumes that you had run LigPrep with -lab flag, like this:
        $SCHRODINGER/ligprep -ismi $ISMI -osd $OSD -epik -ph 7.5 -pht 2.0 -lab -NJOBS 8 -HOST localhost

        :param fname:
        :return:
        """
        if fname.endswith(".sdf"):  # if this is an .sdf file
            print("SDF file detected.")
            with open(fname, 'r') as f:
                contents = f.readlines()

            lignames = []
            newlignames = []
            ligname = contents[0].strip()   # name of the 1st mol in the file
            ligname = replace_alt(ligname, ['@', ' '], '_') # replace @ and gaps in the name
            lignames.append(ligname)
            for i in range(0, len(contents) - 1):   # save the names of the rest mols
                if re.match("\$\$\$\$", contents[i]):
                    ligname = contents[i + 1].strip()
                    ligname = replace_alt(ligname, ['@', ' '], '_')
                    lignames.append(ligname)
                elif not is_structvar(lignames[-1]):
                    m = re.search("^.*_neutralizer_([0-9]+)_epik_([0-9]+)_stereoizer_([0-9]+)", contents[i])
                    if m:
                        ion, tau, stereo = m.groups()
                        lignames[-1] += "_stereo%s_ion%s_tau%s" % (stereo, ion, tau)   # add suffix to the molname

            # If the file has replicate molnames but not info about stereo, ion or tau, then consider each replicate to
            # be an alternative tautomer.
            ligname_count_dict = defaultdict(int)   # starts counting from 0
            for i in range(len(lignames)):
                if '_stereo' not in lignames[i]:
                    ligname_count_dict[lignames[i]] += 1
                    lignames[i] += "_stereo%s_ion%s_tau%s" % (1, 1, ligname_count_dict[lignames[i]])  # add suffix to the molname

            listcycle = cycle(lignames)
            contents[0] = next(listcycle) + "\n"  # get the next ligname
            for i in range(0, len(contents) - 1):
                if re.match("\$\$\$\$", contents[i]):
                    contents[i + 1] = next(listcycle) + "\n"  # get the next ligname

            fout = open(fname.replace(".sdf", ".renstereo_ion_tau.sdf"), 'w')
            fout.writelines(contents)
            fout.close()

    @staticmethod
    def rename_replicates(fname, suffix='_iso', rename_unique=True, get_molnum=False):
        """

        :param fname:
        :param suffix:
        :param rename_unique: add the suffix also to molecules that have only one variant in the fname.
        :return:
        """
        if fname.endswith(".sdf"):  # if this is an .sdf file
            print("SDF file detected.")

            print("Getting all molnames from file %s ." % fname)
            f = open(fname, 'r')
            ligname_count_dict = defaultdict(int)
            lignames = []
            newlignames = []
            next_line = next(f).strip() # 1st line is a molname
            ligname = replace_alt(next_line, ['@', ' '], '_')
            lignames.append(ligname)
            while True:
                try:
                    line = next(f)
                    if line.startswith("$$$$"):
                        next_line = next(f).strip()
                        ligname = replace_alt(next_line, ['@', ' '], '_')
                        lignames.append(ligname)
                        ligname_count_dict[ligname] += 1
                except StopIteration:
                    break

            print("Renaming the molnames.")
            addedligands = []
            addedligname_count_dict = defaultdict(int)
            for i in range(0, len(lignames)):
                ligname = lignames[i]
                isomerNo = ligname_count_dict[ligname]
                addedligands.append(ligname)
                addedligname_count_dict[ligname] += 1
                if isomerNo == 1 and not rename_unique:
                    newlignames.append(ligname)
                elif isomerNo >= 1:
                    isomer = addedligname_count_dict[ligname]
                    newlignames.append(ligname + suffix + str(isomer))

            out_fname = fname.replace(".sdf", ".ren%s.sdf" % suffix.replace("_",""))
            print("Writing file %s with new unique molnames." % out_fname)
            listcycle = cycle(newlignames)
            fout = open(out_fname, 'w')
            f.seek(0)   # for to the beginning of the input file.
            line = next(f)  # skip the old molname to avoid being written
            fout.write(next(listcycle) + "\n")  # 1st line is a molname and is not preceded by '$$$$'
            fout.flush()
            while True:
                try:
                    line = next(f)
                    fout.write(line)
                    if line.startswith("$$$$"):
                        next(f)  # skip the next line with the old molname
                        fout.write(next(listcycle) + "\n")
                        fout.flush()
                except StopIteration:
                    break
            fout.close()
            f.close()

        if fname.endswith(".mol2"):  # if this is a .mol2 file
            print("MOL2 file detected.")
            f = open(fname, 'r')

            lignames = []
            newlignames = []

            print("Getting all molnames from file %s ." % fname)
            ligname_count_dict = defaultdict(int)
            while True:
                try:
                    line = next(f)
                except StopIteration:
                    break
                if line.startswith("@<TRIPOS>MOLECULE"):
                    next_line = next(f).strip()
                    ligname = replace_alt(next_line, ['@', ' '], '_')
                    lignames.append(ligname)
                    ligname_count_dict[ligname] += 1

            print("Renaming the molnames.")
            addedligands = []
            addedligname_count_dict = defaultdict(int)
            for i in range(0, len(lignames)):
                ligname = lignames[i]
                # isomerNo = lignames.count(ligname)    # OBSOLETE and slow
                isomerNo = ligname_count_dict[ligname]
                addedligands.append(ligname)
                addedligname_count_dict[ligname] += 1
                if isomerNo == 1 and not rename_unique:
                    newlignames.append(ligname)
                elif isomerNo >= 1:
                    # isomer = addedligands.count(ligname)  # OBSOLETE and slow
                    isomer = addedligname_count_dict[ligname]
                    newlignames.append(ligname + suffix + str(isomer))

            out_fname = fname.replace(".mol2", ".ren%s.mol2" % suffix.replace("_",""))
            print("Writing file %s with new unique molnames." % out_fname)
            listcycle = cycle(newlignames)
            fout = open(out_fname, 'w')
            f.seek(0)   # for to the beginning of the input file.
            while True:
                try:
                    line = next(f)
                except StopIteration:
                    break
                fout.write(line)
                if line.startswith("@<TRIPOS>MOLECULE"):
                    next(f)  # skip the next line with the old molname
                    fout.write(next(listcycle) + "\n")
                    fout.flush()
            fout.close()
            f.close()

        # return the name of the output file
        if get_molnum:
            return fout.name, len(newlignames)
        else:
            return fout.name