#!/usr/bin/env python

from operator import itemgetter
from lib.global_fun import *

from lib.utils.print_functions import ColorPrint, Debuginfo

try:    # Necessary only to run MD
    import pytraj as pt
except (ModuleNotFoundError, ImportError):
    ColorPrint("WARNING: module pytraj was not found.", "OKRED")
    pass
from sklearn.preprocessing import minmax_scale


class MD_Analysis:
    
    def __init__(self, total_contr_thres=0.0, stdev_contr_thres=0.0):
        # TOTAL_CONTRIBUTION_THRESHOLD = 0.01
        # STDEV_CONTRIBUTION_THRESHOLD = 0.1
        self.TOTAL_CONTRIBUTION_THRESHOLD = total_contr_thres
        self.STDEV_CONTRIBUTION_THRESHOLD = stdev_contr_thres
        code3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
        code1 = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        self.aa1to3_dict = dict((c1,c3) for c1,c3 in zip(code1,code3))
        self.aa3to1_dict = dict((c3,c1) for c3,c1 in zip(code3,code1))
        self.protein_resid_RMSF_dict = {}
        self.protein_resid_bfactors_dict = {}
        self.compound_RMSF_dict = {}
        self.compound_bfactors_dict = {}
        self.complex_compound_resid_RMSF_mdict = tree()
        self.complex_compound_resid_bfactors_mdict = tree()
        self.protein_resid_chitype_chival_mdict = tree()
        self.complex_compound_resid_chitype_chival_mdict = tree()
        self.protein_resids = set()    # all protein resids
            
    
    def read_fluctuation_files(self, compound):
        """
        FUNCTION to read the B-factor, RMSD, chi RMSF, phipsi RMSF values and populate the respective dictionaries.
        """
        
        print("Reading compound", compound, " and files: protein_RMSF.tab, complex_RMSF.tab, protein_bfactors.tab, complex_bfactors.tab, protein_chi.tab, \
        complex_chi.tab, protein_phipsi.tab, complex_phipsi.tab, Bfactor_perturbations.tab, RMSF_perturbations.tab")
        
        with open (compound + '/protein_RMSF.tab', 'r') as f:
            for line in f:
                if not line[0] == '#':
                    word_list = line.split()
                    resid = int(float(word_list[0]))
                    if not resid in list(self.protein_resid_RMSF_dict.keys()):
                        self.protein_resid_RMSF_dict[resid] = [ float(word_list[1]) ]
                    elif not float(word_list[1]) in self.protein_resid_RMSF_dict[resid]:   # if
                        self.protein_resid_RMSF_dict[resid].append(float(word_list[1]))
       
        with open (compound + '/complex_RMSF.tab', 'r') as f:
            for line in f:
                if not line[0] == '#':
                    word_list = line.split()
                    self.complex_compound_resid_RMSF_mdict[compound][int(float(word_list[0]))] = float(word_list[1])
        
        with open (compound + '/ligand_RMSF.tab', 'r') as f:
            for line in f:
                if not line[0] == '#':
                    word_list = line.split()
                    self.compound_RMSF_dict[compound] = float(word_list[1])
        
        with open (compound + '/protein_bfactors.tab', 'r') as f:
            for line in f:
                if not line[0] == '#':
                    word_list = line.split()
                    resid = int(float(word_list[0]))
                    if not resid in list(self.protein_resid_bfactors_dict.keys()):
                        self.protein_resid_bfactors_dict[resid] = [ float(word_list[1]) ]
                    elif not float(word_list[1]) in self.protein_resid_bfactors_dict[resid]:
                        self.protein_resid_bfactors_dict[resid].append(float(word_list[1]))
        
        with open (compound + '/complex_bfactors.tab', 'r') as f:
            for line in f:
                if not line[0] == '#':
                    word_list = line.split()
                    self.complex_compound_resid_bfactors_mdict[compound][int(float(word_list[0]))] = float(word_list[1])
        
        with open (compound + '/ligand_bfactors.tab', 'r') as f:
            for line in f:
                if not line[0] == '#':
                    word_list = line.split()
                    self.compound_bfactors_dict[compound] = float(word_list[1])
        
        # calculate the chi angle fluctuations of the protein trajectory
        with open(compound + "/protein_chi.tab", 'r') as f:
            contents = f.readlines()
            word_list = contents[0].split()
            column_info = []    # list of the form [(resid, chi_type), ...]
            for label in word_list[1:]:
                chi_type = label.split(':')[0]
                resid = int(label.split(':')[1])
                self.protein_resids.add(resid)
                column_info.append((resid, chi_type))
                self.protein_resid_chitype_chival_mdict[resid][chi_type] = []    # initialize it to an empty list where you will save the values
            for line in contents[1:]:
                chival_list = line.split()[1:]
                for chival, (resid, chi_type) in zip(chival_list, column_info):                 
                    self.protein_resid_chitype_chival_mdict[resid][chi_type].append(float(chival))
                    
        # calculate the chi angle fluctuations of the complex trajectory
        with open(compound + "/complex_chi.tab", 'r') as f:
            contents = f.readlines()
            word_list = contents[0].split()
            column_info = []    # list of the form [(resid, chi_type), ...]
            for label in word_list[1:]:
                chi_type = label.split(':')[0]
                resid = int(label.split(':')[1])
                if not resid in self.protein_resids:    # skip any non-protein residue
                    continue
                column_info.append((resid, chi_type))
                self.complex_compound_resid_chitype_chival_mdict[compound][resid][chi_type] = []    # initialize it to an empty list where you will save the values
            for line in contents[1:]:
                chival_list = line.split()[1:]
                for chival, (resid, chi_type) in zip(chival_list, column_info):
                    self.complex_compound_resid_chitype_chival_mdict[compound][resid][chi_type].append(float(chival))
    
    
    def get_min_angle_dist(self, x, xprev):
        """
            FUNCTION to calculate the distance between two angles (of the current and of the previous frame) by periodicity by checking the shortest path
            from the previous frame. A positive sign will indicate the current angle moved clockwise wrt the previous angle, whereas a negative sign
            will indicate that the current angle moved anti-clockwise wrt the previous angle.
            Examples:
            1. x=-170, x_prev=120 --> min( abs(-180-(-170))+(180-120), abs(-170)+120) = +25
        """
        
        if x == xprev:  # include the case x=xprev=0
            return 0
        
        if x < 0 and xprev > 0:
            # calc the distance of each other from 180 and from 0
            d180 = x + 180
            dprev180 = 180 - xprev
            d0 = abs(d180)
            dprev0 = xprev
            dist180 = d180 + dprev180
            dist0 = d0 + dprev0
            if dist180 < dist0:
                return dist180
            elif dist180 > dist0:
                return -1*dist0 
            elif dist180 == dist0:
                return dist180
        
        elif x > 0 and xprev < 0:
            d180 = 180 - x
            dprev180 = xprev + 180
            d0 = d180
            dprev0 = abs(xprev)
            dist180 = d180 + dprev180
            dist0 = d0 + dprev0
            if dist180 < dist0:
                return -1*dist180
            elif dist180 > dist0:
                return dist0 
            elif dist180 == dist0:
                return dist0
            
        elif x < 0 and xprev < 0:
            if x < xprev:   # anti-clockwise
                return x-xprev  # always negative
            elif x > xprev: # clockwise
                return x-xprev  # always positive
            # the alternative route is always longer
        elif x > 0 and xprev > 0:
            if x < xprev:   # anti-clockwise
                return x-xprev  # always negative
            elif x > xprev: # clockwise
                return x-xprev  # always positive
            # the alternative route is always longer
        elif x == 0 and xprev != 0:
            return -1*xprev     # if xprev>0 then anti-clockwise, elif xprev<0 then clockwise
        
        elif x != 0 and xprev == 0:
            return x     # if x>0 then clockwise, elif x<0 then anti-clockwise
        
    
    def standarize_angles(self, angle_list):
        """
            Center all angles to x0 (angle value of the first frame)...
        """
        
        stdangle_list = [None]*len(angle_list)  # standarized angle_list
        x0 = angle_list[0]
        stdangle_list[0] = x0
        for i in range(1, len(angle_list)):
            xprev = angle_list[i-1]
            x = angle_list[i]
            dist = self.get_min_angle_dist(x, xprev)
            stdangle_list[i] = stdangle_list[i-1] + dist    # shift the previous standarized angle by the distance (decrease if anti-clockwise rotation or increase otherwise)
        
        return stdangle_list
    
    
    def get_chi_perturbations(self, scale=True):
        
        protein_resid_chitype_stdchival_mdict = tree()  # standarized chi values
        complex_compound_resid_chitype_stdchival_mdict = tree()
        
        
        # Standarize the angle values to break the periodicity
        for resid in list(self.protein_resid_chitype_chival_mdict.keys()):
            for chi_type in list(self.protein_resid_chitype_chival_mdict[resid].keys()):
                protein_resid_chitype_stdchival_mdict[resid][chi_type] = self.standarize_angles(self.protein_resid_chitype_chival_mdict[resid][chi_type])
        for compound in list(self.complex_compound_resid_chitype_chival_mdict.keys()):
            for resid in list(self.complex_compound_resid_chitype_chival_mdict[compound].keys()):
                for chi_type in list(self.complex_compound_resid_chitype_chival_mdict[compound][resid].keys()):
                    complex_compound_resid_chitype_stdchival_mdict[compound][resid][chi_type] = self.standarize_angles(self.complex_compound_resid_chitype_chival_mdict[compound][resid][chi_type])
        
        # Calculate the average RMSF and B-factor perturbations for each residue from the multiple trajectories (if applicable).
        for resid in list(self.protein_resid_RMSF_dict.keys()):
            self.protein_resid_RMSF_dict[resid] = np.mean(self.protein_resid_RMSF_dict[resid])
        for resid in list(self.protein_resid_bfactors_dict.keys()):
            self.protein_resid_bfactors_dict[resid] = np.mean(self.protein_resid_bfactors_dict[resid])
        
        # Find all the residues that had chi value in all complexes
        resid_set = set()   # include the ligand resid (the same in every compound complex)
        compound_list = []
        for compound in list(complex_compound_resid_chitype_stdchival_mdict.keys()):  # these resids are the same for B-factors
            compound_list.append(compound)
            for resid in list(complex_compound_resid_chitype_stdchival_mdict[compound].keys()):
                resid_set.add(resid)
        
        # Add missing residues to each compound with chi value 0 (==> chi RMSF will be 0 too)
        for compound in list(complex_compound_resid_chitype_stdchival_mdict.keys()):
            for resid in resid_set:
                if not resid in list(complex_compound_resid_chitype_stdchival_mdict[compound].keys()):
                    complex_compound_resid_chitype_stdchival_mdict[compound][resid]["chi"] = [0.0]
        
        # calculate the RMSF (stdev) of each protein chi angles
        protein_resid_meanRMSF_dict = {}
        protein_chiRMSF_fp = [] # the protein trajectory chi RMSF fingerprint
        for resid in resid_set:
            if not resid in list(protein_resid_chitype_stdchival_mdict.keys()):   # skip the ligand resid
                print("ERROR: resid=", resid, "not in  protein_resid_chitype_stdchival_mdict!")
                sys.exit(1)
            mean_RMSF = 0   # mean chi value fuctuation for this resid
            N = 0   # number of chi angle types for this resid
            for chi_type in list(protein_resid_chitype_stdchival_mdict[resid].keys()):
                mean_RMSF += np.std(protein_resid_chitype_stdchival_mdict[resid][chi_type])
                N += 1
            mean_RMSF = mean_RMSF/float(N)
            protein_resid_meanRMSF_dict[resid] = mean_RMSF
            protein_chiRMSF_fp.append(mean_RMSF)
        protein_chiRMSF_fp = np.array(protein_chiRMSF_fp)
        
        # calculate the RMSF (stdev) of each complex chi angles
        chiRMSF_fp_list = []
        for compound in list(complex_compound_resid_chitype_stdchival_mdict.keys()):
            chiRMSF_fp = []
            for resid in resid_set:
                mean_RMSF = 0   # mean chi value fuctuation for this resid
                N = 0   # number of chi angle types for this resid
                for chi_type in list(complex_compound_resid_chitype_stdchival_mdict[compound][resid].keys()):
                    mean_RMSF += np.std(complex_compound_resid_chitype_stdchival_mdict[compound][resid][chi_type])
                    N += 1
                mean_RMSF = mean_RMSF/float(N)
                chiRMSF_fp.append(mean_RMSF)
            chiRMSF_fp_list.append(np.array(chiRMSF_fp))
        
        # calculate the chi RMSF perturbations
        chiRMSF_perturbation_fp_list = []  # [(resid, chi perturbation), (resid, chi perturbation), ...]
        for complex_chiRMSF_fp in chiRMSF_fp_list:
            chiRMSF_perturbation_fp_list.append( protein_chiRMSF_fp - complex_chiRMSF_fp )
        
        if scale:
            chiRMSF_fp_list = list(minmax_scale(chiRMSF_fp_list))
            chiRMSF_perturbation_fp_list = list(minmax_scale(chiRMSF_perturbation_fp_list))
        
        return chiRMSF_fp_list, chiRMSF_perturbation_fp_list
    

    def get_COM(self, selection, prmtop, coordfile):
        """
            FUNCTION to find the COM coordinates of the specified selection. Coordfile can be inpcrd or rst or pdb.
        """
        
        getCOM = """
        trajin """+coordfile+"""
        vector COM center """+selection+""" out COM.tab
        go
        quit
        """
        write2file(getCOM, "getCOM.ptrj")
        run_commandline("cpptraj -p "+prmtop+" -i getCOM.ptrj")
        with open('COM.tab', 'r') as f:
            contents = f.readlines()
            word_list = contents[-1].split()
            COM_coord = word_list[1:4]
            
        return COM_coord
    
    def write_pocket(self, start_FLEX, total_frames, offset, cutdist=6.0):
        """
        Method to save the binding pocket snapshots (including waters) to a .pdb file for SIFT calculation
        :param start_FLEX:
        :param total_frames:
        :param offset:
        :param cutdist:
        :return:
        """
        save_pocket = """
         reference complex.prod_GB.nc lastframe
         trajin complex.prod_GB.nc """ + str(start_FLEX) + """ """ + str(total_frames) + """ """ + str(offset) + """
         strip '(!:LIG<:%.3f | :LIG)'
         trajout pocket.pdb
         go
         quit
         """ % cutdist
        write2file(save_pocket, "save_pocket.ptrj")
        run_commandline("cpptraj -p complex.prmtop -i save_pocket.ptrj")

    @staticmethod
    def write_pdb_from_prmtop(prmtop, coord, outpdb):
        save_pdb = """
        trajout %s
        go
        quit
        """ % outpdb
        write2file(save_pdb, "save_pdb.ptrj")
        run_commandline("cpptraj -p %s -y %s -i save_pdb.ptrj" % (prmtop, coord))

    @staticmethod
    def get_charge(prmtop, coordfile, mask):
        """
        Method to get that net charge of the selection in mask from an AMBER trajectory.
        :param prmtop:
        :param coordfilem:
        :param mask:
        :return:
        """
        traj = pt.load(coordfile, prmtop)
        return int(round(traj[mask].top.charge.sum()))

    @staticmethod
    def prepend_charges_to_pdb(prmtop, coordfile, pdb, rec_mask="!:LIG", lig_mask=":LIG"):
        """
        Method to prepend to the specified pdb file as comments the net charges of the receptor and the ligand
        specified by rec_mask and lig_mask, respectively.
        :param prmtop:  it can be the AMBER pramtop file or the path to 'pdb_charge_header.txt'
        :param coordfile:   it can be the AMBER coord file or the path to 'pdb_charge_header.txt'
        :param pdb:
        :param rec_mask:
        :param lig_mask:
        :return:
        """
        if os.path.exists(prmtop) == False:
            raise FileNotFoundError(Debuginfo("%s file doesn't exist!" % prmtop, fail=True))

        if prmtop.endswith("pdb_charge_header.txt") and coordfile.endswith("pdb_charge_header.txt"):    # if a header file
            with open(prmtop, 'r') as f:
                charge_header = f.readlines()
        else:   # if they are real AMBER topology and coord files
            lig_charge = MD_Analysis.get_charge(prmtop, coordfile, mask=lig_mask)
            rec_charge = MD_Analysis.get_charge(prmtop, coordfile, mask=rec_mask)
            # ColorPrint("Prepending charges to pdb %s" % pdb, "OKBLUE")
            charge_header = "HEADER ligand net charge = %i\nHEADER receptor net charge = %i\n" % (lig_charge, rec_charge)

        # Prepend the charges to the pdb
        with open(pdb, 'r') as f:
            with open(pdb + ".tmp", 'w') as fout:
                fout.write(charge_header)
                for line in f:  # memory effective
                    fout.write(line)
        os.rename(pdb + ".tmp", pdb)

    @staticmethod
    def save_charges_to_file(prmtop, coordfile, outfile, rec_mask="!:LIG", lig_mask=":LIG"):
        """
        Method to save to a file the net charges of the receptor and the ligand
        specified by rec_mask and lig_mask, respectively.
        :param prmtop:
        :param coordfile:
        :param pdb:
        :param rec_mask:
        :param lig_mask:
        :return:
        """
        lig_charge = MD_Analysis.get_charge(prmtop, coordfile, mask=lig_mask)
        rec_charge = MD_Analysis.get_charge(prmtop, coordfile, mask=rec_mask)
        # ColorPrint("Saving charges to file %s" % pdb, "OKBLUE")
        charge_header = "HEADER ligand net charge = %i\nHEADER receptor net charge = %i\n" % (lig_charge, rec_charge)
        with open(outfile, 'w') as f:
            f.write(charge_header)

    @staticmethod
    def copy_charges_from_pdb(inpdb, outpdb):
        """
        Method to prepend to the specified pdb file as comments the net charges of the receptor and the ligand
        specified by rec_mask and lig_mask, respectively.
        :param prmtop:
        :param coordfile:
        :param pdb:
        :param rec_mask:
        :param lig_mask:
        :return:
        """
        lig_charge, rec_charge = MD_Analysis.read_charges_from_pdb(inpdb)
        # ColorPrint("Prepending charges to pdb %s" % pdb, "OKBLUE")
        with open(outpdb, 'r') as f:
            with open(outpdb + ".tmp", 'w') as fout:
                fout.write("HEADER ligand net charge = %i\n" % lig_charge)
                fout.write("HEADER receptor net charge = %i\n" % rec_charge)
                for line in f:  # memory effective
                    fout.write(line)
        os.rename(outpdb + ".tmp", outpdb)

    @staticmethod
    def read_charges_from_pdb(pdb):
        """
        Returns the net receptor and ligand charges from a pdb file. If they don't exists it returns False.
        :param pdb:
        :return:
        """
        lig_charge, rec_charge = None, None
        with open(pdb, 'r') as f:
            for lineNum, line in enumerate(f):
                if lineNum > 10:
                    break
                if line.startswith('HEADER receptor net charge ='):
                    # rec_charge = int(round(float(line.split()[5])))   # charge is already integer
                    rec_charge = int(line.split()[5])
                elif line.startswith('HEADER ligand net charge ='):
                    # lig_charge = int(round(float(line.split()[5])))   # charge is already integer
                    lig_charge = int(line.split()[5])
                if lig_charge != None and rec_charge != None:
                    return lig_charge, rec_charge

        ColorPrint("ERROR: pdb file %s does not contain the ligand and receptor charges as comments!" % pdb, "FAIL")
        return False

    def write_lastframe(self, prmtop, coordfile, fileformat="ncrestart", outfname="", write_charges=True,
                        write_ligand_mol2=False, ligname=""):
        """
        Method to write the last frame of the specified AMBER trajectory to a pdb file, including the net charges of the ligand
        and the receptor.
        :param prmtop:
        :param coordfile:
        :param fileformat:
        :param outfname:
        :param write_charges: if fileformat=="pdb" prepends net charges of the ligand and the receptor to the output pdb file
        :param write_ligand_mol2:
        :param ligname:
        :return:
        """
        # TODO: use pytraj!
        if not outfname:
            outfname = os.path.splitext(coordfile)[0] + "_lastframe.%s" % fileformat
        outbasename = os.path.splitext(outfname)[0]

        if write_ligand_mol2:
            assert ligname, \
                Debuginfo("ERROR: along with write_ligand_mol2=True you must provide the molname of the ligand!",
                            fail=True)
            write_lastfrm = "trajin %s lastframe\ntrajout %s\ngo\nstrip !:LIG\ntrajout ligand_forscoring.gaff2.mol2\ngo\nquit" % \
                            (coordfile, outfname)
        else:
            write_lastfrm = "trajin %s lastframe\ntrajout %s\ngo\nquit" % \
                          (coordfile, outfname)

        write2file(write_lastfrm, "write_lastfrm.ptrj")
        run_commandline("cpptraj -p %s -i write_lastfrm.ptrj" % prmtop)

        if write_charges and fileformat == "pdb":
            workdir = os.path.dirname(prmtop)
            if len(workdir) == 0:   workdir = "."
            if os.path.exists(workdir + "/pdb_charge_header.txt") == False:
                MD_Analysis.save_charges_to_file(prmtop,
                                                 coordfile,
                                                 os.path.dirname(prmtop) + "/pdb_charge_header.txt",
                                                 rec_mask="!:LIG",
                                                 lig_mask=":LIG")
            self.prepend_charges_to_pdb(prmtop, coordfile, pdb=outfname, rec_mask="!:LIG", lig_mask=":LIG")

        if write_ligand_mol2:
            run_commandline("antechamber -i ligand_forscoring.gaff2.mol2 -fi mol2 -o %s_lig.mol2 "
                            "-fo mol2 -rn LIG -at sybyl -dr n -cf charge_file.mol2" % outbasename)
            run_commandline("perl -pi -e \"s/^LIG$/" + ligname + "/\" %s_lig.mol2" % outbasename)  # fix molecule name
            os.remove("ligand_forscoring.gaff2.mol2")

    def minimize_and_write_frames(prmtop,
                                  coordfile,
                                  start_frame,
                                  end_frame,
                                  stride,
                                  min_steps,
                                  outfname="",
                                  write_charges=True,
                                  write_ligand_mol2=False,
                                  ligname="",
                                  GPU_DEVICES="0",
                                  PLATFORM="CUDA"):

        print("The total number of frames that will be exported to PDB files and minimized is", len(range(start_frame, end_frame+1, stride)))

        if not outfname:
            outfname = os.path.splitext(coordfile)[0] + "_frm"

        traj = pt.load("complex.prod_GB.nc", "complex.prmtop")
        pt.write_traj(outfname,
                      traj,
                      format="PDB",
                      frame_indices=list(range(start_frame, end_frame+1, stride)),
                      options='multi',
                      overwrite=True)
        # Get the output file names
        folder = os.path.dirname(outfname)
        if len(folder) == 0:
            folder = "."
        basefname = os.path.basename(outfname)
        frame_pdbs = list_files(folder=folder, pattern=basefname + "\.[0-9]+")
        frame_pdbs.sort()

        # Minimize each of the frames
        for pdb in frame_pdbs:
            # minimization with OpenMM
            # ALWAYS use CPU or CUDA double precision for minimization, otherwise it doesn't work!
            ColorPrint("Minimizing %s" % pdb, "OKBLUE")
            commandline = "openmm_em_GB.py -prmtop %s -coord %s -igb 0 " \
                          "-min_steps %i -prefix %s -nbcut 18.000000 -device %s -platform %s -prec double" % \
                          (prmtop, pdb, min_steps, pdb, GPU_DEVICES, PLATFORM)
            run_commandline(commandline)

        # Convert the restart output files to PDB format
        for i in range(len(frame_pdbs)):
            rst_file = "%s.min_GB.rst" % frame_pdbs[i]
            traj = pt.load(rst_file, prmtop)
            pt.write_traj(filename="%s.min.pdb" % frame_pdbs[i],
                          traj=traj,
                          format="PDB",
                          overwrite=True)
            frame_pdbs[i] = "%s.min.pdb" % frame_pdbs[i]    # update the pdb name

        if write_charges:
            for pdb in frame_pdbs:
                MD_Analysis.prepend_charges_to_pdb(prmtop, pdb, pdb=pdb, rec_mask="!:LIG", lig_mask=":LIG")

        return frame_pdbs

    def get_bfactor_perturbations(self, scale=True):
        """
            FUNCTION to calculate and return file B-factor and RMSF perturbations.outfname
        """
        
        # Calculate the average RMSF and B-factor perturbations for each residue from the multiple trajectories (if applicable).
        for resid in list(self.protein_resid_RMSF_dict.keys()):
            self.protein_resid_RMSF_dict[resid] = np.mean(self.protein_resid_RMSF_dict[resid])
        for resid in list(self.protein_resid_bfactors_dict.keys()):
            self.protein_resid_bfactors_dict[resid] = np.mean(self.protein_resid_bfactors_dict[resid])
        
        # Find all the residues that had RMSF or B-factor value in all complexes
        resid_set = set()   # include the ligand resid (the same in every compound complex)
        compound_list = []
        for compound in list(self.complex_compound_resid_RMSF_mdict.keys()):  # these resids are the same for B-factors
            compound_list.append(compound)
            for resid in list(self.complex_compound_resid_RMSF_mdict[compound].keys()):
                resid_set.add(resid)
        
        # Add missing residues to each compound with RMSF or B-factor value 0
        for compound in list(self.complex_compound_resid_RMSF_mdict.keys()):
            for resid in resid_set:
                if not resid in list(self.complex_compound_resid_RMSF_mdict[compound].keys()):
                    self.complex_compound_resid_RMSF_mdict[compound][resid] = 0.0
                if not resid in list(self.complex_compound_resid_bfactors_mdict[compound].keys()):
                    self.complex_compound_resid_bfactors_mdict[compound][resid] = 0.0
        
        # Create fingerprints from RMSF and B-factor fluctuations
        protein_resids_list = list(self.protein_resid_bfactors_dict.keys())    # all the resids of the protein
        resid_list = list(resid_set)
        resid_list.sort()   # the resids of all the flexible residues in all the complexes, including the resid of the ligand which is always the same
        self.Bfactor_perturbation_fp_list = []
        self.RMSF_perturbation_fp_list = []
        self.Bfactors_fp_list = []
        self.RMSF_fp_list = []
        for compound in compound_list:
            Bfactor_perturbation_list = []  # [Bfactor perturbation of resid1, Bfactor perturbation of resid2, ...]
            RMSF_perturbation_list = []
            Bfactors_list = []
            RMSF_list = []
            for resid in resid_list:
                try:
                    bf_perturbation = self.protein_resid_bfactors_dict[resid] - self.complex_compound_resid_bfactors_mdict[compound][resid]
                    RMSF_perturbation = self.protein_resid_RMSF_dict[resid] - self.complex_compound_resid_RMSF_mdict[compound][resid]
                except KeyError:
                    bf_perturbation = self.compound_bfactors_dict[compound] - self.complex_compound_resid_bfactors_mdict[compound][resid]
                    RMSF_perturbation = self.compound_RMSF_dict[compound] - self.complex_compound_resid_RMSF_mdict[compound][resid]
                Bfactor_perturbation_list.append( bf_perturbation )
                RMSF_perturbation_list.append( RMSF_perturbation )
                Bfactors_list.append(self.complex_compound_resid_bfactors_mdict[compound][resid])
                RMSF_list.append(self.complex_compound_resid_RMSF_mdict[compound][resid])
            self.Bfactors_fp_list.append(np.array(Bfactors_list))
            self.RMSF_fp_list.append(np.array(RMSF_list))
            self.Bfactor_perturbation_fp_list.append(np.array(Bfactor_perturbation_list))   # convert the list of perturbations to array (fingerprint) and save it
            self.RMSF_perturbation_fp_list.append(np.array(RMSF_perturbation_list))
    
        if scale:
            self.Bfactor_perturbation_fp_list = list(minmax_scale(self.Bfactor_perturbation_fp_list))
            self.RMSF_perturbation_fp_list = list(minmax_scale(self.RMSF_perturbation_fp_list))
            self.Bfactors_fp_list = list(minmax_scale(self.Bfactors_fp_list))
            self.RMSF_fp_list = list(minmax_scale(self.RMSF_fp_list))
        
        return self.Bfactor_perturbation_fp_list, self.RMSF_perturbation_fp_list, self.Bfactors_fp_list, self.RMSF_fp_list
    
    
    def write_perturbations(self,
                            ligfile="ligand_bfactors.tab",
                            protfile="protein_bfactors.tab",
                            complexfile="complex_bfactors.tab",
                            outfname="Bfactor_perturbations.tab"):
        """
            FUNCTION to calculate and save to a file B-factor and RMSF perturbations.
        """
        protein_bfactors_dict = {}  # resid->mass-weighted B-factor
        complex_bfactors_dict = {}
        with open(protfile, 'r') as f:
            for line in f:
                if not line[0] == '#':
                    word_list = line.split()
                    protein_bfactors_dict[int(float(word_list[0]))] = float(word_list[1])
        with open(complexfile, 'r') as f:
            for line in f:
                if not line[0] == '#':
                    word_list = line.split()
                    complex_bfactors_dict[int(float(word_list[0]))] = float(word_list[1])
        
        protein_residue_set = set(protein_bfactors_dict.keys())
        complex_residue_set = set(complex_bfactors_dict.keys())
        ligand_set = complex_residue_set.intersection(protein_residue_set)
        ligand_resid = list(ligand_set)[0]  # because in the ligand trajectory the ligand resid is 1.000
        with open(ligfile, 'r') as f:
            for line in f:
                if not line[0] == '#':
                    word_list = line.split()
                    ligand_bf =  float(word_list[1])
        
        Bfactor_perturbation_list = []  # [(resid, Bfactor perturbation), (resid, Bfactor perturbation), ...]
        for resid in list(complex_bfactors_dict.keys()):
            try:
                bf_perturbation = protein_bfactors_dict[resid] - complex_bfactors_dict[resid]
            except KeyError:
                bf_perturbation = ligand_bf - complex_bfactors_dict[resid]
            Bfactor_perturbation_list.append( (resid, bf_perturbation) )
        Bfactor_perturbation_list.sort(key=itemgetter(0))
        with open(outfname, 'w') as f:
            for (resid, bf_perturbation) in Bfactor_perturbation_list:
                f.write(str(resid)+"\t"+str(bf_perturbation)+"\n")

    def get_closests_residues(self, pdb, cutdist=6.0):
        """
        Method to return the list of non-water residues that are within cutdist distance from the ligand.
        Residues are returned in "${resname}_${resid}" format and they can include ions and other HETATMs.
        :param pdb:
        :param cutdist:
        :return:
        """
        workdir = os.path.dirname(pdb)
        mask = """
reference %s lastframe
trajin %s lastframe
mask '(:LIG <:%.3f) &! :WAT' maskout %s/closestAtoms.out
go
quit
        """ % (pdb, pdb, cutdist, workdir)
        write2file(mask, "%s/mask_traj.ptrj"%workdir)
        run_commandline("cpptraj -p %s -i %s/mask_traj.ptrj" % (pdb, workdir), verbose=False)

        closest_residues = []   # includes non-protein atoms apart from waters
        with open("%s/closestAtoms.out" % workdir, 'r') as f:
            contents = f.readlines()
            for i in range(1, len(contents)):
                l = contents[i]
                _, _, _, resid, resname, _ = l.split()
                residue = "%s_%s" % (resname, resid)
                closest_residues.append(residue)
        return set(closest_residues)

    def save_closest_waters(self, start_frame, total_frames, offset, pbradii, cutdist=5.0):
        # TODO: untested and probably incomplete.
        mask_traj = """
reference complex.prod_GB.nc lastframe
trajin complex.prod_GB.nc lastframe
mask '(:LIG <:%.3f) & :WAT@O' maskout closestWAT_mask.txt
mask ':WAT@O' maskout allWAT_mask.txt
go
quit
        """ % cutdist
        write2file(mask_traj, "mask_traj.ptrj")
        run_commandline("cpptraj -p complex.prmtop -i mask_traj.ptrj")

        run_commandline("sort closestWAT_mask.txt > closestWAT_mask.txt.sorted; sort allWAT_mask.txt > allWAT_mask.txt.sorted")
        delWAT4Ang_mask=run_commandline("grep -F -x -v -f closestWAT_mask.txt.sorted allWAT_mask.txt.sorted | awk '{print($4}' "
                                        "| sort | uniq | perl -p -e 's/\n/ /g' | perl -p -e 's/([0-9]) ([0-9])/\1, \2/g'",
                                        return_out=True)
        numWAT=run_commandline("grep WAT closestWAT_mask.txt | wc -l", return_out=True)
        numWAT = numWAT[0].strip()

        closest_waters = """
trajin complex.prod_GB.nc %i %i %i
closest %i :LIG first outprefix closest
trajout complex.MMGBSA_GB.closestWAT.nc
go
quit
        """ % (start_frame+1, total_frames, offset, numWAT)
        write2file(closest_waters, "closest_waters.ptrj")
        run_commandline("cpptraj -p complex.prmtop -i closest_waters.ptrj")

        # fix the PBradii of the written closest.complex.prmtop file
        run_commandline("""
parmed closest.complex.prmtop << EOF
changeRadii %s
parmout closest.complex2.prmtop
go
quit
EOF
        """ % pbradii)

        for fname in ["complex2.prmtop", "protein2.prmtop", "ligand2.prmtop"]:
            if os.path.exists(fname):   # remove the old files if they exist
                os.remove(fname)

    @staticmethod
    def clean_ligand_folder(folder):
        """
        Clean all ligand folders but those that failed MD for trouble shooting. Essentially only the following files will be left:
            complex_forscoring_frm.*.min.pdb
            protein-ligand.pdb
            ligand_forscoring.mol2
            pdb_charge_header.txt

        :param WORKDIR:
        :param molnames2delete:
        :return:
        """
        if os.path.isdir(folder) and os.path.exists("%s/frcmod.ligand" % folder):
            ColorPrint("Cleaning folder %s" % folder, "OKBLUE")
            for f in list_files(folder=folder, pattern=".*", full_path=False, rel_path=False):
                if (f.startswith("complex_forscoring_frm.") and f.endswith(".min.pdb")) or \
                        f.endswith("protein-ligand.pdb") or \
                        f.endswith("ligand_forscoring.mol2") or \
                        f.endswith("pdb_charge_header.txt"):
                    continue
                os.remove("%s/%s" % (folder, f))