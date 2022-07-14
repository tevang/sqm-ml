from operator import itemgetter
from scipy import stats
from .global_fun import *
from sklearn.preprocessing import minmax_scale

from .utils.print_functions import Debuginfo


class MMGBSA:
    
    def __init__(self, total_contr_thres=0.0, stdev_contr_thres=0.0):
        # TOTAL_CONTRIBUTION_THRESHOLD = 0.01
        # STDEV_CONTRIBUTION_THRESHOLD = 0.1
        self.TOTAL_CONTRIBUTION_THRESHOLD = total_contr_thres
        self.STDEV_CONTRIBUTION_THRESHOLD = stdev_contr_thres
        code3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
        code1 = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        self.aa1to3_dict = dict((c1,c3) for c1,c3 in zip(code1,code3))
        self.aa3to1_dict = dict((c3,c1) for c3,c1 in zip(code3,code1))
        self.LIG_resid = None # the resid of the ligand as string
        self.resid2residue_dict = {}
        self.include_skewness = False
        
        # Just of Information purpose (never used in the code)
        self.headers =  ['Resid 1', 'Resid 2',
                         'Internal mean', 'Internal stdev', 'Internal stderr',
                         'van der Waals mean', 'van der Waals stdev', 'van der Waals stderr',
                         'Electrostatic mean', 'Electrostatic stdev', 'Electrostatic stderr',
                         'Polar Solvation mean', 'Polar Solvation stdev', 'Polar Solvation stderr',
                         'Non-Polar Solv. mean', 'Non-Polar Solv. stdev', 'Non-Polar Solv. stderr',
                         'TOTAL mean', 'TOTAL stdev', 'TOTAL stderr']
        self.compound_residue_LIG_decompositionArray_mdict = tree()    # only this dict has non-zero values
        self.perframe_compound_residue_LIG_decompositionArray_mdict = tree()    # like above but with per frame decomposition
    
    @staticmethod
    def Ki2DeltaG(y, RT=0.5957632602, scale=False, original=False):
        """
        Convert Ki,IC50,Kd or EC50 values to pseudoenergies. y can be a list or an array. The affinities must be in uM!
        :param y:
        :param RT:
        :param scale:
        :param original:    calculate correct DeltaGs, not pseudovalues for favorable training.
        :return:
        """
        from sklearn.preprocessing import minmax_scale

        if original:
            denominator = 1e6
        else:
            denominator = 1e2

        if type(y) == float:
            return RT*np.log(float(y/denominator))
        else:
            new_y = []
            for v in y:
                new_y.append(RT*np.log(float(v/denominator))) # the v must be in uM!!! ; original implementation is with v/1e6!
        
        if scale:
            new_y = minmax_scale(new_y, feature_range=(-1.0, 0.0)).tolist()  # in this range the MLPs perform best
            
        return np.array(new_y)
    
    @staticmethod
    def DeltaG2Kd(y, RT=0.5957632602, scale=False, original=True):
        """
        Convert Free Energies in Kcal/mol back to binding affinities in uM. I presume that the energies were calculated
        from affinities expressed in uM!
        :param y:
        :param RT:
        :param scale:
        :param original: True if the scale of the input DeltaG is in the original values. False if DeltaG was generated with
                            Ki2DeltaG(original=False)
        :return :
        """
        from sklearn.preprocessing import minmax_scale
        if original:
            exponent = 1e6
        else:
            exponent = 1e2

        if isinstance(y, (float, np.floating)):
            return np.exp(float(y)/RT)*exponent
        else:
            new_y = []
            for e in y:
                new_y.append(np.exp(float(e)/RT)*exponent)   # original implementation is with *1e6
        
        if scale:
            new_y = minmax_scale(new_y, feature_range=(-1.0, 0.0)).tolist()  # in this range the MLPs perform best
            
        return np.array(new_y)

    @staticmethod
    def KJoule2Kcal(y):
        """
        Method to convert KJ/mol to Kcal/mol.
        :return:
        """
        if type(y) in [float, int]:
            return y*0.238846
        else:
            return [e*0.238846 for e in y]

    @staticmethod
    def logIt(y, minus=True):
        """
            use minus=True for Inhibitions and minus=False for Activities.
        """
        y = np.array(y)
        if np.min(y) < 0:   # shift the values to make them all positive
            y = y - np.min(y)
        if np.max(y) > 100:
            max_y = np.max(y)+1
        else:
            max_y = 101
        new_y = []
        for Inh in y:
            logitInh = Inh/(max_y-Inh)  # convert the % inhibition values to logit
            new_y.append(logitInh)
        
        if minus:
            new_y = -1.0 * np.array(new_y)
        else:
            new_y = np.array(new_y)
        
        return new_y

    @staticmethod
    def Tm_to_DeltaG(y):
        """
        Method that rescales Tm (or DeltaTm) to [-12, 0], which is the frequent range of deepScaffOpts score.
        :param y:
        :return:
        """
        assert isinstance(y, (list, tuple, np.ndarray)), Debuginfo(
            "ERROR: all Tm values from the same assays must be "
            "transformed to DeltaG together, not each one individually!", fail=True)
        Tm = np.array(y)
        # return -12**(Tm/Tm.max())     ; # exponential scale
        return -12**(minmax_scale(Tm, feature_range=(0.7, 1.0)))     ; # another exponential scale
        # return -1*np.exp(minmax_scale(Tm, feature_range=(0.0, 2.0)))-5     ; # yet another exponential scale
        # return -12*((Tm-Tm.min())/Tm.max())-6        ; # linear scale to [-12,-5]
        # return -5*minmax_scale(Tm)-6        ; # another linear scale to [-12,-5]

    @staticmethod
    def Tm_to_Kd(y):
        return MMGBSA.DeltaG2Kd(MMGBSA.Tm_to_DeltaG(y))

    @staticmethod
    def transform2FE(molname_score_dict, FE=None, Kd=None):
        
        if FE:
            refmolname = FE.split()[0].lower()
            refFE = float(FE.split()[1])
        elif Kd:
            refmolname = FE.split()[0].lower()
            refKd = float(FE.split()[1])   # must be in uM
            refFE = MMGBSA.Ki2DeltaG(refKd)
        
        refscore = molname_score_dict[refmolname]
        scale_factor = refFE/refscore
        for molname in list(molname_score_dict.keys()):
            molname_score_dict[molname] *= scale_factor
        
        return molname_score_dict


    def Inhibition2DeltaG(self, y, molnames_list, mode=1, ref_FE=None, ref_Kd=None):
        """
        There are many ways to transform the Inhibition values.
        preprocessing.scale(y)
        preprocessing.minmax_scale(y, feature_range=(0, 1))
        preprocessing.maxabs_scale(y)
        preprocessing.robust_scale(y, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
        """
        from sklearn.preprocessing import minmax_scale
        from sklearn.preprocessing import maxabs_scale
        from sklearn.preprocessing import scale
        from sklearn.preprocessing import robust_scale
        from sklearn.preprocessing import normalize
        from sklearn.preprocessing.data import quantile_transform
        
        y = np.array(y)
        if mode == 0:
            title, new_y = 'no transformation', y
        elif mode == 1:
            title, new_y = 'standard scaling', -1.0 * scale(y)
        elif mode == 2:
            title, new_y = 'min-max scaling', -1.0 * minmax_scale(y, feature_range=(0, 0.8))
        elif mode == 3:
            title, new_y = 'max-abs scaling', -1.0 * maxabs_scale(y)
        elif mode == 4:
            title, new_y = 'robust scaling', [n[0] for n in -1.0 * robust_scale(y.reshape(-1,1), quantile_range=(25, 75))]
        elif mode == 5:
            title, new_y = 'quantile transformation (uniform pdf)', [n[0] for n in -1.0 * quantile_transform(y.reshape(-1,1), output_distribution='uniform')]
        elif mode == 6:
            title, new_y = 'quantile transformation (gaussian pdf)', [n[0] for n in -1.0 * quantile_transform(y.reshape(-1,1), output_distribution='normal')]
        elif mode == 7:
            title, new_y = 'sample-wise L2 normalizing', [n[0] for n in -1.0 * normalize(y.reshape(-1,1), axis=0)]
        elif mode == 8:
            title, new_y = 'logIt transformation', 1000 * self.logIt(y, minus=True)
        elif mode == 9:
            title, new_y = 'DeltaG transformation', self.Ki2DeltaG(1.0 / (y+1))  # y+1 to avoid devision by zero
        elif mode == 10:
            title, new_y = 'DeltaG transformation', self.Ki2DeltaG(0.1 / (y+1))  # y+1 to avoid devision by zero
        elif mode == 11:
            y2 = minmax_scale(y, feature_range=(0.001, 1000.0))
            title, new_y = 'DeltaG transformation', self.Ki2DeltaG(1 / (y2+1))  # y2+1 to avoid devision by zero
        
        new_y = list(new_y)
        molname_score_dict = {}
        for k,v in zip(molnames_list, new_y):
            molname_score_dict[k] = v
        if not (ref_FE or ref_Kd):
            min_score = min(new_y)  # the best score
            min_indx = new_y.index(min_score)
            refmolname = molnames_list[min_indx]
            ref_FE = refmolname + " -12.0"  # if not given, assume that the lowest free energy is -12 kcal/mol
        
        transformed_y = []
        # molname_score_dict = self.transform2FE(molname_score_dict, FE=ref_FE, Kd=ref_Kd)  
        for molname in molnames_list:
            transformed_y.append(molname_score_dict[molname])
            
        print("Applying " + title + " to the percent Inhibition values.")
        return transformed_y
    
    
    def Activity2DeltaG(self, y, RT=0.5957632602):
        for lig,act in list(ligand_Ki_dict.items()):
            Ki = 1001-1000*act  # if act=1 then Ki=1, else if act=0 then Ki=1001
            ligand_Ki_dict[lig] = Ki
    
    
    def read_decomposition_file(self, compound, decomposition_file):
        """
        FUNCTION to read the MMGBSA energy decomposition file and populate the following dictionaries:
        self.compound_residue_LIG_decompositionArray_mdict:   # e.g. CHEMBL389589 -> Y90 -> array([[ 0., 0., 0.],
                                                                        [-0.09816667,  0.00823441,  0.00336168],
                                                                        [-0.10116667,  0.01739173,  0.00710014],
                                                                        [-0.02983333,  0.02034221,  0.00830467],
                                                                        [-0.014874,    0.0056095,   0.00229007],
                                                                        [-0.24404067,  0.02109776,  0.00861312]])
                                                        These are:      ['Internal mean', 'Internal stdev', 'Internal stderr',
                                                                        'van der Waals mean', 'van der Waals stdev', 'van der Waals stderr',
                                                                        'Electrostatic mean', 'Electrostatic stdev', 'Electrostatic stderr',
                                                                        'Polar Solvation mean', 'Polar Solvation stdev', 'Polar Solvation stderr',
                                                                        'Non-Polar Solv. mean', 'Non-Polar Solv. stdev', 'Non-Polar Solv. stderr',
                                                                        'TOTAL mean', 'TOTAL stdev', 'TOTAL stderr']
        self.contibution_dict:      # e.g. Y90 -> (-0.24404066666666666, 0.021097757259218076, 0.0086131233336973357, 0.02510132318980977, 0.086451809640543831, 0.035293803493259335)
        """
        
        print("Reading compound", compound, " and decomposition file", decomposition_file)
        with open (decomposition_file, 'r') as f:
            contents = f.readlines()
        
        #headers=contents[7].rstrip().split(",")
        LIG_residue_decompositionArray_dict = {}
                
        # Read and save the backbone DeltaG decomposition
        start = contents.index("S,i,d,e,c,h,a,i,n, ,E,n,e,r,g,y, ,D,e,c,o,dist_matrix,p,o,s,i,t,i,o,n,:\r\n") + 3
        prefix = "sc_"
        for line in contents[start:]:
            components = line.rstrip().split(",")
            if len(components) != 20:
                continue
            # just to populate self.resid2residue_dict
            residue1 = components[0]    # e.g. 'ASP  98'
            residue2 = components[1]    # e.g. 'ASP  98'
            resid1 = residue1.split()[1]
            resid2 = residue2.split()[1]
            self.resid2residue_dict[resid1] = residue1
            self.resid2residue_dict[resid2] = residue2
            if "LIG" in residue1.split()[0] and not "LIG" in residue2.split()[0]:
                self.LIG_resid = components[0].split()[1]
                # print("DEBUG: components=", components)
                LIG_residue_decompositionArray_dict[prefix + residue2] = np.zeros((6,3))
                row = -1
                col = 0
                for c in components[2:]:
                    if col % 3 == 0:
                        row +=1
                        col = 0
                    LIG_residue_decompositionArray_dict[prefix + components[1]][row, col] = float(c)
                    col += 1
            elif "LIG" in residue2.split()[0] and not "LIG" in residue1.split()[0]:
                # print("DEBUG: components=", components)
                self.compound_residue_LIG_decompositionArray_mdict[compound][prefix + residue1] = np.zeros((6,3))
                row = -1
                col = 0
                for c in components[2:]:
                    if col % 3 == 0:
                        row +=1
                        col = 0
                    self.compound_residue_LIG_decompositionArray_mdict[compound][prefix + residue1][row, col] = float(c)
                    col += 1
                    
        # Read and save the side-chain DeltaG decomposition
        start = contents.index("B,a,c,k,b,o,n,e, ,E,n,e,r,g,y, ,D,e,c,o,dist_matrix,p,o,s,i,t,i,o,n,:\r\n") + 3
        prefix = "bb_"
        for line in contents[start:]:
            components = line.rstrip().split(",")
            if len(components) != 20:
                continue
            # just to populate self.resid2residue_dict
            residue1 = components[0]    # e.g. 'ASP  98'
            residue2 = components[1]    # e.g. 'ASP  98'
            resid1 = residue1.split()[1]
            resid2 = residue2.split()[1]
            self.resid2residue_dict[resid1] = residue1
            self.resid2residue_dict[resid2] = residue2
            if "LIG" in residue1.split()[0] and not "LIG" in residue2.split()[0]:
                self.LIG_resid = components[0].split()[1]
                # print("DEBUG: components=", components)
                LIG_residue_decompositionArray_dict[prefix + residue2] = np.zeros((6,3))
                row = -1
                col = 0
                for c in components[2:]:
                    if col % 3 == 0:
                        row +=1
                        col = 0
                    LIG_residue_decompositionArray_dict[prefix + components[1]][row, col] = float(c)
                    col += 1
            elif "LIG" in residue2.split()[0] and not "LIG" in residue1.split()[0]:
                # print("DEBUG: components=", components)
                self.compound_residue_LIG_decompositionArray_mdict[compound][prefix + residue1] = np.zeros((6,3))
                row = -1
                col = 0
                for c in components[2:]:
                    if col % 3 == 0:
                        row +=1
                        col = 0
                    self.compound_residue_LIG_decompositionArray_mdict[compound][prefix + residue1][row, col] = float(c)
                    col += 1
        
        
        # print("DEBUG: LIG_residue_decompositionArray_dict=", LIG_residue_decompositionArray_dict)
        # CALCULATE PERCENT CONTIBUTION OF EACH RESIDUE TO THE TOTAL ENERGY
        residue_TOTAL_std_err_tuple_list = [(r, self.compound_residue_LIG_decompositionArray_mdict[compound][r][5][0], self.compound_residue_LIG_decompositionArray_mdict[compound][r][5][1], self.compound_residue_LIG_decompositionArray_mdict[compound][r][5][2]) for r in list(self.compound_residue_LIG_decompositionArray_mdict[compound].keys())]
        residue_TOTAL_std_err_tuple_list.sort(key=itemgetter(1,3))
        # print(residue_TOTAL_std_err_tuple_list)
        TOTAL_ENERGY = 0
        for q in residue_TOTAL_std_err_tuple_list:
            TOTAL_ENERGY += q[1]
        # print("TOTAT ENERGY=", TOTAL_ENERGY)
        self.residue_DeltaG_dict = {}   # residue -> [total E, stdev, stderr]
        self.residue_DeltaGcontribution_dict = {}   # residue -> [total E contribution, stdev_contr, stderr_contr]
        for q in residue_TOTAL_std_err_tuple_list:
            resname = q[0].split()[0]
            resid = q[0].split()[1]
            total_contr = np.divide(q[1], TOTAL_ENERGY)
            stdev_contr = np.divide(q[2], abs(q[1]))
            stderr_contr = np.divide(q[3], abs(q[1]))
            self.residue_DeltaG_dict[resname + resid] = [q[1], q[2], q[3]]
            self.residue_DeltaGcontribution_dict[resname + resid] = [q[1], q[2], q[3], total_contr, stdev_contr, stderr_contr]
        
        # # APPLY FILTERS TO THE TOTAL CONTRIBUTION AND TOTAL STDEV OF EACH RESIDUE
        # for c in contibution_dict:
        #     if c[4] < TOTAL_CONTRIBUTION_THRESHOLD or c[5] > STDEV_CONTRIBUTION_THRESHOLD:
        #         print("DEBUG: excluding ", c)
        #     else:
        #         print(c)
    
    
    def read_perframe_decomposition_file(self, compound, perframe_decomposition_file, include_skewness=False):
        """
        FUNCTION to read the per-frame MMGBSA energy decomposition file and populate the following dictionary:
        self.perframe_compound_residue_LIG_decompositionArray_mdict:   # e.g. CHEMBL389589 -> 3 -> Y90 ->   [
                                                                                                                [0.0, 1.207, -6.483, -6.768, 7.2159552, -4.8280448],
                                                                                                                [0.0,-0.751,-6.509,-8.141,6.9924096,-8.4085904],
                                                                                                                [0.0,-0.358,-6.418,-7.064,7.1629488,-6.6770512],
                                                                                                                [0.0,0.639,-7.26,-7.322,7.1511264,-6.7918736],
                                                                                                                [0.0,-0.563,-7.883,-6.941,7.2378936000000005,-8.1491064],
                                                                                                                [0.0,-0.128,-7.305,-7.148,7.107314399999999,-7.4736856000000005],
                                                                                                                [0.0,-0.822,-5.277,-7.924,7.1322192,-6.8907808],
                                                                                                                [0.0,-1.075,-6.24,-8.215,7.0330032000000005,-8.496996800000002],
                                                                                                                ]
            The terms in the array are [Internal, van der Waals, Electrostatic, Polar Solvation, Non-Polar Solv., TOTAL], one for each frame.
        """
        self.include_skewness = include_skewness
        
        print("Reading compound", compound, " and per-frame decomposition file", perframe_decomposition_file)
        # print("DEBUG read_perframe_decomposition_file: self.compound_residue_LIG_decompositionArray_mdict[compound]=", self.compound_residue_LIG_decompositionArray_mdict[compound])
        with open (perframe_decomposition_file, 'r') as f:
            contents = f.readlines()
        
        #headers=contents[7].rstrip().split(",")
        LIG_residue_decompositionArray_dict = {}
        
        # Read and save the backbone DeltaG per-frame decomposition
        start = contents.index("DELTA,Backbone Energy Decomposition:\r\n") + 2
        prefix = "bb_"
        for line in contents[start:]:
            components = line.rstrip().split(",")
            if not len(components) == 9:
                break
            [frame, residue1, residue2, Internal, vdW, Elec, Pol_Solv, NonPola_Solv, TOTAL] = line.rstrip().split(',')
            residue1 = prefix + residue1
            residue2 = prefix + residue2
            # print("DEBUG: energy line=", line)
            Internal, vdW, Elec, Pol_Solv, NonPola_Solv, TOTAL = float(Internal), float(vdW), float(Elec), float(Pol_Solv), float(NonPola_Solv), float(TOTAL)
            # The values residue1,residue2 are slightly different than the values residue2,residue1, therefore they will be both stored and averaged at the end.
            if "LIG" in residue2 and not "LIG" in residue1:
                # print("DEBUG: components=", components)
                # print("DEBUG: saving energies of residue1=", residue1)
                if not residue1 in list(self.perframe_compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                    self.perframe_compound_residue_LIG_decompositionArray_mdict[compound][residue1] = [ [Internal, vdW, Elec, Pol_Solv, NonPola_Solv, TOTAL] ]
                else:
                    self.perframe_compound_residue_LIG_decompositionArray_mdict[compound][residue1].append( [Internal, vdW, Elec, Pol_Solv, NonPola_Solv, TOTAL] )
            if "LIG" in residue1 and "LIG" in residue2:
                # print("DEBUG: components=", components)
                # print("DEBUG: saving energies of residue2=", residue2)
                if not residue2 in list(self.perframe_compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                    self.perframe_compound_residue_LIG_decompositionArray_mdict[compound][residue2] = [ [Internal, vdW, Elec, Pol_Solv, NonPola_Solv, TOTAL] ]
                else:
                    self.perframe_compound_residue_LIG_decompositionArray_mdict[compound][residue2].append( [Internal, vdW, Elec, Pol_Solv, NonPola_Solv, TOTAL] )
        
        # Read and save the side-chain DeltaG per-frame decomposition
        start = contents.index("DELTA,Sidechain Energy Decomposition:\r\n") + 2
        prefix = "sc_"
        for line in contents[start:]:
            components = line.rstrip().split(",")
            if not len(components) == 9:
                break
            [frame, residue1, residue2, Internal, vdW, Elec, Pol_Solv, NonPola_Solv, TOTAL] = line.rstrip().split(',')
            residue1 = prefix + residue1
            residue2 = prefix + residue2
            # print("DEBUG: energy line=", line)
            Internal, vdW, Elec, Pol_Solv, NonPola_Solv, TOTAL = float(Internal), float(vdW), float(Elec), float(Pol_Solv), float(NonPola_Solv), float(TOTAL)
            # The values residue1,residue2 are slightly different than the values residue2,residue1, therefore they will be both stored and averaged at the end.
            if "LIG" in residue2 and not "LIG" in residue1:
                # print("DEBUG: components=", components)
                # print("DEBUG: saving energies of residue1=", residue1)
                if not residue1 in list(self.perframe_compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                    self.perframe_compound_residue_LIG_decompositionArray_mdict[compound][residue1] = [ [Internal, vdW, Elec, Pol_Solv, NonPola_Solv, TOTAL] ]
                else:
                    self.perframe_compound_residue_LIG_decompositionArray_mdict[compound][residue1].append( [Internal, vdW, Elec, Pol_Solv, NonPola_Solv, TOTAL] )
            if "LIG" in residue1 and not "LIG" in residue2:
                # print("DEBUG: components=", components)
                # print("DEBUG: saving energies of residue2=", residue2)
                if not residue2 in list(self.perframe_compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                    self.perframe_compound_residue_LIG_decompositionArray_mdict[compound][residue2] = [ [Internal, vdW, Elec, Pol_Solv, NonPola_Solv, TOTAL] ]
                else:
                    self.perframe_compound_residue_LIG_decompositionArray_mdict[compound][residue2].append( [Internal, vdW, Elec, Pol_Solv, NonPola_Solv, TOTAL] )
        
        if self.include_skewness == True:
            # Calculate the mean, stdev, stderr and skewness
            # print("DEBUG: point 1 self.perframe_compound_residue_LIG_decompositionArray_mdict[compound]=", self.perframe_compound_residue_LIG_decompositionArray_mdict[compound])
            for residue in list(self.perframe_compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                self.compound_residue_LIG_decompositionArray_mdict[compound][residue] = []  # The residue in this case is just the resid number
                # print("DEBUG: residue=", residue, "self.perframe_compound_residue_LIG_decompositionArray_mdict[compound][residue]=", self.perframe_compound_residue_LIG_decompositionArray_mdict[compound][residue])
                for mean,stdev,stderr,skewness in zip(np.mean(self.perframe_compound_residue_LIG_decompositionArray_mdict[compound][residue], axis=0).tolist(),
                                                        np.std(self.perframe_compound_residue_LIG_decompositionArray_mdict[compound][residue], axis=0).tolist(),
                                                        stats.sem(self.perframe_compound_residue_LIG_decompositionArray_mdict[compound][residue], axis=0).tolist(),
                                                        stats.skew(self.perframe_compound_residue_LIG_decompositionArray_mdict[compound][residue], axis=0).tolist()):
                    # print("DEBUG: appending ", [mean, stdev, stderr, skewness], " to residue ", residue, " and compound ", compound)
                    self.compound_residue_LIG_decompositionArray_mdict[compound][residue].append( [mean, stdev, stderr, skewness] )
                    # print("DEBUG: point 1.5 self.compound_residue_LIG_decompositionArray_mdict[compound][residue]=", self.compound_residue_LIG_decompositionArray_mdict[compound][residue])
                self.compound_residue_LIG_decompositionArray_mdict[compound][residue] = np.array(self.compound_residue_LIG_decompositionArray_mdict[compound][residue])
            # print("DEBUG: point 2 self.compound_residue_LIG_decompositionArray_mdict[compound]", self.compound_residue_LIG_decompositionArray_mdict[compound])
    
    
    def get_MMGBSA_fp_term_decomposition(self, compound_list, scale=True, only_terms=True, get_reslist=False, binnedDeltaG=False,
                                                                                            include_skewness=None):
        """
            FUNCTION to create and return an array with the MMGBSA fingerprints of all the compounds in compound_list. The fingerprint(will
            contain the values of the terms 'van der Waals mean', 'van der Waals stdev', 'Electrostatic mean', 'Electrostatic stdev',
            'Polar Solvation mean', 'Polar Solvation stdev', 'Non-Polar Solv. mean', 'Non-Polar Solv. stdev' and optionally  the 'Total DeltaG'
            and 'Total DeltaG stdev'.
            
            ARGS:
            only_terms: if True then only 'van der Waals mean', 'van der Waals stdev', 'Electrostatic mean', 'Electrostatic stdev',
                         'Polar Solvation mean', 'Polar Solvation stdev', 'Non-Polar Solv. mean', 'Non-Polar Solv. stdev' are returned.
                        if False then 'Internal mean', 'Internal stdev' are included, too.
            scale:      if True then the actual values and the stdev will be scaled separately.
            skewness:   if True, then include the 3rd distribution moment (skewness) along with the average value of each term and its stdev.

        """
            
        residue_set = set()
        # print("DEBUG: self.compound_residue_LIG_decompositionArray_mdict=", self.compound_residue_LIG_decompositionArray_mdict)
        for compound in list(self.compound_residue_LIG_decompositionArray_mdict.keys()):
            for residue in list(self.compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                residue_set.add(residue)
        residue_list  = list(residue_set)   # all the residues that have interactions in all pocket files
        # print("DEBUG: residue_list=", residue_list)
        
        MMGBSA_fp_list = []    # initialize it as a list to append the fingerprints and the convert it to array
        for compound in compound_list:
            # print(compound)
            # print(self.compound_residue_LIG_decompositionArray_mdict[compound].keys())
            if not residue_list[0] in list(self.compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                if only_terms == True and self.include_skewness == False:
                    MMGBSA_fp = np.zeros(4*2)
                elif only_terms == False and self.include_skewness == False:
                    MMGBSA_fp = np.zeros(5*2)
                elif only_terms == True and self.include_skewness == True:
                    MMGBSA_fp = np.zeros(4*3)
                elif only_terms == False and self.include_skewness == True:
                    MMGBSA_fp = np.zeros(5*3)
            else:
                if only_terms == True and self.include_skewness == False:
                    MMGBSA_fp = self.compound_residue_LIG_decompositionArray_mdict[compound][residue_list[0]][1:5,0:2].ravel() # the full MMGBSA fingeprint(for this compound
                elif only_terms == False and self.include_skewness == False:
                    MMGBSA_fp = self.compound_residue_LIG_decompositionArray_mdict[compound][residue_list[0]][:5,0:2].ravel() # the full MMGBSA fingeprint(for this compound
                elif only_terms == True and self.include_skewness == True:
                    # include the 4th column in each line, which is the skewness
                    MMGBSA_fp = self.compound_residue_LIG_decompositionArray_mdict[compound][residue_list[0]][1:5,[0,1,3]].ravel() # the full MMGBSA fingeprint(for this compound
                elif only_terms == False and self.include_skewness == True:
                    # include the 4th column in each line, which is the skewness
                    MMGBSA_fp = self.compound_residue_LIG_decompositionArray_mdict[compound][residue_list[0]][:5,[0,1,3]].ravel() # the full MMGBSA fingeprint(for this compound
            
            for residue in residue_list[1:]:
                if not residue in list(self.compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                    if only_terms == True and self.include_skewness == False:
                        a6 = np.zeros(4*2)
                    elif only_terms == False and self.include_skewness == False:
                        a6 = np.zeros(5*2)
                    elif only_terms == True and self.include_skewness == True:
                        a6 = np.zeros(4*3)
                    elif only_terms == False and self.include_skewness == True:
                        a6 = np.zeros(5*3)
                else:
                    if only_terms == True and self.include_skewness == False:
                        a6 = self.compound_residue_LIG_decompositionArray_mdict[compound][residue][1:5,0:2] # ommit the 1st and 6th rows which are the Internal and Total DeltaG
                    elif only_terms == False and self.include_skewness == False:
                        a6 = self.compound_residue_LIG_decompositionArray_mdict[compound][residue][:5,0:2]   # ommit the 6th row which is the Total DeltaG
                    elif only_terms == True and self.include_skewness == True:
                        a6 = self.compound_residue_LIG_decompositionArray_mdict[compound][residue][1:5,[0,1,3]] # ommit the 1st and 6th rows which are the Internal and Total DeltaG
                    elif only_terms == False and self.include_skewness == True:
                        # print("DEBUG: residue=", residue, "self.compound_residue_LIG_decompositionArray_mdict[compound][residue]=", self.compound_residue_LIG_decompositionArray_mdict[compound][residue].tolist())
                        a6 = self.compound_residue_LIG_decompositionArray_mdict[compound][residue][:5,[0,1,3]]   # ommit the 6th row which is the Total DeltaG
                MMGBSA_fp = np.append(MMGBSA_fp , a6.ravel())
            MMGBSA_fp_list.append(MMGBSA_fp)
        
        if binnedDeltaG and not self.include_skewness:
            MMGBSA_fp_list = self.get_binnedDeltaG(MMGBSA_fp_list)   # MMGBSA_fp_list now contrains only binned DeltaG, no stdev
        
        residue_list = [r.split()[0]+r.split()[1] for r in residue_list]    # make the format compatible with SiFt
        
        if scale == True:
            MMGBSA_fp_array = np.array(MMGBSA_fp_list, dtype=float)
            
            # ## scale separately the means and the stdevs
            # Xmax1 = np.max(MMGBSA_fp_array[:, 0::2])
            # Xmin1 = np.min(MMGBSA_fp_array[:, 0::2])
            # Xmax2 = np.max(MMGBSA_fp_array[:, 1::2])
            # Xmin2 = np.min(MMGBSA_fp_array[:, 1::2])
            # MMGBSA_fp_array[:, 0::2] = (MMGBSA_fp_array[:, 0::2] - Xmin1) / float(Xmax1 - Xmin1)
            # MMGBSA_fp_array[:, 1::2] = (MMGBSA_fp_array[:, 1::2] - Xmin2) / float(Xmax2 - Xmin2)
            # shift1 = (0.0 - Xmin1) / float(Xmax1 - Xmin1)
            # shift2 = (0.0 - Xmin2) / float(Xmax2 - Xmin2)
            # MMGBSA_fp_array = MMGBSA_fp_array[:, 0::2] - shift1
            # MMGBSA_fp_array = MMGBSA_fp_array[:, 1::2] - shift2
            
            ## alternativelly scale them all together (here I assume that there is at least one negative value)
            Xmax = np.max(MMGBSA_fp_array)
            Xmin = np.min(MMGBSA_fp_array)
            MMGBSA_fp_array = (MMGBSA_fp_array - Xmin) / float(Xmax - Xmin)
            shift = (0.0 - Xmin) / float(Xmax - Xmin)
            MMGBSA_fp_array = MMGBSA_fp_array - shift
            multiplier = -1.0/np.min(MMGBSA_fp_array)   # assuming that there is at least one negative value!
            MMGBSA_fp_array = multiplier * MMGBSA_fp_array
            
            if self.include_skewness:
                Xmax3 = np.max(MMGBSA_fp_array[:, 2::2])
                Xmin3 = np.min(MMGBSA_fp_array[:, 2::2])
                MMGBSA_fp_array[:, 2::2] = (MMGBSA_fp_array[:, 2::2] - Xmin3) / float(Xmax3 - Xmin3)
            if get_reslist:
                return list(MMGBSA_fp_array), residue_list
            else:
                return list(MMGBSA_fp_array)
        
        if get_reslist:
            return MMGBSA_fp_list, residue_list
        else:
            return MMGBSA_fp_list
        # return np.array(MMGBSA_fp_list, dtype=float)
    
    
    def get_full_MMGBSA_fp_list(self, compound_list, scale=True):
        """
            FUNCTION to create and return an array with the full MMGBSA fingerprints of all the compounds in compound_list.
        """
        # compound_list  = self.compound_residue_LIG_decompositionArray_mdict.keys()
        residue_list  = list(self.compound_residue_LIG_decompositionArray_mdict[compound_list[0]].keys())
        MMGBSA_fp_list = []    # initialize it as a list to append the fingerprints and the convert it to array
        compounds2remove_set = set()
        for compound in compound_list:
            # print(compound)
            # print(self.compound_residue_LIG_decompositionArray_mdict[compound].keys())
            if not residue_list[0] in list(self.compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                compounds2remove_set.add(compound)
                MMGBSA_fp = np.zeros(18)
            else:
                MMGBSA_fp = self.compound_residue_LIG_decompositionArray_mdict[compound][residue_list[0]].ravel() # the full MMGBSA fingeprint(for this compound
            for residue in residue_list[1:]:
                if not residue in list(self.compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                    compounds2remove_set.add(compound)
                    a6x3 = np.zeros(18)
                else:
                    a6x3 = self.compound_residue_LIG_decompositionArray_mdict[compound][residue]
                MMGBSA_fp = np.append(MMGBSA_fp , a6x3.ravel())
            MMGBSA_fp_list.append(MMGBSA_fp)
        
        print("DDEBUG: compounds2remove_set=", compounds2remove_set)
        
        if scale == True:
            MMGBSA_fp_array = np.array(MMGBSA_fp_list, dtype=float)
            Xmax1 = np.max(MMGBSA_fp_array[:, 0::3])
            Xmin1 = np.min(MMGBSA_fp_array[:, 0::3])
            Xmax2 = np.max(MMGBSA_fp_array[:, 1::3])
            Xmin2 = np.min(MMGBSA_fp_array[:, 1::3])
            Xmax3 = np.max(MMGBSA_fp_array[:, 2::3])
            Xmin3 = np.min(MMGBSA_fp_array[:, 2::3])
            MMGBSA_fp_array[:, 0::3] = (MMGBSA_fp_array[:, 0::3] - Xmin1) / float(Xmax1 - Xmin1)
            MMGBSA_fp_array[:, 1::3] = (MMGBSA_fp_array[:, 1::3] - Xmin2) / float(Xmax2 - Xmin2)
            MMGBSA_fp_array[:, 2::3] = (MMGBSA_fp_array[:, 2::3] - Xmin3) / float(Xmax3 - Xmin3)
            return list(MMGBSA_fp_array), compounds2remove_set
        
        return MMGBSA_fp_list, compounds2remove_set
        # return np.array(MMGBSA_fp_list, dtype=float)
    
    
    def get_MMGBSA_fp_list(self, compound_list, scale=True):
        """
            FUNCTION to create and return an array with the MMGBSA fingerprints (E only) of all the compounds in compound_list.
        """
        # compound_list  = self.compound_residue_LIG_decompositionArray_mdict.keys()
        residue_list  = list(self.compound_residue_LIG_decompositionArray_mdict[compound_list[0]].keys())
        MMGBSA_fp_list = []    # initialize it as a list to append the fingerprints and the convert it to array
        for compound in compound_list:
            # print(compound)
            # print(self.compound_residue_LIG_decompositionArray_mdict[compound].keys())
            if not residue_list[0] in list(self.compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                MMGBSA_fp = np.zeros(6)
            else:
                MMGBSA_fp = self.compound_residue_LIG_decompositionArray_mdict[compound][residue_list[0]][:,0].ravel() # the full MMGBSA fingeprint(for this compound
            for residue in residue_list[1:]:
                if not residue in list(self.compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                    a6 = np.zeros(6)
                else:
                    a6 = self.compound_residue_LIG_decompositionArray_mdict[compound][residue][:,0]
                MMGBSA_fp = np.append(MMGBSA_fp , a6.ravel())
            MMGBSA_fp_list.append(MMGBSA_fp)
        
        if scale == True:
            MMGBSA_fp_array = np.array(MMGBSA_fp_list, dtype=float)
            Xmax = np.max(MMGBSA_fp_array)
            Xmin = np.min(MMGBSA_fp_array)
            X_scaled = (MMGBSA_fp_array - Xmin) / float(Xmax - Xmin)
            return list(X_scaled)
        
        return MMGBSA_fp_list
        # return np.array(MMGBSA_fp_list, dtype=float)
    
    
    def fillin_missing_residues(self):
        """
            Function to make all compounds to have the same set of residues in self.compound_residue_LIG_decompositionArray_mdict. For those that did
            not exist an 6x3 array of zeros will be added.
            
        """
        residue_set = set()
        for compound in list(self.compound_residue_LIG_decompositionArray_mdict.keys()):
            for residue in list(self.compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                residue_set.add(residue)
        
        for compound in list(self.compound_residue_LIG_decompositionArray_mdict.keys()):
            for residue in residue_set:
                if not residue in list(self.compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                    self.compound_residue_LIG_decompositionArray_mdict[compound][residue] = np.zeros((6,3))
                
    
    def get_binnedDeltaG(self, MMGBSA_fp_list):
        """
            It is assumed here that MMGBSA_fp_list contains DeltaG+stdev or DeltaG+stderr. Then the DeltaG is placed into bins with size that of the
            largest stdev or stderr. In this case the noise in the measurements is reduced.
            
            RETURNS:
            binned_mmgbsa_fp_list:  a list of arrays with the binned DeltaG contribution of each residue (without the stdev).
        """
        
        max_stdev = 0.0
        for mmgbsa_fp in MMGBSA_fp_list:
            for stdev in mmgbsa_fp[1::2]:   # it works also for stderr
                if stdev > max_stdev:
                    max_stdev = stdev
        
        binned_mmgbsa_fp_list = []
        for mmgbsa_fp in MMGBSA_fp_list:
            binned_mmgbsa_fp = np.round(mmgbsa_fp[0::2]/max_stdev)
            binned_mmgbsa_fp_list.append(binned_mmgbsa_fp)
            
        return binned_mmgbsa_fp_list
        
        
    def get_DeltaG_fp_list(self, compound_list, scale=True, stdev=True, stderr=False, get_reslist=False, binnedDeltaG=False):
        """
            FUNCTION to create and return an array with the MMGBSA fingerprints of all the compounds in compound_list. The fingerprints will contain
            the DeltaG contribution of each residue to the total DeltaG, and optionally the stdev and/or the stderr. The function takes care to include
            for each compound all the residues that have occured in the energy decomposition files. Those that are absent from a compound, are fill in
            with zeros. Consequently, at the end the mmgbsa_fg of each compound will have the same length
            
            ARGS:
            scale:      if True then the actual values and the stdev will be scaled separately.

        """
        
        # self.fillin_missing_residues()  # function to add for every compound, arrays of zeros in place of residues that were too far to be analyzed 
        if binnedDeltaG:    # we need the stdev to bin the energy!
            stdev=True
            stderr=False
        
        # compound_list  = self.compound_residue_LIG_decompositionArray_mdict.keys()
        residue_set = set()
        for compound in list(self.compound_residue_LIG_decompositionArray_mdict.keys()):
            for residue in list(self.compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                residue_set.add(residue)
        residue_list  = list(residue_set)   # all the residues that have interactions in all pocket files
        MMGBSA_fp_list = []    # initialize it as a list to append the fingerprints and the convert it to array
        for compound in compound_list:
            # print(compound)
            # print(self.compound_residue_LIG_decompositionArray_mdict[compound].keys())
            if not residue_list[0] in list(self.compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                if (stdev == True and stderr == False) or (stdev == False and stderr == True):
                    MMGBSA_fp = np.zeros(2)
                elif stdev == True and stderr == True:
                    MMGBSA_fp = np.zeros(3)
                elif stdev == False and stderr == False:
                    MMGBSA_fp = np.zeros(1)
            else:
                if stdev == True and stderr == False:
                    MMGBSA_fp = self.compound_residue_LIG_decompositionArray_mdict[compound][residue_list[0]][5,0:2].ravel()
                elif stdev == False and stderr == True:
                    MMGBSA_fp = self.compound_residue_LIG_decompositionArray_mdict[compound][residue_list[0]][5,0::2].ravel()
                elif stdev == True and stderr == True:
                    MMGBSA_fp = self.compound_residue_LIG_decompositionArray_mdict[compound][residue_list[0]][5,:].ravel()
                elif stdev == False and stderr == False:
                    MMGBSA_fp = self.compound_residue_LIG_decompositionArray_mdict[compound][residue_list[0]][5,0].ravel()
            for residue in residue_list[1:]:
                if not residue in list(self.compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                    if (stdev == True and stderr == False) or (stdev == False and stderr == True):
                        a1 = np.zeros(2)
                    elif stdev == True and stderr == True:
                        a1 = np.zeros(3)
                    elif stdev == False and stderr == False:
                        a1 = np.zeros(1)
                else:
                    if stdev == True and stderr == False:
                        a1 = self.compound_residue_LIG_decompositionArray_mdict[compound][residue][5,0:2].ravel()   # 1st & 2nd element
                    elif stdev == False and stderr == True:
                        a1 = self.compound_residue_LIG_decompositionArray_mdict[compound][residue][5,0::2].ravel()  # 1st & 3rd element
                    elif stdev == True and stderr == True:
                        a1 = self.compound_residue_LIG_decompositionArray_mdict[compound][residue][5,:].ravel() # all 3 elements
                    elif stdev == False and stderr == False:
                        a1 = self.compound_residue_LIG_decompositionArray_mdict[compound][residue][5,0].ravel() # 1st element
                MMGBSA_fp = np.append(MMGBSA_fp , a1.ravel())
            # MMGBSA_fp[0::2] = -1.0*MMGBSA_fp[0::2]  # change the singe of the DeltaG (NOTHING CHANGES IN THE PRERFORMANCE!)
            # print("DEBUG: append MMGBSA_fp=", MMGBSA_fp.tolist())
            MMGBSA_fp_list.append(MMGBSA_fp)
        
        if binnedDeltaG:
            MMGBSA_fp_list = self.get_binnedDeltaG(MMGBSA_fp_list)   # MMGBSA_fp_list now contrains only binned DeltaG, no stdev
        
        residue_list = [r.split()[0]+r.split()[1] for r in residue_list]    # make the format compatible with SiFt
        
        if scale == True:
            MMGBSA_fp_array = np.array(MMGBSA_fp_list, dtype=float)
            if (stdev == True and stderr == False) or (stdev == False and stderr == True):
                Xmax1 = np.max(MMGBSA_fp_array[:, 0::2])
                Xmin1 = np.min(MMGBSA_fp_array[:, 0::2])
                Xmax2 = np.max(MMGBSA_fp_array[:, 1::2])
                Xmin2 = np.min(MMGBSA_fp_array[:, 1::2])
                MMGBSA_fp_array[:, 0::2] = (MMGBSA_fp_array[:, 0::2] - Xmin1) / float(Xmax1 - Xmin1)
                MMGBSA_fp_array[:, 1::2] = (MMGBSA_fp_array[:, 1::2] - Xmin2) / float(Xmax2 - Xmin2)
            elif stdev == True and stderr == True:
                Xmax1 = np.max(MMGBSA_fp_array[:, 0::3])
                Xmin1 = np.min(MMGBSA_fp_array[:, 0::3])
                Xmax2 = np.max(MMGBSA_fp_array[:, 1::3])
                Xmin2 = np.min(MMGBSA_fp_array[:, 1::3])
                Xmax3 = np.max(MMGBSA_fp_array[:, 2::3])
                Xmin3 = np.min(MMGBSA_fp_array[:, 2::3])
                MMGBSA_fp_array[:, 0::3] = (MMGBSA_fp_array[:, 0::3] - Xmin1) / float(Xmax1 - Xmin1)
                MMGBSA_fp_array[:, 1::3] = (MMGBSA_fp_array[:, 1::3] - Xmin2) / float(Xmax2 - Xmin2)
                MMGBSA_fp_array[:, 2::3] = (MMGBSA_fp_array[:, 2::3] - Xmin3) / float(Xmax3 - Xmin3)
            elif stdev == False and stderr == False:
                Xmax = np.max(MMGBSA_fp_array)
                Xmin = np.min(MMGBSA_fp_array)
                MMGBSA_fp_array = (MMGBSA_fp_array - Xmin) / float(Xmax - Xmin)
            if get_reslist:
                return list(MMGBSA_fp_array), residue_list
            else:
                return list(MMGBSA_fp_array)
        
        if get_reslist:
            return MMGBSA_fp_list, residue_list
        else:
            return MMGBSA_fp_list
        # return np.array(MMGBSA_fp_list, dtype=float)
    
    
    def get_MMGBSA_fp_list3(self, compound_list, scale=True):
        """
            FUNCTION to create and return an array with the MMGBSA fingerprints (E contribution and stderr) of all the compounds in compound_list.
        """
        # compound_list  = self.compound_residue_LIG_decompositionArray_mdict.keys()
        residue_list  = list(self.compound_residue_LIG_decompositionArray_mdict[compound_list[0]].keys())
        MMGBSA_fp_list = []    # initialize it as a list to append the fingerprints and the convert it to array
        for compound in compound_list:
            # print(compound)
            # print(self.compound_residue_LIG_decompositionArray_mdict[compound].keys())
            if not residue_list[0] in list(self.compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                MMGBSA_fp = np.zeros(12)
            else:
                MMGBSA_fp = self.compound_residue_LIG_decompositionArray_mdict[compound][residue_list[0]][:,0::2].ravel() # the full MMGBSA fingeprint(for this compound
            for residue in residue_list[1:]:
                if not residue in list(self.compound_residue_LIG_decompositionArray_mdict[compound].keys()):
                    a6x2 = np.zeros(12)
                else:
                    a6x2 = self.compound_residue_LIG_decompositionArray_mdict[compound][residue][:,0::2]
                MMGBSA_fp = np.append(MMGBSA_fp , a6x2.ravel())
            MMGBSA_fp_list.append(MMGBSA_fp)
        
        if scale == True:
            MMGBSA_fp_array = np.array(MMGBSA_fp_list, dtype=float)
            Xmax1 = np.max(MMGBSA_fp_array[:, 0::2])
            Xmin1 = np.min(MMGBSA_fp_array[:, 0::2])
            Xmax2 = np.max(MMGBSA_fp_array[:, 1::2])
            Xmin2 = np.min(MMGBSA_fp_array[:, 1::2])
            MMGBSA_fp_array[:, 0::2] = (MMGBSA_fp_array[:, 0::2] - Xmin1) / float(Xmax1 - Xmin1)
            MMGBSA_fp_array[:, 1::2] = (MMGBSA_fp_array[:, 1::2] - Xmin2) / float(Xmax2 - Xmin2)
            return list(MMGBSA_fp_array)
        
        return MMGBSA_fp_list
        # return np.array(MMGBSA_fp_list, dtype=float)
        
    def remove_compounds(self, compound_list, compounds2remove_set, dataset, y):
        """
            FUNCTION to remove data of compounds that could not be Energy-decomposed completely.
        """
        new_compound_list = []
        new_dataset = []
        new_y = []
        for c,xi,yi in zip(compound_list, dataset, y):
            if not c in compounds2remove_set:
                new_compound_list.append(c)
                new_dataset.append(xi)
                new_y.append(yi)
        
        return new_compound_list, new_dataset, new_y
    