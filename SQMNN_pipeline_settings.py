class Settings:

    def __init__(self):

        self.HYPER_SQMNN_ROOT_DIR = '/home2/shared_files/sqm-ml'
        self.HYPER_SQM_FOLDER_SUFFIX = '_SQM_MM'
        self.HYPER_EXECUTION_DIR_NAME = 'execution_dir'
        self.HYPER_PROTEIN = 'MK2'
        self.ALL_PROTEINS = [self.HYPER_PROTEIN]    # not hyper-param; just for file naming
        self.HYPER_FORCE_COMPUTATION = False

        # *****
        # TUNABLE HYPER-PARAMETERS

        # LEARNING MODEL
        # --
        self.HYPER_LEARNING_MODEL_TYPE = 'Random Forest'
        self.SAMPLE_WEIGHTS_TYPE = 'featvec_similarity'
        # --

        # POSE FILTERING
        # --
        self.HYPER_RATIO_SCORED_POSES = 0.8
        self.HYPER_REMOVE_OUTLIER_WRT_WHOLE_SET = False
        self.HYPER_OUTLIER_MAD_THRESHOLD = 8.0
        self.HYPER_OUTLIER_MIN_SCORED_POSE_NUM = 0  # keep only structvars with more than this number of scored poses
        self.HYPER_KEEP_MAX_N_POSES = 100  # keep at maximum this number of Glide poses per structvar for SQM scoring
        self.HYPER_KEEP_MAX_DeltaG_POSES = 5.0  # keep per structvar at maximum Glide poses with this energy difference
        # (kcal/mol) from the top scored for SQM scoring
        self.HYPER_ABSOLUTE_MAX_DeltaG = True   # if True then only compounds and poses within HYPER_KEEP_MAX_DeltaG_POSES
        # window from the lowest scores compound/pose will be retained for SQM scoring. Many compounds will be discarded!
        self.HYPER_KEEP_POSE_COLUMN = 'r_i_docking_score'  # use this column to keep the best Glide poses for SQM scoring
        self.HYPER_CONFORMER_ENERGY_CUTOFF_LIST = [6.0, 9.0,
                                                   12.0]  # kcal/mol (according to Chan et al. 'Understanding conformational entropy in small molecules', 2020
        self.HYPER_CONFORMER_RMSD_CUTOFF_LIST = [1.0, 1.5, 2.0]
        self.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY = 'Eint'  # OPTIONS: 'DeltaH', 'DeltaHstar', 'Eint', 'r_i_docking_score, etc.
        self.HYPER_SELECT_BEST_BASEMOLNAME_POSE_BY = 'Eint'  # OPTIONS: 'DeltaH', 'DeltaHstar', 'Eint', 'r_i_docking_score', etc.
        self.HYPER_SELECT_BEST_STRUCTVAR_POSE_BY = 'complexE'  # OPTIONS: 'DeltaH', 'DeltaHstar', 'Eint', 'complexE', 'r_i_docking_score', etc.
        self.HYPER_SFs = ['P6C']  # OPTIONS: 'P6C', 'P6C2', 'P7C'
        self.HYPER_HOW_COMBINE = 'inner'  # OPTIONS: 'inner' or 'outer'
        self.HYPER_FUSION_METHOD = 'nofusion'  # OPTIONS: 'nofusion', 'minrank', 'meanrank', 'geomean', 'harmmean'
        self.HYPER_USE_SCONF = False
        self.HYPER_SCONF_RMSD = 1.0
        self.HYPER_SCONF_ECUTOFF = 6
        self.HYPER_REPORT_PROTEIN_ENERGY_FLUCTUATIONS = True
        # --

        # *****
        # FEATURES FOR TRAINING

        # SQM ENERGY TERMS
        self.HYPER_SQM_FEATURES = ['%s_%s' % (self.HYPER_FUSION_METHOD, self.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY)] + \
                                  ['%s_complexE' % SF for SF in self.HYPER_SFs] + \
                                  ['%s_ligandE_bound' % SF for SF in self.HYPER_SFs] + \
                                  ['%s_min_ligandE_bound' % SF for SF in self.HYPER_SFs] + \
                                  ['%s_proteinE_bound' % SF for SF in self.HYPER_SFs]
                                  # ['P6C_complexE_mean_structvar_stdev',
                                  #  'P6C_proteinE_bound_mean_structvar_stdev',
                                  #  'P6C_ligandE_bound_mean_structvar_stdev',
                                  #  'P6C_complexE_best_basemolname_stdev',
                                  #  'P6C_proteinE_bound_best_basemolname_stdev',
                                  #  'P6C_ligandE_bound_best_basemolname_stdev',
                                  #  'P6C_complexE_overall_basemolname_stdev',
                                  #  'P6C_proteinE_bound_overall_basemolname_stdev',
                                  #  'P6C_ligandE_bound_overall_basemolname_stdev']
        # --

        # Glide DESCRIPTORS COMPLEMENTARY TO SQM ENERGY TERMS
        # http://mgcf.cchem.berkeley.edu/mgcf/schrodinger_2020-2/glide_user_manual/glide_docking_properties.htm
        self.HYPER_GLIDE_DESCRIPTORS = []
        # ["r_epik_State_Penalty", "r_i_glide_lipo", 'r_i_glide_hbond',
        #                                 'r_i_glide_metal', 'r_i_glide_rewards',
        #                                 'r_i_glide_evdw', 'r_i_glide_ecoul', 'r_i_glide_erotb',
        #                                 'r_i_glide_esite', 'r_i_glide_eff_state_penalty']
        # --

        # 2D DESCRIPTORS
        self.HYPER_2D_DESCRIPTORS = ['MW', 'AMW', 'deepFl_logP',
                                     # '0_contiguous_rotbonds',
                                     # '1_contiguous_rotbonds', '2_contiguous_rotbonds', '3_contiguous_rotbonds',
                                     # '4_contiguous_rotbonds', '5_contiguous_rotbonds', '6_contiguous_rotbonds',
                                     # '7_contiguous_rotbonds', '8_contiguous_rotbonds', '9_contiguous_rotbonds',
                                     # '10_contiguous_rotbonds', '11_contiguous_rotbonds', '12_contiguous_rotbonds',
                                     # '13_contiguous_rotbonds', '14_contiguous_rotbonds', '15_contiguous_rotbonds',
                                     # '16_contiguous_rotbonds', '17_contiguous_rotbonds', '18_contiguous_rotbonds',
                                     # '19_contiguous_rotbonds', '20_contiguous_rotbonds', '21_contiguous_rotbonds',
                                     # '22_contiguous_rotbonds', '23_contiguous_rotbonds', '24_contiguous_rotbonds',
                                     # '25_contiguous_rotbonds', '26_contiguous_rotbonds', '27_contiguous_rotbonds',
                                     # '28_contiguous_rotbonds', '29_contiguous_rotbonds', '30_contiguous_rotbonds',
                                     # '31_contiguous_rotbonds', '32_contiguous_rotbonds', '33_contiguous_rotbonds',
                                     # '34_contiguous_rotbonds', '35_contiguous_rotbonds', '36_contiguous_rotbonds',
                                     # '37_contiguous_rotbonds', '38_contiguous_rotbonds', '39_contiguous_rotbonds',
                                     # '40_contiguous_rotbonds', '41_contiguous_rotbonds', '42_contiguous_rotbonds',
                                     # '43_contiguous_rotbonds', '44_contiguous_rotbonds', '45_contiguous_rotbonds',
                                     # '46_contiguous_rotbonds', '47_contiguous_rotbonds', '48_contiguous_rotbonds',
                                     # '49_contiguous_rotbonds', '50_contiguous_rotbonds',
                                     'rotor_count', 'terminal_CH3_count', 'function_group_count',
                                     'ring_flexibility', 'Hbond_foldability', 'pipi_stacking_foldability',
                                     'bondType_UNSPECIFIED', 'bondType_SINGLE', 'bondType_DOUBLE', 'bondType_TRIPLE',
                                     'bondType_QUADRUPLE', 'bondType_QUINTUPLE', 'bondType_HEXTUPLE',
                                     'bondType_ONEANDAHALF', 'bondType_TWOANDAHALF', 'bondType_THREEANDAHALF',
                                     'bondType_FOURANDAHALF', 'bondType_FIVEANDAHALF', 'bondType_AROMATIC',
                                     'bondType_IONIC', 'bondType_HYDROGEN', 'bondType_THREECENTER',
                                     'bondType_DATIVEONE', 'bondType_DATIVE', 'bondType_DATIVEL', 'bondType_DATIVER',
                                     'bondType_OTHER', 'bondType_ZERO']
        # --

        # 3D LIGAND DESCRIPTORS
        self.HYPER_3D_LIGAND_DESCRIPTORS = []
         #    ['Asphericity', 'Eccentricity', 'InertialShapeFactor', 'NPR1', 'NPR2', 'PMI1',
         # 'PMI2', 'PMI3', 'RadiusOfGyration', 'SpherocityIndex']
        # --

        # 3D LIGAND CONFORMATIONAL ENTROPY DESCRIPTORS
        self.HYPER_LIGAND_Sconf_DESCRIPTORS = []
        # ['DeltaG_0to1', 'DeltaG_1to2', 'DeltaG_2to3', 'DeltaG_3to4',
        #                                        'DeltaG_4to5', 'DeltaG_5to6']
        # --

        # 3D DESCRIPTORS
        self.HYPER_3D_COMPLEX_DESCRIPTORS = ['prot_interface_surf', 'prot_interface_SASA', 'lig_interface_surf',
                                     'lig_interface_SASA', 'mean_interface_surf', 'mean_interface_SASA', 'net_charge']
        # --

        # FINGERPRINTS

        # PLEC
        self.HYPER_PLEC = True
        self.HYPER_PLEC_PCA_VARIANCE_EXPLAINED_CUTOFF = 20 # 0.2
        self.HYPER_COMPRESS_PLEC = True
        self.HYPER_COMPRESS_UMP = False
        # --

        # *****

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def generated_file(self, file, protein=None):
        if not protein:
            protein = self.HYPER_PROTEIN
        return self.HYPER_SQMNN_ROOT_DIR + '/' + self.HYPER_EXECUTION_DIR_NAME + '/' + protein + file

    def raw_input_file(self, file, protein=None):
        if not protein:
            protein = self.HYPER_PROTEIN
        return self.HYPER_SQMNN_ROOT_DIR + '/' + protein + '/' + protein + file

    def create_feature_csv_name(self):
        return '_features.SF_%s.SBBSB_%s.SBSPB_%s.HC_%s.FM_%s.RSP_%s.' \
               'OMT_%s.OMSPN_%i.KMNP_%i.KMDP_%f.US_%s.csv.gz' % \
               ('_'.join(self.HYPER_SFs),
                self.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY,
                self.HYPER_SELECT_BEST_STRUCTVAR_POSE_BY,
                self.HYPER_HOW_COMBINE,
                self.HYPER_FUSION_METHOD,
                self.HYPER_RATIO_SCORED_POSES,
                self.HYPER_OUTLIER_MAD_THRESHOLD,
                self.HYPER_OUTLIER_MIN_SCORED_POSE_NUM,
                self.HYPER_KEEP_MAX_N_POSES,
                self.HYPER_KEEP_MAX_DeltaG_POSES,
                self.HYPER_USE_SCONF)


'''
--> Explore how the HYPER_CONFORMER_ENERGY_CUTOFF and HYPER_CONFORMER_RMSD_CUTOFF affect the comformational entropy.

Hyperparameters to adjust based on the perfomrance of DeltaH (including deformation energy):
* thresh for outlier removal on the contexts of the whole set
* thresh for outlier removal on the contexts of the whole set
--> See which of the two yields better results.
* minimum number of scored poses per structvar in order to be included in the set ?
'''