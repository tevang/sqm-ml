import uuid

from lib.global_fun import list_files
import pandas as pd

class Settings:

    def __init__(self):

        self.HYPER_SQMNN_ROOT_DIR = '/home2/thomas/Documents/QM_Scoring/SQM-ML'
        self.HYPER_EXECUTION_DIR_NAME = 'GlideML_execution_dir'
        self.HYPER_PROTEIN = 'MK2'
        self.ALL_PROTEINS = [self.HYPER_PROTEIN]    # not hyper-param; just for file naming
        self.HYPER_FORCE_COMPUTATION = False

        # *****
        # TUNABLE HYPER-PARAMETERS

        # LEARNING MODEL
        # --
        self.HYPER_LEARNING_MODEL_TYPE = 'Logistic Regression'
        self.SAMPLE_WEIGHTS_TYPE = 'featvec_similarity'
        self.HYPER_SELECT_BEST_FEATURES = False
        self.HYPER_MAX_BEST_FEATURES = 31
        # --

        # POSE FILTERING
        # --
        self.HYPER_RATIO_SCORED_POSES = 0.8
        self.HYPER_REMOVE_OUTLIER_WRT_WHOLE_SET = False
        self.HYPER_OUTLIER_MAD_THRESHOLD = 99999999
        self.HYPER_OUTLIER_MIN_SCORED_POSE_NUM = 0  # keep only structvars with more than this number of scored poses
        self.HYPER_KEEP_MAX_N_POSES = 100  # keep at maximum this number of Glide poses per structvar for SQM scoring
        self.HYPER_KEEP_MAX_DeltaG_POSES = 1.0  # keep per structvar at maximum Glide poses with this energy difference
        # (kcal/mol) from the top scored for SQM scoring
        self.HYPER_KEEP_POSE_COLUMN = 'r_i_docking_score'  # use this column to keep the best Glide poses for SQM scoring
        self.HYPER_CONFORMER_ENERGY_CUTOFF_LIST = [6.0, 9.0,
                                                   12.0]  # kcal/mol (according to Chan et al. 'Understanding conformational entropy in small molecules', 2020
        self.HYPER_CONFORMER_RMSD_CUTOFF_LIST = [1.0, 1.5, 2.0]
        self.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY = 'r_i_docking_score'  # OPTIONS: 'r_i_docking_score, etc.
        self.HYPER_SELECT_BEST_BASEMOLNAME_POSE_BY = 'r_i_docking_score'  # OPTIONS: 'r_i_docking_score', etc.
        self.HYPER_SELECT_BEST_STRUCTVAR_POSE_BY = 'r_i_docking_score'  # OPTIONS: 'r_i_docking_score', etc.
        self.HYPER_SFs = ['r_i_docking_score']  # OPTIONS: 'r_i_docking_score'
        self.HYPER_HOW_COMBINE = 'inner'  # OPTIONS: 'inner' or 'outer'
        self.HYPER_FUSION_METHOD = 'nofusion'  # OPTIONS: 'nofusion', 'minrank', 'meanrank', 'geomean', 'harmmean'
        self.HYPER_USE_SCONF = False
        self.HYPER_SCONF_RMSD = 1.0
        self.HYPER_SCONF_ECUTOFF = 6
        self.HYPER_REPORT_PROTEIN_ENERGY_FLUCTUATIONS = True
        # --

        # *****
        # FEATURES FOR TRAINING

        # SQM ENERGY TERMS: always empty in GlideML
        self.HYPER_SQM_FEATURES = []
        # --

        # Glide DESCRIPTORS COMPLEMENTARY TO SQM ENERGY TERMS
        # http://mgcf.cchem.berkeley.edu/mgcf/schrodinger_2020-2/glide_user_manual/glide_docking_properties.htm
        self.HYPER_GLIDE_DESCRIPTORS = ["r_epik_State_Penalty", "r_i_glide_lipo", 'r_i_glide_hbond',
                                        'r_i_glide_metal', 'r_i_glide_rewards',
                                        'r_i_glide_evdw', 'r_i_glide_ecoul', 'r_i_glide_erotb',
                                        'r_i_glide_esite', 'r_i_glide_eff_state_penalty']
        # --

        # 2D DESCRIPTORS
        self.HYPER_2D_DESCRIPTORS = []
            # ['MW', 'AMW', 'deepFl_logP',
            #                          'rotor_count', 'terminal_CH3_count', 'function_group_count',
            #                          'ring_flexibility', 'Hbond_foldability', 'pipi_stacking_foldability',
            #                          'bondType_UNSPECIFIED', 'bondType_SINGLE', 'bondType_DOUBLE', 'bondType_TRIPLE',
            #                          'bondType_QUADRUPLE', 'bondType_QUINTUPLE', 'bondType_HEXTUPLE',
            #                          'bondType_ONEANDAHALF', 'bondType_TWOANDAHALF', 'bondType_THREEANDAHALF',
            #                          'bondType_FOURANDAHALF', 'bondType_FIVEANDAHALF', 'bondType_AROMATIC',
            #                          'bondType_IONIC', 'bondType_HYDROGEN', 'bondType_THREECENTER',
            #                          'bondType_DATIVEONE', 'bondType_DATIVE', 'bondType_DATIVEL', 'bondType_DATIVER',
            #                          'bondType_OTHER', 'bondType_ZERO',
            #
            #                          ]
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

        # 3D COMPLEX DESCRIPTORS
        self.HYPER_3D_COMPLEX_DESCRIPTORS = []
            # ['prot_interface_surf', 'prot_interface_SASA', 'lig_interface_surf',
            #                          'lig_interface_SASA', 'mean_interface_surf', 'mean_interface_SASA', 'net_charge']
        # --

        # FINGERPRINTS

        # PLEC
        self.HYPER_PLEC = True
        self.HYPER_PLEC_PCA_VARIANCE_EXPLAINED_CUTOFF = 0.2 # 0.2
        self.HYPER_COMPRESS_PLEC = True
        # --

        # PMAPPER
        self.HYPER_PMAPPER = False
        self.HYPER_PMAPPER_PCA_VARIANCE_EXPLAINED_CUTOFF = 0.25
        self.HYPER_COMPRESS_PMAPPER = True
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

    # def create_feature_csv_name(self):
    #     query_settings_df = pd.DataFrame.from_dict(self.__dict__, orient='index')
    #     for existing_settings_csv in list_files(folder=self.HYPER_EXECUTION_DIR_NAME,
    #                                          pattern='featvec_settings_.*\.csv.gz'):
    #         existing_settings_df = pd.read_csv(existing_settings_csv)
    #         if query_settings_df.equals(existing_settings_df):
    #             return existing_settings_csv.replace('featvec_settings_', 'featvec_values_')
    #     return query_settings_df.to_csv('featvec_settings_' + str(uuid.uuid4()) + '.csv.gz', index=False)

    def create_feature_csv_name(self):
        return '_features.Glide.HC_%s.FM_%s.RSP_%s.' \
               'OMT_%s.OMSPN_%i.KMNP_%i.KMDP_%f.US_%s.csv.gz' % \
               (self.HYPER_HOW_COMBINE,
                self.HYPER_FUSION_METHOD,
                self.HYPER_RATIO_SCORED_POSES,
                self.HYPER_OUTLIER_MAD_THRESHOLD,
                self.HYPER_OUTLIER_MIN_SCORED_POSE_NUM,
                self.HYPER_KEEP_MAX_N_POSES,
                self.HYPER_KEEP_MAX_DeltaG_POSES,
                self.HYPER_USE_SCONF)


'''
--> Explore how the HYPER_CONFORMER_ENERGY_CUTOFF and HYPER_CONFORMER_RMSD_CUTOFF affect the conformational entropy.

Hyperparameters to adjust based on the perfomrance of DeltaH (including deformation energy):
* thresh for outlier removal on the contexts of the whole set
* thresh for outlier removal on the contexts of the whole set
--> See which of the two yields better results.
* minimum number of scored poses per structvar in order to be included in the set ?
'''
