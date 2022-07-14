import copy

import numpy as np
import scoop
from rdkit import Chem
from scipy.stats import zscore

from deepscaffopt.lib.group_RandomForest import group_RandomForestRegressor
from deepscaffopt.lib.neural_nets.utils import prepare_SMILES_input_for_NNs
from deepscaffopt.lib.neural_nets.train_warheads import train_NGF, train_attentive_fp, train_chemprop_fp
from lib.ConsScoreTK_Statistics import remove_uniform_columns
from lib.MMGBSA_functions import MMGBSA
from lib.USRCAT_functions import calculate_moments
from lib.featvec.csv_utils import load_featvec_from_csv_using_serialnum
from lib.featvec.feature_scale_utils import minmax_scale_crossval_xtest
from lib.featvec.fingerprint_computer import calculate_fingerprints_from_RDKit_mols
from lib.featvec.invariants import Discretize
from lib.featvec.physchem import calculate_physchem_descriptors_from_mols
from lib.global_fun import tree, save_pickle
from lib.utils.print_functions import ColorPrint, Debuginfo, bcolors

try:
    import csfpy
except (ModuleNotFoundError, ImportError):
    ColorPrint("WARNING: module csfpy was not found, therefore CSFPy fingerprints won't be calculated.", "OKRED")
    pass

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import minmax_scale


class Features:

    def __init__(self, datasets=None):
        """

        :param datasets: a DataSet() object carrying the crossval and xtest set data.
        """
        self.datasets = datasets

    def __discretize_descriptors(self, all_featvecs):
        """

        :param all_featvecs: MolNum x FeatureNum
        :return:
        """
        ColorPrint("Discretizing the %i descriptors." % all_featvecs.shape[1], "OKBLUE")
        discretize = Discretize()
        for c in range(all_featvecs.shape[1]):  # iterate over all columns (features)
            values = all_featvecs[:, c]
            discrete_values = discretize.fit_transform(X=values)
            all_featvecs[:, c] = discrete_values    # replace the continuous values with the discretized
        return all_featvecs

    def __keep_most_important_descriptors(self,
                                          molname_featvec_dict,
                                          molname_Kd_dict,
                                          importance_type='combination',
                                          zi_cutoff=2):
        """
        Method to calculate the importance of each physicochemical descriptor according to the training set,
        and keep only those that have importance z-score above the zi_cutoff.

        :param molname_featvec_dict:    contains all physchem descriptors of crossval and xtest molecules
        :param molname_Kd_dict:         contains the experimental affinities of crossval molecules
        TODO: this dict contains only one value from the multiple that may be available. Make the method calculate
        TODO: group importances just list the MetaPredictor.
        :param importance_type: can be 'original' (RF importance), 'MI' (mutual information) or
                                'combination' (z-score of the sum of RF importance and Mutual Information z-scores)
        :param zi_cutoff:   z-score cutoff value. Only features with z-score importance above this value will be retained
        :return:
        """

        if set(molname_Kd_dict.values()) == {0, 1}:    # this is a Classification problem
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1);
            mutual_info = mutual_info_classif
        else:   # this is a Regression problem
            rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, criterion='mae');
            mutual_info = mutual_info_regression

        all_featvecs = np.array(list(molname_featvec_dict.values()), dtype='float32')
        all_molnames = list(molname_featvec_dict.keys())
        crossval_molnames = list(molname_Kd_dict.keys())
        crossval_Kds = list(molname_Kd_dict.values())
        crossval_rindices = [all_molnames.index(m) for m in crossval_molnames]     # row indices of crossval compounds

        i_dim = all_featvecs.shape  # initial dimensions

        # Remove NaN columns (features) TODO: this is dangerous as every time will remove different columns!
        all_featvecs = all_featvecs[:, ~np.isnan(all_featvecs).any(axis=0)]

        # all_featvecs = all_featvecs[~np.isnan(all_featvecs).any(axis=1), :] # remove NaN rows (compounds)

        ColorPrint("Initial dimensions of compound set: %i compounds %i physchem descriptors. "
                   "After nan removal: %i compounds %i physchem descriptors." %
                    (i_dim[0], i_dim[1], all_featvecs.shape[0], all_featvecs.shape[1]), "OKBLUE")

        # Remove features without variance
        selector = VarianceThreshold()
        all_featvecs = selector.fit_transform(all_featvecs)
        ColorPrint("Dimensions of compound set after 0-variance columns removal: %i compounds %i physchem descriptors." %
                   (all_featvecs.shape[0], all_featvecs.shape[1]), "OKBLUE")

        # # Discretize continuous descriptors
        # all_featvecs = self.__discretize_descriptors(all_featvecs)

        # CALCULATE FEATURE IMPORTANCES
        all_featvecs = minmax_scale(all_featvecs)  # necessary to calculate RF importances
        crossval_featvecs = all_featvecs[crossval_rindices,:]   # exctract the crossval featvec from the cleaned featvecs
        if importance_type in ['MI', 'combination']:
            # Calculate Mutual Information
            mi_importances = zscore(mutual_info(crossval_featvecs, crossval_Kds))
        if importance_type in ['original', 'combination']:
            # Calculate original importances
            original_importances = zscore(rf.fit(crossval_featvecs, crossval_Kds).feature_importances_)

        if importance_type == 'combination':
            # importances = zscore( mi_importances + original_importances )   # z-score of their sum
            importances = np.max(np.vstack([original_importances, mi_importances]), axis=0)  # z-score of their max z-score
        elif importance_type == 'MI':
            importances = mi_importances
        elif importance_type == 'original':
            importances = original_importances

        # Keep only features with importance z-score greater than the given cutoff.
        cindices = [i for i, zi in enumerate(importances) if zi > zi_cutoff]  # column indices to keep
        all_featvecs = np.array([vec[cindices] for vec in all_featvecs], dtype='float32')
        ColorPrint("Dimensions of compound set after low-importance columns removal"
                   " using z-score cutoff %.3f: %i compounds %i physchem descriptors." %
                   (zi_cutoff,all_featvecs.shape[0], all_featvecs.shape[1]), "OKBLUE")

        molname_impfeatvec_dict = {m: vec for m,vec in zip(all_molnames, all_featvecs)} # dict with the remaining features
        return molname_impfeatvec_dict


    def append_physchem_descriptors_to_featvecs(self,
                                                crossval_molname_fingerprint_dict,
                                                xtest_molname_fingerprint_dict,
                                                crossval_molname_SMILES_conformersMol_mdict,
                                                xtest_molname_SMILES_conformersMol_mdict,
                                                selected_descriptors=[],
                                                extra_descriptors_zscore=None,
                                                molID_assayID_Kd_list=[]):
        """
        Method to calculate selected physicochemical descriptors and append them to the existing feature vectors
        in order to increase the accuracy of MLP models.

        :param crossval_molname_fingerprint_dict:
        :param xtest_molname_fingerprint_dict:
        :param crossval_molname_SMILES_conformersMol_mdict:
        :param xtest_molname_SMILES_conformersMol_mdict:
        :param selected_descriptors:
        :return crossval_molname_fingerprint_dict:
        :return xtest_molname_fingerprint_dict:
        """
        if len(selected_descriptors) == 0:   # Nothing to calculate!
            return crossval_molname_fingerprint_dict, xtest_molname_fingerprint_dict

        ColorPrint("Calculating the selected physicochemical descriptors %s and appending them to the "
                   "existing feature vectors..." % selected_descriptors, "OKGREEN")

        # First calculate the selected physchem descriptors of both crossval and xtest in one shot
        mol_args = []
        for molname, SMILES2mol_dict in list(crossval_molname_SMILES_conformersMol_mdict.items()) + \
                       list(xtest_molname_SMILES_conformersMol_mdict.items()):
            mol_args.append( list(SMILES2mol_dict.values())[0] )
        featvecs, molnames = calculate_physchem_descriptors_from_mols(mol_args, return_molnames=True,
                                                                           selected_descriptors=selected_descriptors,
                                                                           nproc=scoop.SIZE)
        # If requested, keep only the most important descriptors
        # TODO: make the following method calculate group importances just like the MetaPredictor.
        if selected_descriptors and extra_descriptors_zscore != None:
            molname_Kd_dict = {}
            for molname in molnames:
                Kds = [i for m,a,i in molID_assayID_Kd_list if m==molname]
                if len(Kds) > 0:
                    molname_Kd_dict[molname] = Kds[0]   # save only the first experimental affinity if multiple were found
            molname_featvec_dict = {m: vec for m, vec in zip(molnames, featvecs)}
            molname_impfeatvec_dict = \
                self.__keep_most_important_descriptors(molname_featvec_dict, molname_Kd_dict,
                                                       importance_type='combination',
                                                       zi_cutoff=extra_descriptors_zscore)  # <== CHANGE ME
            # reform the featvecs (this time contains only important features) to continue normally
            featvecs = np.array([molname_impfeatvec_dict[m] for m in molnames])

        # Then minmax scale them
        featvecs = minmax_scale(featvecs)
        molname_physchemfp_dict = {m: d for m, d in zip(molnames, featvecs)}

        # Then append them to the existing featvecs
        # TODO: if one descriptor value could not be calculated for at least one molname, the program will crash later!
        for cmolname in crossval_molname_fingerprint_dict.keys():
            extended_featvec = np.append(crossval_molname_fingerprint_dict[cmolname], molname_physchemfp_dict[cmolname])
            crossval_molname_fingerprint_dict[cmolname] = extended_featvec
        for xmolname in xtest_molname_fingerprint_dict.keys():
            extended_featvec = np.append(xtest_molname_fingerprint_dict[xmolname], molname_physchemfp_dict[xmolname])
            xtest_molname_fingerprint_dict[xmolname] = extended_featvec

        return crossval_molname_fingerprint_dict, xtest_molname_fingerprint_dict

    def create_featvecs(self,
                        crossval_molname_SMILES_conformersMol_mdict,
                        xtest_molname_SMILES_conformersMol_mdict,
                        featvec_type="ECFP",
                        nBits=4096,
                        maxPath=12,
                        crossval_csv_file=None,
                        xtest_csv_file=None,
                        extra_descriptors=[],
                        extra_descriptors_zscore=None,
                        molID_assayID_Kd_list=[]):
        """

        Method to calculate feature vectors for training and evaluation/screening from two multi-dicts of RDKit Mol objects. It requires both crossval and xtest set
        molecules in order to standarize the features properly and remove redundant columns.
        :param crossval_molname_SMILES_conformersMol_mdict:
        :param xtest_molname_SMILES_conformersMol_mdict:    pass {} if you don't have an xtest.
        :param featvec_type:
        :param nBits:
        :param maxPath:
        :param extra_descriptors: a list of extra physchem descriptor names to be calculated and appended to the feature vectors
        :param extra_descriptors_zscore: if given, keep only descriptors with importance above this z-score
        :return: x_crossval:    list of feature vectors (one for each crossval molecule)
        :return: x_xtest:    list of feature vectors (one for each xtest molecule)

        """
        if not xtest_molname_SMILES_conformersMol_mdict:
            # Do a trick to fool the function, add a few molecules from crossval set to xtest
            sample_molnames = list(crossval_molname_SMILES_conformersMol_mdict.keys())[2]
            xtest_molname_SMILES_conformersMol_mdict = tree()
            for molname in sample_molnames:
                xtest_molname_SMILES_conformersMol_mdict[molname] = crossval_molname_SMILES_conformersMol_mdict[molname]
            empty_xtest = True
        else:
            empty_xtest = False

        if featvec_type == "moments":
            crossval_molname_SMILES_moments_mdict = calculate_moments(crossval_molname_SMILES_conformersMol_mdict,
                                                                  ensemble_mode=1,
                                                                  moment_number=4,
                                                                  onlyshape=False,
                                                                  plumed_colvars=False,
                                                                  use_spectrophores=False,
                                                                  conf_num=1)
            xtest_molname_SMILES_moments_mdict = calculate_moments(xtest_molname_SMILES_conformersMol_mdict,
                                                                  ensemble_mode=1,
                                                                  moment_number=4,
                                                                  onlyshape=False,
                                                                  plumed_colvars=False,
                                                                  use_spectrophores=False,
                                                                  conf_num=1)
            # dump the SMILES (use only the first one for each molecule)
            crossval_molname_fingerprint_dict, xtest_molname_fingerprint_dict = {}, {}
            for molname in list(crossval_molname_SMILES_moments_mdict.keys()):
                SMILES = list(crossval_molname_SMILES_moments_mdict[molname].keys())[0]
                crossval_molname_fingerprint_dict[molname] = crossval_molname_SMILES_moments_mdict[molname][SMILES][0]
            for molname in list(xtest_molname_SMILES_moments_mdict.keys()):
                SMILES = list(xtest_molname_SMILES_moments_mdict[molname].keys())[0]
                xtest_molname_fingerprint_dict[molname] = xtest_molname_SMILES_moments_mdict[molname][SMILES][0]

        elif featvec_type == "physchem":
            # TODO: using the SMILES strings may be dangerous for calculating accurate phychem descriptors.
            # TODO: However you cannot pass Mol objects to scoop, but mordred has a map() function to calculate
            # TODO: descriptors in parallel for multiple molecules.
            ## CALCULATE PHYSICOCHEMICAL DESCRIPTORS (best correlated with IC50s)
            # TODO: Transform the physchem descriptor logarithmically and estimate their importance using
            # TODO: pre-analysis with RF.
            crossval_molname_fingerprint_dict, xtest_molname_fingerprint_dict = {}, {}
            mol_args = []
            for molname in list(crossval_molname_SMILES_conformersMol_mdict.keys()):
                SMILES = list(crossval_molname_SMILES_conformersMol_mdict[molname].keys())[0]
                mol_args.append(crossval_molname_SMILES_conformersMol_mdict[molname][SMILES])
                # crossval_molname_fingerprint_dict[molname] = self.calculate_physicochemical_descriptors(SMILES)
            featvecs, molnames = calculate_physchem_descriptors_from_mols(mol_args, return_molnames=True,
                                                                               nproc=scoop.SIZE)
            crossval_molname_fingerprint_dict = {m: d for m, d in zip(molnames, featvecs)}
            mol_args = []
            for molname in list(xtest_molname_SMILES_conformersMol_mdict.keys()):
                SMILES = list(xtest_molname_SMILES_conformersMol_mdict[molname].keys())[0]
                mol_args.append(xtest_molname_SMILES_conformersMol_mdict[molname][SMILES])
                # xtest_molname_fingerprint_dict[molname] = calculate_physicochemical_descriptors(SMILES)
            featvecs, molnames = calculate_physchem_descriptors_from_mols(mol_args, return_molnames=True,
                                                                               nproc=scoop.SIZE)
            xtest_molname_fingerprint_dict = {m: d for m, d in zip(molnames, featvecs)}
            crossval_molname_fingerprint_dict, \
            xtest_molname_fingerprint_dict = \
                minmax_scale_crossval_xtest(
                    crossval_molname_fingerprint_dict,
                    xtest_molname_fingerprint_dict
                )
            save_pickle("molname_fingerprint_dicts.pkl", crossval_molname_fingerprint_dict,
                        xtest_molname_fingerprint_dict, molID_assayID_Kd_list)

            # If requested, keep only the most important descriptors
            # TODO: make the following method calculate group importances just like the MetaPredictor.
            if extra_descriptors_zscore != None:
                molname_featvec_dict = copy.deepcopy(crossval_molname_fingerprint_dict)   # contains both crossval and xtest featvecs
                for m,vec in xtest_molname_fingerprint_dict.items():
                    molname_featvec_dict[m] = vec
                molname_Kd_dict = {}    # contains only crossval molecules and their affinities
                for molname in crossval_molname_fingerprint_dict.keys():    # only crossval mols have affinities
                    Kds = [i for m, a, i in molID_assayID_Kd_list if m == molname]
                    if len(Kds) > 0:
                        molname_Kd_dict[molname] = Kds[0]  # save only the first experimental affinity if multiple were found
                molname_impfeatvec_dict = \
                    self.__keep_most_important_descriptors(molname_featvec_dict, molname_Kd_dict,
                                                           importance_type='combination',
                                                           zi_cutoff=extra_descriptors_zscore)  # <== CHANGE ME
                # reform the crossval_molname_fingerprint_dict &  xtest_molname_fingerprint_dict
                # (this time contain only important features) to continue normally
                cmolnames, xmolnames = list(crossval_molname_fingerprint_dict.keys()), list(xtest_molname_fingerprint_dict.keys())
                crossval_molname_fingerprint_dict = {m: molname_impfeatvec_dict[m] for m in cmolnames}
                xtest_molname_fingerprint_dict = {m: molname_impfeatvec_dict[m] for m in xmolnames}

        
        elif featvec_type.startswith("NGF"):
            # prepare the input of NGF
            bind_crossval_SMILES, function_crossval_SMILES, xtest_SMILES, bind_labels = \
                prepare_SMILES_input_for_NNs(self.datasets.bind_molID_assayID_Kd_list,
                                      self.datasets.function_molID_assayID_Kd_list,
                                      crossval_molname_SMILES_conformersMol_mdict,
                                      xtest_molname_SMILES_conformersMol_mdict)

            # Now train the NGF model using only the binding data but calculate the NGF fingerprints of all molecules
            fp_length = int(featvec_type.replace("NGF", ""))  # 62 is the default in the paper
            if fp_length > 3968:    # otherwise the GPU will run out of memory
                conv_width = 512
            else:
                conv_width = (fp_length // 62) * 8
            crossval_NGF, xtest_NGF = train_NGF(bind_crossval_SMILES, function_crossval_SMILES, xtest_SMILES,
                                                bind_labels, conv_width, fp_length)
            crossval_molname_fingerprint_dict = {m: fp for m, fp in
                                        zip(list(crossval_molname_SMILES_conformersMol_mdict.keys()), crossval_NGF)}
            xtest_molname_fingerprint_dict = {m: fp for m, fp in
                                        zip(list(xtest_molname_SMILES_conformersMol_mdict.keys()), xtest_NGF)}

        elif featvec_type.startswith("AttentiveFP"):
            # prepare the input of attentive_fp
            bind_crossval_SMILES, function_crossval_SMILES, xtest_SMILES, bind_labels = \
                prepare_SMILES_input_for_NNs(self.datasets.bind_molID_assayID_Kd_list,
                                      self.datasets.function_molID_assayID_Kd_list,
                                      crossval_molname_SMILES_conformersMol_mdict,
                                      xtest_molname_SMILES_conformersMol_mdict)

            # Now train the AttentiveFP model using only the binding data but calculate the fingerprints of all molecules

            crossval_attentive_fp, xtest_attentive_fp = train_attentive_fp(bind_crossval_SMILES, function_crossval_SMILES, xtest_SMILES,
                                                bind_labels)
            crossval_molname_fingerprint_dict = {m: fp for m, fp in
                                                 zip(list(crossval_molname_SMILES_conformersMol_mdict.keys()),
                                                     crossval_attentive_fp)}
            xtest_molname_fingerprint_dict = {m: fp for m, fp in
                                              zip(list(xtest_molname_SMILES_conformersMol_mdict.keys()), xtest_attentive_fp)}

        elif featvec_type.startswith("Chemprop"):
            # prepare the input of chemprop_fp
            bind_crossval_SMILES, function_crossval_SMILES, xtest_SMILES, bind_labels = \
                prepare_SMILES_input_for_NNs(self.datasets.bind_molID_assayID_Kd_list,
                                      self.datasets.function_molID_assayID_Kd_list,
                                      crossval_molname_SMILES_conformersMol_mdict,
                                      xtest_molname_SMILES_conformersMol_mdict)

            # Now train the ChempropFP model using only the binding data but calculate the fingerprints of all molecules

            crossval_chemprop_fp, xtest_chemprop_fp = train_chemprop_fp(bind_crossval_SMILES,
                                                                           function_crossval_SMILES, xtest_SMILES,
                                                                           bind_labels)
            crossval_molname_fingerprint_dict = {m: fp for m, fp in
                                                 zip(list(crossval_molname_SMILES_conformersMol_mdict.keys()),
                                                     crossval_chemprop_fp)}
            xtest_molname_fingerprint_dict = {m: fp for m, fp in
                                              zip(list(xtest_molname_SMILES_conformersMol_mdict.keys()),
                                                  xtest_chemprop_fp)}

        elif featvec_type == "csv":
            assert crossval_csv_file != None and crossval_csv_file != None, \
                Debuginfo("ERROR: you did not provide a csv file with the feature vectors!", fail=True)
            crossval_molname_fingerprint_dict = load_featvec_from_csv_using_serialnum(crossval_csv_file,
                                                                                               crossval_molname_SMILES_conformersMol_mdict)
            xtest_molname_fingerprint_dict = load_featvec_from_csv_using_serialnum(xtest_csv_file,
                                                                                            xtest_molname_SMILES_conformersMol_mdict)

        else:
            # calculate fingerprints for the crossval and xtest set compounds
            crossval_molname_fingerprint_dict = calculate_fingerprints_from_RDKit_mols(crossval_molname_SMILES_conformersMol_mdict,
                                                                                       featvec_type=featvec_type,
                                                                                       nBits=nBits,
                                                                                       maxPath=maxPath)
            xtest_molname_fingerprint_dict = calculate_fingerprints_from_RDKit_mols(xtest_molname_SMILES_conformersMol_mdict,
                                                                                    featvec_type=featvec_type,
                                                                                    nBits=nBits,
                                                                                    maxPath=maxPath)

        ## Calculate and append any extra physchem descriptors (if requested) to the feature vectors
        if extra_descriptors:
            molID_assayID_Kd_list
            crossval_molname_fingerprint_dict, \
            xtest_molname_fingerprint_dict = \
                self.append_physchem_descriptors_to_featvecs(crossval_molname_fingerprint_dict,
                                                             xtest_molname_fingerprint_dict,
                                                             crossval_molname_SMILES_conformersMol_mdict,
                                                             xtest_molname_SMILES_conformersMol_mdict,
                                                             selected_descriptors=extra_descriptors,
                                                             extra_descriptors_zscore=extra_descriptors_zscore,
                                                             molID_assayID_Kd_list=molID_assayID_Kd_list)

        # Concatenate crossval and blind xtest sets to reduce the size of the fingerprints by
        # removing identical columns
        featvec_list, molnames_crossval, function_molnames_crossval, molnames_xtest = [], [], [], []
        for molname, assayID, IC50 in self.datasets.bind_molID_assayID_Kd_list:  # load crossval set molecules in the order the occur in the sorted bind_molID_assayID_Kd_list
            try:
                featvec_list.append(crossval_molname_fingerprint_dict[molname.lower()])
                molnames_crossval.append(molname)
            except KeyError:
                print(bcolors.WARNING + "WARNING: failed to generate " + featvec_type + " fingerprint for BINDING crossvaling set molecule " + molname + bcolors.ENDC)
                pass

        for molname, assayID, IC50 in self.datasets.function_molID_assayID_Kd_list:  # load crossval set molecules in the order the occur in the sorted bind_molID_assayID_Kd_list
            try:
                featvec_list.append(crossval_molname_fingerprint_dict[molname.lower()])
                function_molnames_crossval.append(molname)
            except KeyError:
                print(bcolors.WARNING + "WARNING: failed to generate " + featvec_type + " fingerprint for FUNCTIONAL crossvaling set molecule " + molname + bcolors.ENDC)
                pass

        for molname in list(xtest_molname_fingerprint_dict.keys()):
            try:
                featvec_list.append(xtest_molname_fingerprint_dict[molname])
                molnames_xtest.append(molname)
            except KeyError:
                print(bcolors.WARNING + "WARNING: failed to generate " + featvec_type + " fingerprint for xtest set molecule " + molname + bcolors.ENDC)
                pass

        # filter bind_molID_assayID_Kd_list to remove compounds that don't have feature vectors
        self.datasets.bind_molID_assayID_Kd_list = [t for t in self.datasets.bind_molID_assayID_Kd_list if
                                                      t[0] in list(crossval_molname_fingerprint_dict.keys())]
        self.datasets.function_molID_assayID_Kd_list = [t for t in self.datasets.function_molID_assayID_Kd_list if
                                                      t[0] in list(crossval_molname_fingerprint_dict.keys())]
        cv_end = len(self.datasets.bind_molID_assayID_Kd_list)
        fun_cv_end = len(self.datasets.function_molID_assayID_Kd_list) + cv_end

        # remove identical columns
        initial_length = len(featvec_list[0])   # it's still a list
        ColorPrint("Removing features without diversity (uniform columns) and scaling values "
                   "between 0 and 1. (may be slow)", "OKBLUE")
        featvec_list = list(remove_uniform_columns(featvec_list, noredundant_cols=True))
        featvec_list = list(minmax_scale(featvec_list))  # scale values betwen 0 and 1
        final_length = len(featvec_list[0])
        ColorPrint("Reduced features from %i to %i." % (initial_length, final_length), "OKBLUE")

        # Now recover the original sets
        x_crossval = featvec_list[:cv_end]
        function_x_crossval = featvec_list[cv_end:fun_cv_end]
        x_xtest = featvec_list[fun_cv_end:]

        if empty_xtest:
            return x_crossval, function_x_crossval, [], molnames_crossval, function_molnames_crossval, []
        else:
            return x_crossval, function_x_crossval, x_xtest, molnames_crossval, function_molnames_crossval, molnames_xtest

    def feature_selection_RandomForest(self, datasets, META_ZCUTOFF=0.0, no_groups=False, fvtype_coeff_dict={}):
        """
        This is a method for feature selection based on Mean Square Error.
        :param datasets:
        :return:
        """
        if datasets.is_classification:
            # TODO: develop the group_RandomForestClassifier
            rf = group_RandomForestClassifier(n_jobs=1,
                                              n_estimators=1000)  # random_state=datasets.random_state);  # n_estimators=1000 creates only non-zero importances, use the default! For lhl use n_estimators=1000.
        else:
            rf = group_RandomForestRegressor(n_jobs=1,
                                             n_estimators=1000)  # random_state=datasets.random_state);  # n_estimators=1000 creates only non-zero importances, use the default! For lhl use n_estimators=1000.
        # try:
        rf.group_fit(datasets, no_groups=no_groups, fvtype_coeff_dict=fvtype_coeff_dict)  # fit score predictions
        datasets.importance_list = rf.group_feature_importances_
        # except ValueError:
        #     print("ValueError: Input contains NaN, infinity or a value too large for dtype('float32').")
        #     return

        # Just print(all the MLPS, random_state and importance value for user's information
        datasets.print_importances_and_tau(normalize=False)

        return datasets.importance_list

    def feature_selection_Variance(self, datasets, zthreshold=0.0):
        """
        NOT GOOD!
        :param datasets:
        :param zthreshold:
        :return:
        """
        X_nogroups = []
        # this must be a high quality 'Binding' assay, because selection is based on RMSE not Kendall's tau!
        for i in range(len(datasets.bind_molID_assayID_Kd_list)):
            molID, assayID, value = datasets.bind_molID_assayID_Kd_list[i]
            X_nogroups.append( np.array(datasets.pred_crossval_dict[molID]) )
        for j in range(len(datasets.function_molID_assayID_Kd_list)):
            molID, assayID, value = datasets.function_molID_assayID_Kd_list[j]
            X_nogroups.append( np.array(datasets.pred_crossval_dict[molID]) )

        # Do this trick, save the variances of the scores of each MLP (feature) as importances and then clip the MLPS by a z-score value
        datasets.importance_list = np.std(X_nogroups, axis=0)**2  # variance per MLP (they are converted to z-score within the clip function)
        datasets.CXsim_list = [1.5] * datasets.importance_list.shape[0] # deactive CXsim
        # datasets.clip_models_by_importance_and_CXsimilarity(zscore_threshold=zthreshold)
        ColorPrint("Features (MLPs) sorted by the z-score the variance of their prediction scores:", "BOLDBLUE")
        datasets.print_importances_and_tau()

        return datasets # returned the clipped datasets

    def feature_selection_Univariate(self, datasets):
        """
        BAD!
        The options for univariate selection are: 1) f_regression, 2) mutual_info_regression
        for mode='percentile' and param=50, it selects the 50% most significant features.
        """
        from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_regression

        X_nogroups, Y_nogroups = [], []
        # this must be a high quality 'Binding' assay, because selection is based on RMSE & R not Kendall's tau!
        for i in range(len(datasets.bind_molID_assayID_Kd_list)):
            molID, assayID, value = datasets.bind_molID_assayID_Kd_list[i]
            X_nogroups.append( np.array(datasets.pred_crossval_dict[molID]) )
            Y_nogroups.append( datasets.y_crossval['scores'][i] )
        # print("X_nogroups=", X_nogroups   )# for DEBUGGING)
        # print("Y_nogroups=", Y_nogroups)

        # featsel = GenericUnivariateSelect(score_func=f_regression, mode='percentile', param=50)
        # featsel.fit(X_nogroups, Y_nogroups)
        # scores = -np.log10(featsel.pvalues_)
        # scores /= scores.max()

        featsel = GenericUnivariateSelect(score_func=mutual_info_regression, mode='percentile')
        featsel.fit(X_nogroups, Y_nogroups)
        scores = featsel.scores_

        # clf = svm.LinearSVR()
        # clf.fit(X_nogroups, Y_nogroups)
        # svm_weights = (clf.coef_ ** 2)
        # svm_weights /= svm_weights.max()
        # scores = svm_weights

        # featsel = LassoCV()
        # featsel.fit(X_nogroups, Y_nogroups)
        # scores = featsel.coef_
        # scores /= scores.max()

        datasets.importance_list = scores
        datasets.CXsim_list = [1.5] * datasets.importance_list.shape[0] # deactive CXsim
        ColorPrint("Features (MLPs) sorted by the Regression P-value of their prediction scores:", "BOLDBLUE")
        datasets.print_importances_and_tau(normalize=False)


