import math
import sys
from collections import OrderedDict
from operator import itemgetter

import croc
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from yard import BinaryClassifierData, ROCCurve, CROCCurve

from library.utils.print_functions import bcolors, ColorPrint


class Classification_Metric():

    def __init__(self, y, x):
        """

        :param y: iterable of true labels, which must be 0 or 1 only! No multi-label is supported in this class!
        :param x: iterable of predicted labels, 0 or 1.
        """
        mcm = multilabel_confusion_matrix(y, x)
        self.tp = mcm[0, 0, 0]
        self.tn = mcm[0, 1, 1]
        self.fp = mcm[0, 1, 0]
        self.fn = mcm[0, 0, 1]

    def TPR(self):
        """
        True Positive rate (TPR) aka Sensitivity.
        :param x:
        :param y:
        :return:
        """
        return self.tp/(self.tp+self.fn)

    def FPR(self):
        """
        False Positive Rate (FPR)
        :param x:
        :param y:
        :return:
        """
        return 1-self.TNR()
    
    def TNR(self):
        """
        True Negative Rate (TNR) aka Specificity.
        :param x:
        :param y:
        :return:
        """
        return self.tn/(self.tn+self.fp)
    
    def FNR(self):
        """
        False Negative Rate (FNR).
        :param x:
        :param y:
        :return:
        """
        return 1-self.TPR()

    def LRplus(self):
        """
        Positive likelihood ratio (LR+).
        :param x:
        :param y:
        :return:
        """
        return self.TPR()/self.FPR()
    
    def LRminus(self):
        """
        Negative likelihood ratio (LR−).
        :param x:
        :param y:
        :return:
        """
        return self.FNR()/self.TNR()
    
    def DOR(self):
        """
        Diagnostic odds ratio (DOR).
        The diagnostic odds ratio ranges from zero to infinity, although for useful tests it is
        greater than one, and higher diagnostic odds ratios are indicative of better test performance.
        Diagnostic odds ratios less than one indicate that the test can be improved by simply inverting
        the outcome of the test – the test is in the wrong direction, while a diagnostic odds ratio of
        exactly one means that the test is equally likely to predict a positive outcome whatever the
        true condition – the test gives no information.
        :param x:
        :param y:
        :return:
        """
        return self.LRplus()/self.LRminus()

    def PPV(self):
        """
        Positive predictive value (PPV) aka Precision.
        :return:
        """
        return self.tp/(self.tp+self.fp)

    def NPV(self):
        """
        Negative predictive value (NPV).
        :return:
        """
        return self.tn/(self.tn+self.fn)

    def MK(self):
        """
        Markedness (MK).
        :return:
        """
        return self.PPV()+self.NPV()-1


class Create_Curve():
    """
        Class with functions that create ROC, CROC and BEDROC curves.
    """

    # from croc import ScoredData, ROC, BEDROC
    # TODO: THIS CLASS NEEDS MAJOR REFORMATING TO BECOME MORE USER-FRIENDLY!!!

    def __init__(self, sorted_ligand_ExperimentalDeltaG_dict,
                 sorted_ligand_IsomerEnergy_dict,
                 ENERGY_THRESHOLD=None,
                 ENSEMBLE_DOCKING_HOME=None,
                 write_results_list=[],
                 actives_list=[],
                 molname_list=[],
                 molname2scaffold_dict={}):
        """ The Constructor
            The threshold Ki value to consider a ligand as a true binder binder must be up to 100 times lower than the Ki of the strongest binder.

            :param sorted_ligand_ExperimentalDeltaG_dict:   an OrderedDict object with keys the ligand names and values the experimental DeltaG (derived from Ki/IC50)
                                                            If it is an array, sorted_ligand_IsomerEnergy_dict must also be an array in the same order!
            :param sorted_ligand_IsomerEnergy_dict:    an OrderedDict object with keys the ligand names and values the docking energies/raw docking scores/Z-scores.
                                                            The ligands in sorted_ligand_ExperimentalDeltaG_dict and sorted_ligand_IsomerEnergy_dict
                                                            DO NOT NEED TO BE IN THE SAME ORDER.
            :param ENERGY_THRESHOLD:    a DeltaG value used as a threshold to judge if a ligand is consired an active or inactive according to its
                                        experimental DeltaG. By dedault it is RT*log(100)*minimum experimental DeltaG.
            :ENSEMBLE_DOCKING_HOME:     the full path of the reports/ folder where you run this script
            :write_results_list:        a list with elements 1) Scoring Function Name, 2) cluster ID, 3) lowestEisomer|averageIsomerE|consensusZscore 4) ligand->isomer
                                        dictionary (either the lowest Energy isomer or ligand->ligand dictionary in case of consensusZscore). In the case of NNscore1 & NNscore2
                                        the elements are 1) Scoring Function Name, 2) network number, 3) cluster ID, 4) lowestEisomer|averageIsomerE, 4) ligand->isomer dictionary.
                                        If the list is empty, then no curve coordinates are written.
            :returns:   the area under ROC curve, the area under CROC curve, the AUC-ROC using CROC package algorithm, the BEDROC value
        """
        ## TODO ##
        ## You must normalize the Energy Threshold and the Docking Energies/Scores before comparison!

        self.classification_data = None
        self.scored_data = None
        self.expected = []
        self.outcomes = []
        self.write_results_list = write_results_list
        self.write_results_list_length = len(self.write_results_list)
        self.ENSEMBLE_DOCKING_HOME = ENSEMBLE_DOCKING_HOME

        # print("DEBUG: ENERGY_THRESHOLD=",ENERGY_THRESHOLD)
        # print("DEBUG: sorted_ligand_ExperimentalDeltaG_dict:",sorted_ligand_ExperimentalDeltaG_dict)
        # print("DEBUG: sorted_ligand_IsomerEnergy_dict:",sorted_ligand_IsomerEnergy_dict)
        # print("DEBUG: type(sorted_ligand_ExperimentalDeltaG_dict)=", type(sorted_ligand_ExperimentalDeltaG_dict))

        ## The following code creates two binary lists "expected" and "outcomes". "expected" consists of digits (0 or 1) indicating if each ligand is an active or an inactive.
        ## "outcomes" consists of digits (0 or 1) indicating if each ligand is classified as an active or an inactive by the scoring function.

        strong_and_weak_binder_dict = {}
        if isinstance(sorted_ligand_ExperimentalDeltaG_dict, (dict, OrderedDict)):
            if ENERGY_THRESHOLD == None:
                RT = 0.5957632602
                ENERGY_THRESHOLD = np.min(list(sorted_ligand_ExperimentalDeltaG_dict.values())) + RT * math.log(100)

            for ligand in list(sorted_ligand_ExperimentalDeltaG_dict.keys()):
                if (float(sorted_ligand_ExperimentalDeltaG_dict[
                              ligand]) < ENERGY_THRESHOLD):  # if this is an active (strong binder)
                    self.expected.append(1)
                else:  # if this is an inactive (weak binder)
                    self.expected.append(0)
                self.outcomes.append(sorted_ligand_IsomerEnergy_dict[ligand])

            if (self.expected.count(0) == len(self.expected)):
                sys.stdout.write(
                    bcolors.BOLDBLUE + "N/A under CROC curve N/A - NO ACTIVES WERE FOUND WITH ENERGY THRESHOLD " + str(
                        ENERGY_THRESHOLD) + ", INCREASE ENERGY THRESHOLD USING --threshold \n" + bcolors.ENDBOLD)
                return
            elif (self.expected.count(1) == len(self.expected)):
                sys.stdout.write(
                    bcolors.BOLDBLUE + "N/A under CROC curve N/A - NO INACTIVES WERE FOUND WITH ENERGY THRESHOLD " + str(
                        ENERGY_THRESHOLD) + ", LOWER ENERGY THRESHOLD USING --threshold \n" + bcolors.ENDBOLD)
                return

            self.sorted_outcome_expected_molname_tuple_list = []  # list of the form (score, activity, compound name); score may be a docking score of Z-score activity is 0 or 1
            for score, activity, ligand in zip(self.outcomes, self.expected,
                                               list(sorted_ligand_ExperimentalDeltaG_dict.keys())):
                self.sorted_outcome_expected_molname_tuple_list.append((score, activity, ligand))
            self.sorted_outcome_expected_molname_tuple_list.sort(key=itemgetter(0))

        elif isinstance(sorted_ligand_ExperimentalDeltaG_dict, (np.ndarray, np.generic, list, tuple,
                                                                pd.core.series.Series)):
            if ENERGY_THRESHOLD == None:
                RT = 0.5957632602
                ENERGY_THRESHOLD = np.min(sorted_ligand_ExperimentalDeltaG_dict) + RT * math.log(100)

            self.molname2scaffold_dict = molname2scaffold_dict
            # In this case both sorted_ligand_ExperimentalDeltaG_dict and sorted_ligand_IsomerEnergy_dict are in the same order
            if ENERGY_THRESHOLD == "ACTIVITIES":  # this means than the sorted_ligand_ExperimentalDeltaG_dict array consists of activity (0 & 1 values)
                self.expected = list(sorted_ligand_ExperimentalDeltaG_dict)
                self.outcomes = list(sorted_ligand_IsomerEnergy_dict)
            else:  # othrwise find which are the actives and which the inactives and save them
                for score, ExpDG in zip(sorted_ligand_IsomerEnergy_dict, sorted_ligand_ExperimentalDeltaG_dict):
                    if (ExpDG < ENERGY_THRESHOLD):  # if this is an active (strong binder)
                        self.expected.append(1)
                    else:  # if this is an inactive (weak binder)
                        self.expected.append(0)
                    self.outcomes.append(score)

            self.sorted_outcome_expected_molname_tuple_list = []  # list of the form (score, activity, compound name); score may be a docking score of Z-score activity is 0 or 1

            if (self.expected.count(0) == len(self.expected)):
                sys.stdout.write(
                    bcolors.BOLDBLUE + "N/A under CROC curve N/A - NO ACTIVES WERE FOUND WITH ENERGY THRESHOLD " + str(
                        ENERGY_THRESHOLD) + ", INCREASE ENERGY THRESHOLD USING --threshold \n" + bcolors.ENDBOLD)
                return
            elif (self.expected.count(1) == len(self.expected)):
                sys.stdout.write(
                    bcolors.BOLDBLUE + "N/A under CROC curve N/A - NO INACTIVES WERE FOUND WITH ENERGY THRESHOLD " + str(
                        ENERGY_THRESHOLD) + ", LOWER ENERGY THRESHOLD USING --threshold \n" + bcolors.ENDBOLD)
                return

            if molname2scaffold_dict != {}:
                for score, activity, molname in zip(self.outcomes, self.expected, molname_list):
                    if molname in actives_list:
                        scaffold_SMILES = molname2scaffold_dict[molname]
                    else:
                        scaffold_SMILES = ""
                    self.sorted_outcome_expected_molname_tuple_list.append(
                        (score, activity, "", scaffold_SMILES))  # empty ligand name because we use arrays
            elif molname2scaffold_dict == {}:
                for score, activity, molname in zip(self.outcomes, self.expected, molname_list):
                    self.sorted_outcome_expected_molname_tuple_list.append(
                        (score, activity, "", ""))  # empty ligand name because we use arrays
            self.sorted_outcome_expected_molname_tuple_list.sort(key=itemgetter(0))

        ## Recall that NN1 & NN2 scores have been negated (-1*) to comply with Vina & DSX scoring (the lowest the better). In order to use CROC functions correctly
        ## you have to negate them again (-1*) in order to become the higher the better.
        self.outcomes = [-1 * float(x) for x in self.outcomes]

        # print("DEBUG: outcomes = ",self.outcomes)
        # print("DEBUG: expected = ",self.expected)
        ColorPrint("Total number of actives with scores = %i , total number of inactives with scores = %i" %
                   (self.expected.count(1), self.expected.count(0)), "BOLDGREEN")

    def Enrichment_Factor(self, x):
        """
            Function to calculate the enrichment factor at a given percentage x of the database screened.
        """
        from scipy import stats

        # print("DEBUG: Calculating EF("+str(x)+"%)..")
        N_total = len(self.sorted_outcome_expected_molname_tuple_list)  # total num of compounds in the entire dataset
        N_sel0 = int((x / 100.0) * N_total)  # num of compounds in the top x %
        if N_sel0 == 0:
            ColorPrint(
                "FAIL: given Enrichment Factor percentage (%f%%) is too low for just %i molecules!" % (x, N_total),
                "FAIL")
            return ('N/A', 0, 0, 0)
        N_sel = N_sel0  # number of compounds within all the ranks that are <N_sel0.
        # MEASURE ALL THE COMPOUNDS THAT BELONG TO ALL THE RANKS THAT ARE < x. DON'T LEAVE ANY RANK INCOMPLETE (MEASURE ALL ITS COMPOUNDS).
        outcome_rankings = stats.rankdata([o[0] for o in
                                           self.sorted_outcome_expected_molname_tuple_list])  # array with the rankings of the sorted docking scores
        rank = outcome_rankings[N_sel0 - 1]  # get the rank of the xth compound
        # print("DEBUG: N_sel=", N_sel, " N_sel0=", N_sel0)
        while rank < N_sel0:
            # print("DEBUG: N_sel=", N_sel, " N_sel0=", N_sel0)
            N_sel += 1
            rank = outcome_rankings[N_sel - 1]

        Hits_x = 0  # num of actives in the top x %
        for triplet in self.sorted_outcome_expected_molname_tuple_list[:N_sel]:
            if triplet[1] == 1:
                Hits_x += 1

        Hits_total = 0  # num of actives in the entire dataset
        for triplet in self.sorted_outcome_expected_molname_tuple_list:
            if triplet[1] == 1:
                Hits_total += 1

        # print("DEBUG: N_sel=", N_sel, "N_total=", N_total, "Hits_x=", Hits_x, "Hits_total=", Hits_total)
        EF = (float(Hits_x) / N_sel) / (float(Hits_total) / N_total)
        # print("DEBUG: Hits_x=%i N_set=%i Hits_total=%i N_total=%i" % (Hits_x, N_sel, Hits_total, N_total))
        norm_EF = EF / ((float(N_sel) / N_sel) / (
                    float(Hits_total) / N_total))  # divide by the maximum value the EF can take
        return (norm_EF, EF, Hits_x, N_sel)  # N_sel is the number of compounds in the x% of the database

    def scaffold_Enrichment_Factor(self, x):
        """
            Function to calculate the scaffold enrichment factor at a given percentage x of the database screened.
        """
        from scipy import stats

        # print("DEBUG: Calculating EF("+str(x)+"%)..")
        N_total = len(self.sorted_outcome_expected_molname_tuple_list)  # total num of compounds in the entire dataset
        N_sel0 = int((x / 100.0) * N_total)  # num of compounds in the top x %
        if N_sel0 == 0:
            ColorPrint(
                "FAIL: given Enrichment Factor percentage (%f%%) is too low for just %i molecules!" % (x, N_total),
                "FAIL")
            return ('N/A', 0)
        N_sel = N_sel0  # number of compounds within all the ranks that are <N_sel0.
        # MEASURE ALL THE COMPOUNDS THAT BELONG TO ALL THE RANKS THAT ARE < x. DON'T LEAVE ANY RANK INCOMPLETE (MEASURE ALL ITS COMPOUNDS).
        outcome_rankings = stats.rankdata([o[0] for o in
                                           self.sorted_outcome_expected_molname_tuple_list])  # array with the rankings of the sorted docking scores
        rank = outcome_rankings[N_sel0 - 1]  # get the rank of the xth compound
        # print("DEBUG: N_sel=", N_sel, " N_sel0=", N_sel0)
        while rank < N_sel0:
            # print("DEBUG: N_sel=", N_sel, " N_sel0=", N_sel0)
            N_sel += 1
            rank = outcome_rankings[N_sel - 1]

        scaffolds_x = []  # scaffolds in the top x %
        for quadruplet in self.sorted_outcome_expected_molname_tuple_list[:N_sel]:
            if quadruplet[1] == 1:  # if this is an active then the scaffold_SMILES string will not be empty
                scaffolds_x.append(quadruplet[3])
        Hits_x = len(set(scaffolds_x))  # num of diverse scaffolds for the actives in the top x %

        Hits_total = len(
            set(self.molname2scaffold_dict.values()))  # num of unique scafolds for the actives in the entire database

        # print("DEBUG: N_sel=", N_sel, "N_total=", N_total, "Hits_x=", Hits_x, "Hits_total=", Hits_total)
        EF = (float(Hits_x) / N_sel) / (float(Hits_total) / N_total)
        norm_EF = EF / ((float(N_sel) / N_sel) / (float(Hits_total) / N_total))
        return (norm_EF, N_sel)

    def get_number_of_actives(self, x):
        """
            Function to return the number of actives in the top-scored x molecules.
            RETURNS:
            N_sel:  number of top scored compound. If the precision of the scoring function is low this number will be > x, because the function must return all the compounds
                    belonging to each rank that is < x.
            Hits:   number of actives in the N_sel compounds.
        """
        from scipy import stats

        if len(self.sorted_outcome_expected_molname_tuple_list) <= x:
            print(bcolors.WARNING + "WARNING: the number of molecules in the dataset is smaller than " + str(
                x) + "!" + bcolors.ENDC)
            return ('N/A', 'N/A', [])

        # MEASURE ALL THE COMPOUNDS THAT BELONG TO ALL THE RANKS THAT ARE < x. DON'T LEAVE ANY RANK INCOMPLETE (MEASURE ALL ITS COMPOUNDS).
        outcome_rankings = stats.rankdata([o[0] for o in
                                           self.sorted_outcome_expected_molname_tuple_list])  # array with the rankings of the sorted docking scores
        rank = outcome_rankings[x - 1]  # get the rank of the xth compound
        N_sel = x
        # print("DEBUG: N_sel=", N_sel, " x=", x)
        while rank < x:
            # print("DEBUG: N_sel=", N_sel, " x=", x)
            N_sel += 1
            rank = outcome_rankings[N_sel - 1]

        Hits = 0
        found_actives_list = []  # list of the actives ligand names found in the top x molecules
        for triplet in self.sorted_outcome_expected_molname_tuple_list[:x]:
            if triplet[1] == 1:
                Hits += 1
                found_actives_list.append(triplet[2])

        return (Hits, N_sel, found_actives_list)

    def ROC_curve(self, plotfile_name=None):
        """
            Function to draw ROC curves and calculate the AU-ROC
        """
        if not self.classification_data:
            self.classification_data = BinaryClassifierData(list(zip(self.outcomes, self.expected)))
        # print("DEBUG: classification_data:",classification_data.get_positive_ranks())
        ROC = ROCCurve(self.classification_data)
        if (
                self.write_results_list_length > 0):  # if write_results_list is not empty write the coordinates of the curve into a file for plotting
            # print("DEBUG: self.write_results_list=",self.write_results_list
            if not self.os.path.exists(self.ENSEMBLE_DOCKING_HOME + "/plots"):
                self.os.makedirs(self.ENSEMBLE_DOCKING_HOME + "/plots")
            if (self.write_results_list_length == 4):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME + "/plots/" + self.write_results_list[0] + "_cluster" + \
                                self.write_results_list[1] + "_" + self.write_results_list[2] + "_ROC_curve.dat"
            elif (self.write_results_list_length == 5):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME + "/plots/" + self.write_results_list[0] + "_" + \
                                self.write_results_list[2] + "_cluster" + self.write_results_list[1] + "_" + \
                                self.write_results_list[3] + "_ROC_curve.dat"
        if plotfile_name:
            with open(plotfile_name, 'w') as f:
                f.write('\n'.join('%s %s' % x for x in ROC.points))
            ## Commandline: yard-plot -t fscore -t roc -t croc test_curve.txt --show-auc -o test_curves.pdf
        return ROC.auc()

    def CROC_curve(self, plotfile_name=None):
        """
            Function to draw CROC curves and calculate the AU-ROC
        """
        if not self.classification_data:
            self.classification_data = BinaryClassifierData(list(zip(self.outcomes, self.expected)))
        CROC = CROCCurve(self.classification_data)
        if (
                self.write_results_list_length > 0):  # if write_results_list is not empty write the coordinates of the curve into a file for plotting
            # print("DEBUG: self.write_results_list=",self.write_results_list
            if (self.write_results_list_length == 4):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME + "/plots/" + self.write_results_list[0] + "_cluster" + \
                                self.write_results_list[1] + "_" + self.write_results_list[2] + "_CROC_curve.dat"
            elif (self.write_results_list_length == 5):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME + "/plots/" + self.write_results_list[0] + "_" + \
                                self.write_results_list[2] + "_cluster" + self.write_results_list[1] + "_" + \
                                self.write_results_list[3] + "_CROC_curve.dat"
        if plotfile_name:
            with open(plotfile_name, 'w') as f:
                f.write('\n'.join('%s %s' % x for x in CROC.points))
        return CROC.auc()

    def BEDROC_curve(self, plotfile_name=None):
        """
            Function to draw BEDROC curves and calculate the AU-BEDROC
        """
        ## Now use CROC python package to calculate ROC and BEDROC curves.
        if not self.scored_data:
            self.scored_data = croc.ScoredData(list(zip(self.outcomes, self.expected)))
        croc_BEDROC_dict = croc.BEDROC(self.scored_data, 20)
        if (
                self.write_results_list_length > 0):  # if write_results_list is not empty write the coordinates of the curve into a file for plotting
            # print("DEBUG: self.write_results_list=",self.write_results_list
            if (self.write_results_list_length == 4):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME + "/plots/" + self.write_results_list[0] + "_cluster" + \
                                self.write_results_list[1] + "_" + self.write_results_list[2] + "_croc_BEDROC_curve.dat"
            elif (self.write_results_list_length == 5):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME + "/plots/" + self.write_results_list[0] + "_" + \
                                self.write_results_list[2] + "_cluster" + self.write_results_list[1] + "_" + \
                                self.write_results_list[3] + "_croc_BEDROC_curve.dat"
        if plotfile_name:
            with open(plotfile_name, 'w') as f:
                f.write('\n'.join('%s %s' % x for x in croc_BEDROC_dict['curve']))
        return croc_BEDROC_dict['BEDROC']

    def croc_ROC_curve(self, plotfile_name=None):
        """
            Function to draw CROC curves and calculate the AU-ROC
        """
        ## Now use CROC python package to calculate ROC and BEDROC curves.
        if not self.scored_data:
            self.scored_data = croc.ScoredData(list(zip(self.outcomes, self.expected)))
        croc_ROC = croc.ROC(
            self.scored_data.sweep_threshold())  # I use sweep_threshold() because it reproduces YARD's AUC-ROC value
        if (
                self.write_results_list_length > 0):  # if write_results_list is not empty write the coordinates of the curve into a file for plotting
            # print("DEBUG: self.write_results_list=",self.write_results_list
            if not self.os.path.exists(self.ENSEMBLE_DOCKING_HOME + "/plots"):
                self.os.makedirs(self.ENSEMBLE_DOCKING_HOME + "/plots")
            if (self.write_results_list_length == 4):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME + "/plots/" + self.write_results_list[0] + "_cluster" + \
                                self.write_results_list[1] + "_" + self.write_results_list[2] + "_croc_ROC_curve.dat"
            elif (self.write_results_list_length == 5):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME + "/plots/" + self.write_results_list[0] + "_" + \
                                self.write_results_list[2] + "_cluster" + self.write_results_list[1] + "_" + \
                                self.write_results_list[3] + "_croc_ROC_curve.dat"
        if plotfile_name:
            f = open(plotfile_name, 'w')
            croc_ROC.write_to_file(f)
            f.close()
        return croc_ROC.area()