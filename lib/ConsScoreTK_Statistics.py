
import math
from itertools import cycle
from operator import itemgetter
import croc
from scipy import stats
from scipy.stats import pearsonr
from yard import BinaryClassifierData, ROCCurve, CROCCurve
from yard.utils import parse_size

from .utils.print_functions import bcolors

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)    # workaround
import pandas as pd
# print("DEBUG: pdb=", pdb)
from sklearn.preprocessing import minmax_scale
## IMPORT TensorFlow modules


from .global_fun import *


def get_Zscores(sorted_lowest_docking_Energy_dict, allposes=False):
    """ A function to convert a dictionary values to Z-scores
        ARGS:
        sorted_lowest_docking_Energy_dict
        allposes:   if True sorted_lowest_docking_Energy_dict contains the Energies from all docking poses per ligand. In this case
                    sorted_lowest_docking_Energy_dict must not contain any N/A values!!!
    """
    from itertools import cycle
    
    if allposes == False:
        # print("DEBUG get_Zscores: sorted_lowest_docking_Energy_dict=", sorted_lowest_docking_Energy_dict)
        raw_scores_list_old = list(sorted_lowest_docking_Energy_dict.values())
        raw_scores_list_new = []
        NA_positions = []
        for index in range(0, len(raw_scores_list_old)):
                if  raw_scores_list_old[index] == 'N/A':
                    NA_positions.append(index)
                else:
                    try:
                        raw_scores_list_new.append(float(raw_scores_list_old[index]))
                    except:
                        print(bcolors.FAIL + "ERROR: non-number docking score detected:", raw_scores_list_old[index] + bcolors.ENDC)
                        sys.exit(1)
        raw_scores_array = np.array(raw_scores_list_new)   # now this arraw will contain only floats, not strings (e.g. 'N/A')
        del raw_scores_list_old, raw_scores_list_new
        #print("DEBUG: raw_scores_list_new=",raw_scores_array.tolist()
        zscore_list = zscore(raw_scores_array).tolist()
        sorted_lowest_docking_Zscore_dict = OrderedDict()
        index = 0
        ligands_list = list(sorted_lowest_docking_Energy_dict.keys())
        zscore_list_cycle = cycle(zscore_list)
        for index in range(0, len(ligands_list)):
            ligand = ligands_list[index]
            if index in NA_positions:
                sorted_lowest_docking_Zscore_dict[ligand] = 'N/A'
            else:
                next_zscore = next(zscore_list_cycle)
                sorted_lowest_docking_Zscore_dict[ligand] = next_zscore
        
        return sorted_lowest_docking_Zscore_dict
    
    elif allposes == True:
        # print("DEBUG get_Zscores: sorted_lowest_docking_Energy_dict=", sorted_lowest_docking_Energy_dict)
        # in this case sorted_lowest_docking_Energy_dict is actually isomer_allPoseEnergiesList_dict: isomer->[pose1 score, pose2 score, ..., poseN score]
        # print("DEBUG get_Zscores: isomer_allPoseEnergiesList_dict=", sorted_lowest_docking_Energy_dict)
        # create an Nx10 array to save the docking Energies, where N is the total number of keys in the isomer_allPoseEnergiesList_dict
        raw_scores_list_old = []    # this will become and Nx10 2D list (adding elemts to a list is faster than to an array)
        isomer_numofPoses_dict = OrderedDict() # the total number of poses for each isomer in isomer_allPoseEnergiesList_dict
        for isomer in list(sorted_lowest_docking_Energy_dict.keys()):
            isomer_numofPoses_dict[isomer] = len(sorted_lowest_docking_Energy_dict[isomer])
            for pose in range(isomer_numofPoses_dict[isomer]):
                raw_scores_list_old.append(sorted_lowest_docking_Energy_dict[isomer][pose])
        raw_scores_list_new = [] # this list will contain only floats, not strings (e.g. 'N/A')
        NA_positions = []
        for index in range(0, len(raw_scores_list_old)):
                if raw_scores_list_old[index] == 'N/A':
                    NA_positions.append(index)
                    # print("DEBUG ERROR: sorted_lowest_docking_Energy_dict contains N/A values!!!")
                    # sys.exit(1)
                else:
                    try:
                        raw_scores_list_new.append(float(raw_scores_list_old[index]))
                    except:
                        print(bcolors.FAIL + "ERROR: non-number docking score detected:", raw_scores_list_old[index] + bcolors.ENDC)
                        sys.exit(1)
        # print("DEBUG: raw_scores_list_old=",raw_scores_list_old)
        # print("DEBUG: raw_scores_list_new=",raw_scores_list_new)
        zscore_list = zscore(raw_scores_list_new).tolist()
        # print("DEBUG: zscore_list=", zscore_list)
        del raw_scores_list_new
        sorted_lowest_docking_Zscore_dict = OrderedDict()
        index = 0
        isomers_list = list(sorted_lowest_docking_Energy_dict.keys())
        zscore_list_cycle = cycle(zscore_list)
        for isomer in list(isomer_numofPoses_dict.keys()):
            sorted_lowest_docking_Zscore_dict[isomer] = []
            for d in range(isomer_numofPoses_dict[isomer]):
                if index in NA_positions:
                    sorted_lowest_docking_Zscore_dict[isomer].append('N/A')
                else:
                    next_zscore = next(zscore_list_cycle)
                    sorted_lowest_docking_Zscore_dict[isomer].append(next_zscore)
                index += 1
        # print("DEBUG: sorted_lowest_docking_Zscore_dict=", sorted_lowest_docking_Zscore_dict)
        if len(raw_scores_list_old) != index:
            print("ERROR: not all docking scores were converted to Z-scores!!!")
            # print("DEBUG: len(raw_scores_list_old)=", len(raw_scores_list_old), "index=", index)
            sys.exit(1)
        
        return sorted_lowest_docking_Zscore_dict


def sum_of_squares(array):
        """ Return the sum of squares of the elements in the array """
        total = 0
        for i in array:
            total += i**2
        return total


def find_ties(predicted_rankings, experimental_rankings):
    """ Return a list of ties in two arrays """
    ties = []
    tied = 1
    sorted_predicted_rankings = predicted_rankings
    sorted_predicted_rankings.sort()
    sorted_experimental_rankings = experimental_rankings
    sorted_experimental_rankings.sort()
    for i in range(0, len(sorted_experimental_rankings)-1): # find ties in the rankings based on Ki/IC50/DeltaG
        if(sorted_experimental_rankings[i] == sorted_experimental_rankings[i+1]):
            tied += 1
        else:
            if(tied > 1):
                ties.append(tied)
            tied = 1
    if(tied > 1):
        ties.append(tied)
        tied =1 # reinitialize tied to 1 to find the ties in the rankings based on docking
    
    for i in range(0, len(sorted_predicted_rankings)-1): # find ties in the rankings based on docking
        if(sorted_predicted_rankings[i] == sorted_predicted_rankings[i+1]):
            tied += 1
        else:
            if(tied > 1):
                ties.append(tied)
            tied = 1
    if(tied > 1):
        ties.append(tied)
    
    return ties


def ties_correction(M, ties):
    """ Return the correction for ties in Kendall's W """
    total = 0
    for t in ties:
        total += t**3 - t
    
    return M*total
        
    
def Kendalls_W(sorted_docking_Energy_dict, reordered_ligand_Ki_dict):
    """
        Function to return the Kendall's W and Top-Down Concordance coefficients along with their P-value
        WARNING: the current implementation presupposes that every ligand from the Ki files was docked and was assigned a score.
        :param sorted_docking_Energy_dict:  an OrderedDict object with keys the ligand names and values the respective docking energies/raw scores/Z-scores.
        :param reordered_ligand_Ki_dict:    an OrderedDict object with keys the ligand names and values the respective experimental energies/Ki/IC50/Z-scores.
                                            The order of keys in both OrderedDict objects must be the same!
        :returns:   a list of two numbers, the Kendall's W concordance coefficient and the respective P-value
    """
    from scipy import stats

    ##
    ## Kendall's W
    ##
    if isinstance(sorted_docking_Energy_dict, (dict, OrderedDict)):
        predicted_rankings = stats.rankdata(list(sorted_docking_Energy_dict.values()))
        experimental_rankings = stats.rankdata(list(reordered_ligand_Ki_dict.values()))
        n = len(list(sorted_docking_Energy_dict.values())) # number of data per variable (namely ligands)
    elif isinstance(sorted_docking_Energy_dict, (np.ndarray, np.generic)):
        predicted_rankings = stats.rankdata(sorted_docking_Energy_dict)
        experimental_rankings = stats.rankdata(reordered_ligand_Ki_dict)
        n = len(sorted_docking_Energy_dict) # number of data per variable (namely ligands)
    
    M = 2 # two variables: rankings from docking and rankings from Ki/IC50/DeltaG
    R = predicted_rankings + experimental_rankings # aray with the sums of ranks
    TIES_CORRECTION = ties_correction(M,find_ties(predicted_rankings,experimental_rankings)) # correction for ties in Kendall's W
    W = ( sum_of_squares(R) - (R.sum()**2)/n ) / ( ((n**3-n)*M**2 - TIES_CORRECTION) /12 ) # Kendall's W
    df = n-1 # the degrees of freedom for Kendall's W
    if(df>7):
        chi_squared_r = M*(n-1)*W # equivalent Friedman's chi-squared value to find the significance of W
        W_P_value = stats.chisqprob(chi_squared_r, df)
    else:
        print(bcolors.WARNING + "WARNING: Chi square and probability was not calculated in the usual way when sample size is small (<7). \n\
        Instead, direct probability must be obtained from a table of critical values, which is currently not supported." + bcolors.ENDC)
        W_P_value = 'N/A'
    
    return [W, W_P_value]


def Savage_Score(i, n):
    """
        Returns the Savage Score. See the WARNING in TopDown_Concordance() function.
    """
    S = 0
    for j in range(int(i), int(n+1)): # calculate the Savage Score with i=1
        S += 1/float(j)
    return S


def TopDown_Concordance(sorted_docking_Energy_dict, sorted_ligand_Ki_dict):
    """
        Function to return the Top-Down Condordance coefficient and the respective P-value.
        WARNING: In the current implementation the average of the scores of the tied observations is used. Other possible treatments could be:
                * The highest score in the group of ties is used.
                * The lowest score in the group of ties is used.
                * The tied observations are to be randomly untied using an IMSL random number generator.
    """
    from scipy import stats
    from numpy import array
    
    reordered_ligand_Ki_dict = OrderedDict() # the dictionary MUST BE ORDERED in order to get the correct rankings
    for ligand in list(sorted_docking_Energy_dict.keys()):
        reordered_ligand_Ki_dict[ligand] = sorted_ligand_Ki_dict[ligand]
    
    predicted_rankings = stats.rankdata(list(sorted_docking_Energy_dict.values()))
    experimental_rankings = stats.rankdata(list(reordered_ligand_Ki_dict.values()))
    M = 2 # two variables: rankings from docking and rankings from Ki/IC50/DeltaG
    n = len(list(sorted_docking_Energy_dict.values())) # number of data per variable (namely ligands)
    
    try:
        predicted_savage_scores_list = []
        for rank in predicted_rankings:
            occurence = predicted_rankings.tolist().count(rank)
            if(occurence == 1): # if this rank is not tied
                #print("DEBUG: Savage rank",rank
                predicted_savage_scores_list.append(Savage_Score(rank, n))
            elif(occurence % 2 == 0): # if the number of ties is even, e.g. 2,3,4,5->3.5
                average_savage_score = 0
                #print("DEBUG: rank",rank,"occurence",occurence,"calculating average Savage score of ranks", range(int(rank-float(occurence)/2 +0.5), int(rank+float(occurence)/2 +0.5))
                for r in range(int(rank-float(occurence)/2 +0.5), int(rank+float(occurence)/2 +0.5)):
                    #print("DEBUG: r",r," ",average_savage_score,"+",Savage_Score(r, n)
                    average_savage_score += Savage_Score(r, n)
                predicted_savage_scores_list.append(float(average_savage_score)/occurence)
            elif(occurence % 2 ==1): # if the number of ties is odd,. e.g. 2,3,4,5,6->4
                average_savage_score = 0
                #print("DEBUG: rank",rank,"occurence",occurence,"calculating average Savage score of ranks", range(int(rank-float(occurence)/2), int(rank+float(occurence)/2))
                for r in range(int(rank-float(occurence)/2), int(rank+float(occurence)/2)):
                    #print("DEBUG: r",r," ",average_savage_score,"+",Savage_Score(r, n)
                    average_savage_score += Savage_Score(r, n)
                predicted_savage_scores_list.append(float(average_savage_score)/occurence)
        #print("DEBUG: predicted_savage_scores_list=",predicted_savage_scores_list
        experimental_savage_scores_list = []
        for rank in experimental_rankings:
            occurence = experimental_rankings.tolist().count(rank)
            if(occurence == 1): # if this rank is not tied
                #print("DEBUG: Savage rank",rank
                experimental_savage_scores_list.append(Savage_Score(rank, n))
            elif(occurence % 2 == 0): # if the number of ties is even, e.g. 2,3,4,5->3.5
                average_savage_score = 0
                #print("DEBUG: rank",rank,"occurence",occurence,"calculating average Savage score of ranks", range(int(rank-float(occurence)/2 +0.5), int(rank+float(occurence)/2 +0.5))
                for r in range(int(rank-float(occurence)/2 +0.5), int(rank+float(occurence)/2 +0.5)):
                    #print("DEBUG: r",r," ",average_savage_score,"+",Savage_Score(r, n)
                    average_savage_score += Savage_Score(r, n)
                experimental_savage_scores_list.append(float(average_savage_score)/occurence)
            elif(occurence % 2 ==1): # if the number of ties is odd,. e.g. 2,3,4,5,6->4
                average_savage_score = 0
                #print("DEBUG: rank",rank,"occurence",occurence,"calculating average Savage score of ranks", range(int(rank-float(occurence)/2), int(rank+float(occurence)/2))
                for r in range(int(rank-float(occurence)/2), int(rank+float(occurence)/2)):
                    #print("DEBUG: r",r," ",average_savage_score,"+",Savage_Score(r, n)
                    average_savage_score += Savage_Score(r, n)
                experimental_savage_scores_list.append(float(average_savage_score)/occurence)
        #print("DEBUG: experimental_savage_scores_list=",experimental_savage_scores_list
        
    except ZeroDivisionError:
        print(bcolors.WARNING + "WARNING: RANK ZERO DETECTED. predicted_rankings =", predicted_rankings.tostring(), "experimental_rankings =", experimental_rankings.tostring() + bcolors.ENDC)
        
    R = array(predicted_savage_scores_list) + array(experimental_savage_scores_list) # array with the sums of Savage scores
    predicted_savage_scores_list.extend(experimental_savage_scores_list) # now predicted_savage_scores_list is useless
    Smax = max(predicted_savage_scores_list) # get the maximum Savage score among experimental and predicted rankings
    C = (sum_of_squares(R)-n*M**2)/((n-Smax)*M**2) # Top-Down Concordance coefficient
    df = n-1 # the degrees of freedom
    chi_squared_T = M*(n-1)*C # equivalent Friedman's chi-squared value to find the significance of C
    C_P_value = stats.chisqprob(chi_squared_T, df)
    
    return [C, C_P_value]


def TopDown_Concordance2(sorted_docking_Energy_dict=None, reordered_ligand_Ki_dict=None, print_WARNING=True):
        """
            Function to return the Top-Down Condordance coefficient and the respective P-value. Maximum value is 1 and minimum in between -1 (in case only two ligands
            are available) and -0.645 (in case infinite ligands are available).
            WARNING: In the current implementation the average of the scores of the tied observations is used. Other possible treatments could be:
                    * The highest score in the group of ties is used.
                    * The lowest score in the group of ties is used.
                    * The tied observations are to be randomly untied using an IMSL random number generator.
            :param sorted_docking_Energy_dict:  an array or an OrderedDict object with keys the ligand names and values the respective docking energies/raw scores/Z-scores.
            :param reordered_ligand_Ki_dict:    an array or an OrderedDict object with keys the ligand names and values the respective experimental energies/Ki/IC50/Z-scores.
                                                The order of keys in both OrderedDict objects must be the same!
            :returns:   a list of two values, the Top-Down Condordance coefficient and the respective P-value if the number of ligands is >14, otherwise the P-value=-10000.0.
        """
        from scipy import stats
        from numpy import array
        
        if isinstance(sorted_docking_Energy_dict, (dict, OrderedDict)):
            predicted_rankings = stats.rankdata(list(sorted_docking_Energy_dict.values()))
            experimental_rankings = stats.rankdata(list(reordered_ligand_Ki_dict.values()))
            #print("DEBUG: sorted_docking_Energy_dict=",sorted_docking_Energy_dict
            #print("DEBUG: reordered_ligand_Ki_dict=",reordered_ligand_Ki_dict
            #print("DEBUG: predicted_rankings=",predicted_rankings
            #print("DEBUG: experimental_rankings=",experimental_rankings
            M = 2 # two variables: rankings from docking and rankings from Ki/IC50/DeltaG
            n = len(list(sorted_docking_Energy_dict.values())) # number of data per variable (namely ligands)
        elif isinstance(sorted_docking_Energy_dict, (np.ndarray, np.generic)):
            predicted_rankings = stats.rankdata(sorted_docking_Energy_dict)
            experimental_rankings = stats.rankdata(reordered_ligand_Ki_dict)
            #print("DEBUG: sorted_docking_Energy_dict=",sorted_docking_Energy_dict
            #print("DEBUG: reordered_ligand_Ki_dict=",reordered_ligand_Ki_dict
            #print("DEBUG: predicted_rankings=",predicted_rankings
            #print("DEBUG: experimental_rankings=",experimental_rankings
            M = 2 # two variables: rankings from docking and rankings from Ki/IC50/DeltaG
            n = len(sorted_docking_Energy_dict) # number of data per variable (namely ligands)
        
        try:
            predicted_savage_scores_list = []
            for rank in predicted_rankings:
                #print("DEBUG: predicted_rank=",rank
                occurence = predicted_rankings.tolist().count(rank)
                if(occurence == 1): # if this rank is not tied
                    #print("DEBUG: Savage rank",rank
                    predicted_savage_scores_list.append(Savage_Score(rank, n))
                elif(occurence % 2 == 0): # if the number of ties is even, e.g. 2,3,4,5->3.5
                    average_savage_score = 0
                    #print("DEBUG: rank",rank,"occurence",occurence,"calculating average Savage score of ranks", range(int(rank-float(occurence)/2 +0.5), int(rank+float(occurence)/2 +0.5))
                    for r in range(int(rank-float(occurence)/2 +0.5), int(rank+float(occurence)/2 +0.5)):
                        #print("DEBUG: r",r," ",average_savage_score,"+",Savage_Score(r, n)
                        average_savage_score += Savage_Score(r, n)
                    predicted_savage_scores_list.append(float(average_savage_score)/occurence)
                elif(occurence % 2 ==1): # if the number of ties is odd,. e.g. 2,3,4,5,6->4
                    average_savage_score = 0
                    #print("DEBUG: rank",rank,"occurence",occurence,"calculating average Savage score of ranks", range(int(rank-float(occurence)/2 +0.5), int(rank+float(occurence)/2 -0.5) +1)
                    for r in range(int(rank-float(occurence)/2 +0.5), int(rank+float(occurence)/2 -0.5) +1):  # +1 add the end because range prints up to end-1
                        #print("DEBUG: r",r," ",average_savage_score,"+",Savage_Score(r, n)
                        average_savage_score += Savage_Score(r, n)
                    predicted_savage_scores_list.append(float(average_savage_score)/occurence)
            #print("DEBUG: predicted_savage_scores_list=",predicted_savage_scores_list
            experimental_savage_scores_list = []
            for rank in experimental_rankings:
                #print("DEBUG: experimental_rank=",rank
                occurence = experimental_rankings.tolist().count(rank)
                if(occurence == 1): # if this rank is not tied
                    #print("DEBUG: Savage rank",rank
                    experimental_savage_scores_list.append(Savage_Score(rank, n))
                elif(occurence % 2 == 0): # if the number of ties is even, e.g. 2,3,4,5->3.5
                    average_savage_score = 0
                    #print("DEBUG: rank",rank,"occurence",occurence,"calculating average Savage score of ranks", range(int(rank-float(occurence)/2 +0.5), int(rank+float(occurence)/2 +0.5))
                    for r in range(int(rank-float(occurence)/2 +0.5), int(rank+float(occurence)/2 +0.5)):
                        #print("DEBUG: r",r," ",average_savage_score,"+",Savage_Score(r, n)
                        average_savage_score += Savage_Score(r, n)
                    experimental_savage_scores_list.append(float(average_savage_score)/occurence)
                elif(occurence % 2 ==1): # if the number of ties is odd,. e.g. 2,3,4,5,6->4
                    average_savage_score = 0
                    #print("DEBUG: rank",rank,"occurence",occurence,"calculating average Savage score of ranks", range(int(rank-float(occurence)/2), int(rank+float(occurence)/2))
                    for r in range(int(rank-float(occurence)/2 +0.5), int(rank+float(occurence)/2 -0.5) +1):  # +1 add the end because range prints up to end-1
                        #print("DEBUG: r",r," ",average_savage_score,"+",Savage_Score(r, n)
                        average_savage_score += Savage_Score(r, n)
                    experimental_savage_scores_list.append(float(average_savage_score)/occurence)
            #print("DEBUG: experimental_savage_scores_list=",experimental_savage_scores_list
        
            #print("FOR JULIA TELES: predicted_rankings"
            #print(predicted_rankings
            #print("FOR JULIA TELES: experimental_rankings"
            #print(experimental_rankings
        
        except ZeroDivisionError:
            print(bcolors.WARNING + "WARNING: RANK ZERO DETECTED. predicted_rankings =", predicted_rankings.tostring(), "experimental_rankings =", experimental_rankings.tostring() + bcolors.ENDC)
            #print("DEBUG: sorted_docking_Energy_dict.values():",sorted_docking_Energy_dict.values(),"\n\nDEBUG: reordered_ligand_Ki_dict.values():",reordered_ligand_Ki_dict.values()
            sys.exit(2)   
        #R = array(predicted_savage_scores_list) + array(experimental_savage_scores_list) # array with the sums of Savage scores
        S_product = array(predicted_savage_scores_list) * array(experimental_savage_scores_list)
        #predicted_savage_scores_list.extend(experimental_savage_scores_list) # now predicted_savage_scores_list is useless
        #Smax = max(predicted_savage_scores_list) # get the maximum Savage score among experimental and predicted rankings
        #C = (self.sum_of_squares(R)-n*M**2)/((n-Smax)*M**2) # Top-Down Concordance coefficient for M>2
        S1 = experimental_savage_scores_list[0] # the Savage score of the first rank (I use the experimental_savage_scores_list in case of tied data at the 1st rank)
        C =  (S_product.sum()-n)/(n-S1) # Top-Down Concordance coefficient for M=2 [Iman & Conover 1987, equation (2)]
        if (n>=14 and M==2): # in that case use the Normal approximation
            Z = C/(n-1)**0.5 # the quantile of normal distribution
            C_P_value = stats.norm(loc=0, scale=1).cdf(Z) # get the P-Value (cdf) of the quantile Z from a normal distribution (mean=0, sd=1)
            #print("DEBUG: normal_quantile = ", Z, "S1 = ",S1
        elif(n>14 and M>2): # normally this condition will always be false
            df = n-1 # the degrees of freedom; this holds only if M>2
            chi_squared_T = M*(n-1)*C # equivalent Friedman's chi-squared value to fi
            C_P_value = stats.chisqprob(chi_squared_T, df)
        elif(n<14):
            if print_WARNING:
                print(
                    bcolors.WARNING + "WARNING: P_VALUE OF TOP-DOWN CONCORDANCE FOR LESS THAN 14 LIGANDS CANNOT BE CALCULATED YET" + bcolors.ENDC)
            C_P_value = -10000.0 # initialize P-value to an obviously wrong number for the user to understand if something went wrong
        
        #print("DEBUG: C=",C
        return [C, C_P_value]


def Kendalls_rank_distance(sorted_lowest_Vina_Energy_dict, sorted_ligand_Ki_dict):
    """
        FUNCTION to measure the normalized Kendall's tau rank distance between two rankings.
        If the inputs are arrays then presume that they are in the same order (each index corresponds to the same molecule in both).
    """
    if isinstance(sorted_lowest_Vina_Energy_dict, (np.ndarray, np.generic, list)):
        l1 = list(sorted_lowest_Vina_Energy_dict)
        sorted_lowest_Vina_Energy_dict = OrderedDict()
        for k,v in enumerate(l1):
            sorted_lowest_Vina_Energy_dict[str(k)]=v
        l2 = list(sorted_ligand_Ki_dict)
        sorted_ligand_Ki_dict = OrderedDict()
        for k,v in enumerate(l2):
            sorted_ligand_Ki_dict[str(k)] = v
        
    score = 0
    number_of_comparisons = 0
    ligand_list = list(sorted_ligand_Ki_dict.keys())
    for i in range(0, len(sorted_ligand_Ki_dict)):
        for j in range(i+1, len(sorted_ligand_Ki_dict)):
            potent_ligand = ligand_list[i].lower()
            weak_ligand = ligand_list[j].lower()
            #print("DEBUG: checking if lowest Vina energies for )## "+potent_ligand+" ## and ## "+weak_ligand+" ## exist: ",sorted_lowest_Vina_Energy_dict[potent_ligand]," ",sorted_lowest_Vina_Energy_dict[weak_ligand]
            if(sorted_lowest_Vina_Energy_dict[potent_ligand] != 'N/A' and sorted_lowest_Vina_Energy_dict[weak_ligand] !='N/A'):
                number_of_comparisons += 1
                #print("DEBUG: examining if )## "+potent_ligand+" ## Vina energy is lower than ## "+weak_ligand+" ##"
                if(float(sorted_lowest_Vina_Energy_dict[potent_ligand]) <= float(sorted_lowest_Vina_Energy_dict[weak_ligand])):
                    #print("DEBUG: according to Vina ligand "+potent_ligand+" is more potent than "+weak_ligand+" , namely ",sorted_lowest_Vina_Energy_dict[potent_ligand]," <= ",sorted_lowest_Vina_Energy_dict[weak_ligand]
                    score += 1
    
    return float(score)/number_of_comparisons

    
def Kendalls_tau(sorted_docking_Energy_dict, reordered_ligand_Ki_dict):
    """
        Function to return the tau-b version of Kendall's tau correlation measure for ordinal data, which accounts for ties
        :param sorted_docking_Energy_dict:  an OrderedDict object with keys the ligand names and values the respective docking energies/raw scores/Z-scores.
        :param reordered_ligand_Ki_dict:    an OrderedDict object with keys the ligand names and values the respective experimental energies/Ki/IC50/Z-scores.
                                            The order of keys in both OrderedDict objects must be the same!
        :returns:   two values, the Kendall's tau correlation coefficient and the respective P-value.
    """
    from scipy import stats

    if isinstance(reordered_ligand_Ki_dict, (dict, OrderedDict)):
        #print("DEBUG: Kendall's tau sorted_docking_Energy_dict =",sorted_docking_Energy_dict
        #print("DEBUG: Kendall's tau reordered_ligand_Ki_dict =",reordered_ligand_Ki_dict
        predicted_rankings = stats.rankdata(list(sorted_docking_Energy_dict.values()))
        experimental_rankings = stats.rankdata(list(reordered_ligand_Ki_dict.values()))
    elif isinstance(reordered_ligand_Ki_dict, (np.ndarray, np.generic)):
        #print("DEBUG: Kendall's tau sorted_docking_Energy_dict =",sorted_docking_Energy_dict
        #print("DEBUG: Kendall's tau reordered_ligand_Ki_dict =",reordered_ligand_Ki_dict
        predicted_rankings = stats.rankdata(sorted_docking_Energy_dict)
        experimental_rankings = stats.rankdata(reordered_ligand_Ki_dict)
    
    return stats.kendalltau(predicted_rankings, experimental_rankings)


class Create_Curve():
    
    """
        Class with functions that create ROC, CROC and BEDROC curves.
    """

    #from croc import ScoredData, ROC, BEDROC
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
        self.ENSEMBLE_DOCKING_HOME= ENSEMBLE_DOCKING_HOME
    
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
                if (float(sorted_ligand_ExperimentalDeltaG_dict[ligand]) < ENERGY_THRESHOLD): # if this is an active (strong binder)
                    self.expected.append(1)
                else:   # if this is an inactive (weak binder)
                    self.expected.append(0)
                self.outcomes.append(sorted_ligand_IsomerEnergy_dict[ligand])
            
            if (self.expected.count(0) == len(self.expected)):
                sys.stdout.write(
                    bcolors.BOLDBLUE + "N/A under CROC curve N/A - NO ACTIVES WERE FOUND WITH ENERGY THRESHOLD " + str(ENERGY_THRESHOLD) + ", INCREASE ENERGY THRESHOLD USING --threshold \n" + bcolors.ENDBOLD)
                return
            elif (self.expected.count(1) == len(self.expected)):
                sys.stdout.write(
                    bcolors.BOLDBLUE + "N/A under CROC curve N/A - NO INACTIVES WERE FOUND WITH ENERGY THRESHOLD " + str(ENERGY_THRESHOLD) + ", LOWER ENERGY THRESHOLD USING --threshold \n" + bcolors.ENDBOLD)
                return
            
            self.sorted_outcome_expected_molname_tuple_list = [] # list of the form (score, activity, compound name); score may be a docking score of Z-score activity is 0 or 1
            for score, activity, ligand in zip(self.outcomes, self.expected, list(sorted_ligand_ExperimentalDeltaG_dict.keys())):
                self.sorted_outcome_expected_molname_tuple_list.append((score, activity, ligand))
            self.sorted_outcome_expected_molname_tuple_list.sort(key=itemgetter(0))
        
        elif isinstance(sorted_ligand_ExperimentalDeltaG_dict, (np.ndarray, np.generic, list, tuple,
                                                                pd.core.series.Series)):
            if ENERGY_THRESHOLD == None:
                RT = 0.5957632602
                ENERGY_THRESHOLD = np.min(sorted_ligand_ExperimentalDeltaG_dict) + RT * math.log(100)
            
            self.molname2scaffold_dict = molname2scaffold_dict
            # In this case both sorted_ligand_ExperimentalDeltaG_dict and sorted_ligand_IsomerEnergy_dict are in the same order
            if ENERGY_THRESHOLD == "ACTIVITIES":    # this means than the sorted_ligand_ExperimentalDeltaG_dict array consists of activity (0 & 1 values)
                self.expected = list(sorted_ligand_ExperimentalDeltaG_dict)
                self.outcomes = list(sorted_ligand_IsomerEnergy_dict)
            else:   # othrwise find which are the actives and which the inactives and save them
                for score, ExpDG in zip(sorted_ligand_IsomerEnergy_dict, sorted_ligand_ExperimentalDeltaG_dict):
                    if (ExpDG < ENERGY_THRESHOLD): # if this is an active (strong binder)
                        self.expected.append(1)
                    else:   # if this is an inactive (weak binder)
                        self.expected.append(0)
                    self.outcomes.append(score)

            self.sorted_outcome_expected_molname_tuple_list = [] # list of the form (score, activity, compound name); score may be a docking score of Z-score activity is 0 or 1

            if (self.expected.count(0) == len(self.expected)):
                sys.stdout.write(
                    bcolors.BOLDBLUE + "N/A under CROC curve N/A - NO ACTIVES WERE FOUND WITH ENERGY THRESHOLD " + str(ENERGY_THRESHOLD) + ", INCREASE ENERGY THRESHOLD USING --threshold \n" + bcolors.ENDBOLD)
                return
            elif (self.expected.count(1) == len(self.expected)):
                sys.stdout.write(
                    bcolors.BOLDBLUE + "N/A under CROC curve N/A - NO INACTIVES WERE FOUND WITH ENERGY THRESHOLD " + str(ENERGY_THRESHOLD) + ", LOWER ENERGY THRESHOLD USING --threshold \n" + bcolors.ENDBOLD)
                return
            
            if molname2scaffold_dict != {}:
                for score, activity, molname in zip(self.outcomes, self.expected, molname_list):
                    if molname in actives_list:
                        scaffold_SMILES = molname2scaffold_dict[molname]
                    else:
                        scaffold_SMILES = ""
                    self.sorted_outcome_expected_molname_tuple_list.append((score, activity, "", scaffold_SMILES))   # empty ligand name because we use arrays
            elif molname2scaffold_dict == {}:
                for score, activity, molname in zip(self.outcomes, self.expected, molname_list):
                    self.sorted_outcome_expected_molname_tuple_list.append((score, activity, "", ""))   # empty ligand name because we use arrays
            self.sorted_outcome_expected_molname_tuple_list.sort(key=itemgetter(0))
        
        ## Recall that NN1 & NN2 scores have been negated (-1*) to comply with Vina & DSX scoring (the lowest the better). In order to use CROC functions correctly
        ## you have to negate them again (-1*) in order to become the higher the better.
        self.outcomes = [-1*float(x) for x in self.outcomes]
        
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
        N_sel0 = int((x/100.0)*N_total)  # num of compounds in the top x %
        if N_sel0 == 0:
            ColorPrint("FAIL: given Enrichment Factor percentage (%f%%) is too low for just %i molecules!" % (x, N_total), "FAIL")
            return ('N/A', 0, 0, 0)
        N_sel = N_sel0  # number of compounds within all the ranks that are <N_sel0.
        # MEASURE ALL THE COMPOUNDS THAT BELONG TO ALL THE RANKS THAT ARE < x. DON'T LEAVE ANY RANK INCOMPLETE (MEASURE ALL ITS COMPOUNDS). 
        outcome_rankings = stats.rankdata([o[0] for o in self.sorted_outcome_expected_molname_tuple_list])   # array with the rankings of the sorted docking scores
        rank = outcome_rankings[N_sel0-1]   # get the rank of the xth compound
        # print("DEBUG: N_sel=", N_sel, " N_sel0=", N_sel0)
        while rank < N_sel0:
            # print("DEBUG: N_sel=", N_sel, " N_sel0=", N_sel0)
            N_sel += 1
            rank = outcome_rankings[N_sel-1]
        
        Hits_x = 0   # num of actives in the top x %
        for triplet in self.sorted_outcome_expected_molname_tuple_list[:N_sel]:
            if triplet[1] == 1:
                Hits_x += 1
        
        Hits_total = 0   # num of actives in the entire dataset
        for triplet in self.sorted_outcome_expected_molname_tuple_list:
            if triplet[1] == 1:
                Hits_total += 1
        
        # print("DEBUG: N_sel=", N_sel, "N_total=", N_total, "Hits_x=", Hits_x, "Hits_total=", Hits_total)
        EF = (float(Hits_x)/N_sel)/(float(Hits_total)/N_total)
        # print("DEBUG: Hits_x=%i N_set=%i Hits_total=%i N_total=%i" % (Hits_x, N_sel, Hits_total, N_total))
        norm_EF = EF/( (float(N_sel)/N_sel)/(float(Hits_total)/N_total) )  # divide by the maximum value the EF can take
        return (norm_EF, EF, Hits_x, N_sel) # N_sel is the number of compounds in the x% of the database
    
    
    def scaffold_Enrichment_Factor(self, x):
        """
            Function to calculate the scaffold enrichment factor at a given percentage x of the database screened.
        """
        from scipy import stats

        # print("DEBUG: Calculating EF("+str(x)+"%)..")
        N_total = len(self.sorted_outcome_expected_molname_tuple_list)  # total num of compounds in the entire dataset
        N_sel0 = int((x/100.0)*N_total)  # num of compounds in the top x %
        if N_sel0 == 0:
            ColorPrint("FAIL: given Enrichment Factor percentage (%f%%) is too low for just %i molecules!" % (x, N_total), "FAIL")
            return ('N/A', 0)
        N_sel = N_sel0  # number of compounds within all the ranks that are <N_sel0.
        # MEASURE ALL THE COMPOUNDS THAT BELONG TO ALL THE RANKS THAT ARE < x. DON'T LEAVE ANY RANK INCOMPLETE (MEASURE ALL ITS COMPOUNDS). 
        outcome_rankings = stats.rankdata([o[0] for o in self.sorted_outcome_expected_molname_tuple_list])   # array with the rankings of the sorted docking scores
        rank = outcome_rankings[N_sel0-1]   # get the rank of the xth compound
        # print("DEBUG: N_sel=", N_sel, " N_sel0=", N_sel0)
        while rank < N_sel0:
            # print("DEBUG: N_sel=", N_sel, " N_sel0=", N_sel0)
            N_sel += 1
            rank = outcome_rankings[N_sel-1]

        scaffolds_x = [] # scaffolds in the top x %
        for quadruplet in self.sorted_outcome_expected_molname_tuple_list[:N_sel]:
            if quadruplet[1] == 1:  # if this is an active then the scaffold_SMILES string will not be empty
                scaffolds_x.append(quadruplet[3])
        Hits_x = len(set(scaffolds_x))   # num of diverse scaffolds for the actives in the top x %
        
        Hits_total = len(set(self.molname2scaffold_dict.values()))   # num of unique scafolds for the actives in the entire database
        
        # print("DEBUG: N_sel=", N_sel, "N_total=", N_total, "Hits_x=", Hits_x, "Hits_total=", Hits_total)
        EF = (float(Hits_x)/N_sel)/(float(Hits_total)/N_total)
        norm_EF = EF/( (float(N_sel)/N_sel)/(float(Hits_total)/N_total) )
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
            print(bcolors.WARNING + "WARNING: the number of molecules in the dataset is smaller than " + str(x) + "!" + bcolors.ENDC)
            return ('N/A', 'N/A', [])
        
        # MEASURE ALL THE COMPOUNDS THAT BELONG TO ALL THE RANKS THAT ARE < x. DON'T LEAVE ANY RANK INCOMPLETE (MEASURE ALL ITS COMPOUNDS). 
        outcome_rankings = stats.rankdata([o[0] for o in self.sorted_outcome_expected_molname_tuple_list])   # array with the rankings of the sorted docking scores
        rank = outcome_rankings[x-1]   # get the rank of the xth compound
        N_sel = x
        # print("DEBUG: N_sel=", N_sel, " x=", x)
        while rank < x:
            # print("DEBUG: N_sel=", N_sel, " x=", x)
            N_sel += 1
            rank = outcome_rankings[N_sel-1]
        
        Hits = 0
        found_actives_list = [] # list of the actives ligand names found in the top x molecules
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
        if (self.write_results_list_length > 0):    # if write_results_list is not empty write the coordinates of the curve into a file for plotting
            #print("DEBUG: self.write_results_list=",self.write_results_list
            if not self.os.path.exists(self.ENSEMBLE_DOCKING_HOME+"/plots"):
                self.os.makedirs(self.ENSEMBLE_DOCKING_HOME+"/plots")
            if (self.write_results_list_length == 4 ):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME+"/plots/"+self.write_results_list[0]+"_cluster"+self.write_results_list[1]+"_"+self.write_results_list[2]+"_ROC_curve.dat"
            elif (self.write_results_list_length == 5 ):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME+"/plots/"+self.write_results_list[0]+"_"+self.write_results_list[2]+"_cluster"+self.write_results_list[1]+"_"+self.write_results_list[3]+"_ROC_curve.dat"
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
        if (self.write_results_list_length > 0):    # if write_results_list is not empty write the coordinates of the curve into a file for plotting
            #print("DEBUG: self.write_results_list=",self.write_results_list
            if (self.write_results_list_length == 4 ):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME+"/plots/"+self.write_results_list[0]+"_cluster"+self.write_results_list[1]+"_"+self.write_results_list[2]+"_CROC_curve.dat"
            elif (self.write_results_list_length == 5 ):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME+"/plots/"+self.write_results_list[0]+"_"+self.write_results_list[2]+"_cluster"+self.write_results_list[1]+"_"+self.write_results_list[3]+"_CROC_curve.dat"
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
        if (self.write_results_list_length > 0):    # if write_results_list is not empty write the coordinates of the curve into a file for plotting
            #print("DEBUG: self.write_results_list=",self.write_results_list
            if (self.write_results_list_length == 4 ):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME+"/plots/"+self.write_results_list[0]+"_cluster"+self.write_results_list[1]+"_"+self.write_results_list[2]+"_croc_BEDROC_curve.dat"
            elif (self.write_results_list_length == 5 ):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME+"/plots/"+self.write_results_list[0]+"_"+self.write_results_list[2]+"_cluster"+self.write_results_list[1]+"_"+self.write_results_list[3]+"_croc_BEDROC_curve.dat"
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
        croc_ROC = croc.ROC(self.scored_data.sweep_threshold()) # I use sweep_threshold() because it reproduces YARD's AUC-ROC value
        if (self.write_results_list_length > 0):    # if write_results_list is not empty write the coordinates of the curve into a file for plotting
            #print("DEBUG: self.write_results_list=",self.write_results_list
            if not self.os.path.exists(self.ENSEMBLE_DOCKING_HOME+"/plots"):
                self.os.makedirs(self.ENSEMBLE_DOCKING_HOME+"/plots")
            if (self.write_results_list_length == 4 ):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME+"/plots/"+self.write_results_list[0]+"_cluster"+self.write_results_list[1]+"_"+self.write_results_list[2]+"_croc_ROC_curve.dat"
            elif (self.write_results_list_length == 5 ):
                plotfile_name = self.ENSEMBLE_DOCKING_HOME+"/plots/"+self.write_results_list[0]+"_"+self.write_results_list[2]+"_cluster"+self.write_results_list[1]+"_"+self.write_results_list[3]+"_croc_ROC_curve.dat"
        if plotfile_name:
            f = open(plotfile_name, 'w')
            croc_ROC.write_to_file(f)
            f.close()
        return croc_ROC.area()


def get_figure_for_curves(self, curve_class):
    """Plots curves given by `curve_class` for all the data in `self.data`.
    `curve_class` is a subclass of `BinaryClassifierPerformanceCurve`.
    `self.data` must be a dict of lists, and the ``__class__`` key of
    `self.data` must map to the expected classes of elements. Returns an
    instance of `matplotlib.figure.Figure`."""
    fig, axes = None, None

    data = self.data
    expected = data["__class__"]

    keys = sorted(data.keys())
    keys.remove("__class__")

    styles = ["r-",  "b-",  "g-",  "c-",  "dist_matrix-",  "y-",  "k-", \
              "r--", "b--", "g--", "c--", "dist_matrix--", "y--", "k--"]

    # Plot the curves
    line_handles, labels, aucs = [], [], []
    for key, style in zip(keys, cycle(styles)):
        self.log.info("Calculating %s for %s..." %
                (curve_class.get_friendly_name(), key))
        observed = data[key]

        bc_data = BinaryClassifierData(list(zip(observed, expected)), title=key)
        curve = curve_class(bc_data)
        
        # resample curves before plotting and AUC calculation
        curve.resample(x/2000. for x in range(2001))

        aucs.append(curve.auc())
        labels.append("%s, AUC=%.4f" % (key, aucs[-1]))

        if not fig:
            dpi = self.options.dpi
            fig = curve.get_empty_figure(dpi=dpi,
                    figsize=parse_size(self.options.size, dpi=dpi))
            axes = fig.get_axes()[0]

        line_handle = curve.plot_on_axes(axes, style=style, legend=False)
        line_handles.append(line_handle)

    # Sort the labels of the legend in decreasing order of AUC
    indices = sorted(list(range(len(aucs))), key=aucs.__getitem__,
                     reverse=True)
    line_handles = [line_handles[i] for i in indices]
    labels = [labels[i] for i in indices]
    aucs = [aucs[i] for i in indices]

    if axes:
        legend_pos = "best"

        # Set logarithmic axes if needed
        if "x" in self.options.log_scale:
            axes.set_xscale("log")
            legend_pos = "upper left"
        if "y" in self.options.log_scale:
            axes.set_yscale("log")

        # Plot the legend
        axes.legend(line_handles, labels, loc = legend_pos)

    return fig
    

def report_results(sim_array, ExpDG_array, actives_list, molname_list, molname2scaffold_dict={}, ENERGY_THRESHOLD=None):
    
    from . import USRCAT_functions
    
    # Calculate Statistics
    if molname2scaffold_dict == {}:
        OBJECTIVE_FUNCTION_LIST = ["ROC", "CROC", "BEDROC", "EF1", "EF5", "EF10"]
        AU_ROC, AU_CROC, croc_BEDROC, EF1, EF5, EF10 = USRCAT_functions.calc_Objective_Function(OBJECTIVE_FUNCTION_LIST, sim_array, ExpDG_array, ENERGY_THRESHOLD, actives_list=actives_list, molname_list=molname_list, molname2scaffold_dict=molname2scaffold_dict)
        AU_ROC_stdev, AU_CROC_stdev, croc_BEDROC_stdev, EF1_stdev, EF5_stdev, EF10_stdev = 0,0,0,0,0,0
    else:
        OBJECTIVE_FUNCTION_LIST = ["ROC", "CROC", "BEDROC", "EF1", "EF5", "EF10", "scaffold_EF1", "scaffold_EF5", "scaffold_EF10"]
        AU_ROC, AU_CROC, croc_BEDROC, EF1, EF5, EF10, scaffold_EF1, scaffold_EF5, scaffold_EF10 = USRCAT_functions.calc_Objective_Function(OBJECTIVE_FUNCTION_LIST, sim_array, ExpDG_array, ENERGY_THRESHOLD, actives_list=actives_list, molname_list=molname_list, molname2scaffold_dict=molname2scaffold_dict)
        AU_ROC_stdev, AU_CROC_stdev, croc_BEDROC_stdev, EF1_stdev, EF5_stdev, EF10_stdev, scaffold_EF1_stdev, scaffold_EF5_stdev, scaffold_EF10_stdev = 0,0,0,0,0,0,0,0,0
            
    print(bcolors.HEADER + "Calculating Area Under ROC and CROR curves ... " + bcolors.ENDC)
    ## The default Energy Threshold is the minimum Energy plus the Energy corresponding to a Ki two orders of magnitute higher than the lowest Ki value
    if ENERGY_THRESHOLD == None:
        RT = 0.5957632602
        ENERGY_THRESHOLD = np.min(ExpDG_array) + RT * math.log(100)
    sys.stdout.write(bcolors.BOLDBLUE + "Average area under ROC curve: " + bcolors.ENDBOLD)
    #print("DEBUG: write_results_list=",write_results_lis
    sys.stdout.write(bcolors.BOLDBLUE + str(AU_ROC) + "+-" + str(AU_ROC_stdev) + " under CROC curve " + str(AU_CROC) \
                     +"+-" + str(AU_CROC_stdev) +" BEDROC " + str(croc_BEDROC) +"+-" + str(croc_BEDROC_stdev) + "\n" + bcolors.ENDBOLD)
    sys.stdout.write(bcolors.BOLDBLUE + "average normalized Enrichment Factor 1% " + str(EF1) + "+-" + str(EF1_stdev) \
                     +", average normalized Enrichment Factor 5% " + str(EF5) +"+-" + str(EF5_stdev) \
                     + " , average normalized Enrichment Factor 10% " + str(EF10) +"+-" + str(EF10_stdev) + "\n" + bcolors.ENDBOLD)
    if molname2scaffold_dict != {}:
        sys.stdout.write(
            bcolors.BOLDBLUE + "average normalized scaffold Enrichment Factor 1% " + str(scaffold_EF1) + "+-" + str(scaffold_EF1_stdev) \
            +", average normalized scaffold Enrichment Factor 5% " + str(scaffold_EF5) +"+-" + str(scaffold_EF5_stdev) \
            + " , average normalized scaffold Enrichment Factor 10% " + str(scaffold_EF10) +"+-" + str(scaffold_EF10_stdev) + "\n" + bcolors.ENDBOLD)


def ratio_dist(y, pred):
    """
    ratio_dist = sqrt(Sigma[(yi/xi-t)^2]/N) ; t is the target ratio constant
    """
    y = np.array(y)
    # target_ratio = np.mean(y/pred)
    target_ratio = 1.0  # <== CHANGE ME
    return np.sqrt(np.mean((y/pred - target_ratio)**2))


def weighted_R(y, x, weights):
    """Weighted Correlation"""

    weights /= weights.sum()    # weights must sum to 1.0
    mx = np.sum(np.multiply(weights, x)) / np.sum(weights) # weighted mean X
    my = np.sum(np.multiply(weights, y)) / np.sum(weights) # weighted mean Y
    sx = np.sum(np.multiply(weights, (x-mx)**2)) / np.sum(weights) # weighted variance X
    sy = np.sum(np.multiply(weights, (y-my)**2)) / np.sum(weights) # weighted variance Y
    sxy = np.sum(np.multiply(np.multiply(weights, x-mx), y-my)) / np.sum(weights)  # weighted covariance X*Y
    R = sxy / np.sqrt(sx*sy)

    return R


def PearsonsR(y, x, b_molnum=-1, f_molnum=-1, weights=np.array([])):
    """
    Same as above, but for numpy.
    :param y:
    :param x:
    :param b_molnum:
    :return:
    """

    if b_molnum > 0 and f_molnum < 0:
        # print("DEBUG: shape before slicing y=", y.get_shape())
        # print("DEBUG: shape before slicing x=", x.get_shape())
        # Keep only the BINDING data
        y = y[:b_molnum]
        x = x[:b_molnum]
        # print("DEBUG: shape after slicing y=", y.get_shape())
        # print("DEBUG: shape after slicing x=", x.get_shape())
    elif b_molnum < 0 and f_molnum > 0:
        start = x.shape[0] - f_molnum
        x = x[start:]
        y = y[start:]

    if weights.shape == (0,): # calculate the conventional R
        R = pearsonr(x, y)[0]
    else:   # calculate the weighted R
        R = weighted_R(y, x, weights)

    # Re-scale R
    R = (1.0-R)/2.0    ; # now R has range 0-1, the lowest the better the correlation (R>0.5=anti-corelation)
    # If R < 0, then use this exponential form to corverge faster away from negative values. Otherise use the original R.
    if R < 0.5:
        return R
    else:
        return 10**R/10.0  ; # same range but now high values of R (negative correlation) are disfavored more ==> faster convergence

def RMSEc(targets, predictions):
    """
        FUNCTION to calculate the centered root mean square error as used in D3R Challenge. predictions and targets are vectors
        with predicted and experimental DeltaG, respectively.
    """
    targets = np.array(targets)
    predictions = np.array(predictions)
    diff = predictions - targets
    mean_diff = np.mean(diff)
    translated_diff = diff - mean_diff
    return np.sqrt(np.square(translated_diff).mean())


def RMSE(y, pred, weights=np.array([])):
    if weights.shape == (0,):
        return np.sqrt(((np.array(pred) - np.array(y)) ** 2).mean())
    else:
        weights /= weights.sum()    # weights must sum to 1
        return np.sqrt(np.sum(np.multiply(weights, (np.array(pred) - np.array(y)) ** 2)))


def weighted_group_rmse(Y, X, assaysize_vec, group_matrix, weights=np.array([])):
    """
    Same as above but for numpy
    :param Y:
    :param X:
    :param assaysize_vec:   1xM
    :param group_matrix:    sparse MxN
    :param weights:
    :return:
    """
    # print("assaysize_vec.shape=", assaysize_vec.shape)
    # print("group_matrix.shape=", group_matrix.shape)
    # print("weights.shape", weights.shape)
    if weights.shape == (0,):
        squared_diff = group_matrix.dot(np.transpose(np.square(X - Y)))  # MxN x Nx1 = Mx1
        # TODO: think again about the tf.reduce_mean, because in the weighted RMSE you don't divide by N!
        return np.mean(np.sqrt(squared_diff / assaysize_vec))  # Mx1 / ???; basically the mean of RMSEs
    else:
        # TODO: weight the RMSE of each assay (before the sum) by the sum of weights of its compounds
        weights /= weights.sum()    # weights must sum to 1.0
        squared_diff = group_matrix.dot( np.transpose(np.multiply(weights, np.square(X-Y))) )  # MxN x Nx1 = Mx1
        # TODO: think again about the tf.reduce_mean, because in the weighted RMSE you don't divide by N!
        return np.mean( np.sqrt( squared_diff / assaysize_vec ) )  # Mx1 / ???; basically the mean of RMSEs


def split_trainset2batches(molID_assayID_Kd_list, batch_num=1):
    """
    Function to split the training sets for deepScaffOpt to batches.
    ARGS:
    molID_assayID_Kd_list:    a list of list of the form [molID, assayID, IC50] which must have been sorted like this:
                                molID_assayID_Kd_list.sort(key=itemgetter(2)) # sort by IC50
                                molID_assayID_Kd_list.sort(key=itemgetter(1)) # sort by assayID
    batch_num:                  number of training batches.
    """
    if batch_num == 1:
        return [molID_assayID_Kd_list]
    
    chunked_molID_assayID_Kd_list = []
    approxN = len(molID_assayID_Kd_list)/batch_num
    end = 0
    assayID = None
    next_assayID = None
    for batch in range(batch_num):
        start = end
        i = end + approxN
        # print("DEBUG: start=", start, "i=", i)
        if i >= len(molID_assayID_Kd_list):   # no more compounds for extra chunks
            chunked_molID_assayID_Kd_list.append(molID_assayID_Kd_list[start:])
            return chunked_molID_assayID_Kd_list
        while assayID == next_assayID:
            if batch % 2 == 0:  # if this is an even round, add more compounds that approxN
                i += 1
            else:   # if odd, add less than approxN to balance
                i -= 1
            try:
                assayID = molID_assayID_Kd_list[i][1]
                next_assayID = molID_assayID_Kd_list[i+1][1]
            except IndexError:  # no more compounds for extra chunks
                chunked_molID_assayID_Kd_list.append(molID_assayID_Kd_list[start:])
                return chunked_molID_assayID_Kd_list
        chunked_molID_assayID_Kd_list.append(molID_assayID_Kd_list[start:i+1])
        end = i+1
        assayID = None
        next_assayID = None
    
    return chunked_molID_assayID_Kd_list


def group_Kendalls_tau_loss(pred, bind_matrix=np.array([]), function_matrix=np.array([]), weights=np.array([])):
    """
        Same as above but for numpy. Works for both combined
    """
    # print("DEBUG: before slicing pred.shape=", pred.shape)
    if function_matrix.shape == (0,):
        if bind_matrix.shape == (0,):
            raise Exception(
                ColorPrint("ERROR: you must provide at least one group comparison matrix to tf_Kendalls_tau_loss!", "FAIL"))
        else:
            comparison_matrix = bind_matrix
            N = comparison_matrix.shape[1]  # number of BINDING molecules
            pred = pred[:N]
    else:
        comparison_matrix = function_matrix
        N = comparison_matrix.shape[1]  # number of FUNCTIONAL molecules
        start = pred.shape[0] - N
        pred = pred[start:]
    # print("DEBUG: after slicing pred.shape=", pred.shape)
    if N == 1:
        return 0.0

    # print("DEBUG: bind_matrix.shape=", bind_matrix.shape)
    # print("DEBUG: function_matrix.shape=", function_matrix.shape)
    if weights.shape == (0,):
        tau = Kendalls_tau_correlation(pred, comparison_matrix)
    else:
        tau = weighted_Kendalls_tau_correlation(pred, comparison_matrix, weights)

    tau = (1.0-tau) / 2.0    ;  # now R has range 0-1, the lowest the better the correlation (R>0.5=anti-corelation)
    # If tau < 0, then use this exponential form to corverge faster away from negative values. Otherise use the original tau.
    if tau < 0.5:
        return tau
    else:
        return 10 ** tau / 10.0;  # same range but now high values of tau (low rank agreement) are disfavored more ==> faster convergence


def Kendalls_tau_correlation(pred, comparison_matrix):
    """
        Like the TensorFlow implementation but written in numpy to calculate the real tau value.
        Assumes that both Y and X are sorted based on Y and that the lowest the value the better (E.g. Free Energy)!
        ARGS:
        
        comparison_matrix:  a matrix of 0 and 1 for pairwise value comparison of predictions created with create_comparison_matrix(N). Eg.
                            array([ [ 1., -1.,  0.,  0.,  0.,  0.],
                                    [ 1.,  0., -1.,  0.,  0.,  0.],
                                    [ 1.,  0.,  0., -1.,  0.,  0.],
                                    [ 1.,  0.,  0.,  0., -1.,  0.],
                                    [ 1.,  0.,  0.,  0.,  0., -1.],
                                    [ 0.,  1., -1.,  0.,  0.,  0.],
                                    [ 0.,  1.,  0., -1.,  0.,  0.],
                                    [ 0.,  1.,  0.,  0., -1.,  0.],
                                    [ 0.,  1.,  0.,  0.,  0., -1.],
                                    [ 0.,  0.,  1., -1.,  0.,  0.],
                                    [ 0.,  0.,  1.,  0., -1.,  0.],
                                    [ 0.,  0.,  1.,  0.,  0., -1.],
                                    [ 0.,  0.,  0.,  1., -1.,  0.],
                                    [ 0.,  0.,  0.,  1.,  0., -1.],
                                    [ 0.,  0.,  0.,  0.,  1., -1.]])

    """
    if comparison_matrix.shape == (1,1):    # if there are not FUNCTIONAL data
        return -1000
    N = pred.shape[0]   ; # number of observations
    M = comparison_matrix.shape[0]  ; # number of comparisons
    pairdiff_tensor = comparison_matrix.dot(pred) -0.000001 # MxN x Nx1 = Mx1 (dot and matmul are the same!)
    C = -1.0 * np.sum( (pairdiff_tensor/np.abs(pairdiff_tensor)-1.0)/2.0 )  # -1* because the predicted energies are always negative
    D = np.sum( (pairdiff_tensor/np.abs(pairdiff_tensor)+1.0)/2.0 )
    tau = (C-D)/(C+D)   ; # C = # condordance pairs, D = # disconcordance pairs
    # tau = 10**tau/10.0  ; # same range but now high values of tau (low rank agreement) are disfavored more ==> faster convergence
    return tau

def weighted_Kendalls_tau_correlation(pred, comparison_matrix, weights):
    """
        Like the TensorFlow implementation but written in numpy to calculate the real tau value.
        Assumes that both Y and X are sorted based on Y (ascending order) and that the lowest the value the better (E.g. Free Energy)!
        ARGS:

        comparison_matrix:  a matrix of 0 and 1 for pairwise value comparison of predictions created with create_comparison_matrix(N). Eg.
                            array([ [ 1., -1.,  0.,  0.,  0.,  0.],
                                    [ 1.,  0., -1.,  0.,  0.,  0.],
                                    [ 1.,  0.,  0., -1.,  0.,  0.],
                                    [ 1.,  0.,  0.,  0., -1.,  0.],
                                    [ 1.,  0.,  0.,  0.,  0., -1.],
                                    [ 0.,  1., -1.,  0.,  0.,  0.],
                                    [ 0.,  1.,  0., -1.,  0.,  0.],
                                    [ 0.,  1.,  0.,  0., -1.,  0.],
                                    [ 0.,  1.,  0.,  0.,  0., -1.],
                                    [ 0.,  0.,  1., -1.,  0.,  0.],
                                    [ 0.,  0.,  1.,  0., -1.,  0.],
                                    [ 0.,  0.,  1.,  0.,  0., -1.],
                                    [ 0.,  0.,  0.,  1., -1.,  0.],
                                    [ 0.,  0.,  0.,  1.,  0., -1.],
                                    [ 0.,  0.,  0.,  0.,  1., -1.]])
    :param pred: Nx1 array of prediction scores sorted based on the real experimental values
    :param comparison_matrix: MxN matrix
    :param weights: Mx1 array that has the coefficients for each comparison
    :return:
    """
    if comparison_matrix.shape == (1,1):    # if there are not FUNCTIONAL data
        return -1000
    N = pred.shape[0]   ; # number of observations
    M = comparison_matrix.shape[0]  ; # number of comparisons
    pairdiff_tensor = comparison_matrix.dot(pred) -0.000001 # MxN x Nx1 = Mx1 (dot and matmul are the same!)
    C = -1.0 * np.sum( np.multiply(weights, (pairdiff_tensor/np.abs(pairdiff_tensor)-1.0)/2.0) )  # -1* because the predicted energies are always negative
    D = np.sum( np.multiply(weights, (pairdiff_tensor/np.abs(pairdiff_tensor)+1.0)/2.0) )
    tau = (C-D)/(C+D)   ; # C = # condordance pairs, D = # disconcordance pairs
    # tau = 10**tau/10.0  ; # same range but now high values of tau (low rank agreement) are disfavored more ==> faster convergence
    return tau


def positive_penalty(pred, a=1000.0):
    """
        Same as above, but for numpy arrays.
    """
    np.array(pred)
    N = len(pred)
    return a * np.mean((pred/np.abs(pred) + 1.0)/2.0)/N


def switchfun_rational(r, d0=0.0, r0=1.0, n=6, m=12):
    
    nominator = 1.0 - ((r-d0)/r0)**n
    denominator = 1.0 - ((r-d0)/r0)**m
    rational = nominator/denominator
    return rational


# def remove_uniform_columns(mat, no0=True, no1=False, noredudant_cols=False):
#     """
#         Function to remove all zero or all one columns from a bit matrix.
#     """
#     mat = np.array(mat)
#     if noredudant_cols:
#         col1_values = set(mat[0,:])
#         for val in col1_values:
#             mat = mat[:, ~np.all(mat==val, axis=0)]
#         return mat
#
#     if no0:
#         mat = mat[:, ~np.all(mat==0, axis=0)]
#     if no1:
#         mat = mat[:, ~np.all(mat==1, axis=0)]
#     return mat

def remove_uniform_columns(mat, no0=True, no1=False, noredundant_cols=False):
    """
        NEW FASTER method to remove all identical or all zero or all one columns from a bit matrix.
    """
    mat = np.array(mat)
    mat = mat[:, ~np.isnan(mat).any(axis=0)]    # remove all columns that contain 'nan'
    rownum, colnum = mat.shape
    if noredundant_cols:
        cols2remove = []
        for c in range(colnum):
            val = mat[0, c]
            remove = True
            for v in mat[:, c]:
                if v != val:
                    remove = False
                    break
            if remove:
                cols2remove.append(c)
        mat = np.delete(mat, cols2remove, axis=1)
        return mat

    # TODO: rewrite the following case in the same efficient manner!
    if no0:
        mat = mat[:, ~np.all(mat==0, axis=0)]
    if no1:
        mat = mat[:, ~np.all(mat==1, axis=0)]
    return mat

def remove_lowdiversity_columns(mat, no0=True, no1=False, noredudant_cols=False, tolerance=0.01):
    # TODO: this method is untested and I doubt that it works correctly!!!
    """
        Method to remove all low-diversity or all zero or all one columns from a bit matrix.
    """
    mat = np.array(mat)
    rownum, colnum = mat.shape
    if noredudant_cols:
        cols2remove = []
        for c in range(colnum):
            values_freq_dict = {}
            for v in mat[:, c]:
                try:
                    values_freq_dict[v] += 1
                except KeyError:
                    values_freq_dict[v] = 1
            sorted_vals = sorted(list(values_freq_dict.values()), reverse=True)
            if (np.sum(sorted_vals)-sorted_vals[0])/np.sum(sorted_vals) > tolerance:
                cols2remove.append(c)
            # col_stdev = np.mean()
            # if col_stdev > tolerance:
            #     cols2remove.append(c)
        mat = np.delete(mat, cols2remove, axis=1)
        return mat

    # TODO: rewrite the following case in the same efficient manner!
    if no0:
        mat = mat[:, ~np.all(mat==0, axis=0)]
    if no1:
        mat = mat[:, ~np.all(mat==1, axis=0)]
    return mat

def remove_correlated_binary_columns(X):
    """
    Method to remove both correlated and anti-correlated columns from binary matrices and keep only one of them.
    :param X:
    :return:
    """
    X = np.array(X)
    corrcols_list = []
    all_corrcols = set()
    N = X.shape[0]  # number of samples
    for i in range(X.shape[1]): # number of columns
        if i in all_corrcols:
            continue
        for j in range(i + 1, X.shape[1]):
            if j in all_corrcols:
                continue
            if np.count_nonzero(X[:, i] == X[:, j]) in [0, N]:
                exists = False
                for c in corrcols_list:
                    if i in c:
                        c.add(j)
                        all_corrcols.add(j)
                        exists = True
                        break
                    elif j in c:
                        c.add(i)
                        all_corrcols.add(i)
                        exists = True
                        break
                if not exists:
                    corrcols_list.append({i,j})
                    all_corrcols.add(i)
                    all_corrcols.add(j)
    cols2remove = []
    for c in corrcols_list:
        c = list(c)
        c.sort()    # ascending order
        cols2remove.extend(c[1:])
    X = np.delete(X, cols2remove, axis=1)   # remove the redundant columns
    return X

def remove_correlated_scalar_columns(X):
    """
    Method to remove both correlated and anti-correlated columns from scalar matrices and keep only one of them.
    :param X:
    :return:
    """
    X = np.array(X)
    corrcols_list = []
    all_corrcols = set()
    N = X.shape[0]  # number of samples
    for i in range(X.shape[1]): # number of columns
        if i in all_corrcols:
            continue
        for j in range(i + 1, X.shape[1]):
            if j in all_corrcols:
                continue
            if np.abs(np.corrcoef(X[:, i], X[:, j], rowvar=False)[0][1]) > 0.9999:
                exists = False
                for c in corrcols_list:
                    if i in c:
                        c.add(j)
                        all_corrcols.add(j)
                        exists = True
                        break
                    elif j in c:
                        c.add(i)
                        all_corrcols.add(i)
                        exists = True
                        break
                if not exists:
                    corrcols_list.append({i,j})
                    all_corrcols.add(i)
                    all_corrcols.add(j)
    cols2remove = []
    for c in corrcols_list:
        c = list(c)
        c.sort()    # ascending order
        cols2remove.extend(c[1:])
    X = np.delete(X, cols2remove, axis=1)   # remove the redundant columns
    return X

def borda_count(score_matrix, exponent=1, change_sign=False, energylike=False):
    """
        ARGS:
        score_matrix:   each row corresponds to a different scoring function and each column to a different compound. It is assumed that the good scores
                        are the smallest one.
    """
    (rowNum,colNum) = score_matrix.shape
    # Borda count Method (superior using -1*Z-scores rather than -1*ranks)
    score_matrix = -1*zscore(score_matrix, axis=1)  # now in each row the best scores are the highest
    borda_coeff_array = np.zeros(colNum)
    borda_coeff_array.fill(rowNum**exponent)   # exponent adds more weight to the top ranked predictors
    for k in reversed(list(range(1, rowNum))):
        extra_row = np.zeros(colNum)
        extra_row.fill(k**exponent)
        borda_coeff_array = np.vstack([borda_coeff_array, extra_row]) # stack new row of Borda coefficients
    score_matrix.sort(axis=0)   # sort each column in ascending order
    score_matrix = np.flipud(score_matrix) # flip rows so than columns are sorted in descending order (highest to lowest value)
    score_array = np.sum(-1*score_matrix * borda_coeff_array, axis=0)   # '-1*' to make the best compounds have the lowest score
                                                                                     # for correct statistics calculation
    score_array = minmax_scale(score_array) # scale the scores to be between 0 and 1.
    if change_sign:
        score_array *= -1   # change the sign
    if energylike:
        score_array = 10*score_array - 10   # now they will be between -10 and 0, like energy values
    
    return score_array

def calc_Pareto_score(data_mdict, Pareto_mode):
    """
        FUNCTION to filter out nan values, rescale al R,tau,RMSEc values using all data types (in the correct way), calculate the Pareto score
        for each random state and modify data_mdict accordingly.
    """
    is_input_list = False
    if type(data_mdict) == list:
        is_input_list = True
        seed_C_stdev_list = list(data_mdict) # copy the list
        data_mdict = tree()
        data_mdict[0]['seed_C_stdev_list'] = seed_C_stdev_list  # emulate the multi-dict
    
    all_seeds = []
    all_mean_R = []
    all_stdev_R = []
    all_mean_tau = []
    all_stdev_tau = []
    all_mean_RMSEc = []
    all_stdev_RMSEc = []
    for data in list(data_mdict.keys()):
        new_seed_C_stdev_list = []  # new list without NaN values
        for seed_C_stdev in data_mdict[data]['seed_C_stdev_list']:
            # seed_C_stdev = (random_state, (mean_R, mean_tau, mean_RMSEc), (stdev_R, stdev_tau, stdev_RMSEc))
            [random_state, (mean_R, mean_tau, mean_RMSEc), (stdev_R, stdev_tau, stdev_RMSEc)] = seed_C_stdev
            # Remove NaN and negative correlations
            if not np.isnan(mean_R) and not np.isnan(mean_tau) and not np.isnan(mean_RMSEc) and mean_R>0 and mean_tau>0:
                new_seed_C_stdev_list.append([random_state, (mean_R, mean_tau, mean_RMSEc), (stdev_R, stdev_tau, stdev_RMSEc)])
                all_seeds.append(random_state)
                all_mean_R.append(mean_R)
                all_mean_tau.append(mean_tau)
                all_mean_RMSEc.append(mean_RMSEc)
                all_stdev_R.append(stdev_R)
                all_stdev_tau.append(stdev_tau)
                all_stdev_RMSEc.append(stdev_RMSEc)
        data_mdict[data]['seed_C_stdev_list'] = new_seed_C_stdev_list   ; # new list without NaN
    
    if Pareto_mode == 1 and len(all_mean_R)>0:
        all_mean_R = minmax_scale(all_mean_R, feature_range=(0,1))
        all_mean_tau = minmax_scale(all_mean_tau, feature_range=(0,1))
        all_mean_RMSEc = -1 * minmax_scale(all_mean_RMSEc, feature_range=(-1,0))  # because we discared the MLPs with RMSEc > 4.0; scale it
                    # between [-1,0] to have equivalent value range with R and tau and change its sign because the lowest the error the better
        
    elif Pareto_mode == 2 and len(all_mean_R)>0:  # divide the mean C by the stdev C
        all_mean_R = np.array(all_mean_R)/np.array(all_stdev_R)
        all_mean_tau = np.array(all_mean_tau)/np.array(all_stdev_tau)
        all_mean_RMSEc = np.array(all_mean_RMSEc)/np.array(all_stdev_RMSEc)
        
        all_mean_R = minmax_scale(all_mean_R, feature_range=(0,1))
        all_mean_tau = minmax_scale(all_mean_tau, feature_range=(0,1))
        all_mean_RMSEc = -1 * minmax_scale(all_mean_RMSEc, feature_range=(-1,0))  # because we discared the MLPs with RMSEc > 4.0; scale it
                    # between [-1,0] to have equivalent value range with R and tau and change its sign because the lowest the error the better
        
    elif Pareto_mode == 3 and len(all_mean_R)>0:  # convert the mean values to square root
        all_mean_R = [np.sqrt(v) for v in all_mean_R]
        all_mean_tau = [np.sqrt(v) for v in all_mean_tau]
        all_mean_RMSEc = [np.sqrt(v) for v in all_mean_RMSEc]
        
        all_mean_R = minmax_scale(all_mean_R, feature_range=(0,1))
        all_mean_tau = minmax_scale(all_mean_tau, feature_range=(0,1))
        all_mean_RMSEc = -1 * minmax_scale(all_mean_RMSEc, feature_range=(-1,0))  # because we discared the MLPs with RMSEc > 4.0; scale it
                    # between [-1,0] to have equivalent value range with R and tau and change its sign because the lowest the error the better
    
    elif Pareto_mode == 4 and len(all_mean_R)>0:  # square the mean values
        all_mean_R = [v**2 for v in all_mean_R]
        all_mean_tau = [v**2 for v in all_mean_tau]
        all_mean_RMSEc = [v**2 for v in all_mean_RMSEc]
        
        all_mean_R = minmax_scale(all_mean_R, feature_range=(0,1))
        all_mean_tau = minmax_scale(all_mean_tau, feature_range=(0,1))
        all_mean_RMSEc = -1 * minmax_scale(all_mean_RMSEc, feature_range=(-1,0))  # because we discared the MLPs with RMSEc > 4.0; scale it
                    # between [-1,0] to have equivalent value range with R and tau and change its sign because the lowest the error the better
    
    determ_R_list = all_mean_R
    determ_tau_list = all_mean_tau
    determ_RMSEc_list = all_mean_RMSEc
    
    ## CALCULATE THE PARETO SCORE OF THIS MLP
    # NOTE: It is not a good idead to calculate the R per cross-val repeat because the R and tau may be negative in some repeats only and this
    # effect will not be taken into account due to (fitness)**2!
    WEIGHTS = [2.0, 2.0, 1.0]   # the R and tau correlations have higher weights than the error
    FITNESSLists_list = [determ_R_list, determ_tau_list, determ_RMSEc_list]
    Pareto_score_list = []  # I call like this the R in the equation of the hyper-ellipsoid 
    for i in range(len(FITNESSLists_list[0])):
        Pareto_score = 0
        for j in range(len(FITNESSLists_list)):
            weight = WEIGHTS[j]
            fitness = FITNESSLists_list[j][i]
            Pareto_score += weight*(fitness)**2    # assemble the equation of the hyper-ellipsoid that gives the weighted score of each MPL
        Pareto_score_list.append(Pareto_score)
    
    k = 0
    for data in list(data_mdict.keys()):
        for i in range(len(data_mdict[data]['seed_C_stdev_list'])):
            seed = data_mdict[data]['seed_C_stdev_list'][i][0]
            Pareto_score = Pareto_score_list[k]
            # replace the (random_state, (mean_R, mean_tau, mean_RMSEc), (stdev_R, stdev_tau, stdev_RMSEc)) with
            # (random_state, Pareto_score, 0.0) namely zero stdev_C
            data_mdict[data]['seed_C_stdev_list'][i] = (seed, Pareto_score, 0.0)
            k += 1
    
    if is_input_list:
        return data_mdict[0]['seed_C_stdev_list']
    else:
        return data_mdict
            

def get_nonredundant_seed_C_stdev_list(data_mdict):
    
    is_input_list = False
    if type(data_mdict) == list:
        is_input_list = True
        seed_C_stdev_list = list(data_mdict) # copy the list
        data_mdict = tree()
        data_mdict[0]['seed_C_stdev_list'] = seed_C_stdev_list  # emulate the multi-dict
    
    for data in list(data_mdict.keys()):
        new_seed_C_stdev_list = []  # new list without NaN values
        for seed_C_stdev in data_mdict[data]['seed_C_stdev_list']:
            # seed_C_stdev = (random_state, (mean_C, mean_C, mean_C), (stdev_C, stdev_C, stdev_C))
            [random_state, (mean_C, mean_C, mean_C), (stdev_C, stdev_C, stdev_C)] = seed_C_stdev
            # Remove NaN and negative correlations
            if not np.isnan(mean_C) and mean_C>0:   # basically we want R,tau>0 but RMSEc is always >0, too
                new_seed_C_stdev_list.append([random_state, mean_C, stdev_C])
        data_mdict[data]['seed_C_stdev_list'] = new_seed_C_stdev_list   ; # new list without NaN and negative correlations
    
    if is_input_list:
        return data_mdict[0]['seed_C_stdev_list']
    else:
        return data_mdict


def calc_confidence(final_scores, all_MLP_scores, percentile=95):

    all_MLP_scores = np.array(all_MLP_scores)
    err_down = np.percentile(all_MLP_scores, (100 - percentile) / 2.0, axis=0)
    err_up = np.percentile(all_MLP_scores, 100 - (100 - percentile) / 2.0, axis=0)

    ci = err_up - err_down

    yhat = all_MLP_scores.mean(axis=0)
    y = final_scores

    df = pd.DataFrame()
    df['down'] = err_down
    df['up'] = err_up
    df['final'] = y
    df['average'] = yhat
    df['deviation'] = (df['up'] - df['down']) / df['average']
    df['confidence'] = minmax_scale(df['deviation'])
    # df.reset_index(inplace=True)
    # df_sorted = df.iloc[np.argsort(df['confidence'])[::-1]]
    return df['confidence'].tolist()

def group_multi_loss(Y, X, matrices):
    """
    The 'group_multi' loss function implemented in numpy.
    :param Y:
    :param X:
    :param matrices:
    :return:
    """
    Mb, Nb = matrices.bind_group_comparison_matrix.shape    # Mb: number of comparisons, Nb: number of molecules
    Mf, Nf = matrices.function_group_comparison_matrix.shape
    Lb = float(Mb) / (Mb + Mf)
    Lf = float(Mf) / (Mb + Mf)
    bind_X = X[:Nb]
    bind_Y = Y[:Nb]

    cost = 30.0 * (Lb * group_Kendalls_tau_loss(X,
                                                bind_matrix=matrices.bind_group_comparison_matrix,
                                                weights=matrices.bind_group_weights) \
                 + Lf * group_Kendalls_tau_loss(X,
                                                function_matrix=matrices.function_group_comparison_matrix,
                                                weights=matrices.function_group_weights)) \
           + 20.0 * PearsonsR(bind_Y,
                              bind_X,
                              b_molnum=Nb,
                              weights=matrices.loss_weights) \
           + weighted_group_rmse(bind_Y,
                                 bind_X,
                                 matrices.bind_assaysize_vec,
                                 matrices.bind_group_matrix,
                                 weights=matrices.loss_weights) \
           + positive_penalty(X)
    return cost

def group_multi_loss2(Y, X, matrices):
    """
    The 'group_multi' loss function without Kendall's tau implemented in numpy.
    :param Y:
    :param X:
    :param matrices:
    :return:
    """
    Mb, Nb = matrices.bind_group_comparison_matrix.shape    # Mb: number of comparisons, Nb: number of molecules
    Mf, Nf = matrices.function_group_comparison_matrix.shape
    bind_X = X[:Nb]
    bind_Y = Y[:Nb]

    cost = 50.0 * PearsonsR(bind_Y,
                              bind_X,
                              b_molnum=Nb,
                              weights=matrices.loss_weights) \
           + weighted_group_rmse(bind_Y,
                                 bind_X,
                                 matrices.bind_assaysize_vec,
                                 matrices.bind_group_matrix,
                                 weights=matrices.loss_weights) \
           + positive_penalty(X)
    return cost

