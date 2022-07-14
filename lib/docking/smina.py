from collections import defaultdict
from lib.ConsScoreTK_Statistics import *
from scipy.stats import kendalltau, pearsonr
from random import shuffle
import copy
import pandas as pd

from lib.lig_entropy import load_ligentropy_file
from lib.outliers import remove_outliers_1D

try:
    from rdkit.Chem import SDMolSupplier
except ImportError:
    print("WARNING: rdkit module could not be found!")

from lib.MMGBSA_functions import *
from lib.global_fun import *


class Smina():

    def __init__(self,
                 pose_sdf=None,
                 pose_sdf_pattern=None,
                 score_file=None,
                 pose_score_file=None,
                 max_poses=None):

        self.pose_sdf = pose_sdf
        self.pose_sdf_pattern = pose_sdf_pattern
        self.score_file = score_file
        self.pose_score_file = pose_score_file
        self.max_poses = max_poses
        self.best_results = []
        self.software = "SMINA"
        self.score_property_name='minimizedAffinity' # the name of field in the sdf file that has the
                                                     # docking score. 'minimizedAffinity' for Smina,
                                                     # 'r_i_docking_score' for Glide.
        self.structvar_validPoses_dict = defaultdict(list)     # structvar -> list of valid poseIDs
        if self.max_poses and self.pose_score_file:     # populate structvar_validPoses_dict
            self.find_valid_poses()

    def find_valid_poses(self):
        """
        Method that populates self.structvar_validPoses_dict, which stores the valid max_poses per structural variant
        :return:
        """

        ColorPrint("Finding the valid poseIDs of each structural variant", "OKBLUE")

        with open(self.pose_score_file, 'r') as f:
            for line in f:
                words = line.split()
                try:
                    if len(words) == 6:
                        # line format:  basename    <pose score>   stereoisomer	ionstate	tautomer	pose
                        basename, pose_score, stereo, ion, tau, poseID = words
                        pose_score = float(pose_score)
                        structvar = "%s_stereo%s_ion%s_tau%s" % (basename, stereo, ion, tau)
                    elif len(words) == 2:
                        # line format:  structvar    pose    <pose score>
                        structvar, poseID, pose_score = words
                        pose_score = float(pose_score)
                    else:   # this is probably a comment
                        continue
                except ValueError:  # this was a comment
                    continue
                self.structvar_validPoses_dict[structvar].append((poseID, pose_score))

        # Filter self.structvar_validPoses_dict byt leaving only the self.max_poses
        for structvar, poseID_score_list in list(self.structvar_validPoses_dict.items()):
            poseID_score_list.sort(key=itemgetter(1))
            self.structvar_validPoses_dict[structvar] = [p for p,s in poseID_score_list[:self.max_poses]]  # keep only poseIDs, not scores

    def load_energies_from_sdf(self, activity_file=None, affinities_file=None):
        """
        Can load the specified score property from an sdf pose file coming either from Smina or Glide.
        :param activity_file:
        :return:
        """
        
        ColorPrint("Loading all energies from file %s." % self.pose_sdf, "BOLDBLUE")

        # First load the activities to read only the poses from mols that have activity value
        if activity_file:
            activities_dict = { l.split()[0].lower():int(l.split()[1]) for l in open(activity_file, 'r') }
        elif affinities_file:   # overlook the name 'activities_dict', only the keys will be used in this function
            activities_dict = {l.split()[0].lower(): float(l.split()[1]) for l in open(affinities_file, 'r')}

        all_results = []
        suppl = SDMolSupplier(self.pose_sdf, removeHs=False)
        structvar_maxprop_dict = defaultdict(list) # applicable only to properties: 'i_i_glide_confnum'

        for mol in suppl:
            if mol == None or mol.GetNumAtoms() == 0:
                continue  # skip empty molecules
            assert mol.GetNumConformers() == 1, ColorPrint("ERROR: mol has more that 1 conformers!!!", "FAIL")
            full_molname = mol.GetProp('_Name').lower()
            basename = sub_alt(full_molname, ["_pose[0-9]+", "_iso[0-9]+", "_tau[0-9]+", "_stereo[0-9]+", "_ion[0-9]+",
                                             "_noWAT"], "").lower()
            if basename not in activities_dict.keys():
                ColorPrint("Compound %s not in activities file!" % basename, "OKRED")
                continue

            try:
                energy = float(mol.GetProp(self.score_property_name))
            except KeyError:
                ColorPrint("WARNING: property %s does not exist for molecule %s!" %
                           (self.score_property_name, full_molname), "WARNING")
                continue
            try:
                poseID = re.search(".*_pose([0-9]+).*", full_molname).group(1)
            except AttributeError:
                poseID = 'nan'
            try:
                tautomer = re.search(".*_tau([0-9]+).*", full_molname).group(1)
            except AttributeError:
                tautomer = 'nan'
            try:
                ionstate = re.search(".*_ion([0-9]+).*", full_molname).group(1)
            except AttributeError:
                ionstate = 'nan'
            try:
                stereoisomer = re.search(".*_stereo([0-9]+).*", full_molname).group(1)
            except AttributeError:
                stereoisomer = 'nan'

            # Before you save this pose, check if it is valid (if applicable)
            if len(self.structvar_validPoses_dict) > 0 and 'nan' not in [poseID, tautomer, ionstate, stereoisomer]:
                structvar = "%s_stereo%s_ion%s_tau%s" % (basename, stereoisomer, ionstate, tautomer)
                if poseID not in self.structvar_validPoses_dict[structvar]:
                    continue

            if self.score_property_name == 'i_i_glide_confnum':
                # We want the pose with the higher number of conformers per structvar
                structvar_maxprop_dict[(basename, stereoisomer, ionstate, tautomer)].append(energy)
            else:
                all_results.append((basename, energy, stereoisomer, ionstate, tautomer, poseID))

        if self.score_property_name == 'i_i_glide_confnum':
            # For each structvar keep the highest confnum value, but for the molecule, only the structvar with the
            # lowest confnum will be retained.
            for structvar in structvar_maxprop_dict.keys():
                max_confnum = max(structvar_maxprop_dict[structvar])
                basename, stereoisomer, ionstate, tautomer = structvar
                all_results.append( (basename, max_confnum, stereoisomer, ionstate, tautomer, '1') ) # fake pose

        ColorPrint("Sorting results and keep the best stereoisomer, tautomer and pose from each compound.", "BOLDBLUE")
        all_results.sort(key=itemgetter(1))  # sort from lowest energy to highest
        header = "# Contains all results. For the best result for each compound please refer to file " \
                 "BEST_%s_RESULTS.\n" \
                 "molname\tenergy\tstereoisomer\tionstate\ttautomer\tpose\n" % self.software
        writelist2file(all_results, "ALL_%s_RESULTS" % self.software, header=header)
        # Sort results and keep the best stereoisomer, tautomer and pose from each compound
        basenames = []
        for basename, energy, stereoisomer, ionstate, tautomer, poseID in all_results:
            if basename not in basenames:
                self.best_results.append((basename, energy, stereoisomer, ionstate, tautomer, poseID))
                basenames.append(basename)
        self.best_results.sort(key=itemgetter(1))   # sort by the lowest score
        header = "# Contains only the best result for each compound. For the full list of results please refer to " \
                 "ALL_%s_RESULTS.\n" \
                 "molname\tenergy\tstereoisomer\tionstate\ttautomer\tpose\n" % self.software
        writelist2file(self.best_results, "BEST_%s_RESULTS" % self.software, header=header)

    def load_energies_from_multi_sdfs(self, activity_file):
        # TODO: adapt the code to read "_stereo1_ion1_tau1_pose" molnames

        ColorPrint("Loading all energies from file %s." % self.pose_sdf, "BOLDBLUE")

        # First load the activities to read only the poses from mols that have activity value
        activities_dict = { l.split()[0]:int(l.split()[1]) for l in open(activity_file, 'r') }

        all_results = []
        sdf_list = list_files(".", pattern=self.pose_sdf_pattern)
        for sdf in sdf_list:
            full_molname = sdf.replace("_poses.sdf", "")    # I assume that the docking pose file is named ${molnames}_poses.sdf
            suppl = SDMolSupplier(sdf, removeHs=False)
            for mol in suppl:
                if mol == None or mol.GetNumAtoms() == 0:
                    continue  # skip empty molecules
                assert mol.GetNumConformers() == 1, ColorPrint("ERROR: mol has more that 1 conformers!!!", "FAIL")
                energy = float(mol.GetProp('minimizedAffinity'))
                molname = sub_alt(full_molname, ["_pose[0-9]+", "_iso[0-9]+", "_tau[0-9]+", "_stereo[0-9]+", "_ion[0-9]+",
                                            "_noWAT"], "")

                if molname not in activities_dict.keys():
                    continue

                try:
                    pose = re.search(".*_pose([0-9]+).*", full_molname).group(1)
                except AttributeError:
                    pose = 'nan'
                try:
                    tautomer = re.search(".*_tau([0-9]+).*", full_molname).group(1)
                except AttributeError:
                    tautomer = 'nan'
                try:
                    ionstate = re.search(".*_ion([0-9]+).*", full_molname).group(1)
                except AttributeError:
                    ionstate = 'nan'
                try:
                    stereoisomer = re.search(".*_stereo([0-9]+).*", full_molname).group(1)
                except AttributeError:
                    stereoisomer = 'nan'

                all_results.append((molname, energy, stereoisomer, ionstate, tautomer, pose))

        ColorPrint("Sorting results and keep the best stereoisomer, tautomer and pose from each compound.", "BOLDBLUE")
        all_results.sort(key=itemgetter(1))  # sort from lowest energy to highest
        header = "# Contains all results. For the best result for each compound please refer to " \
                 "file BEST_%s_RESULTS.\n" \
                 "molname\tenergy\tstereoisomer\tionstate\ttautomer\tpose\n" % self.software
        writelist2file(all_results, "ALL_%s_RESULTS" % self.software, header=header)
        # Sort results and keep the best stereoisomer, tautomer and pose from each compound
        molnames = []
        for molname, energy, stereoisomer, ionstate, tautomer, pose in all_results:
            if not molname in molnames:
                self.best_results.append((molname, energy, stereoisomer, ionstate, tautomer, pose))
                molnames.append(molname)
        header = "# Contains only the best result for each compound. For the full list of results please refer to " \
                 "ALL_%s_RESULTS.\n" \
                 "molname\tenergy\tstereoisomer\tionstate\ttautomer\tpose\n" % self.software
        writelist2file(self.best_results, "BEST_%s_RESULTS" % self.software, header=header)

    def evaluate(self, activity_file=None, affinities_file=None, write_plots=False):
        ColorPrint("Evaluating BEST_%s_RESULTS." % self.software, "BOLDGREEN")

        # Load only the energies of the compounds that have activity value for speed
        self.load_energies_from_sdf(activity_file=activity_file,
                                    affinities_file=affinities_file)

        best_results_df = pd.DataFrame([(b[0], b[1]) for b in self.best_results], # keep only molname and energy
                                       columns=['basemolname', 'docking_score'])

        if activity_file:
            self.evaluate_from_scores_classification(best_results_df,
                                                     activity_file,
                                                     score_column='docking_score',
                                                     write_plots=write_plots,
                                                     write_scored_mols=False)
        elif affinities_file:
            # TODO: convert the first argument to dataframe, like above
            self.evaluate_from_scores_regression([(b[0], b[1]) for b in self.best_results], affinities_file)

    def load_score_file(self, score_file, only_iso1=False, frame_selection=None):
        """

        :param score_file:
        :param only_iso1:
        :param frame_selection: this is a boolean expression which will be evaluated to keep only the frames that satisfy it
                                e.g. frame_selection=">250" ignores all frames with number 250 or smaller.
        :return:
        """
        molname_scores_list = []
        ColorPrint("Loading score file %s" % score_file, "OKBLUE")
        # NEW WAY: load the score file into a Pandas table
        # ATTENTION: in Python 3 do not initialize the variables to be assigned by exec(), because otherwise
        # ATTENTION: they will be updates within locals() but as individual variables will retain their original value.
        skiprows = []   # row IDs to skip (comments)
        for r, l in enumerate(open(score_file, 'r')):
            if l.startswith("#"):
                skiprows.append(r)
            else:
                break
        table = pd.read_table(score_file, delim_whitespace=True, skiprows=skiprows, dtype={"molname":str})  # 1st line is a comment
        table_columns = table.columns
        for i, row in table.iterrows():
            # Variable assignemnt
            # ATTENTION: make sure you don't reassign these variable later in this function, otherwise it will raise an error
            # ATTENTION: in the second iteration!
            for varname in ["molname", "stereoisomer", "ionstate", "tautomer", "pose", "frame", "min_complexE", "min_lEfree",
                            "best_structvar", "best_conf", "complexE", "ligandE_bound", "proteinE_bound", "min_lEfree",
                            "Eint", "score", "DeltaH", "energy"]:
                if varname is "molname":    # exec does not apply to string variables
                    molname = row[varname]  # explicitly set string variable values
                elif varname in table_columns:
                    exec("%s = '%s'" % (varname, row[varname]), locals(), globals()) # in Python 3 swap locals() with globals()
                else:
                    exec("%s = None" % (varname), locals(), globals())

            # Check if the frame is valid
            # print("frame=", frame)
            if frame_selection != None and (frame == None or eval("%s %s" % (frame, frame_selection)) == False):
                # ColorPrint("Discarding frame %s because it doesn't satisfly the condition %s" %
                #            (frame, frame_selection), "OKRED")
                continue

            # Type conversions
            for varname in ["min_complexE", "min_lEfree", "complexE", "ligandE_bound", "proteinE_bound", "min_lEfree",
                            "Eint", "score", "DeltaH", "energy"]:
                if eval("%s != None" % varname):
                    exec("%s = float(%s)" % tuple([varname]*2), locals(), globals())

            # If these columns don't exist in the score file they are set to None: stereoisomer, ionstate, tautomer, pose
            molname_suffix = "_stereo%s_ion%s_tau%s_pose%s" % (stereoisomer, ionstate, tautomer, pose)  # assign full molname
            molname_suffix = replace_multi(molname_suffix, {"stereoNone": "stereo1", "ionNone": "ion1", "tauNone": "tau1", "poseNone": "pose1"})
            molname_suffix = replace_multi(molname_suffix, {"stereonan": "stereo1", "ionnan": "ion1", "taunan": "tau1", "posenan": "pose1"})
            molname = molname.lower() + molname_suffix.replace(".0", "")  # convert to lowercase for compatibility

            # There are 3 possible final_scores that have the following priority: score>Eint>energy
            if score != None:       # in case of consensus score
                final_score = score
            elif DeltaH != None:    # in case of DeltaH
                final_score = DeltaH
            elif Eint != None:      # in case of interaction energy
                final_score = Eint
            elif energy != None:    # in case of docking energy
                final_score = energy
            if np.isnan(final_score):
                continue

            # Before you save this pose, check if it is valid (if applicable)
            structvar = get_structvar(molname)
            if len(self.structvar_validPoses_dict) > 0 and pose not in [None, 'nan']:
                if pose not in self.structvar_validPoses_dict[structvar]:
                    continue
            # TODO: this is not True, sometime the dominant structvar is indexed by Epik as 'stereo1_ion1_tau3', etc.
            if only_iso1 and get_structvar_suffix(structvar) != "stereo1_ion1_tau1":
                continue

            # the structvar information and the poseID are within molname. As such, you have all available info in this list
            molname_scores_list.append((molname, final_score, complexE, ligandE_bound, proteinE_bound, min_lEfree, str(frame).replace("None", "1")))
        
        return molname_scores_list

    def evaluate_from_score_file_classification(self,
                                                score_file,
                                                activity_file,
                                                write_plots=False,
                                                write_scored_mols=False,
                                                only_iso1=False,
                                                keep_outliers=False,
                                                frame_selection=None,
                                                ligentropy_file=None,
                                                ligentropy_cutoff=1000):
        ColorPrint("Evaluating results from file %s" % score_file, "BOLDGREEN")
        molname_score_list = self.load_score_file(score_file, only_iso1=only_iso1, frame_selection=frame_selection)
        if ligentropy_file: # if entropies are provided, keep only the molecules below the cutoff
            basename_entropy_dict = load_ligentropy_file(fname=ligentropy_file, lowest_entropy_per_basename=False)
            molname_score_list_copy = molname_score_list.copy()
            molname_score_list = []
            for s in molname_score_list_copy:
                try:
                    if basename_entropy_dict[get_structvar(s[0])] < ligentropy_cutoff:
                        molname_score_list.append(s)
                except KeyError:
                    ColorPrint("WARNING: structural variant %s has score but not entropy value and thus will be excluded!" %
                               s[0], "OKRED")
                    pass
            ColorPrint("%i out of %i structural variants were filtered out by the imposed entropy cutoff criterium (%f)." %
                       (len(molname_score_list_copy)-len(molname_score_list), len(molname_score_list_copy), ligentropy_cutoff), "BOLDGREEN")
        self.evaluate_from_scores_classification(molname_score_list,
                                            activity_file,
                                            write_plots=write_plots,
                                            write_scored_mols=write_scored_mols,
                                            keep_outliers=keep_outliers)

    def equalize_scores_and_activities_NEW(self,
                                           results_df,
                                           score_column="Eint",
                                           sel_basemolnames=[],
                                           keep_outliers=False,
                                           average_energies=False):
        """
        TODO: in progress of re-writing the function.
        """
        # NOTE: shuffling before sorting is necessary if some energy values exist multiple times. Because
        # NOTE: usually the actives are first, in that case the enrichments will be artificially high.
        ColorPrint("Shuffling the order of the molecules to calculate more accurate enrichments.", "OKBLUE")
        results_df = results_df.sample(frac=1).reset_index(drop=True)

        if sel_basemolnames:
            results_df = results_df[results_df["basemolname"].isin(sel_basemolnames)]

        # Keep only the best score of each basemolname
        results_df = results_df.loc[
            results_df.groupby(by="basemolname", as_index=False).apply(lambda g: g.assign(keep_index=g[score_column].idxmin()))['keep_index']] \
            .dropna() \
            .reset_index(drop=True)

        ColorPrint("%i active and %i inactive structural variants with scores were found." %
                   (results_df[results_df["is_active"]==1].shape[0],
                    results_df[results_df["is_active"]==0].shape[0]), "BOLDBLUE")

        return results_df


    def equalize_scores_and_activities(self,
                                       fullmolname_scores_list,
                                       activities_dict,
                                       sel_basenames=[],
                                       keep_outliers=False,
                                       average_energies=False):
        """
        Method to return the scores only of those molnames that have activity value. Works both with bioactivities
        and experimental affinities.

        :param  fullmolname_scores_list: a list of the form [(full molname, score1, [score2, score3, ...])] but only score1 is
                                    necessary.
        :param activities_dict: dict that may contain wither bioactivities or experimental affinities
        :param sel_basenames:   basenames to keep for comparison with other scoring functions (optional)
        :returns basenames:
        :returns structvars:    the base structvar of each basename
        :returns outcomes:
        """
        valid_structvars = []
        for molname in activities_dict.keys():
            if is_structvar(molname):
                valid_structvars.append(molname)
        activities_dict = {get_basemolname(k): v for k, v in activities_dict.items()}

        # If the scores are structural variant specific, then add new entries to activities_dict
        # and create a new_scores list. At the end only the structural variant with the lowest energy will be retained.
        new_scores = []
        for ms in fullmolname_scores_list:
            fullmolname = ms[0]
            basename = get_basemolname(fullmolname)
            structvar = get_structvar( fullmolname)
            new_scores.append(tuple([basename, structvar] + list(ms[1:])))

        fltr_basenames, fltr_structvars, fltr_scores, fltr_activities = [], [], [], []
        # NOTE: shuffling before sorting is necessary if some energy values exist multiple times. Because
        # NOTE: usually the actives are first, in that case the enrichments will be artificially high.
        ColorPrint("Shuffling the order of the molecules to calculate more accurate enrichments.", "OKBLUE")
        shuffle(new_scores)    # TODO: ????? I had shuffle( fullmolname_scores_list) here, never used again!!
        new_scores.sort(key=itemgetter(2))  # sort by overall score
        # Keep only the best score of each basename
        for ns in new_scores:  # assuming the molname is a structvar
            basename, structvar, score = ns[:3] # score is the overall score, it may be Eint, Eint-defE, etc.
            if len(sel_basenames) > 0 and basename not in sel_basenames: # keep only the indicated molecules (if provided)
                continue
            # NOTE: the best score may be an outlier. Therefore, do this step later and here just filter for invalid basenames.
            if basename not in activities_dict.keys() or np.isnan(score):
                continue
            if len(valid_structvars) > 0 and structvar not in valid_structvars:
                continue
            fltr_structvars.append(structvar)
            fltr_scores.append(tuple(ns[2:]))
            fltr_activities.append(activities_dict[basename])
            fltr_basenames.append(basename)
        ColorPrint("%i active and %i inactive structural variants with scores were found." %
                   (fltr_activities.count(1), fltr_activities.count(0)), "BOLDBLUE")

        topfltr_basenames, topfltr_structvars, topfltr_scores, topfltr_activities = [], [], [], []
        if Smina.is_interactionE_file(self.score_file):
            with open("BEST_INTERACTION_ENERGIES_DECOMPOSITION.tab", 'w') as f:
                f.write("molname\tEint\tcomplexE\tligandE_bound\tproteinE_bound\tstereoisomer\tionstate\ttautomer\tpose\tframe\n")
                molname_structvar_pose_frame_prop_mdict, global_min_pEbound = Smina.load_interactionE_file(self.score_file,
                                                                                                     keep_outliers=keep_outliers)
                for structvar, scores in zip(fltr_structvars, fltr_scores):
                    basename = get_basemolname(structvar)
                    structvar_suffix = get_structvar_suffix(structvar)
                    # Save each basename only once, the one with the lowest energy/best score
                    if basename in topfltr_basenames:
                        continue
                    for poseID in molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix].keys():
                        for frameID in molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix][poseID].keys():
                            if molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix][poseID][frameID]['Eint'] == scores[0]:
                                complexE = molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix][poseID][frameID]['cE']
                                ligandE_bound = molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix][poseID][frameID]['lEbound']
                                proteinE_bound = molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix][poseID][frameID]['pEbound']
                                stereo, ion, tautomer = get_structvar_suffix(structvar, as_numbers=True)
                                f.write("%s\t%f\t%f\t%f\t%f\t%s\t%s\t%s\t%s\t%s\n" %
                                        (basename, scores[0], complexE, ligandE_bound, proteinE_bound, stereo, ion, tautomer, poseID, frameID))
                                topfltr_basenames.append(basename)
                                topfltr_structvars.append(structvar)
                                topfltr_scores.append(scores)
                                topfltr_activities.append(activities_dict[basename])
            if average_energies:
                # TODO: modify this!
                with open("AVERAGE_INTERACTION_ENERGIES_DECOMPOSITION.tab", 'w') as f:
                    f.write("molname\tEint\tcomplexE\tligandE_bound\tproteinE_bound\tstereoisomer\tionstate\ttautomer\tpose\tframe\n")
                    molname_structvar_pose_frame_prop_mdict, global_min_pEbound = Smina.load_interactionE_file(self.score_file,
                                                                                                         keep_outliers=keep_outliers)
                    for structvar, scores in zip(fltr_structvars, fltr_scores):
                        basename = get_basemolname(structvar)
                        structvar_suffix = get_structvar_suffix(structvar)
                        # Save each basename only once.
                        if basename in topfltr_basenames:
                            continue
                        for poseID in molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix].keys():
                            for frameID in molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix][poseID].keys():
                                if molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix][poseID][frameID]['Eint'] == scores[0]:
                                    complexE = molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix][poseID][frameID]['cE']
                                    ligandE_bound = molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix][poseID][frameID]['lEbound']
                                    proteinE_bound = molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix][poseID][frameID]['pEbound']
                                    stereo, ion, tautomer = get_structvar_suffix(structvar, as_numbers=True)
                                    f.write("%s\t%f\t%f\t%f\t%f\t%s\t%s\t%s\t%s\t%s\n" %
                                            (basename, scores[0], complexE, ligandE_bound, proteinE_bound, stereo, ion, tautomer, poseID, frameID))
                                    topfltr_basenames.append(basename)
                                    topfltr_structvars.append(structvar)
                                    topfltr_scores.append(scores)
                                    topfltr_activities.append(activities_dict[basename])
        else:
            # TODO: here we haven't filtered for outliers!!!
            # Write the best score of each basename to a file
            with open("BEST_SCORES_PER_BASENAME.tab", 'w') as f:  # <== CHANGE THE FILENAME
                for basename, scores in zip(fltr_basenames, fltr_scores):
                    if basename in topfltr_basenames:
                        continue
                    f.write("%s\t%s\n" % (basename, "\t".join([str(s) for s in scores])))
                    topfltr_basenames.append(basename)
                    topfltr_scores.append(scores)
                    topfltr_activities.append(activities_dict[basename])
                    topfltr_structvars.append(structvar)
            written_structvars = []
            with open("BEST_SCORES_PER_STRUCTVAR.tab", 'w') as f:  # <== CHANGE THE FILENAME
                for structvar, score in zip(fltr_structvars, fltr_scores):
                    if structvar in written_structvars:
                        continue
                    f.write("%s\t%s\n" % (structvar, "\t".join([str(s) for s in scores])))
                    written_structvars.append(structvar)

        return topfltr_basenames, topfltr_structvars, topfltr_scores, topfltr_activities

    def evaluate_from_scores_classification(self,
                                            best_results_df,
                                            activity_file,
                                            score_column="Eint",
                                            write_plots=False,
                                            write_scored_mols=False,
                                            keep_outliers=False):
        """
        Special case, where scores contains:
            lem00018316_stereo1_ion1_tau1 3
            lem00018316_stereo2_ion1_tau1 3
            lem00018316_stereo1_ion1_tau2 3
            lem00018316_stereo2_ion1_tau2 3

        but activity_file only one line:
            lem00018316 1

        :param molname_scores_list: a list of the form [(full molname, score1, [score2, score3, ...])] but only score1 is
                                    necessary.
        :param activity_file:
        :param write_plots:
        :param write_scored_mols:
        :return:
        """
        best_results_df = best_results_df.merge(pd.read_csv(activity_file,
                                          names=["basemolname", "is_active"],
                                          delim_whitespace=True) \
                                                .assign(basemolname=lambda df: df["basemolname"].str.lower()),
                              on="basemolname")

        processed_results_df = self.equalize_scores_and_activities_NEW(best_results_df,
                                                                       score_column=score_column,
                                                                       keep_outliers=keep_outliers)
        outcomes = processed_results_df[score_column]
        ColorPrint("The score values have stdev %f, the normalized score values within [-1,0] have stdev %f." %
                   (np.std(outcomes), np.std(minmax_scale(outcomes, feature_range=(-1,0)))), "OKGREEN")

        curve = Create_Curve(sorted_ligand_ExperimentalDeltaG_dict=processed_results_df["is_active"],
                             sorted_ligand_IsomerEnergy_dict=outcomes,
                             ENERGY_THRESHOLD="ACTIVITIES",
                             molname_list=processed_results_df["basemolname"])
        for x in [1, 2, 3, 4, 5, 10]:
            norm_EF, EF, Hits_x, N_sel = curve.Enrichment_Factor(x)
            if norm_EF == 'N/A':
                continue
            ColorPrint(
                "Normalized Enrichment Factor %i %% = %.3f , Enrichment Factor = %.3f , "
                "number of actives found in the top %i %% (%i) = %i\n" %
                (x, norm_EF, EF, x, N_sel, Hits_x), "BOLDGREEN")
        # Calculate AUC and write coordinates for plotting ROC, CROC, BEDROC
        if write_plots:
            auROC = curve.ROC_curve(plotfile_name="ROC_coordinatestab")
            auCROC = curve.CROC_curve(plotfile_name="CROC_coordinatestab")
            auBEDROC = curve.BEDROC_curve(plotfile_name="BEDROC_coordinatestab")
            ColorPrint("AUC-ROC = %.3f , AUC-CROC = %.3f , AUC-BEDROC = %.3f \n" %
                       (auROC, auCROC, auBEDROC), "BOLDGREEN")
        if write_scored_mols:
            processed_results_df[["basemolname", "is_active", score_column]] \
                .to_csv("scored_activities.csv", index=False)

    def evaluate_from_score_file_regression(self,
                                            score_file,
                                            affinities_file,
                                            only_iso1=False,
                                            frame_selection=None):
        ColorPrint("Evaluating results from file %s" % score_file, "BOLDGREEN")
        # TODO: update the way score_file is loaded to a DataFrame in order to be compatible with evaluate_from_scores_regression()
        # best_results_df = self.load_score_file(score_file, only_iso1=only_iso1, frame_selection=frame_selection)
        best_results_df = pd.read_csv(score_file, delim_whitespace=True, names=["basemolname", "Eint"]) \
            .astype({'basemolname': str, 'Eint': float}) \
            .assign(basemolname=lambda df: df["basemolname"].str.lower())
        self.evaluate_from_scores_regression(best_results_df, affinities_file)

    def evaluate_from_scores_regression(self, best_results_df, affinities_file, write_scored_mols=False):
        """
        :param best_results_df: is a DataFrame
        :param affinities_file:
        :return:
        """
        # affinities_dict = {l.split()[0].lower(): float(l.split()[1])
        #                    for l in open(affinities_file, 'r') if len(l.split())==2}
        best_results_df = best_results_df.merge(pd.read_csv(affinities_file,
                                                            names=["basemolname", "affinity"],
                                                            delim_whitespace=True) \
                                                .astype({'basemolname': str, 'affinity': float}) \
                                                .assign(basemolname=lambda df: df["basemolname"].str.lower()),
                                                on="basemolname") \
            .sort_values(by='Eint').dropna(axis=0)

        # OBSOLETE
        # If the scores are structural variant specific, then add new entries to activities_dict
        # and create a new_scores list. At the end only the structural variant with the lowest energy will be retained.
        # new_scores = []
        # for s in scores:
        #     molname, energy = s[0], s[1]    # the score is always the second element
        #     if '_stereo' in molname:    # then molname is a structvar
        #         basename = get_basemolname(molname)
        #         try:
        #             affinities_dict[molname] = affinities_dict[basename]
        #         except KeyError:    # this molecule has not experimental binding affinity
        #             continue
        #         new_scores.append((basename, energy))
        #     else:
        #         new_scores.append((molname, energy))
        # new_scores.sort(key=itemgetter(1)) # sort by score
        # expected, outcomes, molnames = [], [], []
        # for molname, energy in new_scores:
        #     # if molname in molnames or molname not in affinities_dict.keys() or np.isnan(energy):
        #     #     continue
        #     outcomes.append(float(energy))
        #     expected.append(affinities_dict[molname])
        #     molnames.append(molname)

        expected, outcomes, molnames = \
            best_results_df['affinity'], best_results_df['Eint'], best_results_df['basemolname']
        ColorPrint("The score values have stdev %f." % np.std(outcomes), "OKGREEN")
        # Convert the experimental binding affinities to pseudoenergies to calculate Pearson's R and RMSE
        if np.all(np.array(expected) > 0):    # if they are not binding affinities (positive) do not convert
            expected = MMGBSA.Ki2DeltaG(expected)
        ColorPrint("%i actives with scores were found." % len(molnames), "BOLDGREEN")
        r = pearsonr(expected, outcomes)[0]
        tau = kendalltau(expected, outcomes)[0]
        # rmse = RMSE(expected, outcomes)
        rmsec = RMSEc(expected, outcomes)
        ColorPrint("Pearson's R = %.3f , R^2 = %.3f , Kendall's tau = %.3f , RMSEc = %.3f kcal/mol\n" %
                   (r, r**2, tau, rmsec), "BOLDGREEN")

        if write_scored_mols:
            with open("scored_affinities.txt", 'w') as f:
                for m, e, o in zip(molnames, expected, outcomes):
                    f.write("%s\t%f\t%f\n" % (m, e, o))


    def write_docking_script(self,
                             receptor,
                             ligfile,
                             refligand,
                             pose_filename,
                             script_filename="docking.sh",
                             Nposes=50,
                             Erange=5,
                             cpus=8):
        """
        Method to write a BASH script for docking with Smina.

        :param receptor:
        :param ligfile:
        :param refligand:
        :param pose_filename:
        :param script_filename: full path and name of the docking script
        :param Nposes:
        :param Erange:
        :param cpus:
        :return:
        """

        out = """#!/bin/bash

# NECESSARY TO MAKE SMINA WORK ON UBUNTU 18.04
export LC_ALL=C

smina.static \\
--addH off \\
--num_modes %i \\
--energy_range %f \\
--min_rmsd_filter 1 \\
-r %s \\
-l %s \\
--autobox_ligand %s \\
-o %s \\
--cpu %i
        """ % (Nposes, Erange, receptor, ligfile, refligand, pose_filename, cpus)

        with open(script_filename, 'w') as f:
            f.writelines(out)

        run_commandline("chmod 777 %s" % script_filename)

    @staticmethod
    def is_interactionE_file(file):
        """
        Is this a SQM/COSMO* interaction energy file?
        :param file:
        :return:
        """
        if file == None:
            return False

        with open(file, 'r') as f:
            for line in f:
                if line.startswith("molname"):
                    words = line.split()
                    return np.all([w in words for w in ["molname", "Eint", "complexE", "ligandE_bound",
                                          "proteinE_bound", "stereoisomer", "ionstate", "tautomer", "pose", "frame"]])
        return False

    @staticmethod
    def load_interactionE_file(INTERACTION_E_FILE, keep_outliers=False, frame_selection=None):

        global_min_pEbound = 1000000
        molname_structvar_pose_frame_prop_mdict = tree()

        scores = Smina().load_score_file(INTERACTION_E_FILE, frame_selection=frame_selection)
        for (molname, Eint, complexE, ligandE_bound, proteinE_bound, min_lEfree, frame) in scores:
            molname = replace_alt(molname, ["_forscoring", "_noWAT"], "").lower()
            basename = get_basemolname(molname)
            structvar_suffix = get_structvar_suffix(molname)
            pose = get_poseID(molname)
            molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix][pose][frame]["Eint"] = float(Eint)
            molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix][pose][frame]["lEbound"] = float(ligandE_bound)
            if min_lEfree != None:  # exception for min_lEfree
                molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix][pose][frame]["lEfree"] = float(min_lEfree)
            molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix][pose][frame]["pEbound"] = proteinE_bound
            molname_structvar_pose_frame_prop_mdict[basename][structvar_suffix][pose][frame]["cE"] = float(complexE)
            if proteinE_bound < global_min_pEbound:
                global_min_pEbound = proteinE_bound

        # OBSOLETE
        # with open(INTERACTION_E_FILE, 'r') as f:
        #     for line in f:
        #         if line.startswith("#") or line.startswith("molname"):
        #             continue
        #         words = line.split()
        #         if len(words) == 10:     # NEW FORMAT
        #             molname, Eint, complexE, ligandE_bound, proteinE_bound, stereo, ion, tautomer, pose, frame = words
        #             Eint, complexE, ligandE_bound, proteinE_bound = float(Eint), float(complexE), float(ligandE_bound), float(proteinE_bound)
        #             if np.isnan(Eint):  # skip nan energies
        #                 continue
        #             molname = replace_alt(molname, ["_forscoring", "_noWAT"], "").lower()
        #             proteinE_bound = float(proteinE_bound)
        #             molname_structvar_pose_frame_prop_mdict[molname]["stereo%s_ion%s_tau%s" % (stereo, ion, tautomer)][pose][frame]["Eint"] = float(Eint)
        #             molname_structvar_pose_frame_prop_mdict[molname]["stereo%s_ion%s_tau%s" % (stereo, ion, tautomer)][pose][frame]["lEbound"] = float(ligandE_bound)
        #             molname_structvar_pose_frame_prop_mdict[molname]["stereo%s_ion%s_tau%s" % (stereo, ion, tautomer)][pose][frame]["pEbound"] = proteinE_bound
        #             molname_structvar_pose_frame_prop_mdict[molname]["stereo%s_ion%s_tau%s" % (stereo, ion, tautomer)][pose][frame]["cE"] = float(complexE)
        #             if proteinE_bound < global_min_pEbound:
        #                 global_min_pEbound = proteinE_bound

        # Removing unusually low or high interaction energies
        if keep_outliers == False:
            mdict_copy = copy.deepcopy(molname_structvar_pose_frame_prop_mdict)

            # Decide whether to delete poses instead of frames that have outlier Eint value
            delete_poses = np.all([len(mdict_copy[molname][structvar][pose].keys())==1
                for molname in mdict_copy.keys()
                for structvar in mdict_copy[molname].keys()
                for pose in mdict_copy[molname][structvar].keys()])

            for molname in mdict_copy.keys():
                for structvar in mdict_copy[molname].keys():
                    all_Eint, all_lEbound, all_pEbound, all_cE = [], [], [], []
                    for pose in mdict_copy[molname][structvar].keys():
                        for frame in mdict_copy[molname][structvar][pose].keys():
                            all_Eint.append(mdict_copy[molname][structvar][pose][frame]["Eint"])
                            all_lEbound.append(mdict_copy[molname][structvar][pose][frame]["lEbound"])
                            all_pEbound.append(mdict_copy[molname][structvar][pose][frame]["pEbound"])
                            all_cE.append(mdict_copy[molname][structvar][pose][frame]["cE"])
                        all_frames = list(mdict_copy[molname][structvar][pose].keys())
                        if len(all_frames) >= 10:    # TODO: this is also empirical, the min num of frames required to calculate zscores
                            # TODO: the zmin and zmax values are completely empirical.
                            outlier_indices1, outlier_indices2, outlier_indices3, outlier_indices4 = [], [], [], []
                            # _, outlier_indices1 = remove_outliers_1D(all_lEbound, method='mad', thresh=3.5, get_outlier_indices=True)
                            # _, outlier_indices2 = remove_outliers_1D(all_pEbound, method='mad', thresh=3.5, get_outlier_indices=True)
                            # _, outlier_indices3 = remove_outliers_1D(all_cE, method='mad', thresh=3.5, get_outlier_indices=True)
                            _, outlier_indices4 = remove_outliers_1D(all_Eint, method='mad', thresh=3.5, get_outlier_indices=True)
                            outlier_indices_set = set( outlier_indices1 + outlier_indices2 + outlier_indices3 + outlier_indices4 )
                            for i in outlier_indices_set:
                                ColorPrint("Removing frame %s_%s_%s_%s due to unusual interaction Energy %f ." %
                                           (molname, structvar, pose, all_frames[i], all_Eint[i]), "OKRED")
                                del molname_structvar_pose_frame_prop_mdict[molname][structvar][pose][all_frames[i]]

                        if delete_poses == False:   # inialize the lists for the next pose frames
                            all_Eint, all_lEbound, all_pEbound, all_cE = [], [], [], []

                    all_poses = list(mdict_copy[molname][structvar].keys())
                    if delete_poses == True and len(all_poses) >= 10:    # TODO: this is also empirical, the min num of pose required to calculate zscores
                        # TODO: the thresh value is completely empirical.
                        outlier_indices1, outlier_indices2, outlier_indices3, outlier_indices4 = [], [], [], []
                        # _, outlier_indices1 = remove_outliers_1D(all_lEbound, method='mad', thresh=3.5, get_outlier_indices=True)
                        # _, outlier_indices2 = remove_outliers_1D(all_pEbound, method='mad', thresh=3.5, get_outlier_indices=True)
                        # _, outlier_indices3 = remove_outliers_1D(all_cE, method='mad', thresh=3.5, get_outlier_indices=True)
                        _, outlier_indices4 = remove_outliers_1D(all_Eint, method='mad', thresh=3.5, get_outlier_indices=True)
                        outlier_indices_set = set(outlier_indices1 + outlier_indices2 + outlier_indices3 + outlier_indices4)
                        for i in outlier_indices_set:
                            ColorPrint("Removing pose %s_%s_%s due to unusual interaction Energy %f ." %
                                       (molname, structvar, all_poses[i], all_Eint[i]), "OKRED")
                            del molname_structvar_pose_frame_prop_mdict[molname][structvar][all_poses[i]]
                        # Initialize the lists for the next structvar poses
                        all_Eint, all_lEbound, all_pEbound, all_cE = [], [], [], []

        return molname_structvar_pose_frame_prop_mdict, global_min_pEbound