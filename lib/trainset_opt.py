"""
Functions to optimize the training set.
"""
from scipy.optimize import minimize
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import *

from deepscaffopt.lib.data_loader import *
from lib.utils.print_functions import ColorPrint


class TrainsetOptimizer():

    def __init__(self, args):
        # LOAD THE TRAINING DATA
        self.datasets = GroupDataLoader(args=args)
        args = self.datasets.args
        # THIS OPERATIONS SUPPORTS ONLY ONE FEATVEC TYPE
        self.featvec_type = args.featvec_type_list[0]
        self.cost = None
        self.maxcost1 = None
        self.maxcost2 = None
        self.dataset_mdict = None
        self.rf = None
        self.rf1 = None
        self.rf2 = None
        self.criterion = 'mse'
        self.n_estimators = 96
        self.n_jobs = -1
        self.random_state = 2018
        self.existing_connections = set()  # set of (assay1,assay2), (assay2,assay1) [ it will be populated within populate_leaves_and_ancestors()
        self.common_molnames_mdict = tree()  # assay1 ID (uppercase) -> assay2 ID (uppercase) -> molname (lowercase)

        fname = args.WORK_DIR + "/train_set." + self.featvec_type + ".pickle"
        if os.path.exists(fname) and not args.test_IC50s_file:
            ColorPrint("Loading feature vectors from file %s." % fname, "BOLDGREEN")
            self.datasets.x_crossval[self.featvec_type], \
            self.datasets.function_x_crossval[self.featvec_type], \
            self.datasets.y_crossval[self.featvec_type], \
            self.datasets.function_y_crossval[self.featvec_type], \
            self.datasets.bind_molID_assayID_Kd_list, \
            self.datasets.function_molID_assayID_Kd_list \
             = load_pickle(args.WORK_DIR + "/train_set." + self.featvec_type + ".pickle")

        else:
            ColorPrint("Calculating feature vectors from scratch.", "BOLDGREEN")

            # load the target values from the file
            self.datasets.load_group_affinities()

            train_molname_SMILES_conformersMol_mdict, \
            test_molname_SMILES_conformersMol_mdict = \
                StructureLoader().load_ligand_structure_files(args)   # ignore the xtest mols
            # CREATE FEATURE VECTORS, CROSSVAL AND XTEST SETS
            feat = Features(self.datasets)

            self.datasets.x_crossval[self.featvec_type], \
            self.datasets.function_x_crossval[self.featvec_type], \
            self.datasets.x_xtest[self.featvec_type], \
            self.datasets.molnames_crossval[self.featvec_type], \
            self.datasets.function_molnames_crossval[self.featvec_type], \
            self.datasets.molnames_xtest[self.featvec_type] = \
                feat.create_featvecs(train_molname_SMILES_conformersMol_mdict,
                                     test_molname_SMILES_conformersMol_mdict,
                                     featvec_type=self.featvec_type,
                                     extra_descriptors=args.EXTRA_DESCRIPTORS)
            self.datasets.prepare_for_MLP_training(featvec_type=self.featvec_type)    # remove redundant features

            if args.test_IC50s_file:    # must be loaded after self.datasets.molnames_xtest is populated
                self.datasets.load_xtest_affinities(args.test_IC50s_file, self.featvec_type)

            self.datasets.y_crossval[self.featvec_type] = [float(t[2]) for t in self.datasets.bind_molID_assayID_Kd_list]
            self.datasets.function_y_crossval[self.featvec_type] = [float(t[2]) for t in self.datasets.function_molID_assayID_Kd_list]
            mmgbsa = MMGBSA()
            self.datasets.y_crossval[self.featvec_type] = mmgbsa.Ki2DeltaG(self.datasets.y_crossval[self.featvec_type],
                                                                 scale=False)  # convert the IC50s to pseudoenergies
            self.datasets.function_y_crossval[self.featvec_type] = mmgbsa.Ki2DeltaG(self.datasets.function_y_crossval[self.featvec_type],
                                                                          scale=False)  # convert the IC50s to pseudoenergies

        self.datasets.molnames_crossval[self.featvec_type] = [t[0] for t in
                                                    self.datasets.bind_molID_assayID_Kd_list]  # needed to save the last hidden layers
        self.datasets.function_molnames_crossval[self.featvec_type] = [t[0] for t in
                                                             self.datasets.function_molID_assayID_Kd_list]  # needed to save the last hidden layers

        # create a dict which returns all the molnames of each assayID (ONLY ONCE IS ENOUGH)
        self.assayID_molnames_dict = {}
        for t in self.datasets.bind_molID_assayID_Kd_list:
            molname, assayID = t[0], t[1]
            try:
                self.assayID_molnames_dict[assayID].append(molname)
            except KeyError:
                self.assayID_molnames_dict[assayID] = [molname]

        self.populate_dataset_mdict()

        if not os.path.exists(fname):
            # Save the training and test sets
            ColorPrint("Saving the training set into file %s." % fname, "BOLDBLUE")
            save_pickle(fname, self.datasets.x_crossval[self.featvec_type],
                        self.datasets.function_x_crossval[self.featvec_type],
                        self.datasets.y_crossval[self.featvec_type],
                        self.datasets.function_y_crossval[self.featvec_type],
                        self.datasets.bind_molID_assayID_Kd_list, self.datasets.function_molID_assayID_Kd_list)

    def load_new_train_IC50s_file(self, new_train_IC50s_file):
        self.datasets.args.train_IC50s_file = new_train_IC50s_file
        # load the new target values from the file
        self.datasets.load_group_affinities()
        self.cost = None

    def populate_leaves_and_ancestors(self, Assignment_Tree, longest_branch=True):
        """

        :param Assignment_Tree: The Tree structure with connectivities
        :param datasetID:
        :param longest_branch: don't add leaves to those leaves that were not expanded in previous iterations to save memory
        :return: (Assignment_Tree, BOOLEAN):    A tuple with elements the input Tree structure with new branches (if applicable), and a BOOLEAN value which is True if the function added
                                           new leaves to the Tree, or False otherwise
        """

        number_of_new_leaves = 0
        # ATTENTION: never use Assignment_Tree.iter_leaf_names(), it doesn't return the names in the order
        # ATTENTION: corresponding to Assignment_Tree.get_leaves()!!!
        for leaf in Assignment_Tree.get_leaves():
            # print("DEBUG: leaf=", leaf.get_ascii(), "leaf.name=", leaf.name)
            # try:
            for assayID in self.common_assays_set:
                overlap = len(self.common_molnames_mdict[leaf.name][assayID])  # common molecules between the two assays
                if overlap > 0 and \
                        not (leaf.name, assayID) in self.existing_connections and \
                        not (assayID, leaf.name) in self.existing_connections:
                    new_child = leaf.add_child(name=assayID)  # add a new brach
                    # print("DEBUG: adding new leaf %s with overlap %i molecules." % (assayID, overlap))
                    new_child.add_features(overlap=overlap,
                                           is_alive=True)
                    number_of_new_leaves += 1
                    self.existing_connections.add((leaf.name, assayID))
                    self.existing_connections.add((assayID, leaf.name))
            # Add leaves to all intermediate nodes, too
            ancestor_list = [ancestor for ancestor in
                             leaf.get_ancestors()]  # list with all the datasetIDs currently in the branch starting from this leaf
            for ancestor in ancestor_list:
                for assayID in self.common_assays_set:
                    overlap = len(
                        self.common_molnames_mdict[ancestor.name][assayID])  # common molecules between the two assays
                    if overlap > 0 and \
                            not (ancestor.name, assayID) in self.existing_connections and \
                            not (assayID, ancestor.name) in self.existing_connections:
                        new_child = ancestor.add_child(name=assayID)  # add a new brach
                        # print("DEBUG: adding new leaf %s with overlap %i molecules." % (assayID, overlap))
                        new_child.add_features(overlap=overlap,
                                               is_alive=True)
                        number_of_new_leaves += 1
                        self.existing_connections.add((ancestor.name, assayID))
                        self.existing_connections.add((assayID, ancestor.name))
            # except KeyError:
            #     continue

        # print(Assignment_Tree.get_ascii(show_internal=True, compact=False, attributes=["name", "overlap"]))
        # print(Assignment_Tree.get_ascii(show_internal=True, compact=False))
        if number_of_new_leaves > 0:
            return (Assignment_Tree, True)
        else:
            return (Assignment_Tree, False)

    def find_assay_combinations(self):
        """
            Method to return the largest possible __assay__ combination according to the overlapping molecules.
            WARNING: it is not exactly the same as in combine_datasets.py!
        """

        # Find the common molnames between datasets
        self.common_molnames_mdict = tree()  # assay1 ID (uppercase) -> assay2 ID (uppercase) -> molname (lowercase)
        common_molnames_set = set() # REDUNDANT
        overlap_dict = {}
        all_assayIDs = list(self.dataset_mdict.keys())
        for i in range(len(all_assayIDs)):
            assayID1 = all_assayIDs[i]
            for j in range(i + 1, len(all_assayIDs)):
                assayID2 = all_assayIDs[j]
                common_molnames = []
                for molname1 in list(self.dataset_mdict[assayID1]['Kd_dict'].keys()):
                    if molname1 in list(self.dataset_mdict[assayID2]['Kd_dict'].keys()):
                        common_molnames.append(molname1)
                        common_molnames_set.add(molname1)
                self.common_molnames_mdict[assayID1][assayID2] = common_molnames

        # print("DEBUG: self.common_molnames_mdict=", self.common_molnames_mdict)
        # First remove from self.common_molnames_mdict assays without overlap with any other assay
        assays2remove = set()
        for assayID1 in list(self.common_molnames_mdict.keys()):
            common_compounds = []
            for assayID2 in list(self.common_molnames_mdict[assayID1].keys()):
                common_compounds += self.common_molnames_mdict[assayID1][assayID2]
            if len(common_compounds) == 0:
                # print("assay %s has no overlap with any other assay!" % assayID1)
                assays2remove.add(assayID1)
        for assayID in assays2remove:  # remove those assays
            del self.common_molnames_mdict[assayID]

        self.common_assays_set = set()   # all assays with common molnames with others
        for assayID1 in list(self.common_molnames_mdict.keys()):
            for assayID2 in list(self.common_molnames_mdict[assayID1].keys()):
                if len(self.common_molnames_mdict[assayID1][assayID2]) > 0:
                    self.common_assays_set.add(assayID1)
                    self.common_assays_set.add(assayID2)

        ColorPrint("Searching for assay combinations... (memory demanding!)", "BOLDGREEN")
        all_assay_combinations = []  # each tree gives one assay combination
        # for permutation in permutations(self.dataset_mdict.keys()):
        # for permutation in permutations(self.common_assays_set):
        for root_assayID in self.common_assays_set:
            # print("DEBUG: building new tree using ClusterID permutation:", permutation)
            # sys.stdout.write("DEBUG: Expanding tree from level ")
            self.existing_connections = set()  # set of (assay1,assay2), (assay2,assay1) [ it will be populated within populate_leaves_and_ancestors()
            expand_tree = True
            Assignment_Tree = Tree()
            Root = Assignment_Tree.get_tree_root()
            # Root.add_features(name=permutation[0], overlap=-1, is_alive=True)
            Root.add_features(name=root_assayID, overlap=-1, is_alive=True)
            while expand_tree:
                Assignment_Tree, expand_tree = self.populate_leaves_and_ancestors(Assignment_Tree)

            ## WORKING WITH SETS
            # print("\nSaving chains from Tree...")
            # print(Assignment_Tree.get_ascii(show_internal=True, compact=False, attributes=["name", "overlap"]))
            # print(Assignment_Tree.get_ascii(show_internal=True, compact=False))
            assay_tree_set = set()
            for leaf in Assignment_Tree.iter_leaves():
                assay_tree_set.add(leaf.name)
                # print("DEBUG: leaf.name=", leaf.name)
                # print("DEBUG: leaf.overlap=", leaf.overlap)
                for ancestor in leaf.get_ancestors():
                    # print("DEBUG: ancestor.name=", ancestor.name)
                    assay_tree_set.add(ancestor.name)
            if len(assay_tree_set) > 1 and not assay_tree_set in all_assay_combinations:
                all_assay_combinations.append(assay_tree_set)
        # TODO: temporarily deactived
        # return list(flatten(all_assay_combinations))    # return a list of all assays that belong to at least one combination

        # print("DEBUG: all_assay_combinations=", all_assay_combinations)
        # Concatenate combinations if they share at least one common assayID
        # Example:
        # The scaled assays chembl3874241 chembl3404380 chembl908048 chembl832973 are:
        # The scaled assays chembl1806078 chembl2338980 are:
        # The scaled assays chembl1100282a chembl1015017 are:
        # Will become chembl3874241 chembl3404380 chembl908048 chembl832973 chembl1806078, chembl1100282a chembl1015017
        refined_assay_combinations = []
        for comb1 in all_assay_combinations:
            union = copy.deepcopy(comb1)
            for comb2 in all_assay_combinations:
                for assay in comb2:
                    if assay in union:
                        union = union.union(comb2)
                        break
            if not union in refined_assay_combinations:
                refined_assay_combinations.append(union)
        refined_assay_combinations = [tuple(c) for c in refined_assay_combinations]
        # print("DEBUG: refined_assay_combinations=", refined_assay_combinations)
        return refined_assay_combinations  # return all assay combinations

    def find_assay_groups(self):
        assay_combinations = self.find_assay_combinations()
        assays_in_groups = list(set(flatten(assay_combinations)))
        self.assayGroups_dict = OrderedDict()  # assay group ID -> assayIDs with common compounds
        assayID_list = list(self.dataset_mdict.keys())
        groupID = 0
        for comb in assay_combinations:
            self.assayGroups_dict[groupID] = comb
            groupID += 1
        for assayID in assayID_list:
            if assayID in assays_in_groups:
                continue
            self.assayGroups_dict[groupID] = (assayID,)
            groupID += 1

    def train_GradientBoostingRegressor(self, y=[], n_estimators=96, loss='lad', random_state=2018):
        ColorPrint("Training Gradient Boosting Regressor...", "OKBLUE")
        if len(y) == 0:
            y = self.datasets.y_crossval[self.featvec_type]

        self.rf = GradientBoostingRegressor(n_estimators=n_estimators, loss=loss,
                                        random_state=random_state)
        self.rf.fit(X=self.datasets.x_crossval[self.featvec_type], y=y)
        if loss == 'mse':
            self.cost = mean_squared_error(y,
                                           self.rf.predict(self.datasets.x_crossval[self.featvec_type]))
        elif loss == 'mae':
            self.cost = mean_absolute_error(y,
                                            self.rf.predict(self.datasets.x_crossval[self.featvec_type]))
        ColorPrint("Cost after Gradient Boosting Regressor training = %f" % self.cost, "BOLDGREEN")

    def eval_GradientBoostingRegressor(self, n_estimators=96, loss='lad', random_state=2018):

        from scipy.stats import pearsonr, kendalltau

        self.rf = GradientBoostingRegressor(n_estimators=n_estimators,
                                            loss=loss,
                                            random_state=random_state)
        self.rf.fit(X=self.datasets.x_crossval[self.featvec_type], y=self.datasets.y_crossval[self.featvec_type])
        y = self.datasets.y_xtest[self.featvec_type]
        preds = self.rf.predict(self.datasets.x_xtest[self.featvec_type])
        r = pearsonr(preds, y)[0]
        t = kendalltau(preds, y)[0]
        rmse = np.sqrt(mean_squared_error(y, preds))
        ColorPrint("Gradient Boosting Regressor evaluation on xtest: R=%f tau=%f RMSE=%f" % (r, t, rmse), "BOLDGREEN")

    def calc_maxcost1_maxcost2(self, y=[], n_estimators=96, n_jobs=-1, criterion='mse_mae', random_state=2018):
        ColorPrint("Training Random Forest Regressor...", "OKBLUE")
        if len(y) == 0:
            y = self.datasets.y_crossval[self.featvec_type]
        if criterion == "mse_mae":
            self.rf1 = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, criterion="mse",
                                             random_state=random_state)
            self.rf1.fit(X=self.datasets.x_crossval[self.featvec_type], y=y)
            self.rf2 = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, criterion="mae",
                                             random_state=random_state)
            self.rf2.fit(X=self.datasets.x_crossval[self.featvec_type], y=y)
            self.maxcost1 = mean_squared_error(y, self.rf1.predict(self.datasets.x_crossval[self.featvec_type]))
            self.maxcost2 = mean_absolute_error(y, self.rf2.predict(self.datasets.x_crossval[self.featvec_type]))
            ColorPrint("Random Forest Regressor max mse=%f"%self.maxcost1, "BOLDGREEN")
            ColorPrint("Random Forest Regressor max mae=%f"%self.maxcost2, "BOLDGREEN")


    def train_RandomForestRegressor(self, y=[], n_estimators=96, n_jobs=-1, criterion='mse', random_state=2018):
        if len(y) == 0:
            y = self.datasets.y_crossval[self.featvec_type]

        if criterion == 'mse':
            ColorPrint("Training mse Random Forest Regressor...", "OKBLUE")
            self.rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, criterion=criterion,
                                            random_state=random_state)
            self.rf.fit(X=self.datasets.x_crossval[self.featvec_type], y=y)
            self.cost = mean_squared_error(y,
                                           self.rf.predict(self.datasets.x_crossval[self.featvec_type]))
        elif criterion == 'mae':
            ColorPrint("Training mae Random Forest Regressor...", "OKBLUE")
            self.rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, criterion=criterion,
                                            random_state=random_state)
            self.rf.fit(X=self.datasets.x_crossval[self.featvec_type], y=y)
            self.cost = mean_absolute_error(y,
                                      self.rf.predict(self.datasets.x_crossval[self.featvec_type]))
        elif criterion == "mse_mae":
            ColorPrint("Training mse Random Forest Regressor...", "OKBLUE")
            self.rf1 = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, criterion="mse",
                                             random_state=random_state)
            self.rf1.fit(X=self.datasets.x_crossval[self.featvec_type], y=y)
            ColorPrint("Training mae Random Forest Regressor...", "OKBLUE")
            self.rf2 = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, criterion="mae",
                                             random_state=random_state)
            self.rf2.fit(X=self.datasets.x_crossval[self.featvec_type], y=y)
            cost1 = mean_squared_error(y, self.rf1.predict(self.datasets.x_crossval[self.featvec_type]))
            cost2 = mean_absolute_error(y, self.rf2.predict(self.datasets.x_crossval[self.featvec_type]))
            self.cost = (cost1/self.maxcost1 + cost2/self.maxcost2)/2.0

        ColorPrint("Cost after Random Forest Regressor training = %f" % self.cost, "BOLDGREEN")

    def eval_RandomForestRegressor(self, n_estimators=96, n_jobs=-1, criterion='mae', random_state=2018):

        from scipy.stats import pearsonr, kendalltau

        self.rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, criterion=criterion,
                                        random_state=random_state)
        self.rf.fit(X=self.datasets.x_crossval[self.featvec_type], y=self.datasets.y_crossval[self.featvec_type])
        y = self.datasets.y_xtest[self.featvec_type]
        preds = self.rf.predict(self.datasets.x_xtest[self.featvec_type])
        r = pearsonr(preds, y)[0]
        t = kendalltau(preds, y)[0]
        rmse = np.sqrt(mean_squared_error(y, preds))
        ColorPrint("Random Forest Regressor evaluation on xtest: R=%f tau=%f RMSE=%f" % (r,t,rmse), "BOLDGREEN")

    def populate_dataset_mdict(self):
        self.dataset_mdict = tree()
        for molname, assayID, IC50 in self.datasets.bind_molID_assayID_Kd_list:
            # TODO: add control for molnames2keep
            self.dataset_mdict[assayID]['Kd_dict'][molname] = IC50
            self.dataset_mdict[assayID]['scale'] = 1.0   # default values
            self.dataset_mdict[assayID]['shift'] = 0.0

    def save_optimizer(self):
        """
        For the nelder-mead optimizer only.
        :return:
        """
        if os.path.exists("nelder-mead.resume.pickle"):
            scales_dict, \
            cost = load_pickle("nelder-mead.resume.pickle", pred_num=2)
        else:
            cost = 1000 # trick to always save if no checkpoint file exists
        if self.cost < cost:
            scales_dict = { assayID:self.dataset_mdict[assayID]['scale'] for assayID in list(self.dataset_mdict.keys()) }
            save_pickle("nelder-mead.resume.pickle", scales_dict, self.cost)

    def load_optimizer(self):
        ColorPrint("Resuming Nelder-Mead optimization.", "BOLDGREEN")
        if os.path.exists("nelder-mead.resume.pickle"):
            scales_dict, \
            cost = load_pickle("nelder-mead.resume.pickle", pred_num=2)
            for assayID, scale in list(scales_dict.items()):
                self.dataset_mdict[assayID]['scale'] = scale

    def Cost(self, *inargs):
        """
            The same function but without using shifts!
        """
        # Save the input scale factors and shifts for each dataset
        scales = inargs[0]**1000
        for groupID, assays in list(self.assayGroups_dict.items()):  # this is Ordereddict
            # FORCE IMPLICITLY THE SCALE AND SHIFT TO BE POSITIVE
            # if inargs[0][i] <= 0:  # THIS WORKS BETTER THAN CONSTRAINS!
            if scales[groupID] < 0.001 or scales[groupID] > 1000:  # THIS WORKS BETTER THAN CONSTRAINS!
                ColorPrint("Invalid scale values. Generating new scales and continuing to new iteration.", "WARNING")
                return 1000.0
            for assayID in assays:
                self.dataset_mdict[assayID]['scale'] = inargs[0][groupID]
        print("DEBUG: scales=", scales)
        # Iterate over all pairs of common molecules, scale them accordingly (depending on the assay they belong to) and save them into
        # two lists for RMSD calculation.
        new_y = []
        for molname, assayID, IC50 in self.datasets.bind_molID_assayID_Kd_list:
            # print("molname=", molname, "scale1=",self.dataset_mdict[i]['scale'], "value1=", self.dataset_mdict[i]['Kd_dict'][molname])
            new_y.append((self.dataset_mdict[assayID]['scale']**1000) * IC50)
        # Convert the IC50s to DeltaGs before training
        mmgbsa = MMGBSA()
        # print("DEBUG: new_y=", new_y)
        new_y = mmgbsa.Ki2DeltaG(new_y, scale=False)  # convert the IC50s to pseudoenergies
        self.train_RandomForestRegressor(y=new_y,
                                         n_estimators=self.n_estimators,
                                         n_jobs=self.n_jobs,
                                         criterion=self.criterion,
                                         random_state=self.random_state)   # slow part
        # Save the scales and the cost (if applicable)
        self.save_optimizer()

        return self.cost

    def optimize(self, exclude_assayIDs=[], n_estimators=96, n_jobs=-1, criterion='mse', random_state=2018,
                 optimizer='nelder-mead', resume=False, tol=0.001):

        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.criterion = criterion
        self.random_state = random_state
        self.find_assay_groups()

        if criterion == "mse_mae":
            self.calc_maxcost1_maxcost2()

        if optimizer == "nelder-mead":
            if resume:  # resume optimization from scales that yielded the lowest cost
                self.load_optimizer()

            def con(inargs):
                """Constraints for method=SLSQP"""
                for i in range(len(inargs)):
                    if inargs[i] <= 0:  # THIS WORKS BETTER THAN CONSTRAINS!
                        return 1
                return 0
            constrains = {'type':'eq', 'fun': con}
            # Find the optimum scale factors and shifts
            inargs0 = []
            for groupID, assays in list(self.assayGroups_dict.items()):
                # do not create scales and shifts for assays that are not in the specified standard types
                # if not assayID in exclude_assayIDs:
                scale = self.dataset_mdict[assays[0]]['scale']
                inargs0.append(scale)  # initial scale factor for every dataset
                    # inargs0.append(0.0)  # initial shift for every dataset
            # for constrained
            # Warning: Method nelder-mead cannot handle constraints nor bounds!
            res = minimize(self.Cost, inargs0, method='nelder-mead',
                           options={'maxiter': 1000000, 'maxfev': 1000000, 'xatol': tol, 'fatol': tol, 'disp': True},
                           constraints=constrains)
        elif optimizer == "de": # Differencial Evolution Solver
            ColorPrint("Launching optimization with Differencial Evolution Solver.", "BOLDGREEN")
            upper = 1.00694   # because 1.00694**1000 = 1008.3080849148517
            lower = 0.9932      # because 0.9932**1000 = 0.0010882054146647891
            bounds = [(lower,upper) for groupID in list(self.assayGroups_dict.keys())]
            print("DEBUG: bounds=", bounds)
            solver = DifferentialEvolutionSolver(self.Cost, bounds, recombination=0.3)
            for i in range(100):
                best_x, best_cost = next(solver)
                save_pickle('solver.pkl', solver)

    def write_optimized_IC50s(self, fname=""):

        contents = [l.split() for l in open(self.datasets.args.train_IC50s_file, 'r')]
        if not fname:
            fname = self.datasets.args.train_IC50s_file+".opt"
        with open(fname, 'w') as f:
            for l in contents:
                if l[2] == 'B':
                    l[3] = str(self.dataset_mdict[l[1]]['scale']**1000 * float(l[3]))
                f.write(" ".join(l)+"\n")
