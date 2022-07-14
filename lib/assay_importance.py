from .featvec.fingerprint_computer import calculate_fingerprints_from_RDKit_mols
from sklearn.model_selection import cross_val_score, ShuffleSplit

from .USRCAT_functions import *
from .global_fun import *
from .utils.print_functions import ColorPrint


class AssayImportance():

    def __init__(self, bind_molID_assayID_Kd_list, molID2SMILES_dict, featvec_type="RDK5L"):
        self.bind_molID_assayID_Kd_list = bind_molID_assayID_Kd_list
        self.molID2SMILES_dict = molID2SMILES_dict
        self.featvec_type = featvec_type
        self.assayID_importance = {}
        self.assayIDs = set([a[1] for a in bind_molID_assayID_Kd_list])
        molID_SMILES_conformersMol_mdict = load_structures_from_SMILES(self.molID2SMILES_dict, N=0, keep_SMILES=False)
        self.molID_fingerprint_dict = calculate_fingerprints_from_RDKit_mols(molID_SMILES_conformersMol_mdict,
                                                                          featvec_type=self.featvec_type,
                                                                          as_array=True,
                                                                          nBits=4096,
                                                                          maxPath=5)

    def prepare_RF_input(self, assayID):
        mmgbsa = MMGBSA()
        molID_DeltaG_dict = {}
        molID_IC50_list = [(a[0],a[2]) for a in self.bind_molID_assayID_Kd_list if a[1]==assayID]
        for (molID, IC50) in molID_IC50_list:
            molID_DeltaG_dict[molID] = mmgbsa.Ki2DeltaG([float(IC50)])[0]

        X = [self.molID_fingerprint_dict[molID] for molID in list(molID_DeltaG_dict.keys())]
        Y = [molID_DeltaG_dict[molID] for molID in list(molID_DeltaG_dict.keys())]
        return X, Y

    def calc_assay_importance(self, X, Y, assayID):
        """
        Method to calculate the prediction errors (implicitly the importance) of one assay by cross-validation.
        :param X:
        :param Y:
        :return:
        """
        ColorPrint("Calculating importance of assay %s (%i compounds)" % (assayID, len(Y)), "OKBLUE")
        rf = RandomForestRegressor(n_jobs=1)
        lpo = ShuffleSplit(n_splits=100, test_size=0.3)
        errors = -1.0 * cross_val_score(rf, X, Y, cv=lpo, n_jobs=1, scoring='neg_mean_squared_error')
        return assayID, errors.mean(), errors.std()
        # return errors.mean()/errors.std()

    def calc_all_assay_importances(self, assayIDs=[]):
        """
        Method to calculate the importance of each assay in the dataset.
        :return:
        """
        if not assayIDs:
            assayIDs = self.assayIDs

        X_args, Y_args, assayID_args = [], [], []
        assayID_size_dict = {}
        for assayID in self.assayIDs:
            X, Y = self.prepare_RF_input(assayID)
            assayID_size_dict[assayID] = len(Y)
            if len(Y) < 3:
                ColorPrint("WARNING: cannot calculate importance of assay %s as it contains only %i compounds!" % (assayID, len(Y)),
                           "WARNING")
                continue
            # # Serial execution
            # self.assayID_importance[assayID] = self.calc_assay_importance(X, Y)
            X_args.append(X)
            Y_args.append(Y)
            assayID_args.append(assayID)
        # Parallel execution
        results = list(futures.map(self.calc_assay_importance, X_args, Y_args, assayID_args))
        self.assayID_importance = {assayID:(mu, stdev) for assayID,mu,stdev in results}
        # assayIDs, importances = self.assayID_importance.keys(), self.assayID_importance.values()  # convert errors to z-scores
        # self.assayID_importance = {k:v for k,v in zip(assayIDs, zscore(importances))}   # convert errors to z-scores
        print("assayID\tsize\tmean_importance\tstdev_importance\tnorm_mean_importance\n")
        assayID_size_importance_stdev_list = [(assayID, assayID_size_dict[assayID], self.assayID_importance[assayID][0], self.assayID_importance[assayID][1])
                                        for assayID in list(self.assayID_importance.keys())]
        assayID_size_importance_stdev_list.sort(key=itemgetter(2))
        for assayID, size, importance, stdev in assayID_size_importance_stdev_list:
            print("%s\t%i\t%f\t%f\t%f\n" % (assayID, size, importance, stdev, importance/np.sqrt(size), ))

    def clip_training_set(self, importance_threshold=1.0):
        new_bind_molID_assayID_Kd_list = []
        for a in self.bind_molID_assayID_Kd_list:
            molID, assayID = str(a[0].lower()), str(a[1].lower())
            try:
                if self.assayID_importance[assayID][0] > importance_threshold:
                    ColorPrint("Omitting compound %s from assay %s with RMSE ( %f +- %f ) above the threshold %f" %
                               (molID, assayID, self.assayID_importance[assayID][0], self.assayID_importance[assayID][1],
                                importance_threshold), "FAIL")
                    continue
                new_bind_molID_assayID_Kd_list.append(a)
            except KeyError:    # skip assays with < 3 compounds and thus no importance
                continue
        return new_bind_molID_assayID_Kd_list

