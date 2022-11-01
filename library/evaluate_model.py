import logging
import os

import pandas as pd

from library.global_fun import save_pickle
from library.model_evaluation.classification_metrics import Classification_Metric, Create_Curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import minmax_scale

lg = logging.getLogger(__name__)


def evaluate_without_learning(evaluate_df, column, use_basemolnames_from_df=None):
    if isinstance(use_basemolnames_from_df, pd.DataFrame):
        evaluate_df = evaluate_df[evaluate_df["basemolname"].isin(
            use_basemolnames_from_df["basemolname"].dropna().unique())]

    # auc_roc = 1 - roc_auc_score(evaluate_df["is_active"], evaluate_df[column])
    curve = Create_Curve(sorted_ligand_ExperimentalDeltaG_dict=evaluate_df["is_active"],
                         sorted_ligand_IsomerEnergy_dict=evaluate_df[column],
                         ENERGY_THRESHOLD="ACTIVITIES",
                         molname_list=evaluate_df["basemolname"])
    auc_roc = curve.ROC_curve()
    auCROC = curve.CROC_curve()
    auBEDROC = curve.BEDROC_curve()

    num_of_actives, num_of_inactives, min_score, mean_score, max_score, stdev_score, range_score, \
    scaled_stdev \
        = evaluate_df.loc[evaluate_df["is_active"] == 1, "basemolname"].unique().size,\
          evaluate_df.loc[evaluate_df["is_active"] == 0, "basemolname"].unique().size,\
          evaluate_df[column].min(), \
          evaluate_df[column].mean(), \
          evaluate_df[column].max(), \
          evaluate_df[column].std(), \
          abs(evaluate_df[column].min()-evaluate_df[column].max()), \
          minmax_scale(evaluate_df[column]).std()

    lg.info("number of actives = %i number of inactives = %i AUC-ROC of %s = %f AUC-CROC = %f AUC-BEDROC = %f" %
            (num_of_actives, num_of_inactives, column, auc_roc, auCROC, auBEDROC))
    print("AUC-ROC of %s = %f AUC-CROC = %f AUC-BEDROC = %f" % (column, auc_roc, auCROC, auBEDROC))     # number of actives and inactives is printed by Create_Curve()
    print("Value range of %s: min = %f , mean = %f , max = %f , stdev = %f , range = %f "
          "kcal/mol scaled_stdev = %f\n" %
          (column, min_score, mean_score, max_score, stdev_score, range_score, scaled_stdev))
    return column, num_of_actives, num_of_inactives, auc_roc, auCROC, auBEDROC, min_score, mean_score, max_score, stdev_score, range_score, scaled_stdev

def _DOR(estimator, X, y):
    preds = estimator.predict(X).flatten()
    return Classification_Metric(y, preds).DOR()

def _MK(estimator, X, y):
    preds = estimator.predict(X).flatten()
    return Classification_Metric(y, preds).MK()

def evaluate_learning_model(model, features_df, sel_columns, execution_dir):
    print(f"Writing scores to file " + os.path.join(execution_dir, f"{features_df['protein'].iloc[0]}_features_SQM-ML_scores.csv.gz"))
    features_df[['basemolname', 'structvar', 'pose', 'is_active', 'plec_ump1',
                 'plec_ump2', 'plec_ump3', 'plec_ump4', 'plec_ump5', 'plec_ump6',
                 'plec_ump7', 'plec_ump8', 'plec_ump9', 'plec_ump10', 'plec_ump11',
                 'plec_ump12', 'plec_ump13', 'plec_ump14', 'plec_ump15', 'plec_ump16',
                 'plec_ump17', 'plec_ump18', 'plec_ump19', 'plec_ump20', 'plec_ump21',
                 'plec_ump22', 'plec_ump23', 'plec_ump24', 'plec_ump25', 'plec_ump26',
                 'plec_ump27', 'plec_ump28', 'plec_ump29', 'plec_ump30', 'plec_ump31',
                 'plec_ump32', 'plec_ump33', 'plec_ump34', 'plec_ump35', 'plec_ump36',
                 'plec_ump37', 'plec_ump38', 'plec_ump39', 'plec_ump40', 'nofusion_Eint',
                 'bondType_SINGLE', 'bondType_AROMATIC', 'MW', 'ring_flexibility',
                 'AMW', 'deepFl_logP', 'function_group_count']] \
        .assign(SQM_ML_score=-model.predict_proba(features_df[sel_columns])[:, 1]) \
        .to_csv(os.path.join(execution_dir, f"{features_df['protein'].iloc[0]}_features_SQM-ML_scores.csv.gz"), index=False)

    auc_roc = 1 - roc_auc_score(y_true=features_df["is_active"],
                                y_score=-model.predict_proba(features_df[sel_columns])[:, 1])
    DOR = _DOR(model, features_df[sel_columns], features_df["is_active"])
    MK = _MK(model, features_df[sel_columns], features_df["is_active"])
    lg.info("AUC-ROC of %s model = %f DOR = %f MK = %f" %
          (model.__class__ , auc_roc, DOR, MK))
    print("AUC-ROC of %s model = %f DOR = %f MK = %f" %
          (model.__class__, auc_roc, DOR, MK))
    return pd.DataFrame([[auc_roc, DOR, MK]], columns=['AUC-ROC', 'DOR', 'MK'])


# class_metric = Classification_Metric(scores_df["is_active"], scores_df["P6C_Eint"])
# class_metric.DOR()
# class_metric.MK()
