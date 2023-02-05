import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
    StratifiedKFold,
)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from skfeature.function.information_theoretical_based import LCSI

FILEPATH = os.path.dirname(os.path.realpath(__file__))
# ROOTPATH = os.path.dirname(FILEPATH)
sys.path.append(FILEPATH)
from utils_path import results_path
from utils_type import Results_Data

seed = 0


def metrics_cv(results: Results_Data, save_name, t_used=None):

    """
    Function that calculate Metrics for labelled dataset obtained after Cross Validation. Metrics Calculated :
        Precision,Recall,MCC,F1-score,AUC ROC,AUC PR,Sensitivity
        Except AUC ROC and PR,most of previous Metrics will be calculated using threshold from Maximum MCC value,
        of each CV folds.

    Inputs :
        y_label [List of 1D Arrays] : List containing the reference array used during each iteration/fold of
             Cross Validation
        prob_predicted [list of 2D Arrays] : List containing the probability array calculated at each
            iteration/fold of Cross Validation.
            Numpy array shape : [size_CV_fold*probabilities].

        T [Float] (default : None) : Threshold given by the user. If no threshold given, MAX MCC threshold used.
    Ouput :
        df [Pandas Dataframe] : Pandas Dataframe containing, for each binary label, a tuple with the mean and SD of
            each metrics calculated. The tuple with the mean and SD of the threshold obtained at each folds, is also given.
        df is also printed at the end of the function
    """

    list_metrics = [
        "auc_roc",
        "auc_pr",
        "precision",
        "recall",
        "f1",
        "specificity",
        "accuracy",
        "mcc",
        "T",
    ]
    y_pred = results.proba_unacceptable
    y_label = results.label
    dict_metrics = dict([(key, np.empty([len(y_label)])) for key in list_metrics])

    for i, (y_label_cv, y_pred_cv) in enumerate(zip(y_label, y_pred)):

        dict_metrics["auc_roc"][i] = roc_auc_score(y_label_cv, y_pred_cv)
        precision, recall, threshold = precision_recall_curve(
            y_label_cv, y_pred_cv, pos_label=1
        )
        dict_metrics["auc_pr"][i] = auc(recall, precision)
        if t_used is not None:
            dict_metrics["mcc"][i] = matthews_corrcoef(
                y_label_cv, (y_pred_cv >= t_used).astype(int)
            )
            dict_metrics["T"][i] = t_used
            label_pred = (y_pred_cv >= t_used).astype(int)
        else:
            mcc_tmp = np.array(
                [
                    matthews_corrcoef(y_label_cv, (y_pred_cv >= t).astype(int))
                    for t in threshold
                ]
            )
            dict_metrics["mcc"][i] = np.max(mcc_tmp)
            t_used = threshold[np.argmax(mcc_tmp)]
            dict_metrics["T"][i] = t_used
            label_pred = (y_pred_cv >= t_used).astype(int)

        dict_reports = classification_report(
            y_label_cv.astype(int), label_pred, output_dict=True
        )
        dict_metrics["precision"][i] = dict_reports["1"]["precision"]
        dict_metrics["recall"][i] = dict_reports["1"]["recall"]
        dict_metrics["f1"][i] = dict_reports["1"]["f1-score"]
        dict_metrics["specificity"][i] = dict_reports["0"][
            "recall"
        ]  # specificity is the recall of label 0
        dict_metrics["accuracy"][i] = dict_reports["accuracy"]

    df_results = pd.DataFrame(index=list_metrics, columns=["mean", "std"])
    for key, val in dict_metrics.items():
        df_results.loc[key, "mean"] = np.round(val.mean(), 2)
        df_results.loc[key, "std"] = np.round(val.std(), 2)

    if save_name is not None:
        df_results.to_csv(
            os.path.join(results_path, "evaluation_metrics", f"{save_name}.csv")
        )
    return df_results
