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
    multilabel_confusion_matrix,
)
from sklearn.metrics import (
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

sys.path.append(os.path.join(os.getcwd(), ".."))
from shared_utils.Custom_Logit import Logit_binary

seed = 0


def feature_checker(df_features: pd.DataFrame) -> bool:
    """Function that check if the features in your feature dataset have the
        good range ([0;1]) in your columns set

    Args:
        df_features (pd.DataFrame): Dataframe with features to be checked

    Raises:
        ValueError: Raise an error if the features are not between 0 and 1

    Returns:
        bool:  True if the features are between 0 and 1
    """
    columns_remove = np.array([])
    for (colname, colval) in df_features.iteritems():
        if not (np.min(colval) >= 0 and np.max(colval) <= 1):
            columns_remove = np.append(columns_remove, colname)
    if len(columns_remove) > 0:
        raise ValueError("The features ", columns_remove, "are not between 0 and 1")
    else:
        print("Features have the correct scale!")
    return True


def evaluate_index(df_index: pd.DataFrame, df_label: pd.DataFrame, thres_metric=None):

    if df_index.shape[1] > 1:
        raise ValueError(
            "One index is evaluated at each pass. df_index must have only one column"
        )
    feature_checker(df_index)
    np_index = df_index.to_numpy()
    # Index are defined as 1 = acceptable and 0 = not acceptable, i.e. reflect
    # probability of class 0 (class 1 = unacceptable)
    np_prob = np.c_[np_index, 1 - np_index, df_label.to_numpy()]

    index_ml_calculator_cv([np_prob[:, -1]], [np_prob[:, 1]], t_used=thres_metric)


def classification_metrics(y_true, y_pred):
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=[0, 1])
    mcm_0 = mcm[0, :, :]
    mcm_1 = mcm[1, :, :]

    tn_0, fn_0, tp_0, fp_0 = mcm_0[0, 0], mcm_0[1, 0], mcm_0[1, 1], mcm_0[0, 1]
    tn_1, fn_1, tp_1, fp_1 = mcm_1[0, 0], mcm_1[1, 0], mcm_1[1, 1], mcm_1[0, 1]

    if tp_0 == 0 and fp_0 == 0:
        prec_0 = 1
        rec_0 = 0
    elif fp_0 == 0:
        prec_0 = 1
        rec_0 = tp_0 / (tp_0 + fn_0)
    elif tp_0 == 0:
        prec_0 = 0
        rec_0 = 0
    else:
        prec_0 = tp_0 / (fp_0 + tp_0)
        rec_0 = tp_0 / (tp_0 + fn_0)

    if tp_1 == 0 and fp_1 == 0:
        prec_1 = 1
        rec_1 = 0
    elif fp_1 == 0:
        prec_1 = 1
        rec_1 = tp_1 / (tp_1 + fn_1)
    elif tp_1 == 0:
        prec_1 = 0
        rec_1 = 0
    else:
        prec_1 = tp_1 / (fp_1 + tp_1)
        rec_1 = tp_1 / (tp_1 + fn_1)

    f1_0, f1_1 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0), 2 * (prec_1 * rec_1) / (
        prec_1 + rec_1
    )
    fpr_0, fpr_1 = fp_0 / (fp_0 + tn_0), fp_1 / (fp_1 + tn_1)
    spec_0, spec_1 = 1 - fpr_0, 1 - fpr_1
    acc_0, acc_1 = (tp_0 + tn_0) / (tp_0 + tn_0 + fp_0 + fn_0), (tp_1 + tn_1) / (
        tp_1 + tn_1 + fp_1 + fn_1
    )

    return (
        np.array([prec_0, prec_1]),
        np.array([rec_0, rec_1]),
        np.array([f1_0, f1_1]),
        np.array([spec_0, spec_1]),
        np.array([fpr_0, fpr_1]),
        np.array([acc_0, acc_1]),
    )


def index_ml_calculator_cv(y_label: list, y_pred: list, t_used=None):

    """
    Function that calculate ML Metrics for labelled dataset obtained after Cross Validation. Metrics Calculated : Precision,Recall,MCC,F1-score,AUC ROC,AUC PR,Sensitivity
    Except AUC ROC and PR,most of previous Metrics will be calculated using threshold from Maximum MCC value, of each CV folds.

    Inputs :
        y_label [List of 1D Arrays] : List containing the reference array used during each iteration/fold of Cross Validation
        prob_predicted [list of 2D Arrays] : List containing the probability array calculated at each iteration/fold of Cross Validation.
                                             Numpy array shape : [size_CV_fold*probabilities].
                                             The probabilities in the array must be order as such : first column = probability of label 0 and second column = probability of label 1
        T [Float] (default : None) : Threshold given by the user. If no threshold given, MAX MCC threshold used.

    Ouput :
        df [Pandas Dataframe] : Pandas Dataframe containing, for each binary label, a tuple with the mean and SD of each metrics calculated. The tuple with the mean and SD of the threshold obtained at each folds, is also given.
        df is also printed at the end of the function
    """
    auc_roc = np.empty([len(y_label)])
    auc_pr = np.empty([len(y_label)])
    prec = np.empty([len(y_label)])
    rec = np.empty([len(y_label)])
    F1 = np.empty([len(y_label)])
    Spec = np.empty([len(y_label)])
    ACC = np.empty([len(y_label)])
    MCC = np.empty([len(y_label)])
    T = np.empty([len(y_label)])
    for i, (y_label_cv, y_pred_cv) in enumerate(zip(y_label, y_pred)):

        auc_roc[i] = roc_auc_score(y_label_cv, y_pred_cv)
        precision, recall, threshold = precision_recall_curve(
            y_label_cv, y_pred_cv, pos_label=1
        )
        auc_pr[i] = auc(recall, precision)
        if t_used is not None:
            thresh = "(set T)"
            MCC[i] = matthews_corrcoef(y_label_cv, (y_pred_cv >= t_used).astype(int))
            T[i] = t_used
            label_pred = (y_pred >= t_used).astype(int)
        else:
            thresh = "(Max MCC)"
            mcc_tmp = np.array(
                [
                    matthews_corrcoef(y_label_cv, (y_pred_cv >= t).astype(int))
                    for t in threshold
                ]
            )
            MCC[i] = np.max(mcc_tmp)
            t_used = threshold[np.argmax(mcc_tmp)]
            T[i] = t_used
            label_pred = (y_pred >= t_used).astype(int)

        dict_metrics = classification_report(
            y_label_cv, (y_pred_cv > t_used).astype(int), output_dict=True
        )
        breakpoint()
        precision, recall, f1, specificity, _, acc = classification_metrics(
            y_label, label_pred
        )
        prec[i, 0], prec[i, 1] = precision[0], precision[1]
        rec[i, 0], rec[i, 1] = recall[0], recall[1]
        F1[i, 0], F1[i, 1] = f1[0], f1[1]
        Spec[i, 0], Spec[i, 1] = specificity[0], specificity[1]
        ACC[i, 0], ACC[i, 1] = acc[0], acc[1]

    df = pd.DataFrame(index=["0", "1"])

    df["Precision (mean,std) " + thresh] = [
        (
            np.around(prec[:, 0].mean(), 2),
            np.around(prec[:, 0].std(), 2) if len(y_label) > 1 else 0,
        ),
        (
            np.around(prec[:, 1].mean(), 2),
            np.around(prec[:, 1].std(), 2) if len(y_label) > 1 else 0,
        ),
    ]
    df["Recall (mean,std) " + thresh] = [
        (
            np.around(rec[:, 0].mean(), 2),
            np.around(rec[:, 0].std(), 2) if len(y_label) > 1 else 0,
        ),
        (
            np.around(rec[:, 1].mean(), 2),
            np.around(rec[:, 1].std(), 2) if len(y_label) > 1 else 0,
        ),
    ]
    df["Specificity (TNR) (mean,std) " + thresh] = [
        (
            np.around(Spec[:, 0].mean(), 2),
            np.around(Spec[:, 0].std(), 2) if len(y_label) > 1 else 0,
        ),
        (
            np.around(Spec[:, 1].mean(), 2),
            np.around(Spec[:, 1].std(), 2) if len(y_label) > 1 else 0,
        ),
    ]
    df["F1 score (mean,std) " + thresh] = [
        (
            np.around(F1[:, 0].mean(), 2),
            np.around(F1[:, 0].std(), 2) if len(y_label) > 1 else 0,
        ),
        (
            np.around(F1[:, 1].mean(), 2),
            np.around(F1[:, 1].std(), 2) if len(y_label) > 1 else 0,
        ),
    ]
    df["Accuracy (mean,std) " + thresh] = [
        (
            np.around(ACC[:, 0].mean(), 2),
            np.around(ACC[:, 0].std(), 2) if len(y_label) > 1 else 0,
        ),
        (
            np.around(ACC[:, 1].mean(), 2),
            np.around(ACC[:, 1].std(), 2) if len(y_label) > 1 else 0,
        ),
    ]
    df["MCC (mean,std) " + thresh] = [
        (
            np.around(MCC[:, 0].mean(), 2),
            np.around(MCC[:, 0].std(), 2) if len(y_label) > 1 else 0,
        ),
        (
            np.around(MCC[:, 1].mean(), 2),
            np.around(MCC[:, 1].std(), 2) if len(y_label) > 1 else 0,
        ),
    ]
    df["AUC ROC (mean,std)"] = [
        (
            np.around(auc_roc[:, 0].mean(), 2),
            np.around(auc_roc[:, 0].std(), 2) if len(y_label) > 1 else 0,
        ),
        (
            np.around(auc_roc[:, 1].mean(), 2),
            np.around(auc_roc[:, 1].std(), 2) if len(y_label) > 1 else 0,
        ),
    ]
    df["AUC PR (mean,std)"] = [
        (
            np.around(auc_pr[:, 0].mean(), 2),
            np.around(auc_pr[:, 0].std(), 2) if len(y_label) > 1 else 0,
        ),
        (
            np.around(auc_pr[:, 1].mean(), 2),
            np.around(auc_pr[:, 1].std(), 2) if len(y_label) > 1 else 0,
        ),
    ]
    df["Optimal Threshold from " + thresh] = [
        (np.around(T[:, 0].mean(), 2), np.around(T[:, 0].std(), 2)),
        (np.around(T[:, 1].mean(), 2), np.around(T[:, 1].std(), 2)),
    ]
    return df
