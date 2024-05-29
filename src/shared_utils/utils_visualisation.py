import os
import pickle as pkl
import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def plot_roc_pr_curve(y_label: list, y_pred: list):
    """

    Plot individual ROC and PR curve obtained for each fold, for a specific model evaluated

    Args:
        y_label (list): Prediction label associated to each fold (shape : [n_fold,n_fold_sample])
        y_pred (list): Predicted label by the model to each fold (shape : [n_fold,n_fold_sample])
    """
    plt.rcParams.update({"font.size": 22})
    k_cv = len(y_label)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
    color = iter(plt.cm.rainbow(np.linspace(0, 1, k_cv)))
    mean_fpr = np.linspace(0, 1, 500)
    mean_recall = np.linspace(0, 1, 500)
    tprs = []
    precs = []
    aucs_roc = []
    aucs_pr = []
    for i, (y_label_cv, y_pred_cv) in enumerate(zip(y_label, y_pred)):

        fpr, tpr, _ = roc_curve(y_label_cv, y_pred_cv)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0
        tprs.append(interp_tpr)
        aucs_roc.append(auc(fpr, tpr))

        precision, recall, _ = precision_recall_curve(y_label_cv, y_pred_cv)
        index_rec = np.argsort(recall)
        interp_prec = np.interp(mean_recall, np.sort(recall), precision[index_rec])
        # interp_prec[0] = 1
        precs.append(interp_prec)
        aucs_pr.append(auc(recall, precision))

        precision_avg = np.mean(precs, axis=0)
        # mean_recall = np.mean(recs,axis = 0)

    mean_auc_pr = np.mean(aucs_pr)  # auc(mean_fpr, mean_tpr)
    std_auc_pr = np.std(aucs_pr)
    std_precs = np.std(precs, axis=0)
    precs_upper = np.minimum(precision_avg + std_precs, 1)
    precs_lower = np.maximum(precision_avg - std_precs, 0)

    tpr_avg = np.mean(tprs, axis=0)
    # mean_fpr = np.mean(fprs,axis = 0)
    mean_auc_roc = np.mean(aucs_roc)  # auc(mean_fpr, mean_tpr)
    std_auc_roc = np.std(aucs_roc)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(tpr_avg + std_tpr, 1)
    tprs_lower = np.maximum(tpr_avg - std_tpr, 0)

    ax[0].plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
    )
    ax[1].plot(
        [0, 1], [0, 0], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
    )

    for e in range(k_cv):
        c = next(color)
        ax[0].plot(
            mean_fpr,
            tprs[e],
            label=f"ROC fold {e} with AUC = {aucs_roc[e]:.2f}",
            color=c,
            alpha=0.3,
            lw=1,
        )
        ax[1].plot(
            mean_recall,
            precs[e],
            label=f"PR fold {e} with AUC = {aucs_pr[e]:.2f}",
            color=c,
            alpha=0.3,
            lw=1,
        )

        ax[0].plot(
            mean_fpr,
            tpr_avg,
            label=rf"Mean ROC (AUC = {mean_auc_roc:.2f} $\pm$ {std_auc_roc:.2f})",
            color="b",
        )
    ax[0].fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    # ax[0].set_title(f"ROC curve for {index}")
    ax[0].grid()
    ax[0].legend(loc="best")
    ax[1].plot(
        mean_recall,
        precision_avg,
        label=rf"Mean PR (AUC = {mean_auc_pr:.2f} $\pm$ {std_auc_pr:.2f})",
        color="b",
    )
    ax[1].fill_between(
        mean_recall,
        precs_lower,
        precs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    # ax[1].set_title(f"PR curve for {index}")
    ax[1].grid()
    ax[1].legend(loc="best")
    plt.show()


def comparison_roc_pr_mean_curve(path_results, methods):
    """

    Comparative ROC and PR curve for different methods tested

    Args:
        path_results (String): Path toward folder containing probability predicted by each method
        methods (String List):  List containning the methods to be compared
    """
    plt.rcParams.update({"font.size": 40})
    plt.rcParams["legend.fontsize"] = 40
    dict_results = {}
    for method in methods:
        name_file = f"proba_{method}.pkl"
        path_file = os.path.join(path_results, name_file)
        if os.path.isfile(path_file):
            with open(path_file, "rb") as f:
                dict_results[method] = pkl.load(f)
        else:
            print(f"file {path_file} not found")
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(40, 20), constrained_layout=True)
    fig.tight_layout(w_pad=5)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(dict_results))))
    for name_model, dict_proba in dict_results.items():
        c = next(color)
        mean_fpr = np.linspace(0, 1, 500)
        mean_recall = np.linspace(0, 1, 500)
        tprs = []
        precs = []
        aucs_roc = []
        aucs_pr = []
        y_pred = dict_proba["proba_unacceptable"]
        y_label = dict_proba["label"]
        for i, (y_label_cv, y_pred_cv) in enumerate(zip(y_label, y_pred)):

            fpr, tpr, _ = roc_curve(y_label_cv, y_pred_cv)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0
            tprs.append(interp_tpr)
            aucs_roc.append(auc(fpr, tpr))

            precision, recall, _ = precision_recall_curve(y_label_cv, y_pred_cv)
            index_rec = np.argsort(recall)
            interp_prec = np.interp(mean_recall, np.sort(recall), precision[index_rec])
            precs.append(interp_prec)
            aucs_pr.append(auc(recall, precision))

        precision_avg = np.mean(precs, axis=0)
        mean_auc_pr = np.mean(aucs_pr)
        std_auc_pr = np.std(aucs_pr)
        tpr_avg = np.mean(tprs, axis=0)
        mean_auc_roc = np.mean(aucs_roc)
        std_auc_roc = np.std(aucs_roc)
##: AUC = {mean_auc_roc:.2f} +- {std_auc_roc:.2f}
        ax[0].plot(
            mean_fpr,
            tpr_avg,
            color=c,
            label=f"{name_model}",
            linewidth=5
        )
## AUC = {mean_auc_pr:.2f} +- {std_auc_pr:.2f}
        ax[1].plot(
            mean_recall,
            precision_avg,
            color=c,
            label=f"{name_model}",
            linewidth=5
        )
    
    ax[0].plot([0, 1], [0, 1], "--k", label="Reference line")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title("Testing mean ROC Curve for all indexes created ")

    #ax[0].legend(bbox_to_anchor = (1.50, 0.6), loc='center left',fancybox=False, shadow=False)
    ax[0].grid()

    ax[1].plot([0, 1], [0, 0], "--k", label="Reference line")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Testing mean PR Curve for all indexes created ")
    #ax[1].legend(loc='lower rig',fancybox=False, shadow=False)
    ax[1].grid()
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor = (1.28, 0.5), loc='center right')
    plt.show()
