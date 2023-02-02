import os

import matplotlib.pyplot as plt
import numpy as np


def roc_pr_cv_curve_index(path_indexes, names_methods=list([]), pos_label=0):

    if len(os.listdir(path_indexes)) == 0:
        raise AttributeError("No Model found!")

    if len(indexes) == 0:

        print("No model given! We will use all of them")
        name_model = os.listdir(path_indexes)
    else:
        name_model = indexes
        ref_model = os.listdir(path_indexes)

        for r in name_model:
            if r not in ref_model:
                raise AttributeError(f"The index {r} do not exist")

    for i in indexes:
        print(f"Performance of {i}:")
        plt.figure()
        plot_graph_roc_pr_indexes(path_indexes, i, pos_label)


def plot_graph_roc_pr_indexes(y_label: list, y_pred: list, name_model):

    k_cv = len(y_label)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
    color = iter(plt.cm.rainbow(np.linspace(0, 1, k_cv)))
    mean_fpr = np.linspace(0, 1, 500)
    mean_recall = np.linspace(0, 1, 500)
    tprs = []
    precs = []
    aucs_roc = []
    aucs_pr = []
    for i, y_ori in enumerate(original_lab):

        y_score = prob_lab[i]

        fpr, tpr, _ = roc_curve(y_ori, y_score[:, pos_lab], pos_label=pos_lab)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0
        tprs.append(interp_tpr)
        aucs_roc.append(auc(fpr, tpr))

        precision, recall, _ = precision_recall_curve(
            y_ori, y_score[:, pos_lab], pos_label=pos_lab
        )
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
            label="ROC fold {} with AUC = {:.2f}".format(e, aucs_roc[e]),
            color=c,
            alpha=0.3,
            lw=1,
        )
        ax[1].plot(
            mean_recall,
            precs[e],
            label="PR fold {} with AUC = {:.2f}".format(e, aucs_pr[e]),
            color=c,
            alpha=0.3,
            lw=1,
        )

        ax[0].plot(
            mean_fpr,
            tpr_avg,
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_roc, std_auc_roc),
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
    ax[0].set_title(f"ROC curve for {index}")
    ax[0].grid()
    ax[0].legend(loc="best")
    ax[1].plot(
        mean_recall,
        precision_avg,
        label=r"Mean PR (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_pr, std_auc_pr),
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
    ax[1].set_title(f"PR curve for {index}")
    ax[1].grid()
    ax[1].legend(loc="best")

    plt.show()
