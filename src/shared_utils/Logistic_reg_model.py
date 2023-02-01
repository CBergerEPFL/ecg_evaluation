import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from IPython.display import display
import statsmodels.api as sm
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    multilabel_confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
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

sys.path.append(os.path.join(os.getcwd(), ".."))
from shared_utils.Custom_Logit import Logit_binary
from skfeature.function.information_theoretical_based import LCSI

seed = 0

list_name_features = [
    "Corr_interlead",
    "Corr_intralead",
    "wPMF",
    "SNRECG",
    "HR",
    "Kurtosis",
    "Flatline",
    "TSD",
]

dico_T_opt = {
    "Corr_interlead": 0.39,
    "Corr_intralead": 0.67,
    "wPMF": 0.116,
    "SNRECG": 0.48,
    "Kurtosis": 2.16,
    "Flatline": 0.51,
    "TSD": 0.42,
}

path_save = "/workspaces/maitrise/results"


def feature_selector(X_data, y_data, cols, model_type="Logistic", already_model=False):
    """
    Function that prepare your feature dataset for a specific model you want to create.

    Inputs :
        X_data [2D Pandas Dataframe] : Dataframe with the rows being the patients ECG score and the columns the
            features you evaluated (with their name)
        y_data [1D Pandas Dataframe] : Dataframe containing the label assigned to your patient ECG recording
        cols [String list] : Array of string containing the features name you tested. The name contains must
            be the same as X_data column (Otherwise an error will be thrown)
        model_type [String variable] : The name of the model you want to train. Include : Custom logistic regression,
            Logistic regression, ExtraClassifier, RandomForestClassifier from sklearn
        **kwargs : Any additional arguments
                - model : your custom model (herited from class sklearn)

    Outputs :
        X [2D Numpy Array] : Your prepared feature Dataset of size [number_of_patient*features_selected]
        y [1D Numpy Array] : Your true label assigned to your dataset
        model : The model you selected (or gave)
        Hindex [int] : index where the columns correspond to the Heart Rate in your dataset
    """
    y = y_data.values
    if cols is None:
        print("Using Backward model selection")
        cols = backward_model_selection(X_data, y_data)
        if "HR" in cols and len(cols) > 1:
            Hindex = list(X_data[cols].columns.values).index("HR")

        else:
            Hindex = None
        X = X_data[cols].values
    else:
        if "HR" in cols and len(cols) > 1:
            Hindex = list(X_data[cols].columns.values).index("HR")
        else:
            Hindex = None
        X = X_data[cols].values

    if already_model:
        return X, y, None, Hindex
    else:
        if model_type == "ExtraTreeClassifier":
            model = ExtraTreesClassifier(random_state=seed)
        elif model_type == "RandomTreeClassifier":
            model = RandomForestClassifier(random_state=seed)
        elif model_type == "Logistic" and Hindex is not None:
            model_type = model_type + " Binary"
            model = Logit_binary(HR_index=Hindex, random_state=seed)
        else:
            model = LogisticRegression(random_state=seed)

        return X, y, model, Hindex


def feature_checker(df_features):

    """
    Function that check if the features in your feature dataset have the good range ([0;1]) in your columns set

    Inputs :
    X [2D Numpy Array] : Your feature dataset (shape : [number_of_patient*feature_selected])
    cols [String list] : A string list containng the columns you want to select. The index of each feautre in your list must be the same than the index of your geature dataset

    Outputs :
    X[2D Numpy Array] : Feature dataset with correct feature
    cols [String List] : String list with selected columns
    """
    columns_remove = np.array([])
    for j in range(df_features.shape[1]):
        if not (np.min(X[:, j]) >= 0 and np.max(X[:, j]) <= 1):
            columns_remove = np.append(columns_remove, j)
    if len(columns_remove) > 0:
        raise ValueError(
            "The features ",
            np.array(cols)[columns_remove.astype(int)],
            "are not between 0 and 1",
        )
    else:
        print("No features were removed!")
    return True


def save_model_index(X_data, y_data, save_path, cols, **kwargs):
    """
    Function that save a sklearn model using a a dataset and labels, given a specific set of features you want to create your model around
    Inputs :
        X_data [2D Pandas Dataframe] : Dataframe containing your features [shape : [number_of_patients*number_of_features]. The columns must have the name of your feature]
        y_data [1D Pandas Dataframe] : Dataframe containing your labels for each patient
        cols [String list] : Columns name you want to use. The index of each feature must be the same than X_data columns

    """
    if len(save_path) == 0:
        raise AttributeError("you didn't give a path where to save!")

    ##Create Folder where to stock our data
    model_folder = os.path.join(save_path, "Models")
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    index_folder = os.path.join(save_path, "Indexes")
    if not os.path.exists(index_folder):
        os.mkdir(index_folder)

    ##Now let the fun begin : We do cv on all indexes and store their results into a pandas then csv
    cv = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
    y = y_data.values
    if len(os.listdir(index_folder)) == 0:
        X, reference_col = feature_checker(X_data.values, list(X_data.columns.values))
        for j, c in enumerate(reference_col):
            X_s = X[:, j]

            path_folder_index = os.path.join(index_folder, c)

            if not os.path.exists(path_folder_index):
                os.mkdir(path_folder_index)

            for i, (_, test) in enumerate(cv.split(X, y.ravel())):
                index_val_test = X_s[test]
                lab_test = y[test].ravel()
                der_data = np.c_[1 - index_val_test, index_val_test, lab_test]
                df = pd.DataFrame(
                    der_data,
                    columns=["proba_label_0", "proba_label_1", "ref_test_label"],
                )
                df.to_csv(os.path.join(path_folder_index, f"Test_Fold_{i}.csv"))
    else:
        print(
            "Indexes already present! if you need to add another index, please use the following function : Index_saver"
        )

    if len(cols) != 0:
        ##See if user proposer a model (already trained or not)
        if kwargs.get("model"):
            model = kwargs["model"]
            X, y, _, _ = feature_selector(X_data, y_data, cols, already_model=True)
        else:
            X, y, model, _ = feature_selector(X_data, y_data, cols)

        if kwargs.get("Model_name"):
            name_model = kwargs["Model_name"]
        else:
            print("Auto Generation name model")
            name_model = "Model_"
            for c in cols:
                name_model += c + "_"
            print(f"This will be the name of your model : {name_model}")

        path_folder_model = os.path.join(model_folder, name_model)

        if not os.path.exists(path_folder_model):
            os.mkdir(path_folder_model)

        path_model_fold_cv = os.path.join(path_folder_model, "Fold_CV")
        if not os.path.exists(path_model_fold_cv):
            os.mkdir(path_model_fold_cv)
        X_train, _, y_train, _ = train_test_split(X, y.ravel())

        if not kwargs.get("model"):
            model.fit(X_train, y_train)

        for i, (_, test) in enumerate(cv.split(X, y.ravel())):
            y_pred = model.predict(X[test])
            proba_model = model.predict_proba(X[test])
            y_ref = y[test].ravel()
            Hugues_data = np.c_[proba_model[:, 0], proba_model[:, 1], y_pred, y_ref]
            df_m = pd.DataFrame(
                Hugues_data,
                columns=[
                    "proba_label_0",
                    "proba_label_1",
                    "predicted_test_label",
                    "ref_test_label",
                ],
            )
            df_m.to_csv(os.path.join(path_model_fold_cv, f"Fold_{i}.csv"))

        filename = name_model + ".sav"
        pickle.dump(model, open(os.path.join(path_folder_model, filename), "wb"))
    else:
        print("You didn't give any cols. No model were created")


def csv_reader(main_folder, file_csv):
    fold_n = os.path.join(main_folder, file_csv)
    df = pd.read_csv(fold_n)
    return df["proba_label_0"], df["proba_label_1"], df["ref_test_label"]


def list_creator_cv(folder_csv):
    original_lab = []
    prob_lab = []
    for csv_file in os.listdir(folder_csv):
        prob_0, prob_1, y_test = csv_reader(folder_csv, csv_file)
        y_prob = np.c_[prob_0, prob_1]
        original_lab.append(y_test)
        prob_lab.append(y_prob)
    return original_lab, prob_lab


def classification_report_index(path_indexes, name_feature=list([]), **kwargs):
    ref_feature = os.listdir(path_indexes)

    if len(ref_feature) == 0:
        raise AttributeError(
            "No folder for indexes found! Please run save_model_index before using this function."
        )

    if len(name_feature) == 0:
        print("all models will be tested!")
        name_feature = ref_feature
    else:
        for feature in name_feature:
            original_lab = []
            prob_lab = []
            index_path = os.path.join(path_indexes, feature)
            original_lab, prob_lab = list_creator_cv(index_path)
            print(f"For index {feature} : ")
            if kwargs.get("T"):
                display(
                    index_ml_calculator_cv(original_lab, prob_lab, t_used=kwargs["T"])
                )
            else:
                display(index_ml_calculator_cv(original_lab, prob_lab))


def classification_report_model(path_model, name_model=list([]), **kwargs):
    ref_model = os.listdir(path_model)

    if len(ref_model) == 0:
        raise AttributeError(
            "No folder for models found! Please run save_model_index before using this function."
        )

    if len(name_model) == 0:
        print("all models will be tested!")
        name_model = ref_model

    for models in name_model:
        original_lab = []
        prob_lab = []
        folder = os.path.join(path_model, models)
        # path_to_save = os.path.join(folder,name_model + ".sav")
        path_CSV_folder = os.path.join(folder, "Fold_CV")
        original_lab, prob_lab = list_creator_cv(path_CSV_folder)
        print(f"For model {models} : ")
        if kwargs.get("T"):
            display(index_ml_calculator_cv(original_lab, prob_lab, t_used=kwargs["T"]))
        else:
            display(index_ml_calculator_cv(original_lab, prob_lab))


def index_ml_calculator_cv(original_label, prob_predicted, t_used=None):

    """
    Function that calculate ML Metrics for labelled dataset obtained after Cross Validation. Metrics Calculated : Precision,Recall,MCC,F1-score,AUC ROC,AUC PR,Sensitivity
    Except AUC ROC and PR,most of previous Metrics will be calculated using threshold from Maximum MCC value, of each CV folds.

    Inputs :
        original_label [List of 1D Arrays] : List containing the reference array used during each iteration/fold of Cross Validation
        prob_predicted [list of 2D Arrays] : List containing the probability array calculated at each iteration/fold of Cross Validation.
                                             Numpy array shape : [size_CV_fold*probabilities].
                                             The probabilities in the array must be order as such : first column = probability of label 0 and second column = probability of label 1
        T [Float] (default : None) : Threshold given by the user. If no threshold given, MAX MCC threshold used.

    Ouput :
        df [Pandas Dataframe] : Pandas Dataframe containing, for each binary label, a tuple with the mean and SD of each metrics calculated. The tuple with the mean and SD of the threshold obtained at each folds, is also given.
        df is also printed at the end of the function
    """
    AUCROC = np.empty([len(original_label), 2])
    AUCPR = np.empty([len(original_label), 2])
    prec = np.empty([len(original_label), 2])
    rec = np.empty([len(original_label), 2])
    F1 = np.empty([len(original_label), 2])
    Spec = np.empty([len(original_label), 2])
    ACC = np.empty([len(original_label), 2])
    MCC = np.empty([len(original_label), 2])
    T = np.empty([len(original_label), 2])
    for i, y_ori in enumerate(original_label):
        rocauc_0, rocauc_1 = roc_auc_score(
            1 - y_ori, prob_predicted[i][:, 0]
        ), roc_auc_score(y_ori, prob_predicted[i][:, 1])
        AUCROC[i, 0], AUCROC[i, 1] = rocauc_0, rocauc_1
        precision_1, recall_1, threshold_1 = precision_recall_curve(
            y_ori, prob_predicted[i][:, 1], pos_label=1
        )
        precision_0, recall_0, threshold_0 = precision_recall_curve(
            y_ori, prob_predicted[i][:, 0], pos_label=0
        )
        AUCPR[i, 0], AUCPR[i, 1] = auc(recall_0, precision_0), auc(
            recall_1, precision_1
        )

        if t_used is not None:
            thresh = "(set T)"
            MCC[i, 0], MCC[i, 1] = matthews_corrcoef(
                1 - y_ori, (prob_predicted[i][:, 0] >= t_used).astype(int)
            ), matthews_corrcoef(y_ori, (prob_predicted[i][:, 1] >= t_used).astype(int))
            i_1, i_0 = t_used, t_used
            T[i, 0], T[i, 1] = i_0, i_1
            y_pred = (prob_predicted[i][:, 1] >= i_1).astype(int)
        else:
            thresh = "(Max MCC)"
            MCC_0, MCC_1 = np.array(
                [
                    matthews_corrcoef(
                        1 - y_ori, (prob_predicted[i][:, 0] >= t).astype(int)
                    )
                    for t in threshold_0
                ]
            ), np.array(
                [
                    matthews_corrcoef(y_ori, (prob_predicted[i][:, 1] >= t).astype(int))
                    for t in threshold_1
                ]
            )
            MCC[i, 0], MCC[i, 1] = np.max(MCC_0), np.max(MCC_1)
            i_1 = threshold_1[np.argmax(MCC_1)]
            i_0 = threshold_0[np.argmax(MCC_0)]
            T[i, 0], T[i, 1] = i_0, i_1
            y_pred = (prob_predicted[i][:, 1] >= i_1).astype(int)

        precision, recall, f1, specificity, _, acc = roc_pr_curve_multilabel(
            y_ori, y_pred
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
            np.around(prec[:, 0].std(), 2) if len(original_label) > 1 else 0,
        ),
        (
            np.around(prec[:, 1].mean(), 2),
            np.around(prec[:, 1].std(), 2) if len(original_label) > 1 else 0,
        ),
    ]
    df["Recall (mean,std) " + thresh] = [
        (
            np.around(rec[:, 0].mean(), 2),
            np.around(rec[:, 0].std(), 2) if len(original_label) > 1 else 0,
        ),
        (
            np.around(rec[:, 1].mean(), 2),
            np.around(rec[:, 1].std(), 2) if len(original_label) > 1 else 0,
        ),
    ]
    df["Specificity (TNR) (mean,std) " + thresh] = [
        (
            np.around(Spec[:, 0].mean(), 2),
            np.around(Spec[:, 0].std(), 2) if len(original_label) > 1 else 0,
        ),
        (
            np.around(Spec[:, 1].mean(), 2),
            np.around(Spec[:, 1].std(), 2) if len(original_label) > 1 else 0,
        ),
    ]
    df["F1 score (mean,std) " + thresh] = [
        (
            np.around(F1[:, 0].mean(), 2),
            np.around(F1[:, 0].std(), 2) if len(original_label) > 1 else 0,
        ),
        (
            np.around(F1[:, 1].mean(), 2),
            np.around(F1[:, 1].std(), 2) if len(original_label) > 1 else 0,
        ),
    ]
    df["Accuracy (mean,std) " + thresh] = [
        (
            np.around(ACC[:, 0].mean(), 2),
            np.around(ACC[:, 0].std(), 2) if len(original_label) > 1 else 0,
        ),
        (
            np.around(ACC[:, 1].mean(), 2),
            np.around(ACC[:, 1].std(), 2) if len(original_label) > 1 else 0,
        ),
    ]
    df["MCC (mean,std) " + thresh] = [
        (
            np.around(MCC[:, 0].mean(), 2),
            np.around(MCC[:, 0].std(), 2) if len(original_label) > 1 else 0,
        ),
        (
            np.around(MCC[:, 1].mean(), 2),
            np.around(MCC[:, 1].std(), 2) if len(original_label) > 1 else 0,
        ),
    ]
    df["AUC ROC (mean,std)"] = [
        (
            np.around(AUCROC[:, 0].mean(), 2),
            np.around(AUCROC[:, 0].std(), 2) if len(original_label) > 1 else 0,
        ),
        (
            np.around(AUCROC[:, 1].mean(), 2),
            np.around(AUCROC[:, 1].std(), 2) if len(original_label) > 1 else 0,
        ),
    ]
    df["AUC PR (mean,std)"] = [
        (
            np.around(AUCPR[:, 0].mean(), 2),
            np.around(AUCPR[:, 0].std(), 2) if len(original_label) > 1 else 0,
        ),
        (
            np.around(AUCPR[:, 1].mean(), 2),
            np.around(AUCPR[:, 1].std(), 2) if len(original_label) > 1 else 0,
        ),
    ]
    df["Optimal Threshold from " + thresh] = [
        (np.around(T[:, 0].mean(), 2), np.around(T[:, 0].std(), 2)),
        (np.around(T[:, 1].mean(), 2), np.around(T[:, 1].std(), 2)),
    ]
    return df


def roc_pr_cv_curve_model(path_models, models=list([]), pos_label=0):

    if len(models) == 0:
        if len(os.listdir(path_models)) == 0:
            raise AttributeError("No Model found!")
        else:
            print("No model given! We will use all of them")
            name_model = os.listdir(path_models)
    else:
        name_model = models
        ref_model = os.listdir(path_models)

        if len(ref_model) == 0:
            raise AttributeError("No Model found")

        for r in name_model:
            if r not in ref_model:
                raise AttributeError(f"The model {r} do not exist")

    for i in models:
        print(f"Model {i} Performance : ")
        plt.figure()
        plot_graph_roc_pr_model(path_models, i, pos_label)


def roc_pr_cv_curve_index(path_indexes, indexes=list([]), pos_label=0):

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


def plot_graph_roc_pr_model(path_models, model, pos_label=1):

    pos_lab = pos_label
    mean_fpr = np.linspace(0, 1, 500)
    mean_recall = np.linspace(0, 1, 500)
    tprs = []
    precs = []
    aucs_roc = []
    aucs_pr = []
    inside_path = os.path.join(path_models, model)
    path_to_csv = os.path.join(inside_path, os.listdir(inside_path)[0])
    original_lab, prob_lab = list_creator_cv(path_to_csv)
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

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 25))
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(original_lab))))

    ax[0].plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
    )
    ax[1].plot(
        [0, 1], [0, 0], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
    )

    for j in range(len(original_lab)):
        c = next(color)
        ax[0].plot(
            mean_fpr,
            tprs[j],
            label="ROC fold {} with AUC = {:.2f}".format(j, aucs_roc[j]),
            color=c,
            alpha=0.3,
            lw=1,
        )
        ax[1].plot(
            mean_recall,
            precs[j],
            label="PR fold {} with AUC = {:.2f}".format(j, aucs_pr[j]),
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
    ax[0].set_title(f"ROC curve for {model}")
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
    ax[1].set_title(f"PR curve for {model}")
    ax[1].grid()
    ax[1].legend(loc="best")

    plt.show()


def plot_graph_roc_pr_indexes(path_indexes, index, pos_label=1):
    pos_lab = pos_label
    inside_folder = os.path.join(path_indexes, index)
    original_lab, prob_lab = list_creator_cv(inside_folder)
    k_cv = len(original_lab)
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


def global_comp_roc_pr_mean_curve(
    path_models, path_indexes, models=list([]), indexes=list([]), pos_label=0
):

    if len(os.listdir(path_models)) == 0 or len(os.listdir(path_indexes)) == 0:
        raise AttributeError(
            "You don't have all the folders necessary to use this function"
        )

    ref_models = os.listdir(path_models)
    ref_indexes = os.listdir(path_indexes)

    if len(models) == 0:
        print("No models given! All the models in the folder will be tested")
        models = ref_models

    if len(indexes) == 0:
        print("No indexes given! All the indexes in the folder will be tested")
        indexes = ref_indexes

    for r, i in zip(models, indexes):
        if r not in ref_models:
            raise AttributeError(f"The model {r} does not exist")
        elif i not in ref_indexes:
            raise AttributeError(f"The index {i} does not exist")

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 20))
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(models) + len(indexes))))
    dict_indexes = {}
    dict_models = {}

    for ind in indexes:
        ##indexes :
        inside_path_index = os.path.join(path_indexes, ind)
        original_label, proba_label = list_creator_cv(inside_path_index)
        dict_indexes[ind] = (original_label, proba_label)
    for mod in models:
        inside_path_models = os.path.join(path_models, mod)
        path_CSV_folder = os.path.join(inside_path_models, "Fold_CV")
        original_label_m, proba_label_m = list_creator_cv(path_CSV_folder)
        dict_models[mod] = (original_label_m, proba_label_m)

    for i in indexes:
        c = next(color)
        mean_fpr = np.linspace(0, 1, 500)
        mean_recall = np.linspace(0, 1, 500)
        tprs = []
        precs = []
        aucs_roc = []
        aucs_pr = []
        original_lab, proba_lab = dict_indexes[i][0], dict_indexes[i][1]
        for j, y_ori in enumerate(original_lab):
            y_score = proba_lab[j]

            fpr, tpr, _ = roc_curve(
                y_ori.ravel(), y_score[:, pos_label], pos_label=pos_label
            )
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0
            tprs.append(interp_tpr)
            aucs_roc.append(auc(fpr, tpr))

            precision, recall, _ = precision_recall_curve(
                y_ori, y_score[:, pos_label], pos_label=pos_label
            )
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

        ax[0].plot(
            mean_fpr,
            tpr_avg,
            color=c,
            label="Mean ROC curve {} : AUC = {:.2f} +- {:.2f}".format(
                i, mean_auc_roc, std_auc_roc
            ),
        )
        ax[1].plot(
            mean_recall,
            precision_avg,
            color=c,
            label="Mean PR curve {}: AUC = {:.2f} +- {:.2f}".format(
                i, mean_auc_pr, std_auc_pr
            ),
        )

    for m in models:
        c = next(color)
        original_lab_m, proba_lab_m = dict_models[m][0], dict_models[m][1]
        mean_fpr = np.linspace(0, 1, 500)
        mean_recall = np.linspace(0, 1, 500)
        tprs = []
        precs = []
        aucs_roc = []
        aucs_pr = []
        for w, y_ori_m in enumerate(original_lab_m):
            y_score = proba_lab_m[w]
            fpr, tpr, _ = roc_curve(
                y_ori_m.ravel(), y_score[:, pos_label], pos_label=pos_label
            )
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0
            tprs.append(interp_tpr)
            aucs_roc.append(auc(fpr, tpr))

            precision, recall, _ = precision_recall_curve(
                y_ori_m.ravel(), y_score[:, pos_label], pos_label=pos_label
            )
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
        ax[0].plot(
            mean_fpr,
            tpr_avg,
            color=c,
            label="Mean ROC curve {} : AUC = {:.2f} +- {:.2f}".format(
                m, mean_auc_roc, std_auc_roc
            ),
        )
        ax[1].plot(
            mean_recall,
            precision_avg,
            color=c,
            label="Mean PR curve {}: AUC = {:.2f} +- {:.2f}".format(
                m, mean_auc_pr, std_auc_pr
            ),
        )

    ax[0].plot([0, 1], [0, 1], "--k", label="Reference line")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title(f"Testing mean ROC Curve for all indexes created ")
    ax[0].legend(loc=4)
    ax[0].grid()

    ax[1].plot([0, 1], [0, 0], "--k", label="Reference line")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title(f"Testing mean PR Curve for all indexes created ")
    ax[1].legend(loc=4)
    ax[1].grid()
    plt.show()


def backward_model_selection(X, y, threshold_out=0.001):
    initial_feature_set = list(X.columns.values)
    logit_model = sm.Logit(y.values.ravel(), X)
    result = logit_model.fit()
    sumsum = results_summary_to_dataframe(result)
    list_pval = np.array(sumsum["pvals"].values)
    max_pval = sumsum["pvals"].max()
    while max_pval >= threshold_out:
        idx_maxPval = np.array(initial_feature_set)[list_pval == max_pval]
        initial_feature_set.remove(idx_maxPval)
        logit_mod = sm.Logit(y, X[initial_feature_set])
        result = logit_mod.fit()
        sumsum = results_summary_to_dataframe(result)
        max_pval = sumsum["pvals"].max()
        list_pval = np.array(sumsum["pvals"].values)
    return initial_feature_set


def evaluate_model(X_data, y_data, repeats):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=repeats, random_state=seed)
    if "HR" in X_data.columns.values:
        index = list(X_data.columns.values).index("HR")
        X = X_data.values
        y = y_data.values.ravel()
        model = Logit_binary(index, random_state=seed)

    else:
        X = X_data.values
        y = y_data.values.ravel()
        model = LogisticRegression(random_state=seed)

    scores = cross_val_score(model, X, y, scoring="f1", cv=cv, n_jobs=-1)

    return scores


def f1_score_CV_estimates(X, y, repeats):
    results = list()
    for r in range(1, repeats):
        # evaluate using a given number of repeats
        scores = evaluate_model(X, y, r)
        # summarize
        print(">%d mean=%.4f se=%.3f" % (r, np.mean(scores), np.std(scores)))
        # store
        results.append(scores)
    plt.boxplot(results, labels=[str(r) for r in range(repeats)], showmeans=True)
    plt.show()


def results_summary_to_dataframe(results):
    """take the result of an statsmodel results table and transforms it into a dataframe"""
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame(
        {
            "pvals": pvals,
            "coeff": coeff,
            "conf_lower": conf_lower,
            "conf_higher": conf_higher,
        }
    )

    # Reordering...
    results_df = results_df[["coeff", "pvals", "conf_lower", "conf_higher"]]
    return results_df


def roc_pr_curve_multilabel(y_true, y_pred):
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


def extra_tree_classifier_cv_feature_selection(X_data, y_data, k_cv=10):
    model = ExtraTreesClassifier(random_state=seed)
    cv = StratifiedKFold(n_splits=k_cv)
    cols = X_data.columns.values
    df = pd.DataFrame(index=X_data.columns)
    X = X_data.values
    y = y_data.values
    for i, (train, test) in enumerate(cv.split(X, y.ravel())):
        model.fit(X[train], y[train].ravel())
        feat_importances = pd.Series(model.feature_importances_, index=X_data.columns)
        df[f"{i+1} fold"] = feat_importances
    df_n = df.to_numpy()
    mean_val = np.mean(df_n, axis=1)
    std_val = np.std(df_n, axis=1)
    plt.figure()
    plt.bar(cols, mean_val)
    plt.errorbar(
        cols,
        mean_val,
        yerr=std_val,
        alpha=0.5,
        fmt="o",
        color="r",
        ecolor="black",
        capsize=10,
    )
    plt.title(
        f"Feature importance from ExtraTreeClassifier for {k_cv} Fold CV on training set"
    )
    plt.xlabel("Features")
    plt.ylabel("Gini Score")
    plt.grid()
    plt.tight_layout()
    plt.show()


def kbest_mutual_information_cv(X_data, y_data, k_cv=10):
    model = SelectKBest(score_func=mutual_info_classif, k=len(X_data.columns.values))
    cv = StratifiedKFold(n_splits=k_cv)
    cols = X_data.columns.values
    df = pd.DataFrame(index=X_data.columns)
    X = X_data.values
    y = y_data.values
    for i, (train, test) in enumerate(cv.split(X, y.ravel())):
        fit = model.fit(X[train], y[train].ravel())
        df[f"{i+1} fold"] = pd.DataFrame(fit.scores_, index=cols)

    df_n = df.to_numpy()
    mean_val = np.mean(df_n, axis=1)
    std_val = np.std(df_n, axis=1)
    plt.figure()
    plt.bar(cols, mean_val)
    plt.errorbar(
        cols,
        mean_val,
        yerr=std_val,
        alpha=0.5,
        fmt="o",
        color="r",
        ecolor="black",
        capsize=10,
    )
    plt.title(f"Mutual information for {k_cv} Fold CV on training set")
    plt.xlabel("Features")
    plt.ylabel("Mutual Information")
    plt.grid()
    plt.tight_layout()
    plt.show()


###Link for the following code : https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/information_theoretical_based/JMI.py


def jmi(X, y, **kwargs):
    """
    This function implements the JMI feature selection
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response
    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """
    if "n_selected_features" in kwargs.keys():
        n_selected_features = kwargs["n_selected_features"]
        F, J_CMI, MIfy = LCSI.lcsi(
            X, y, function_name="JMI", n_selected_features=n_selected_features
        )
    else:
        F, J_CMI, MIfy = LCSI.lcsi(X, y, function_name="JMI")
    return F, J_CMI, MIfy


def discretize_data(X_data):
    X_dis = np.zeros_like(X_data.values)
    for j in X_data.columns.values:
        i = list(X_data.columns.values).index(j)
        if j == "HR":
            X_dis[:, i] = X_data[j]
        else:
            X_dis[:, i] = np.digitize(X_data[j], bins=[dico_T_opt[j]])
    return X_dis


def jmi_calculator(X_data, y_data, k_cv=10):

    cv = StratifiedKFold(n_splits=k_cv, shuffle=True, random_state=seed)
    df_jmi_train = pd.DataFrame(index=X_data.columns)
    df_Fy_jmi_train = pd.DataFrame(index=X_data.columns)
    df_jmi_test = pd.DataFrame(index=X_data.columns)
    df_Fy_jmi_test = pd.DataFrame(index=X_data.columns)

    X_dis = discretize_data(X_data)
    for i, (train, test) in enumerate(cv.split(X_dis, y_data.values.ravel())):

        F_importance_train, F_JMI_train, Fy_JMI_train = jmi(
            X_dis[train],
            y_data.values[train].ravel(),
            n_selected_features=(len(X_data.columns.values)),
        )
        F_importance_test, F_JMI_test, Fy_JMI_test = jmi(
            X_dis[test],
            y_data.values[test].ravel(),
            n_selected_features=(len(X_data.columns.values)),
        )
        df_jmi_train[f"{i+1} fold"] = pd.DataFrame(
            F_JMI_train, index=X_data.columns.values[F_importance_train]
        )
        df_Fy_jmi_train[f"{i+1} fold"] = pd.DataFrame(
            Fy_JMI_train, index=X_data.columns.values[F_importance_train]
        )
        # print(F_importance_test)
        # print(pd.DataFrame(F_JMI_train,index=X_data.columns.values[F_importance_train]))
        df_jmi_test[f"{i+1} fold"] = pd.DataFrame(
            F_JMI_test, index=X_data.columns.values[F_importance_test]
        )
        df_Fy_jmi_test[f"{i+1} fold"] = pd.DataFrame(
            Fy_JMI_test, index=X_data.columns.values[F_importance_test]
        )

    df_jmi_train_n = df_jmi_train.to_numpy()
    df_Fy_jmi_train_n = df_Fy_jmi_train.to_numpy()
    df_jmi_test_n = df_jmi_test.to_numpy()
    df_Fy_jmi_test_n = df_Fy_jmi_test.to_numpy()
    mean_jmi_train = np.mean(df_jmi_train_n, axis=1)
    std_jmi_train = np.std(df_jmi_train_n, axis=1)
    mean_Fy_jmi_train = np.mean(df_Fy_jmi_train_n, axis=1)
    std_Fy_jmi_train = np.std(df_Fy_jmi_train_n, axis=1)
    mean_jmi_test = np.mean(df_jmi_test_n, axis=1)
    std_jmi_test = np.std(df_jmi_test_n, axis=1)
    mean_Fy_jmi_test = np.mean(df_Fy_jmi_test_n, axis=1)
    std_Fy_jmi_test = np.std(df_Fy_jmi_test_n, axis=1)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    fig.tight_layout(h_pad=4)
    ax[0, 0].bar(X_data.columns.values, mean_jmi_train)
    ax[0, 0].errorbar(
        X_data.columns.values,
        mean_jmi_train,
        yerr=std_jmi_train,
        lw=2,
        capsize=10,
        capthick=2,
        color="r",
        ecolor="black",
        linestyle="",
    )
    ax[0, 0].set_ylabel("JMI value")
    ax[0, 0].set_title(
        "JMI between each features for a {} fold stratified CV from training set".format(
            k_cv
        )
    )
    ax[0, 0].grid()
    plt.setp(ax[0, 0].get_xticklabels(), rotation=30, horizontalalignment="right")
    ax[1, 0].bar(X_data.columns.values, mean_Fy_jmi_train)
    ax[1, 0].errorbar(
        X_data.columns.values,
        mean_Fy_jmi_train,
        yerr=std_Fy_jmi_train,
        lw=2,
        capsize=10,
        capthick=2,
        color="r",
        ecolor="black",
        linestyle="",
    )
    ax[1, 0].set_xlabel("Features")
    ax[1, 0].set_ylabel("MI value")
    ax[1, 0].set_title(
        "MI between selected features and response for a {} fold stratified CV from training set".format(
            k_cv
        )
    )
    ax[1, 0].grid()
    plt.setp(ax[1, 0].get_xticklabels(), rotation=30, horizontalalignment="right")
    ax[0, 1].bar(X_data.columns.values, mean_jmi_test)
    ax[0, 1].errorbar(
        X_data.columns.values,
        mean_jmi_test,
        yerr=std_jmi_test,
        lw=2,
        capsize=10,
        capthick=2,
        color="r",
        ecolor="black",
        linestyle="",
    )
    ax[0, 1].set_ylabel("JMI value")
    ax[0, 1].set_title(
        "JMI between each features for a {} fold stratified CV from testing set".format(
            k_cv
        )
    )
    ax[0, 1].grid()
    plt.setp(ax[0, 1].get_xticklabels(), rotation=30, horizontalalignment="right")
    ax[1, 1].bar(X_data.columns.values, mean_Fy_jmi_test)
    ax[1, 1].errorbar(
        X_data.columns.values,
        mean_Fy_jmi_test,
        yerr=std_Fy_jmi_test,
        lw=2,
        capsize=10,
        capthick=2,
        color="r",
        ecolor="black",
        linestyle="",
    )
    ax[1, 1].set_xlabel("Features")
    ax[1, 1].set_ylabel("MI value")
    ax[1, 1].set_title(
        "MI between selected features and response for a {} fold stratified CV from testing set".format(
            k_cv
        )
    )
    ax[1, 1].grid()
    plt.setp(ax[1, 1].get_xticklabels(), rotation=30, horizontalalignment="right")
    fig.subplots_adjust(top=1.10)
