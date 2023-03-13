import numpy as np
import statsmodels.api as sm
import pandas as pd
import os
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from skfeature.utility.entropy_estimators import midd, cmidd
from skfeature.function.information_theoretical_based.LCSI import lcsi
import sys
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append(os.path.join(os.getcwd(), ".."))
from kneed import KneeLocator


def backward_model_selection(X, y):
    """
    Perform p-value based backward selection on the feature set given

    Args:
        X (2D Numpy array): Matrix features (shape : [n_sample,n_feature])
        y (1D Numpy array): Response vector (containing the label assign)

    Returns:
        String List : List of the feature selected by the algorithm
    """
    initial_feature_set = list(X.columns.values)
    logit_model = sm.Logit(y.values.ravel(), X)
    result = logit_model.fit()
    sumsum = results_summary_to_dataframe(result)
    bi_1 = result.aic
    bi_0 = result.aic + 1
    list_pval = np.array(sumsum["pvals"].values)
    max_pval = sumsum["pvals"].max()
    while bi_0 >= bi_1:
        idx_maxPval = np.array(initial_feature_set)[list_pval == max_pval]
        initial_feature_set.remove(idx_maxPval)
        logit_mod = sm.Logit(y.values.ravel(), X[initial_feature_set])
        result = logit_mod.fit()
        bi_0 = bi_1
        bi_1 = result.aic
        sumsum = results_summary_to_dataframe(result)
        max_pval = sumsum["pvals"].max()
        list_pval = np.array(sumsum["pvals"].values)
    return initial_feature_set


def JMI_score(X, y):
    """
    Feature selection using JMI score method

    Args:
        X (2D Numpy array): Matrix features (shape : [n_sample,n_feature])
        y (1D Numpy array): Response vector (containing the label assign)

    Returns:
        String List : List of the feature selected by the algorithm
    """
    initial_feature_set = list(X.columns.values)
    X_dis = discretize_data(X)
    F, _, _ = lcsi(X_dis, y.values.ravel(), function_name="JMI", n_selected_features=3)
    F = list_noduplicate(F)
    S = [initial_feature_set[i] for i in F]
    return S


def model_selection_L2reg(X, y):
    """

    Feature selection using L2 regularization (applied on a logistic regression model with constraint C = 1)

    Args:
        X (2D Numpy array): Matrix features (shape : [n_sample,n_feature])
        y (1D Numpy array): Response vector (containing the label assign)

    Returns:
        String List : List of the feature selected by the algorithm
    """
    sel_ = SelectFromModel(LogisticRegression(C=1, penalty="l2"))
    sel_.fit(X, y.values.ravel())
    selected_feat = X.columns[(sel_.get_support())]
    return selected_feat.values


def results_summary_to_dataframe(results):
    """
    Take the result of an statsmodel results table and transforms it into a dataframe

    Args:
        results (statmodels api): Results object obtained with statmodels

    Returns:
        Pandas : Pandas dataframe with the pvalues, the features weigth coefficient, lower and upper bound of confidence interval
    """

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


def hjmi_selection(X, y, max_iteration=20, print_plot=False):
    """

    Feature selection using the Historical JMI method

    Args:
        X (2D Numpy array): Matrix features (shape : [n_sample,n_feature])
        y (1D Numpy array): Response vector (containing the label assign)
        max_iteration (int, optional): Number of iteration to repeat the process. Defaults to 20.
        print_plot (bool, optional): Boolean indicating if a elbow plot is necessary. Defaults to False.

    Returns:
        String List : List of the feature selected by the algorithm
    """
    select_features = []
    collect_hjmi = []
    diff_m = []
    X_dis = discretize_data(X)
    initial_feature_set = list(X.columns.values)
    j_h = 0
    jmi = np.zeros([len(initial_feature_set)])
    for i in range(max_iteration):
        for p in range(len(initial_feature_set)):
            if initial_feature_set[p] in select_features:
                continue
            JMI_1 = midd(X_dis[:, p], y.values.ravel())
            JMI_2 = 0
            for j in range(len(select_features)):
                tmp1 = midd(X_dis[:, p], X_dis[:, j])
                tmp2 = cmidd(X_dis[:, p], X_dis[:, j], y.values.ravel())
                JMI_2 = JMI_2 + tmp1 - tmp2
            jmi[p] = j_h + JMI_1
            if i > 1:
                jmi[p] = jmi[p] - (JMI_2) / (i - 1)
        if i == 0:
            j_h = np.max(jmi)
            hjmi = j_h
            ind = np.argmax(jmi)
            select_features.append(initial_feature_set[ind])
            collect_hjmi.append(j_h)

        else:  # ((j_h-hjmi)/hjmi)>1e-10)
            j_h = np.max(jmi)
            ind = np.argmax(jmi)
            if (j_h - hjmi) / (hjmi) > 0.13 and len(select_features) < len(
                initial_feature_set
            ):
                diff_m.append((j_h - hjmi) / (hjmi))
                hjmi = j_h
                select_features.append(initial_feature_set[ind])
                collect_hjmi.append(hjmi)
            else:
                break
    if print_plot:
        elbow_plot(diff_m)
    return select_features, collect_hjmi


def elbow_plot(elbow):
    """
    Function that calculate elbow plot and the elbow point

    Args:
        elbow (1D Numpy array): Array containing the delta values in function of the number of features added
    """
    n_feat = range(2, len(elbow) + 2)
    elbow_1 = KneeLocator(n_feat, elbow, curve="convex", direction="decreasing")
    fig, ax = plt.subplots()

    ax.set_xlabel("number of features in the subset")
    ax.set_ylabel(r"$\delta$")
    ax.plot(n_feat, elbow, "xb-")
    ax.grid()
    ax.vlines(
        elbow_1.knee, plt.ylim()[0], plt.ylim()[1], linestyles="dashed", color="k"
    )
    ax.set_title("Elbow plot for finding optimal threshold")


def discretize_data(X_data):
    """

    Function that discretize the feature matrix (using Friedman draconis method)

    Args:
        X_data (2D Numpy array): Matrix features (shape : [n_sample,n_feature])

    Returns:
        X_dis : Feature matrix with the associated discretize value for each feature (shape : [n_sample,n_feature])
    """
    ##Calculating number of bins necessary :
    X_dis = np.zeros_like(X_data.values)
    for j in X_data.columns.values:
        i = list(X_data.columns.values).index(j)
        if j == "HR" or j == "der_label":
            X_dis[:, i] = X_data[j]
        else:
            X_f = X_data[j].values.copy()
            Dx = freedman_diaconis(X_f, returnas="bins")
            new_ref = np.linspace(0, 1, Dx)
            ind = np.digitize(X_f, bins=np.linspace(0, 1, Dx))
            X_dis[:, i] = [new_ref[i] for i in ind]
    return X_dis


def freedman_diaconis(data, returnas="width"):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin width.
    ``returnas`` can be one of "width" or "bins", indicating whether
    the bin width or number of bins should be returned respectively.


    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    returnas: {"width", "bins"}
        If "width", return the estimated width for each histogram bin.
        If "bins", return the number of bins suggested by rule.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR = stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
    N = data.size
    bw = (2 * IQR) / np.power(N, 1 / 3)

    if returnas == "width":
        result = bw
    else:
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        result = int(np.ceil((datrng / bw) + 1))
    return result


def list_noduplicate(seq):
    """
    Suppress duplicate present in the list

    Args:
        seq (List): List to be processed

    Returns:
        List : List without duplicates
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
