import numpy as np
import statsmodels.api as sm
import pandas as pd
import os
from skfeature.function.information_theoretical_based import LCSI
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


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


def backward_model_selection_MI_JMI(X, y, path_result, threshold_out=0.5):
    initial_feature_set = list(X.columns.values)
    dic_T = {}
    for i in initial_feature_set:
        f = pd.read_csv(os.path.join(path_result, f"{i}.csv"))
        dic_T[i] = f.loc[len(f.index) - 1, "mean"]
    X_dis = discretize_data(X, dic_T)
    F, JMI, MI = jmi(X_dis, y)
    F = f7(F)
    JMI = f7(JMI)
    MI = f7(MI)
    if len(F) != len(MI):
        MI = MI[:-1]
    if len(F) != len(JMI):
        JMI = JMI[:-1]
    min_mi = np.min(MI)
    min_jmi = np.min(JMI)
    min_arg = F[-1]
    min_arg_name = initial_feature_set[min_arg]

    while min_mi <= threshold_out and min_jmi <= threshold_out:
        initial_feature_set.remove(min_arg_name)
        X_dis = np.delete(X_dis, min_arg, axis=1)
        F, JMI, MI = jmi(X_dis, y)
        F = f7(F)
        JMI = f7(JMI)
        MI = f7(MI)
        if len(F) != len(MI):
            MI = MI[:-1]
        if len(F) != len(JMI):
            JMI = JMI[:-1]
        min_mi = np.min(MI)
        min_jmi = np.min(JMI)
        min_arg = F[-1]
        min_arg_name = initial_feature_set[min_arg]

    ##Save the MI JMI results in csv file, in folder name (MI JMI score):
    folder_path = os.path.join(os.path.join(path_result, ".."), "JMI_MI_score")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    jmimi_dataframe = pd.DataFrame(
        index=np.array(initial_feature_set)[F], columns=["JMI", "MI"]
    )
    jmimi_dataframe["JMI"] = JMI
    jmimi_dataframe["MI"] = MI
    jmimi_dataframe.to_csv(os.path.join(folder_path, "jmi_mi_selection.csv"))
    return initial_feature_set


def model_selection_L2reg(X, y):
    sel_ = SelectFromModel(LogisticRegression(C=1, penalty="l2"))
    sel_.fit(X, y.values.ravel())
    selected_feat = X.columns[(sel_.get_support())]
    return selected_feat.values


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


def discretize_data(X_data, dico_T):
    X_dis = np.zeros_like(X_data.values)
    for j in X_data.columns.values:
        i = list(X_data.columns.values).index(j)
        if j == "HR":
            X_dis[:, i] = X_data[j]
        else:
            X_dis[:, i] = np.digitize(X_data[j], bins=[dico_T[j]])
    return X_dis


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
