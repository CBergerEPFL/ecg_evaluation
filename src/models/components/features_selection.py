import numpy as np
import statsmodels.api as sm
import pandas as pd
import os
from tqdm import tqdm
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef,f1_score
from sklearn.pipeline import Pipeline
from skfeature.utility.entropy_estimators import midd, cmidd
from skfeature.function.information_theoretical_based.LCSI import lcsi
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sys
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append(os.path.join(os.getcwd(), ".."))
from kneed import KneeLocator

def cal_vif(x):
    """

    Perform Variance Inflation Factor (VIF) Feature selection 
    ##TODO Change it to JMI VIF feature selection

    Args:
        x (Pandas Dataframe): Design Matrix containing your feature

    Returns:
        output (Numpy Array 1D): Array containing the selected feature
    """
    thresh_up = 10
    thresh_down = 5
    output = pd.DataFrame()
    k = x.shape[1]
    vif = [variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
    for i in range(1,k):
        a = np.argmax(vif)
        print('Max vif is for variable no:',a)
        if (vif[a]<=thresh_up and vif[a]>=thresh_down):
            break
        if (i==1):
            output = x.drop(x.columns[a],axis=1)
            vif = [variance_inflation_factor(output.values,j) for j in range(output.shape[1])]
        elif (i>1):
            output = output.drop(output.columns[a],axis=1)
            vif = [variance_inflation_factor(output.values,j) for j in range(output.shape[1])]
            
    return output

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
    X_dis = discretize_data(X)
    feature_set = list(X_dis.columns.values)
    X_dis = X_dis.values
    F, _, _ = lcsi(
        X_dis,
        y.values.ravel(),
        function_name="JMI",
        n_selected_features=len(feature_set),
    )
    F = list_noduplicate(F)
    S = [feature_set[i] for i in F]
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


def hjmi_selection(X, y, max_iteration=20, print_plot=True):
    """

    Feature selection using the Historical JMI method

    Args:
        X (2D Numpy array): Matrix features (shape : [n_sample,n_feature])
        y (1D Numpy array): Response vector (containing the label assign)
        max_iteration (int, optional): Number of iteration to repeat the process. Defaults to 20.
        print_plot (bool, optional): Boolean indicating if a elbow plot is necessary. Defaults to False.

    Returns:
        String List : List of the feature selected by the algorithm
        Float List : List containing the HJMI value when considering each feature
        FLoat List : List containing the delta value of each feature added.
    """
    X = discretize_data(X)
    columns_name = X.columns
    X = X.values
    y = y.values
    select_features = []
    collect_hjmi = []
    diff_m = []
    j_h = 0
    for n in range(max_iteration):
        jmi = np.zeros([X.shape[1]])
        print("Iteration : ", n)
        for k in range(X.shape[1]):
            if k in select_features:
                continue
            JMI_1 = midd(X[:, k], y.ravel())
            JMI_2 = 0
            for j in select_features:
                tmp1 = midd(X[:, k], X[:, j])
                tmp2 = cmidd(X[:, k], X[:, j], y.ravel())
                JMI_2 = JMI_2 + tmp1 - tmp2
            if len(select_features) == 0:
                jmi[k] = j_h + JMI_1
            else:
                jmi[k] = j_h + JMI_1 - JMI_2 / len(select_features)

        if n == 0:
            j_h = np.max(jmi)
            hjmi = j_h
            ind = np.argmax(jmi)
            select_features.append(ind)
            collect_hjmi.append(j_h)

        else:  # ((j_h-hjmi)/hjmi)>1e-10)
            j_h = np.max(jmi)
            ind = np.argmax(jmi)
            print("diff : ", (j_h - hjmi) / (hjmi))
            if (j_h - hjmi) / (hjmi) > 0.03 and len(select_features) < X.shape[1]:
                diff_m.append((j_h - hjmi) / (hjmi))
                hjmi = j_h
                select_features.append(ind)
                collect_hjmi.append(j_h)
            else:
                break
        print("Name features added : ", columns_name[ind])

    return select_features, collect_hjmi, diff_m


def elbow_plot(elbow):
    """
    Function that calculate elbow plot and the elbow point

    Args:
        elbow (1D Numpy array): Array containing the delta values in function of the number of features added
    """
    print(elbow)
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

    X_dis = X_data.copy()
    for col in X_dis.columns:
        if col == "sex":
            continue
        bins = np.histogram_bin_edges(X_dis[col].values, bins="fd")
        ind = np.digitize(X_dis[col].values, bins=bins)
        X_dis[col] = [bins[i - 1] for i in ind]
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

def feature_selection_nested_cross_val(X,y,num_trial = 30):
    ##TODO : Done only for logistic regression. Need to implement it for other model (SVM, decision tree,...)
    columns_name = X.columns
    X = X.values
    y = y.values



    ##Nested cross validation : 
    features_name = np.array([])
    mcc_score = np.array([])
    f1score = np.array([])

    for n in tqdm(range(num_trial),desc = "Number of trials done"):
        
        ##initialize fold : 
        cv_outward = StratifiedKFold(n_splits = 5,shuffle=True,random_state=1)
        cv_inner = StratifiedKFold(n_splits = 4,shuffle = True,random_state=1)
        ## Access outter fold : 
        for train,test in cv_outward.split(X,y):
            select = SelectFromModel(LogisticRegression(penalty  = "elasticnet",solver="saga",l1_ratio=0.5))
            pipeline_model = Pipeline([("select",select),("model",LogisticRegression(random_state=2))])
            p_grid = {"select_max_features":np.linspace(1,len(columns_name)+1,10) ,"model__C":np.linspace(1,100,10)}
            search = RandomizedSearchCV(pipeline_model, p_grid, scoring='f1', cv=cv_inner, refit=True)
            results = search.fit(X[train],y[train])
            best_model = results["model"].best_estimator_
            prediction = best_model.predict(X[test])
            mcc =matthews_corrcoef(y[test],prediction)
            f1 = f1_score(y[test],prediction)
            mcc_score = np.append(mcc_score,mcc)
            f1score = np.append(f1score,f1)
            features_name = np.append(features_name,columns_name[results["select"].get_support()])
            if n%10 == 0:
                print('>F1-score=%.3f, MCC = %.3f, est=%.3f, cfg=%s' % (f1,mcc, results.best_score_, results.best_params_))
            # summarize the estimated performance of the model)
    print(features_name)

                
