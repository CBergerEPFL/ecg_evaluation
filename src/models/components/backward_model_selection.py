import numpy as np
import statsmodels.api as sm
import pandas as pd


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
