import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import pickle as pkl

FILEPATH = os.path.dirname(os.path.realpath(__file__))
ROOTPATH = os.path.dirname(FILEPATH)
sys.path.append(os.path.join(ROOTPATH))

from shared_utils.utils_data import feature_checker, extract_index_label
from shared_utils.utils_evaluation import metrics_cv
from shared_utils.utils_path import results_path


def evaluate_index(
    df_index: pd.DataFrame, df_label: pd.DataFrame, save_name=None, thres_metric=None
):
    if df_index.ndim > 1:
        raise ValueError(
            "One index is evaluated at each pass. df_index must have only one column"
        )
    feature_checker(df_index)
    np_index = df_index.to_numpy()
    # Index are defined as 1 = acceptable and 0 = not acceptable,
    # index are created such that initially they reflect probability of class 0,
    #  i.e need to be inverted according to convention
    if np_index.shape[0] != df_label.shape[0]:
        raise ValueError(
            "The number of samples in the index and the number of labels are not equal"
        )
    np_prob = np.c_[np_index, 1 - np_index, df_label.to_numpy()]
    if save_name is not None:
        dict_results = {
            "proba_class_0": [np_prob[:, 0]],
            "proba_class_1": [np_prob[:, 1]],
            "label": [np_prob[:, -1]],
        }
        with open(
            os.path.join(results_path, "proba_methods", f"proba_{save_name}.pkl"), "wb"
        ) as f:
            pkl.dump(dict_results, f)

    metrics_cv(
        dict_results["label"],
        dict_results["proba_class_1"],
        save_name,
        t_used=thres_metric,
    )


def evaluate_list_indices(input_data_path, list_features):
    """Evaluate a list of indices

    Args:
        input_data_path (str): Path to the input data with ECG to be analysed
        list_features (list): List of features to be evaluated
    """
    ds_metrics = xr.load_dataset(input_data_path)
    pbar = tqdm(list_features)
    for feature in pbar:
        pbar.set_description(f"Evaluating {feature}")
        df_X, df_y = extract_index_label(ds_metrics, feature)
        evaluate_index(df_X, df_y, save_name=feature, thres_metric=None)