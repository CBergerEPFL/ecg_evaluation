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
from shared_utils.utils_type import Results_Data


def evaluate_index(
    df_index: pd.DataFrame, df_label: pd.DataFrame, save_name=None, thres_metric=None
):

    """Evaluate and compute metrics of a list of indices.

    Args:
        df_index (Pandas Dataframe) : Dataframe containing the indexes vaues for each patient (and each lead).
        df_label (Pandas Dataframe) : Dataframe containing the quality label associated to each patient (for the 12 ECG lead).
        save_name (String) : Name of folder where you want to save your results.
        thres_metric (Float) : Threshold value given by the user (Default : None).
    """
    if df_index.shape[1] > 1:
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
    data_results = Results_Data(save_name)
    data_results.append_results(np_prob[:, 1], np_prob[:, -1])
    if save_name is not None:
        data_results.dump_to_file(save_name)

    metrics_cv(
        data_results.dict_results,
        save_name,
        t_used=thres_metric,
    )


def evaluate_list_indices(input_data_path, list_features, aggregation_methods="mean"):
    """Evaluate a list of indices

    Args:
        input_data_path (str): Path to the input data with ECG to be analysed
        list_features (list): List of features to be evaluated
    """
    ds_metrics = xr.load_dataset(input_data_path)
    pbar = tqdm(list_features)
    for feature in pbar:
        pbar.set_description(f"Evaluating {feature} using {aggregation_methods}")
        df_X, df_y = extract_index_label(
            ds_metrics, feature, aggregation_method=aggregation_methods
        )
        evaluate_index(
            df_X, df_y, save_name=feature + f"_{aggregation_methods}", thres_metric=None
        )
