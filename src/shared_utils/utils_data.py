import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from petastorm import make_reader
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)


def format_data_to_xarray(data_path: str, save_path: str | None = None):
    """Format data to xarray format.

    Args:
        data_path (str): Path to data.
        save_path (str): Path to save data. If save_path is not None, data will be saved to save_path.

    Returns:
        ds_ecg (xarray): Data in xarray format.
    """
    if (save_path is not None) and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    if "file://" not in data_path:
        path_petastorm = f"file:///{data_path}"
    else:
        path_petastorm = data_path
    # Load data
    array_signal = []
    array_name = []
    array_quality = []
    array_fs = []
    array_sex = []
    with make_reader(path_petastorm) as reader:
        for idx, sample in enumerate(reader):
            if idx == 0:
                lead_names = sample.signal_names.astype(str)
            array_signal.append(sample.signal)
            array_name.append(sample.noun_id.decode("utf-8"))
            array_quality.append(sample.signal_quality.decode("utf-8"))
            array_fs.append(sample.sampling_frequency)
            array_sex.append(sample.sex.decode("utf-8"))

    ds_ecg = xr.Dataset(
        data_vars=dict(
            signal=(["id", "time", "lead_name"], np.array(array_signal)),
            data_quality=(["id"], np.array(array_quality)),
            fs=(["id"], np.array(array_fs)),
            sex=(["id"], np.array(array_sex)),
        ),
        coords=dict(
            id=(["id"], np.array(array_name)),
            time=(["time"], np.arange(0, 5000)),
            lead_names=(["lead_names"], lead_names),
        ),
        attrs=dict(description="ecg with quality description"),
    )

    if save_path is not None:
        path_to_file = os.path.join(save_path, "ecg_data.nc")
        print(f"Saving ECG in netCDF format in: {path_to_file}")
        ds_ecg.to_netcdf(path_to_file)

    return ds_ecg


def format_data_to_xarray_2020(data_path: str, save_path: str | None = None):
    """Format data to xarray format.This is the same function as before but adapted for the Classification of 12-leads ECGgs ; the physionet/computing in Cardiology Challenge 2020” dataset

    Args:
        data_path (str): Path to data.
        save_path (str): Path to save data. If save_path is not None, data will be saved to save_path.

    Returns:
        ds_ecg (xarray): Data in xarray format.
    """
    if (save_path is not None) and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    if "file://" not in data_path:
        path_petastorm = f"file:///{data_path}"
    else:
        path_petastorm = data_path
    # Load data
    array_signal = []
    array_name = []
    array_fs = []
    array_sex = []
    with make_reader(path_petastorm) as reader:
        for idx, sample in enumerate(reader):
            if idx == 0:
                lead_names = sample.signal_names.astype(str)
            if len(sample.signal[:, 0]) != 5000:
                continue

            array_signal.append(sample.signal)
            array_name.append(sample.noun_id.decode("utf-8"))
            array_fs.append(sample.sampling_frequency)
            array_sex.append(sample.sex.decode("utf-8"))

    ds_ecg = xr.Dataset(
        data_vars=dict(
            signal=(["id", "time", "lead_name"], np.array(array_signal)),
            fs=(["id"], np.array(array_fs)),
            sex=(["id"], np.array(array_sex)),
        ),
        coords=dict(
            id=(["id"], np.array(array_name)),
            time=(["time"], np.arange(0, 5000)),
            lead_names=(["lead_names"], lead_names),
        ),
        attrs=dict(description="ecg with pathologies description"),
    )

    if save_path is not None:
        ds_ecg.to_netcdf(os.path.join(save_path, "ecg_data_2020.nc"))

    return ds_ecg


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
        raise ValueError("The features are not between 0 and 1")
    return True


def extract_index_label(ds_data, required_index=None, aggregation_method="mean"):
    """Extract index and label from xarray dataset

    Args:
        ds_data (_type_): Data in xarray format.
        required_index (list, optional): List of index to extract.
        aggregation_method (str, optional): Aggregation method to use. Defaults to "mean".
            One of ["mean", "min", "max", "median", "None"]

    Returns:
        df_X (pd.DataFrame):dataframe with requested index
    """

    if not isinstance(required_index, list):
        required_index = [required_index]

    ds_filtered = ds_data.where(ds_data.data_quality != "unlabeled").dropna(dim="id")

    np_metrics = ds_filtered.quality_metrics.values
    metrics_names = ds_filtered.metric_name.values.tolist()
    np_label = ds_filtered.data_quality.values

    np_label[np_label == "acceptable"] = 0
    np_label[np_label == "unacceptable"] = 1
    np_label = np_label.astype(int)

    if "HR" in metrics_names:
        HR_index = metrics_names.index("HR")
        HR_metrics = np_metrics[:, :, HR_index].min(axis=1)

    if aggregation_method == "mean":
        X = np_metrics.mean(axis=1)
    elif aggregation_method == "min":
        X = np_metrics.min(axis=1)
    elif aggregation_method == "max":
        X = np_metrics.max(axis=1)
    elif aggregation_method == "median":
        X = np.median(np_metrics, axis=1)
    elif aggregation_method is not None:
        raise ValueError("Aggregation method not supported")

    X[:, HR_index] = HR_metrics
    df_X = pd.DataFrame(X, columns=metrics_names)
    df_y = pd.DataFrame(np_label, columns=["y"])

    if required_index is not None:
        df_X = df_X.loc[:, required_index]
    else:
        required_index = df_X.columns.tolist()

    if "HR" in required_index:
        df_X.loc[:, "HR"] = HR_metrics

    return df_X, df_y
