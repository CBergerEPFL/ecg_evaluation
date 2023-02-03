import os
import sys

import numpy as np
import xarray as xr
from tqdm import tqdm

FILEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FILEPATH))
from .methods import method_registry

list_normalization = ["SNRECG", "TSD", "Flatline"]


def compute_metrics(
    signal: np.ndarray,
    fs: np.ndarray,
    list_methods: list,
    normalization=True,
    verbose=False,
):
    """Wrapper function to calculate metrics.

    Args:
        signal (np.array): Input signal.
        fs (int): Sampling frequency.
        list_methods (list): List of methods to calculate.
        normalization (bool): Whether to normalize the metrics.

    Returns:
        dict_metrics (dict): Dictionary of metrics.
    """

    if signal.ndim == 2:
        signal = signal[np.newaxis, ...]

    signal = np.transpose(signal, (0, 2, 1))
    X = np.zeros([signal.shape[0], signal.shape[1], len(list_methods)])
    for idx_method, name_method in enumerate(list_methods):
        if name_method not in list(method_registry.keys()):
            raise ValueError(f"The feature {name_method} is not implemented!")
        func_method = method_registry[name_method]

        for idx_signal in tqdm(
            range(signal.shape[0]),
            desc=f"Calculating {name_method}",
            disable=not verbose,
        ):
            if name_method in list_normalization:
                X[idx_signal, :, idx_method] = func_method(
                    signal[idx_signal, ...],
                    fs=fs[idx_signal],
                    normalization=normalization,
                )

            else:
                X[idx_signal, :, idx_method] = func_method(
                    signal[idx_signal, ...], fs=fs[idx_signal]
                )

    return X


def save_metrics_to_xarray(
    ds_data: xr.Dataset, list_methods: list, save_path: str, verbose=False
):
    """Compute and save metrics to xarray format.

    Args:
        ds_data (xarray): Data in xarray format.
        list_methods (list): List of methods to calculate.
        save_path (str): Path to save data.
        verbose (bool): Whether to show progress bar.

    Returns:
        ds_data (xarray): Initial xarray merged with requested metrics in xarray format.
    """

    if (save_path is not None) and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    signal = ds_data.signal.values
    fs = ds_data.fs.values
    # name_method = ["Corr_interlead","Corr_intralead","wPMF","SNRECG","HR","Kurtosis","Flatline","TSD"]
    # Compute metrics
    np_metrics = compute_metrics(
        signal, fs, list_methods, normalization=True, verbose=verbose
    )
    da_metric = xr.DataArray(
        np_metrics,
        dims=["id", "lead_names", "metric_name"],
        coords=[ds_data.id, ds_data.lead_names, list_methods],
    )
    ds_data["quality_metrics"] = da_metric

    if save_path is not None:
        path_to_file = os.path.join(save_path, "quality_metrics.nc")
        print(f"Saving computed metrics in netCDF format in: {path_to_file}")
        ds_data.to_netcdf(path_to_file)

    return ds_data
