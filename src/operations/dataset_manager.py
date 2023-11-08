import numpy as np
import wfdb
import argparse
import os
import sys
import warnings
import xarray as xr
import pandas as pd

##Custom import
from shared_utils.utils_path import data_path


def add_file_condition(path, file):
    """
    Check if the file you are adding a file and it has either .dat,.hea,.atr or .xws

    Args:
        path (String): Path to your Physionet data folder
        file (String): Name of the file

    Returns:
        Boolean : True if all conditions stated before confirmed
    """
    return os.path.isfile(os.path.join(path, file)) and (
        file.endswith(".dat")
        or file.endswith(".hea")
        or file.endswith(".atr")
        or file.endswith(".xws")
    )


def get_name_files(name_dataset, ignore_inner_folder):

    """
    Get the name of all the files present in your Physionet dataset. If needed, it takes care of other data present in folder inside it.

    Args:
        name_dataset (String): The name of your Physionet dataset
        ignore_inner_folder (Boolean): Indicate if data present in folder inside your Physionet must be verified

    Returns:
        x (numpy array string): numpy array of files name (preceded by the name of the inner folder if checked)
    """
    path_to_folder = os.path.join(data_path, name_dataset)
    if not os.path.isdir(path_to_folder):
        raise OSError("Please indicate the correct name of your dataset!")
    if ignore_inner_folder:
        files = np.array(
            [
                f.split(".")[0]
                for f in os.listdir(path_to_folder)
                if add_file_condition(path_to_folder, f)
            ]
        )
    else:
        files = np.array([])
        for f in os.listdir(path_to_folder):
            if os.path.isdir(os.path.join(path_to_folder, f)):
                new_path = os.path.join(path_to_folder, f)
                for f1 in os.listdir(new_path):
                    if add_file_condition(new_path, f1):
                        files = np.append(files, os.path.join(f, f1).split(".")[0])
            else:
                if add_file_condition(path_to_folder, f):
                    files = np.append(files, f.split(".")[0])

    files = np.unique(files)

    return files


def resampling_data(data_ref, fs=500):
    """
    Resample your physionet data at your desired sampling frequency and divided it into different time window

    Args:
        data_ref (Tuple): Tuple containing your physionet data
        time_window (int) : The time window (in sec) you want your signal to have
        fs (int) : Your sampling frequency

    Returns:
        data (Tuple) : Return your physionet data at your desired sampling frequency (format : (numpy array shape [number_signal,number_of_time_window,signal_length],dictionnary containing updated physionet metadata))
    """
    ##Convert data into a list
    data = list(data_ref)
    ## Check sampling frequency and time window given
    if not isinstance(fs, int):
        raise TypeError("Your sampling frequency must be a non null integer")
    elif fs == 0 or fs < 0:
        raise ValueError("Your sampling frequency must be strictly positive")

    ## number of signal
    nb_signal = data[1]["n_sig"]
    ##Original sampling frequency
    fs_ori = data[1]["fs"]
    ##new length signal
    N_old = data[1]["sig_len"]

    ## Calculate duration of the total signal
    time_tot = N_old / fs_ori
    ## First, we have to resample the signal into 500 Hz.
    scale = fs / fs_ori
    ##new sample length
    N_res = int(N_old * (scale))
    resample_data = np.zeros([N_res, nb_signal])

    for sig in range(nb_signal):
        sig_to_resamp = data[0][:, sig]
        resample_data[:, sig] = np.interp(
            np.linspace(0, time_tot, N_res, endpoint=False),
            np.linspace(0, time_tot, N_old, endpoint=False),
            sig_to_resamp,
        )
    ##Changing the value now :
    data[0] = resample_data
    data[1]["sig_len"] = N_res
    data[1]["fs"] = fs
    return tuple(data)


def segment_signal(data, fs=500, time_window=10):
    """
    Segment your signal into multiple sub signal of define time window.

    Args:
        data_ref (Numpy array): Numpy array containing your data (shape : [signal_length,nb_signal])
        time_window (int) : The time window (in sec) you want your signal to have
        fs (int) : Your sampling frequency

    Returns:
        data (Numpy array) : Segmented signal (format : (numpy array shape [number_signal,number_of_time_window,signal_length],dictionnary containing updated physionet metadata))
    """
    if not isinstance(fs, int):
        raise TypeError("Your sampling frequency must be a non null integer")
    elif fs == 0 or fs < 0:
        raise ValueError("Your sampling frequency must be strictly positive")

    if not isinstance(time_window, int):
        raise TypeError("Your time_window must be a non null integer")
    elif time_window < 0:
        raise ValueError("Your time_window must be strictly positive")

    N_new = fs * time_window
    N_res = data.shape[0]
    if N_res < N_new:
        raise ValueError(
            "Please give a time window lower than you signal recording time"
        )
    nb_signal = data.shape[1]
    n = int(N_res / N_new)
    ## Array containing all the new sample (shape : [nb_signal,nb_chunk,N_new])
    new_data = np.zeros([nb_signal, n, N_new])
    for s in range(nb_signal):
        sig_study = data[:, s]
        for chunks in range(n):
            new_data[s, chunks, :] = sig_study[N_new * chunks : N_new * (chunks + 1)]
    return new_data


def format_architecture_data(data, patient_id):
    """
    Format the architecture of your Physionet into an xarray

    Args:
        data (Tuple) : Your physionet data at your desired sampling frequency (format : (numpy array shape [number_signal,number_of_time_window,signal_length],dictionnary containing updated physionet metadata))
        patient_id (String) : The ID number of your patient

    Returns:
        data (xarray) : xarray format of your data
    """
    xr_data = xr.DataArray(data[0], dims=("signal_length", "nb_signal"))
    xr_data.attrs["ID"] = patient_id
    for key, value in data[1].items():
        xr_data.attrs[key] = value
    return xr_data


def get_dataset(name_dataset, ignore_subdfolder=True, fs=None):
    """
    Get your physionet dataset (at the desired sampling frequency and time window)

    Args:
        name_dataset (String): Name of your physionet dataset
        ignore_subfolder (Boolean : Defaul = True): Boolean if you want to look into any folder inside your physionet dataset
        fs (int : Default = None) : your sampling frequency (in Hz)


    Returns:
        dataset (List) : List containing the entire dataset under the xarray format
    """

    files = get_name_files(name_dataset, ignore_inner_folder=ignore_subdfolder)
    dico_data = {}
    path_to_folder = os.path.join(data_path, name_dataset)
    for data in files:
        dico_data[data.split("/")[-1]] = wfdb.rdsamp(os.path.join(path_to_folder, data))

    if fs is not None:
        for data in files:
            dico_data[data.split("/")[-1]] = resampling_data(
                dico_data[data.split("/")[-1]], fs=fs
            )

    dataset = [
        format_architecture_data(dico_data[key], key) for key, _ in dico_data.items()
    ]

    return dataset
