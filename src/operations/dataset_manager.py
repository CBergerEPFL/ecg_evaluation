import numpy as np
from numpy.random import seed
import wfdb
import os
from pyspark.sql import SparkSession
from petastorm.unischema import dict_to_spark_row
from petastorm.etl.dataset_metadata import materialize_dataset

##Custom import
from shared_utils.utils_path import data_path

##Set seed :

seed(1)

##Functions


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


def has_numbers(inputString, want_noise_data):
    """
       Check if string variable has number (int) in it, only if the user wants only the noisy data

        Args:
            name_dataset (String): String to check if there is number (int number)
            want_noise_data (Boolean) : Boolean indicating if the user want only the noise data

    Returns:
        (Bool): Boolean indicating if there is a number in it.
    """
    if want_noise_data:
        return any(char.isdigit() for char in inputString)
    else:
        return False


def get_name_files(name_dataset, ignore_inner_folder, only_noise_data):

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
                and not has_numbers(f, only_noise_data)
            ]
        )
    else:
        files = np.array([])
        for f in os.listdir(path_to_folder):
            if os.path.isdir(os.path.join(path_to_folder, f)):
                new_path = os.path.join(path_to_folder, f)
                for f1 in os.listdir(new_path):
                    if add_file_condition(new_path, f1) and not has_numbers(
                        os.path.join(f, f1), only_noise_data
                    ):
                        files = np.append(files, os.path.join(f, f1).split(".")[0])
            else:
                if add_file_condition(path_to_folder, f) and not has_numbers(
                    f, only_noise_data
                ):
                    files = np.append(files, f.split(".")[0])

    files = np.unique(files)

    return files


def resampling_data(data_ref, fs=500):
    """
    Resample your physionet data at your desired sampling frequency and divided it into different time window

    Args:
        data_ref (Tuple): Tuple containing your physionet data
        fs (int) : Your sampling frequency

    Returns:
        data (Tuple) : Return your physionet data at your desired sampling frequency (format : (numpy array shape [number_signal,1,signal_length],dictionnary containing updated physionet metadata))
    """
    ##Convert data into a list
    data = list(data_ref)
    ## Check sampling frequency
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
    resample_data = np.zeros([nb_signal, N_res, 1])

    for sig in range(nb_signal):
        sig_to_resamp = data[0][sig, :, 0]
        resample_data[sig, :, 0] = np.interp(
            np.linspace(0, time_tot, N_res, endpoint=False),
            np.linspace(0, time_tot, N_old, endpoint=False),
            sig_to_resamp,
        )
    ##Changing the value now :
    data[0] = resample_data
    data[1]["sig_len"] = N_res
    data[1]["fs"] = fs
    return data


def segment_signal(data, fs=500, time_window=10):
    """
    Segment your signal into multiple sub signal of define time window.

    Args:
        data (List [Numpy array, dict]): List containing Numpy array containing your data (shape : [nb_signal,1,signal_length]) and a dict with the metadata
        time_window (int) : The time window (in sec) you want your signal to have
        fs (int) : Your sampling frequency

    Returns:
        data (List [Numpy array, dict]) : List containing Numpy array containing your data (shape : [nb_signal,nb_time_window,signal_length]) and a dict with the metadata
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
    N_res = data[0].shape[1]
    if N_res < N_new:
        raise ValueError(
            "Please give a time window lower than you signal recording time"
        )
    nb_signal = data[0].shape[0]
    n = int(N_res / N_new)
    ## Array containing all the new sample (shape : [nb_signal,nb_chunk,N_new])
    new_data = np.zeros([nb_signal, N_new, n])
    for s in range(nb_signal):
        sig_study = data[0][s, :, 0]
        for chunks in range(n):
            new_data[s, :, chunks] = sig_study[N_new * chunks : N_new * (chunks + 1)]
    data[0] = new_data
    data[1]["sig_len"] = N_new
    data[1]["nb_time_window"] = n
    return data


def format_architecture_data(data):
    """
    Format the architecture of your Physionet into an list fo dictionnaries

    Args:
        data (Tuple) : Your physionet data at your desired sampling frequency (format : (numpy array shape [number_signal,number_of_time_window,signal_length],dictionnary containing updated physionet metadata))

    Returns:
        dataset (List) : List containing all the data in the dictionnary format
    """

    dataset = []
    for keys, values in data.items():
        signal = values[0]
        metadata = values[1]
        for sn in range(len(metadata["sig_name"])):
            new_data = {}
            new_data["noun_id"] = f"{keys}_{metadata['sig_name'][sn]}"
            new_data["signal"] = signal[sn, :, :]
            for key, value in values[1].items():
                if type(value) == list:
                    value = np.asarray(value)
                    if all(isinstance(value[j], str) for j in range(len(value))):
                        value = value.astype(np.bytes_)
                new_data[key] = value
            new_data["sig_name"] = np.asarray([metadata["sig_name"][sn]]).astype(
                np.bytes_
            )
            dataset.append(new_data)

    return dataset


def format_architecture_data_noisy(data, N_lead=12):
    """
    Format the architecture of your Physionet into a dict adapated for noisy dataset.

    Args:
        data (Tuple) : Your physionet data at your desired sampling frequency (format : (numpy array shape [number_signal,number_of_time_window,signal_length],dictionnary containing updated physionet metadata))
        N_lead (int) : The number of fake lead to create (Default : 12)

    Returns:
        data (Dict) : Dictionnary format of your data
    """
    dataset = []
    for keys, values in data.items():
        signals = values[0]
        metadata = values[1]

        ## Adapt metadata :
        metadata["units"] = ["mV"] * N_lead
        metadata["sig_name"] = [f"S{i}" for i in range(1, N_lead + 1)]
        metadata["comments"] = [
            f"Patient created using randomly selected segments from {keys}'s signal "
        ]
        metadata["n_sig"] = N_lead
        metadata["nb_time_window"] = 1

        nb_patient = int((signals.shape[0] * signals.shape[2]) / N_lead)
        index_segment = np.random.randint(
            0, signals.shape[2], signals.shape[0] * signals.shape[2]
        )
        index_lead_sel = np.random.randint(
            0, signals.shape[0], signals.shape[0] * signals.shape[2]
        )
        for i, j in zip(
            range(nb_patient), range(0, signals.shape[0] * signals.shape[2], N_lead)
        ):
            new_data = {}
            new_data["noun_id"] = f"{keys}_{i}"
            new_data["signal"] = np.transpose(
                signals[
                    index_lead_sel[j : j + N_lead], :, index_segment[j : j + N_lead]
                ]
            )
            for key, value in metadata.items():
                if type(value) == list:
                    value = np.asarray(value)
                    if all(isinstance(value[j], str) for j in range(len(value))):
                        value = value.astype(np.bytes_)
                new_data[key] = value

            dataset.append(new_data)
    return dataset


def get_dataset(
    name_dataset,
    ignore_subdfolder=True,
    only_noise_data=False,
    fs=None,
    time_window=None,
    N_lead=12,
):
    """
    Get your physionet dataset (at the desired sampling frequency and time window)

    Args:
        name_dataset (String): Name of your physionet dataset
        ignore_subfolder (Boolean : Defaul = True): Boolean if you want to look into any folder inside your physionet dataset
        fs (int : Default = None) : your sampling frequency (in Hz)
        time_window (int : Default = None) : The time window lenght you want your signal to have.


    Returns:
        dataset (List) : List containing the entire dataset (list of dictionary)
    """

    files = get_name_files(
        name_dataset,
        ignore_inner_folder=ignore_subdfolder,
        only_noise_data=only_noise_data,
    )
    print("The following files will be in the dataset : ", files)
    dico_data = {}
    path_to_folder = os.path.join(data_path, name_dataset)
    for data in files:
        d = wfdb.rdsamp(os.path.join(path_to_folder, data))
        signal = np.transpose(d[0])
        signal = signal[:, :, np.newaxis]
        metadata = d[1].copy()
        metadata["nb_time_window"] = 1
        dico_data[data.split("/")[-1]] = (signal, metadata)

    if fs is not None:
        for data in files:
            dico_data[data.split("/")[-1]] = resampling_data(
                dico_data[data.split("/")[-1]], fs=fs
            )

    if time_window is not None:
        for data in files:
            dico_data[data.split("/")[-1]] = segment_signal(
                dico_data[data.split("/")[-1]], fs=fs, time_window=time_window
            )

    if only_noise_data:
        return format_architecture_data_noisy(dico_data, N_lead=N_lead)

    else:
        return format_architecture_data(dico_data)


def get_path_petastorm_format(name_dataset, name_folder):

    """
    Create a save path variable towards a desired folder, with the format adapted for petastorm.

    Args:
        name_datast (String): Name of your dataset (no a path to a directorie. Just the name of the dataset)
        name_folder (String) : Name of the folder where you want to save your petastorm data

    Returns:
        path_petastorm (String) : Formated path toward the folder where to save dataset.
    """
    save_path = os.path.join(data_path, name_dataset, name_folder)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    path_petastorm = f"file://{save_path}"
    return path_petastorm


def save_to_parquet_petastorm(
    dataset, name_dataset, sparksession, schema, row_generator, noise_data=False
):

    """
    Save your dataset into petastorm adapted parquet file, given your Unischema.

    Args:
        dataset (List) : List containing the entire dataset (list of dictionary)
        name_datast (String): Name of your dataset (no a path to a directorie. Just the name of the dataset)
        sparksession (SparkSession.builder) : The spark session your are using.
        schema (Petastorm Unischema) : The Unischema you want to use
        row_generator (function) : The row generator function for accessing one data of your dataset.

    """

    ## Parameter initialization
    row_group_size_mb = 256
    sc = sparksession.sparkContext
    range_dataset = range(len(dataset))

    ## Path to save parquet
    if noise_data:
        path_petastorm = get_path_petastorm_format(name_dataset, "ParquetFileNoise")
    else:
        path_petastorm = get_path_petastorm_format(name_dataset, "ParquetFile")
    ##Now : let's save the dataset into Parquet
    with materialize_dataset(sparksession, path_petastorm, schema, row_group_size_mb):
        rows_rdd = (
            sc.parallelize(range_dataset)
            .map(row_generator)
            .map(lambda x: dict_to_spark_row(schema, x))
        )
        sparksession.createDataFrame(rows_rdd, schema.as_spark_schema()).coalesce(
            10
        ).write.mode("overwrite").parquet(path_petastorm)
