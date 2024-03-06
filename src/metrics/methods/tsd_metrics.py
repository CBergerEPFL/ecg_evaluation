import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import isnan
from numba import njit


def system_coordinates_reader(Path_to_data, Attractor_name, num_attractor=0):
    """

    Read csv file and extract x,y,z time evolution of a dynamical system

    Args:
        Path_to_data (String): Path toward your dataset
        Attractor_name (String): Attractor name
        num_attractor (int, optional): Attractor type. Defaults to 0.

    Returns:
        Tuple : Tuple containing the x,y,z time evolution and the timestep
    """
    path = Path_to_data + f"/{Attractor_name}_attractors"
    if num_attractor is None:
        df = pd.read_csv(path + ".csv")
    else:
        df = pd.read_csv(path + f"/{Attractor_name}__{num_attractor}.csv")
    df_n = df.to_numpy()
    xyzs = df_n[:, 1:4]
    t = df_n[:, 0]
    return xyzs, t


@njit
def taux_Mean_fast(signal, taux, hprime, h=0):
    """

    Calculate the Mean of the observed signal follwoing the definition given by Takumi Sase et al in
    "Estimating the level of dynamical noise in time series by using fractal dimensions"

    Args:
        signal (1D Numpy array): Signal
        taux (int): Time step
        hprime (int): Interval bound
        h (int, optional): Segment timestep. Defaults to 0.

    Returns:
        Float : E(Xobs(t))
    """
    return np.mean((signal[int(h + taux) : int(taux + hprime + h)]))


@njit
def taux_var_fast(signal, taux, hprime, h=0):
    """
    Calculate the variance of the observed signal following the definition given by Takumi Sase et al in
    "Estimating the level of dynamical noise in time series by using fractal dimensions"
    Args:
        signal (1D Numpy array): Signal
        taux (int): Time step
        hprime (int): Interval bound
        h (int, optional): Segment timestep. Defaults to 0.

    Returns:
        Float : Var(Xobs(t))
    """
    return np.var(signal[int(h + taux) : int(taux + hprime + h)])


@njit(parallel=True)
def adapted_c(c_val, fs, h, hprime, signal):
    """
    Return the adequate c interval for the calculation of I1 adn I2

    Args:
        c_val (1D Numpy array): Timestep valus
        fs (int): Sampling frequency
        h (Float): Segment timestep
        hprime (Float): Interval bound
        signal (1D Numpy array): Signal studied

    Returns:
        1D numpy array : Adapted timestep interval
    """
    for l in c_val:
        if (
            l * fs + hprime * len(signal) + h * len(signal) > len(signal)
            and l * fs + h * len(signal) > len(signal) - 1
        ):
            c = c_val[c_val < l]
            break
    return c


@njit
def I1(c, signal, fs, h, hprime, step_c, t0=0):
    """

    I1 value as define by Takumi Sase et al in
    "Estimating the level of dynamical noise in time series by using fractal dimensions"

    Args:
        c (1D Numpy array ): Timestep interval
        signal (1D Numpy array): Signal considered (lead)
        fs (int): Sampling frequency
        h (Float): Segment timestep
        hprime (Float): Interval bound
        step_c (Float): step used to get c
        t0 (int, optional): Intial timestep. Defaults to 0.

    Returns:
        1D Numpy array : I1 values
    """
    tab = np.zeros_like(c)
    for count in range(len(tab)):
        if count == 0:
            I1c = (
                (1 / (h * len(signal)))
                * step_c
                * np.abs(
                    taux_Mean_fast(
                        signal, t0 * fs, hprime * len(signal), h * len(signal)
                    )
                    - taux_Mean_fast(signal, t0 * fs, hprime * len(signal))
                )
            )
            tab[count] = I1c
        else:
            I1c = tab[count - 1]
            I1c = I1c + (
                (1 / (h * len(signal)))
                * step_c
                * np.abs(
                    taux_Mean_fast(
                        signal, t0 * fs, hprime * len(signal), h * len(signal)
                    )
                    - taux_Mean_fast(signal, t0 * fs, hprime * len(signal))
                )
            )
            tab[count] = I1c
    return tab[:-1]


@njit
def I2(c, signal, fs, h, hprime, step_c, t0=0):
    """
    I2 value as define by Takumi Sase et al in
    "Estimating the level of dynamical noise in time series by using fractal dimensions"

    Args:
        c (1D Numpy array ): Timestep interval
        signal (1D Numpy array): Signal considered (lead)
        fs (int): Sampling frequency
        h (Float): Segment timestep
        hprime (Float): Interval bound
        step_c (Float): step used to get c
        t0 (int, optional): Intial timestep. Defaults to 0.

    Returns:
        1D Numpy array : I2 values
    """
    tab = np.zeros_like(c)
    for count in range(len(tab)):
        if count == 0:
            I1c = (
                (1 / (h * len(signal)))
                * step_c
                * np.abs(
                    taux_var_fast(
                        signal, t0 * fs, hprime * len(signal), h * len(signal)
                    )
                    - taux_var_fast(signal, t0 * fs, hprime * len(signal))
                )
            )
            tab[count] = I1c
        else:
            I1c = tab[count - 1]
            I1c = I1c + (
                (1 / (h * len(signal)))
                * step_c
                * np.abs(
                    taux_var_fast(
                        signal, t0 * fs, hprime * len(signal), h * len(signal)
                    )
                    - taux_var_fast(signal, t0 * fs, hprime * len(signal))
                )
            )
            tab[count] = I1c
    return tab[:-1]


@njit
def discrepancies_mean_curve(signal_tot, fs, h, hprime, t0=0):
    """
    Calculate I1 and I2 coefficients as defined by Takumi Sase et al in
    "Estimating the level of dynamical noise in time series by using fractal dimensions"
    Args:
        signal_tot (1D Numpy array): Signal studied
        fs (int): Sampling frequency
        h (Float): Segment timestep
        hprime (Float): Interval bound
        t0 (int, optional): Initial timestep point. Defaults to 0.

    Returns:
        Tuple : Tuple containing I1,I2 and the interval array on which they have been calculated (in that order)
    """
    c1 = np.linspace(t0, int((len(signal_tot) / fs) + t0), len(signal_tot))
    c_adapted = adapted_c(c1, fs, h, hprime, signal_tot)
    I1_t = I1(c_adapted, signal_tot, fs, h, hprime, 1 / fs, t0)
    I2_t = I2(c_adapted, signal_tot, fs, h, hprime, 1 / fs, t0)
    return I1_t, I2_t, c_adapted


@njit
def Interval_calculator_lead(signal, fs, t0=0):
    """

    Calculate the optimal interval for applying the TSD on the signal, wiht the method defined by Takumi Sase et al in
    "Estimating the level of dynamical noise in time series by using fractal dimensions"

    Args:
        signal (2D Numpy array): Signal studied
        fs (int): Sampling frequency
        t0 (int, optional): Initial timestep point. Defaults to 0.

    Returns:
        _type_: _description_
    """
    h = 0.001
    hprime = 0.005
    I1c, I2c, c = discrepancies_mean_curve(signal, fs, h, hprime)
    c1 = c[np.where(I1c < 0.5)]  # np.max(I1c)/2
    c2 = c[np.where(I2c < 1.25)]
    if np.isnan(c1).any():
        c1 = c1[~np.isnan(c1)]
    elif np.isnan(c2).any():
        c2 = c2[~np.isnan(c2)]
    if len(c1) == 0:
        cs = c2[-1]
    elif len(c2) == 0:
        cs = c1[-1]
    elif len(c1) == 0 and len(c2) == 0:
        cs = 0.2
    else:
        cs = np.minimum(c1[-1], c2[-1])
    dic_segment_lead = (cs - t0) * fs

    return dic_segment_lead


def is_segment_flatline(sig):
    """
    Check if the signal has more than 50% of its values that are horizontale

    Args:
        sig (1D Numpy array): signal considered

    Returns:
        Bool : Boolean value indicating if the signal is mostly a flatline (True if this is the case)
    """
    cond = np.where(np.diff(sig.copy(), 1) != 0.0, False, True)
    if len(cond[cond == True]) < 0.50 * len(sig):
        return False
    return True


def tsd_index_solo(dico_signal, name_lead, fs):
    """
    Calculate the TSD index for one lead

    Args:
        dico_signal (1D Numpy array): Lead studied
        name_lead (String): Name of the lead
        fs (int): Sampling frequency

    Returns:
        dictionary : Dictionary containing the mean TSD value of the lead
    """
    Dv, _ = tsd_mean_calculator(dico_signal, 100, fs)
    if Dv < 1:
        Dv = 1
    return {name_lead: Dv}


def tsd_index(signals, fs, **norm):
    """

    Calculate the TSD score for signal quality assessment

    Args:
        signals (2D Numpy array): Numpy array containing all the signal (expected shape : [num_feature (ex : #lead),signal_length])
        fs (int): Sampling frequency

    Returns:
        1D Numpy array : 1D numpy array containing the index for each lead (shape [num_feature])
    """
    D_arr = np.array([])

    for i in range(signals.shape[0]):
        Dv, _ = tsd_mean_calculator(signals[i, :], 100, fs)

        if (
            Dv < 1
        ):  ##Reason : Since we use an Approximation of Fractal index, we can have some bad approximation and thus have values outside predefined range
            Dv = 1

        if norm.get("normalization") == True:
            D_arr = np.append(D_arr, (2 - Dv))
        else:
            D_arr = np.append(D_arr, Dv)
    return D_arr


def tsd_index_dico(dico_signal, name_lead, fs):
    """
    Calculate the TSD score for signal quality assessment using dictionaries

    Args:
        dico_signal (Dictionary): Dictionary containing each lead (key : Lead name ; Value : 1D Numpy array)
        name_lead (1D numpy array String or String list): List of the leads name
        fs (int): Sampling frequency

    Returns:
        Tuple : Tuple with a dictionary of the TSD value of each lead and the mean TSD value of the signals set
    """
    ###Index Creation :TSD
    ###The label will be as follow : mean(TSD) < 1.25 = Acceptable;mean(SDR of all lead) >1.25 = Unacceptable
    ##For each lead, we will return a more precise classification based on the folloying rules:
    ## TSD<1.25 = Good quality ; 1.25<TSD<1.40 = Medium quality; TSD>1.4 = Bad quality
    # dico_seg = Interval_calculator(dico_signal,name_lead,fs,t0)
    dico_D = {}
    D_arr = np.array([])
    # dic_segment = Interval_calculator_all(dico_signal,name_lead,fs)
    # dic_segment = 2500
    for i in name_lead:
        Dv, _ = tsd_mean_calculator(dico_signal[i], 100, fs)
        if Dv < 1:
            Dv = 1
        dico_D[i] = (Dv, dico_signal[i])
        D_arr = np.append(D_arr, Dv)
    return dico_D, np.mean(D_arr)


def tsd_index_lead(signal, segment, fs):
    """
    Calculate the TSD score for signal quality assessment for one lead only

    Args:
        signal (1D Numpy array): Signal
        segment (int): Segment length desired
        fs (int): Sampling frequency

    Returns:
        Float64 : Return the Mean TSD value of the signal
    """
    ###Index Creation :TSD for 1 lead
    ###The label will be as follow : mean(TSD) < 1.25 = Acceptable;mean(SDR of all lead) >1.25 = Unacceptable
    ##For each lead, we will return a more precise classification based on the folloying rules:
    ## TSD<1.25 = Good quality ; 1.25<TSD<1.40 = Medium quality; TSD>1.4 = Bad quality
    # dico_seg = Interval_calculator(dico_signal,name_lead,fs,t0)
    Dv, _ = tsd_mean_calculator(signal, segment, fs)
    return Dv


@njit
def lm_q(signal1, m, k, fs):
    """
    Calculate the Lm_q term in the Higuchi method. Adapted from : https://github.com/hiroki-kojima/HFDA

    Args:
        signal1 (1D Numpy array): Signal under study
        m (int): Total number of segment indexes
        k (int): Total number of segment
        fs (int): Sampling frequency

    Returns:
        Float64 : Lm_q
    """
    N = len(signal1)
    n = np.floor((N - m) / k)
    norm = (N - 1) / (n * k * (1 / fs))
    # sum = np.sum(np.abs(np.diff(signal1[m::k], n=1)))
    sum1 = 0
    for i in range(1, n):
        sum1 = sum1 + np.absolute(signal1[m + i * k] - signal1[m + (i - 1) * k])
    Lmq = (sum1 * norm) / k
    return Lmq


@njit
def lq_k(signal, k, fs):
    """
    Calculate the Lq_k term in the Higuchi method. Adapted from : https://github.com/hiroki-kojima/HFDA


    Args:
        signal (1D Numpy array): Signal under study
        k (int): Total number of segment
        fs (int): Sampling frequency

    Returns:
       Float64 : Averge length (l(k))
    """
    # calc_L_series = np.frompyfunc(lambda m: Lm_q(signal, m, k, fs), 1, 1)
    calc_L_series = np.zeros(k)
    for m in range(1, k + 1):
        calc_L_series[m - 1] = lm_q(signal, m, k, fs)
    L_average = np.mean(calc_L_series)
    return L_average


def Dq(signal, kmax, fs):
    """
    Calculate Dimension of time series. Adapted from : https://github.com/hiroki-kojima/HFDA

    Args:
        signal (1D Numpy array): Signal
        kmax (int): Total number of segment
        fs (int): Sampling Frequency

    Returns:
        Float64 : Time series dimension
    """
    calc_L_average_series = np.frompyfunc(lambda k: lq_k(signal, k, fs), 1, 1)

    k = np.arange(1, kmax + 1)
    L = calc_L_average_series(k).astype(np.float64)

    D_t, _ = -1 * np.polyfit(np.log2(k), np.log2(L), 1)

    return D_t


def tsd_plot(dico_lead, name_lead, fs):
    """
    Plot TSD time evolution for a set of signal given
    Args:
        dico_lead (Dictionary): Dictionary containing each lead (key : Lead name ; Value : 1D Numpy array)
        name_lead (1D String Numpy array or list of string): Name of each lead
        fs (int): Sampling frequency
    """
    D_lead = {}
    for i in name_lead:
        sig = dico_lead[i]
        segment_length = 100
        X = np.c_[
            [
                sig[int((w - 1)) : int((w) + segment_length)]
                for w in range(1, int(len(sig) - segment_length))
            ]
        ]
        L1 = np.array([lq_k(X[i, :], 1, fs) for i in range(X.shape[0])])
        L2 = np.array([lq_k(X[i, :], 2, fs) for i in range(X.shape[0])])
        Ds = (np.log(L1) - np.log(L2)) / (np.log(2))
        D_lead[i] = Ds

    for i in name_lead:
        plt.figure()
        _, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
        w_length = range(len(D_lead[i]))
        ax[0].plot(w_length, D_lead[i], label=i)
        ax[0].set_title(f"TSD time Evolution of Lead {i.decode('utf8')}")
        ax[0].set_xlabel("Segment number")
        ax[0].set_ylabel("TSD value")
        ax[0].grid()
        ax[1].plot(
            np.linspace(0, int(len(dico_lead[i]) / fs), len(dico_lead[i])),
            dico_lead[i],
            label=i,
        )
        ax[1].set_title(f"Lead {i.decode('utf8')}")
        ax[1].set_xlabel("Time (sec)")
        ax[1].set_ylabel("Voltage Amplitude")
        ax[1].grid()
        plt.show()


@njit
def tsd_mean_calculator(signal2, segment_length, fs):
    """
        Calculate the mean TSD of the signal

    Args:
        signal2 (1D Numpy array ): Signal
        segment_length (int): Segment size used to calculate the TSD
        fs (int): Sampling Frequency

    Returns:
        Tuple : Tuple containing the mean TSD value of the signal and its SD
    """
    Ds = np.zeros(int(len(signal2) - segment_length) - 1)
    for w in range(1, int(len(signal2) - segment_length)):
        sig_true = signal2[int((w - 1)) : int((w) + segment_length)]
        L1 = lq_k(sig_true, 1, fs)
        L2 = lq_k(sig_true, 2, fs)
        Ds[w - 1] = (np.log(L1) - np.log(L2)) / (np.log(2))
        ##Thresholding necessary since we use an approximation of the Higuchi method
        if Ds[w - 1] > 2 or np.isnan(Ds[w - 1]):
            Ds[w - 1] = 2
        elif Ds[w - 1] < 1:
            Ds[w - 1] = 1
    return np.mean(Ds[~np.isnan(Ds)]), np.std(Ds[~np.isnan(Ds)])


@njit
def TSD_calculator_time_series(signal2, segment_length, fs):
    """
        Calculate the TSD time evolution of the signal.

    Args:
        signal2 (1D Numpy array ): Signal
        segment_length (int): Segment size used to calculate the TSD
        fs (int): Sampling Frequency

    Returns:
        Tuple : Tuple containing the time evolution TSD value of the signal and its SD
    """
    Ds = np.zeros(int(len(signal2) - segment_length) - 1)
    for w in range(1, int(len(signal2) - segment_length)):
        sig_true = signal2[int((w - 1)) : int((w) + segment_length)]
        L1 = lq_k(sig_true, 1, fs)
        L2 = lq_k(sig_true, 2, fs)
        Ds[w - 1] = (np.log(L1) - np.log(L2)) / (np.log(2))
        ##Thresholding necessary since we use an approximation of the Higuchi method
        if Ds[w - 1] > 2 or np.isnan(Ds[w - 1]):
            Ds[w - 1] = 2
        elif Ds[w - 1] < 1:
            Ds[w - 1] = 1
    return Ds, np.mean(Ds[~np.isnan(Ds)])


@njit
def tsd_calculator(signal2, segment_length, fs):
    """
    Calculate the TSD time evolution of the signal and return the TSD time evolution

    Args:
        signal2 (1D Numpy array ): Signal
        segment_length (int): Segment size used to calculate the TSD
        fs (int): Sampling Frequency

    Returns:
        Tuple : Tuple containing the TSD time evolution array and its mean value
    """
    Ds = np.zeros(int(len(signal2) - segment_length) - 1)
    for w in range(1, int(len(signal2) - segment_length)):
        sig_true = signal2[int((w - 1)) : int((w) + segment_length)]
        L1 = lq_k(sig_true, 1, fs)
        L2 = lq_k(sig_true, 2, fs)
        Ds[w - 1] = (np.log(L1) - np.log(L2)) / (np.log(2))
        ##Thresholding necessary since we use an approximation of the Higuchi method
        if Ds[w - 1] > 2 or np.isnan(Ds[w - 1]):
            Ds[w - 1] = 2
        elif Ds[w - 1] < 1:
            Ds[w - 1] = 1
    return Ds, np.mean(Ds[~np.isnan(Ds)])


def add_observational_noise(sig, SNR):
    """
    Add to the signal an specific white noise according to a desired SNR value

    Args:
        sig (1D Numpy array): Signal
        SNR (Float): SNR of the output signal

    Returns:
        1D Numpy array : Signal with noise added
    """
    Power_sig = (1 / len(sig)) * np.sum(np.abs(sig) ** 2, dtype=np.float64)
    P_db = 10 * np.log10(Power_sig)
    noisedb = P_db - SNR
    sd_db_watts = 10 ** (noisedb / 10)
    # sd_noise = np.sqrt(Power_sig/(SNR))
    noise = np.random.normal(0, np.sqrt(sd_db_watts), len(sig))
    sig_noisy = sig + noise
    return sig_noisy


def tsd_vs_noiseLevel_array(noise_level, path_to_data, list_attractor):
    """
    Calculate TSD of systems (from csv file) with different noise levels

    Args:
        noise_level (1D Numpy Float array): Range of the noise level studied
        path_to_data (String):  Path to your systems
        list_attractor (List of String): Name of your system

    Returns:
        Tuple : Tuple containing dictionnary of TSD mean value for different noise level and their STD
    """
    Dmean = {}
    SD_D = {}

    for i in noise_level:
        if i == 0:
            for name in list_attractor:
                mid_Dmean = np.array([])
                mid_SD = np.array([])
                for j in range(
                    0, len(os.listdir(path_to_data + f"/{name}_attractors"))
                ):
                    coord, _ = system_coordinates_reader(path_to_data, name, j)
                    Obs = coord[:, 0]
                    Mean_TSD, SD_TSD = tsd_mean_calculator(Obs, 100)
                    mid_Dmean = np.append(mid_Dmean, Mean_TSD)
                    mid_SD = np.append(mid_SD, SD_TSD)
                Dmean[name] = np.array([np.mean(mid_Dmean)])
                SD_D[name] = np.array([np.mean(mid_SD)])

        else:
            for name in list_attractor:
                mid_Dmean = np.array([])
                mid_SD = np.array([])
                for j in range(
                    0, len(os.listdir(path_to_data + f"/{name}_attractors"))
                ):
                    coord, _ = system_coordinates_reader(path_to_data, name, j)
                    Obs = coord[:, 0]
                    noise_obs = add_observational_noise(Obs, 1 / i)
                    Mean_TSD, SD_TSD = tsd_mean_calculator(noise_obs, 100)
                    mid_Dmean = np.append(mid_Dmean, Mean_TSD)
                    mid_SD = np.append(mid_SD, SD_TSD)
                Dmean[name] = np.append(Dmean[name], np.mean(mid_Dmean))
                SD_D[name] = np.array(SD_D[name], np.mean(mid_SD))

    return Dmean, SD_D


def plt_TSDvsNoise(noise_lev, path_to_data, attractors_sel):
    """

    Pot a heatmap representing the variation of observational noise in function of dynamical noise using TSD

    Args:
        noise_level (1D Numpy Float array): Range of the noise level studied
        path_to_data (String):  Path to your systems
        list_attractor (List of String): Name of your system
    """
    Great_mean, Great_SD = tsd_vs_noiseLevel_array(
        noise_lev, path_to_data, attractors_sel
    )
    fig, ax = plt.subplots(len(attractors_sel) - 1, 2, figsize=(20, 10))
    for i, j in zip(attractors_sel, range(len(attractors_sel))):
        ax[j].errorbar(noise_lev, Great_mean[i], Great_SD[i])
        ax[j].set_xlabel("Noise level")
        ax[j].set_ylabel("mean TSD value")
        ax[j].set_title(f"TSD vs noise level for {i} system")
        ax[j].set_ylim([1.9, 2.1])
        ax[j].grid()

    plt.figure()
    for i in attractors_sel:
        plt.plot(noise_lev, Great_mean[i])
    plt.legend([i for i in attractors_sel])
    plt.title("Mean TSD value evolution with noise level for both system")
    plt.xlabel("Noise level")
    plt.ylabel("mean TSD value")
    plt.ylim([1.9, 2.1])
    plt.grid()
    plt.show()


def Comparative_lead_plot(
    Synth_data,
    Acc_data,
    Unacc_data,
    SD_synth,
    SD_acc,
    SD_unacc,
    S_level,
    name_lead,
    name="TSD",
):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
    # plt.rcParams.update({'font.size':20})
    fig.tight_layout(h_pad=4)
    coordinates = [(0, y) for y in range(3)]
    for i, c in zip(name_lead[:3], coordinates):

        lead_synth, lead_acc, lead_unacc = Synth_data[i], Acc_data[i], Unacc_data[i]
        e_synth, e_acc, e_unacc = SD_synth[i], SD_acc[i], SD_unacc[i]
        if c[1] == 0:
            ax[c[1]].errorbar(S_level, lead_synth, e_synth, label=" Synthethic lead ")
            ax[c[1]].errorbar(S_level, lead_acc, e_acc, label=" Acceptable lead ")
            ax[c[1]].errorbar(S_level, lead_unacc, e_unacc, label=" Unacceptable lead ")
        else:
            ax[c[1]].errorbar(S_level, lead_synth, e_synth)
            ax[c[1]].errorbar(S_level, lead_acc, e_acc)
            ax[c[1]].errorbar(S_level, lead_unacc, e_unacc)

        ax[c[1]].set_xlabel("SNR (db)")
        ax[c[1]].set_ylabel(f"mean {name} value")
        ax[c[1]].set_title(f"Lead {i.decode('utf8')}")
        ax[c[1]].grid()
    handles, labels = ax[0].get_legend_handles_labels()
    plt.figlegend(
        handles,
        labels,
        loc=(0.84, 0.7),
        labelspacing=1.0,
        handletextpad=0.0,
        handlelength=1.5,
        fancybox=True,
        shadow=True,
    )
    fig.suptitle(
        f"{name} vs SNR (dB) for average {name} value for 100 patients", fontsize=20
    )
    fig.subplots_adjust(top=0.90)
