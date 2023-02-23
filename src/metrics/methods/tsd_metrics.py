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
    c1 = np.linspace(t0, int((len(signal_tot) / fs) + t0), len(signal_tot))
    c_adapted = adapted_c(c1, fs, h, hprime, signal_tot)
    I1_t = I1(c_adapted, signal_tot, fs, h, hprime, 1 / fs, t0)
    I2_t = I2(c_adapted, signal_tot, fs, h, hprime, 1 / fs, t0)
    return I1_t, I2_t, c_adapted


@njit
def Interval_calculator_lead(signal, fs, t0=0):
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
    # if dic_segment_lead <100 :
    # dic_segment_lead = 100
    return dic_segment_lead


# def Interval_calculator_all(dico_signal, name_signal, fs):
#     dic_segment_lead = {}
#     for i in name_signal:
#         dic_segment_lead[i] = Interval_calculator_lead(dico_signal[i], fs)
#     return dic_segment_lead


def is_segment_flatline(sig):
    cond = np.where(np.diff(sig.copy(), 1) != 0.0, False, True)
    if len(cond[cond == True]) < 0.50 * len(sig):
        return False
    return True


def tsd_index_solo(dico_signal, name_lead, fs):

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


def tsd_index(signals, fs, **norm):

    ###Index Creation :TSD
    ###The label will be as follow : mean(TSD) < 1.25 = Acceptable;mean(SDR of all lead) >1.25 = Unacceptable
    ##For each lead, we will return a more precise classification based on the folloying rules:
    ## TSD<1.25 = Good quality ; 1.25<TSD<1.40 = Medium quality; TSD>1.4 = Bad quality
    # dico_seg = Interval_calculator(dico_signal,name_lead,fs,t0)
    D_arr = np.array([])
    # dic_segment = Interval_calculator_all(dico_signal,name_lead,fs)
    # dic_segment = 2500
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

    ###Index Creation :TSD for 1 lead
    ###The label will be as follow : mean(TSD) < 1.25 = Acceptable;mean(SDR of all lead) >1.25 = Unacceptable
    ##For each lead, we will return a more precise classification based on the folloying rules:
    ## TSD<1.25 = Good quality ; 1.25<TSD<1.40 = Medium quality; TSD>1.4 = Bad quality
    # dico_seg = Interval_calculator(dico_signal,name_lead,fs,t0)
    Dv, _ = tsd_mean_calculator(signal, segment, fs)
    return Dv


@njit
def lm_q(signal1, m, k, fs):
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
    # calc_L_series = np.frompyfunc(lambda m: Lm_q(signal, m, k, fs), 1, 1)
    calc_L_series = np.zeros(k)
    for m in range(1, k + 1):
        calc_L_series[m - 1] = lm_q(signal, m, k, fs)
    L_average = np.mean(calc_L_series)
    return L_average


def Dq(signal, kmax, fs):
    calc_L_average_series = np.frompyfunc(lambda k: lq_k(signal, k, fs), 1, 1)

    k = np.arange(1, kmax + 1)
    L = calc_L_average_series(k).astype(np.float64)

    D_t, _ = -1 * np.polyfit(np.log2(k), np.log2(L), 1)

    return D_t


def tsd_plot(dico_lead, name_lead, fs):

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
def tsd_calculator(signal2, segment_length, fs):
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
    Power_sig = (1 / len(sig)) * np.sum(np.abs(sig) ** 2, dtype=np.float64)
    P_db = 10 * np.log10(Power_sig)
    noisedb = P_db - SNR
    sd_db_watts = 10 ** (noisedb / 10)
    # sd_noise = np.sqrt(Power_sig/(SNR))
    noise = np.random.normal(0, np.sqrt(sd_db_watts), len(sig))
    sig_noisy = sig + noise
    return sig_noisy


def tsd_vs_noiseLevel_array(noise_level, path_to_data, list_attractor):
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
