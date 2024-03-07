import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import rfft, irfft
from numba import njit
from metrics.methods.tsd_metrics import (
    lq_k,
    discrepancies_mean_curve,
    Interval_calculator_lead,
    system_coordinates_reader,
    is_segment_flatline,
)


def plot_attractor_timevscoord(dico_xyzs, t, name):

    """
    Function that plots the attractors signal in function of the time.

    Inputs :
        dico_xyzs (dict) : dictionnary containing the attractors coordinates generated (for x,y,z coordinates)
        t (Numpy Array) : Array containing the time input use to generate the attractors
        name (string) : Name of the attractor to be plotted
    """
    for i in name:
        plt.plot(t, dico_xyzs[i], label=i)
        plt.xlabel("time")
        plt.ylabel("function value")
        plt.legend(loc="best", bbox_to_anchor=(1, 1))
        plt.grid()
        plt.plot()


def Plot_attractors(xyzs_attractor, name_attractor):
    """

    Functino that plot the 3D representation of a given xyz coordinates of a specific attractor

    Args:
        xyzs_attractor (3D numpy array): Array containing the xyz coordinates value at different time step (shape : time_length*coordinates)
        name_attractor (string): Name of the plotted attractor
    """
    ax = plt.figure().add_subplot(projection="3d")

    ax.plot(*xyzs_attractor.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(f"{name_attractor} Attractor")

    plt.show()


def add_observational_noise(sig, SNR):
    """

    Add, to a given signal, a noise level define by a specific SNR value

    Args:
        sig (1D numpy Array) : Signal you want to add noise
        SNR (Float): SNR level you want to add

    Returns:
        1D numpy array: signal with noise level define by SNR
    """
    Power_sig = (1 / len(sig)) * np.sum(np.abs(sig) ** 2, dtype=np.float64)
    P_db = 10 * np.log10(Power_sig)
    noisedb = P_db - SNR
    sd_db_watts = 10 ** (noisedb / 10)
    noise = np.random.normal(0, np.sqrt(sd_db_watts), len(sig))
    sig_noisy = sig + noise
    return sig_noisy


def add_observational_noise_val(sig, SNR):
    """
    Function that return the noise level added at a given SNR for a specific signal

    Args:
        sig (1D numpy Array) : Signal you want to add noise
        SNR (Float): SNR level you want to add

    Returns:
        float : Noise level added.
    """
    Power_sig = np.mean(np.abs(sig) ** 2)
    P_db = 10 * np.log10(Power_sig)
    noisedb = P_db - SNR
    sd_db_watts = 10 ** (noisedb / 10)
    # sd_noise = np.sqrt(Power_sig/(SNR))
    noise = np.random.normal(0, np.sqrt(sd_db_watts), (len(sig), 3))
    # sig_noisy = sig+noise
    return noise


@njit
def TSD_mean_calculator(signal2, segment_length, fs):
    """
        Calculate the TSD time evolution of the signal

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


def get_interval_length_c_val(dico_xyzs, t, name, fs_l):
    """

    Return the optimal length interval for a coordinates systems and timestemps coming from chaotic system

    Args:
        dico_xyzs (dict): dictionnary containing the time evolution of x,y,z coordinates of the system
        t (1D numpy array): Time values at which the system evolved (t axis)
        name (str): Name of the systme
        fs_l (int): Frequency with which the timestep were created.
    """
    for i in name:
        I1_c, I2_c, c = discrepancies_mean_curve(
            dico_xyzs[i], fs_l, 0.0001, 0.0005, t0=t[0]
        )
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax[0].set_title(i)
        ax[0].plot(c[:-1], I1_c)
        ax[0].set_xlabel("signal length [s]")
        ax[0].set_ylabel("I1 value")
        ax[0].grid()
        ax[1].set_title(i)
        ax[1].plot(c[:-1], I2_c)
        ax[1].set_xlabel("signal length [s]")
        ax[1].set_ylabel("I2 value")
        ax[1].grid()

    I1_cx, I2_cx, c = discrepancies_mean_curve(
        dico_xyzs[name[0]], fs_l, 0.0001, 0.0005, t0=int(t[0])
    )
    I1_cx = I1_cx[~np.isnan(I1_cx)]
    I2_cx = I2_cx[~np.isnan(I2_cx)]
    I1_cx = np.append(I1_cx, I1_cx[-1])
    I2_cx = np.append(I2_cx, I2_cx[-1])
    c1 = c[np.isclose(I1_cx, [0.01], atol=0.0001)]
    c2 = c[np.isclose(I2_cx, [0.005], atol=0.0001)]
    if np.isnan(np.mean(c1)).any():
        cs = np.mean(c2[~np.isnan(c2)])
    elif np.isnan(np.mean(c2)).any():
        cs = np.mean(c1[~np.isnan(c1)])
    else:
        cs = np.minimum(np.mean(c1[~np.isnan(c1)]), np.mean(c2[~np.isnan(c2)]))
    print(
        f"distances (c1,c2) : {np.mean(c1[~np.isnan(c1)]),np.mean(c2[~np.isnan(c2)])}"
    )
    print(f"smallest c* : {cs}")
    print(f"Highest length short time series :{(cs-t[0])*fs_l}")


def Interval_calculator_all(dico_signal, name_signal, fs):
    """
    Calculate the optimal interval for applying the TSD on all signals store in a dictionnary, wiht the method defined by Takumi Sase et al in
    "Estimating the level of dynamical noise in time series by using fractal dimensions"

    Args:
        dico_signal (dict): dictionnary containing the signals with their name (shape of dict : {signal_name : 1D numpy array signal})
        name_signal (1D numpy array str): Array containing the names of the signals
        fs (int): Sampling frequency of the signals

    Returns:
        dict : dictionnary containnig the optimal length for each signal (format : {signal_name : optimal_length_segment})
    """
    dic_segment_lead = {}
    for i in name_signal:
        dic_segment_lead[i] = Interval_calculator_lead(dico_signal[i], fs)
    return dic_segment_lead


def TSD_plot(dico_lead, name_lead, segment_length, fs, t):
    """

    Plot the TSD time evolution for a set of signals (with a predefined segment legnth)

    Args:
        dico_lead (dict): dictionnary containing the signals with their name (shape of dict : {signal_name : 1D numpy array signal})
        name_lead (1D numpy array str): Array containing the names of the signals
        segment_length (int): segment length
        fs (int): sampling frequency of the signals
        t (1D numpy array): time duration of the signal at the sampling frequency fs
    """
    D_lead = {}
    for i in name_lead:
        w = 1
        Ds = np.array([])
        sig = dico_lead[i]
        while (w * segment_length * fs) <= len(sig):
            sig_c = sig[
                int((w - 1) * segment_length * fs) : int((w) * segment_length * fs)
            ]
            L1 = lq_k(sig_c, 1, fs)
            L2 = lq_k(sig_c, 2, fs)
            Dv = (np.log(L1) - np.log(L2)) / (np.log(2))
            Ds = np.append(Ds, Dv)
            w += 1
        D_lead[i] = Ds

    w_length = [
        w * segment_length for w in range(0, int((len(t) / fs) * (1 / segment_length)))
    ]

    for i in name_lead:
        plt.plot(w_length, D_lead[i], label=i)
    plt.xlabel("Time interval")
    plt.ylabel("TSD value")
    plt.legend(loc="best", bbox_to_anchor=(1, 1))
    plt.grid()
    plt.show()


def TSDvsNoiseLevel(
    noise_level, path_to_data, fs, list_attractor=["lorenz", "rossler"]
):
    """

    Calculate the TSD value of dynamical chaotic system considered for different value of noise level (in dB).

    Args:
        noise_level (1D numpy array ): Array containing the SNR level value you want your signal to have
        path_to_data (str): Path to your data folder
        fs (int): Sampling frequency used for your signals
        list_attractor (list, optional): List containing the name (in str) of the signal you want to study.. Defaults to ["lorenz", "rossler"].

    Returns:
        Tuple : Tuple dicts of mean TSD value and their SD value, at each SNR level for each signal
    """
    Dmean = {name: np.array([]) for name in list_attractor}
    SD_D = {name: np.empty([2, len(noise_level)]) for name in list_attractor}
    for i, n in zip(noise_level, range(len(noise_level))):
        for name in list_attractor:
            mid_Dmean = np.array([])
            coord, _ = system_coordinates_reader(path_to_data, name, num_attractor=None)
            Obs = coord[:, 0].copy()
            noise_obs = add_observational_noise(Obs.copy(), i)
            Mean_TSD, _ = TSD_mean_calculator(noise_obs, 100, fs)
            mid_Dmean = np.append(mid_Dmean, Mean_TSD)

            Dmean[name] = np.append(Dmean[name], np.mean(mid_Dmean.copy()))
            SD_D[name][:, n] = np.array(
                [
                    np.abs(
                        np.mean(mid_Dmean.copy()) - np.percentile(mid_Dmean.copy(), 25)
                    ),
                    np.abs(
                        np.mean(mid_Dmean.copy()) - np.percentile(mid_Dmean.copy(), 75)
                    ),
                ]
            )
    return Dmean, SD_D


def TSDvsNoiseLevel_array(noise_level, dico_signal, name_lead, fs):
    """
    Calculate the TSD value of for ECG recordings, different value of noise level (in dB).


    Args:
        noise_level (1D numpy array ): Array containing the SNR level value you want your signal to have
        dico_signal (dict): dictionnary containing the signals with their name (shape of dict : {signal_name : 1D numpy array signal})
        name_lead (1D numpy array): Array contzaining the name of the lead
        fs (int): Sampling frequency

    Returns:
        Tuple: Tuple containing dicts of the TSD mean value and its SD for all lead of each patient , and 2 numpy arrays containing the average of TSD mean and SD value obtained
    """
    Dmean = {}
    SD_D = {}
    the_mean_lead_calculator = np.array([])
    the_SDmean_lead_calculator = np.array([])

    for name in name_lead:
        Dmean[name.decode("utf8")] = np.array([])
        SD_D[name.decode("utf8")] = np.array([])
    for i in noise_level:
        inter_Dmean = np.array([])
        inter_SD = np.array([])
        for name in name_lead:

            Obs = dico_signal[name]
            noise_obs = add_observational_noise(Obs.copy(), i)
            seg = Interval_calculator_lead(noise_obs, fs)
            Mean_TSD, SD_TSD = TSD_mean_calculator(noise_obs, seg, fs)
            inter_Dmean = np.append(inter_Dmean, Mean_TSD)
            Dmean[name.decode("utf8")] = np.append(Dmean[name.decode("utf8")], Mean_TSD)
            SD_D[name.decode("utf8")] = np.append(SD_D[name.decode("utf8")], SD_TSD)
        the_mean_lead_calculator = np.append(
            the_mean_lead_calculator, np.mean(inter_Dmean)
        )
        the_SDmean_lead_calculator = np.append(
            the_SDmean_lead_calculator, np.mean(inter_SD)
        )

    return Dmean, SD_D, the_mean_lead_calculator, the_SDmean_lead_calculator


def TSDvsNoiseLevel_100ECG(noise_level, theBIGdataset, name_lead, fs):
    """
    Calculate the TSD value of for a dataset of 100 ECG recordings, different value of noise level (in dB).

    Args:
        noise_level (1D numpy array ): Array containing the SNR level value you want your signal to have
        theBIGdataset (nD numpy array): Array containing your data in the format of dictionnaries (shape of dict : {signal_name : 1D numpy array signal})
        name_lead (1D numpy array): Array contzaining the name of the lead
        fs (int): Sampling frequency

    Returns:
        Tuple : Tuple of dictionnary containing TSD mean and SD value, at each SNR level for each signal
    """
    Big_Dmean = {}
    Big_SDmean = {}
    N = len(theBIGdataset)
    for name in name_lead:
        Big_Dmean[name] = np.array([])
        Big_SDmean[name] = np.empty([2, len(noise_level)])
        arr = np.vstack([theBIGdataset[j][name] for j in range(N)])
        for i, n in zip(noise_level, range(len(noise_level))):
            arr_noise = np.vstack(
                [add_observational_noise(arr[j, :].copy(), i) for j in range(N)]
            )
            inter_Dmean = np.array([])
            for b in range(arr_noise.shape[0]):
                sig = arr_noise[b, :].copy()
                # seg = TSD.Interval_calculator_lead(sig,fs)
                Mean_TSD, _ = TSD_mean_calculator(sig, 100, fs)
                inter_Dmean = np.append(inter_Dmean, Mean_TSD)
            m, p25, p75 = (
                np.mean(inter_Dmean.copy()),
                np.percentile(inter_Dmean.copy(), 25),
                np.percentile(inter_Dmean.copy(), 75),
            )
            Big_Dmean[name] = np.append(Big_Dmean[name], m)
            Big_SDmean[name][:, n] = np.array([np.abs(m - p25), np.abs(m - p75)])

    return Big_Dmean, Big_SDmean


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
    iter_lead=-1,
):
    """_summary_

    Args:
        Synth_data (nD numpy array): Array containing your TSD mean value from your synthetic ECG recording data in the format of dictionnaries (shape of dict : {signal_name : 1D numpy array signal})
        Acc_data (nD numpy array): Array containing your TSD mean value from your "Acceptable" labelled ECG recording data in the format of dictionnaries (shape of dict : {signal_name : 1D numpy array signal})_
        Unacc_data (nD numpy array): Array containing your TSD mean value from your "Unacceptable" ECG recording data in the format of dictionnaries (shape of dict : {signal_name : 1D numpy array signal})
        SD_synth (nD numpy array): Array containing your TSD SD value from your synthetic ECG recording data in the format of dictionnaries (shape of dict : {signal_name : 1D numpy array signal})
        SD_acc (nD numpy array): Array containing your TSD SD value from your "Acceptable" labelled ECG recording in the format of dictionnaries (shape of dict : {signal_name : 1D numpy array signal})
        SD_unacc (nD numpy array): Array containing your TSD SD value from your "Unacceptable" labelled ECG recording in the format of dictionnaries (shape of dict : {signal_name : 1D numpy array signal})
        S_level (1D numpy array ): Array containing the SNR level you used to collect your data
        name_lead (1D numpy array): Array contzaining the name of the lead
        name (str, optional): Index you used to create your measurement. Defaults to "TSD".
        iter_lead (int,optional) : Int variable to indicate how many lead you want to plot (can be positive or negative). Defaults to -1
    """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
    # plt.rcParams.update({'font.size':20})
    fig.tight_layout(h_pad=4)
    if iter_lead > 0:
        coordinates = [(0, y) for y in range(iter_lead)]
    else:
        coordinates = [(0, y) for y in range(len(name_lead) + iter_lead)]
    for i, c in zip(name_lead[:iter_lead], coordinates):

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


def TSDvsObsNoise_plot_100ECG(noise_level, dergrossdataset, name_lead, fs):
    """

    Plot mean TSD value evolution, in function of the SNR level, for your entire dataset (for our case : 100 12 lead ECG recording with the same labelling)

    Args:
        noise_level (1D numpy array ): Array containing the SNR level value you want your signal to have
        dergrossdataset (nD numpy array): Array containing your data in the format of dictionnaries (shape of dict : {signal_name : 1D numpy array signal})
        name_lead (1D numpy array): Array contzaining the name of the lead
        fs (int): Sampling frequency
    """
    BDM, BP = TSDvsNoiseLevel_100ECG(noise_level, dergrossdataset, name_lead, fs)
    plt.figure()
    plt.gca().set_prop_cycle(
        plt.cycler("color", plt.cm.jet(np.linspace(0, 1, len(name_lead))))
    )
    labels = []
    for i in name_lead:
        plt.errorbar(noise_level, BDM[i], BP[i])
        labels.append(i.decode("utf8"))
    plt.legend(
        labels,
        ncol=4,
        loc="best",
        columnspacing=1.0,
        labelspacing=0.0,
        handletextpad=0.0,
        handlelength=1.5,
        fancybox=True,
        shadow=True,
    )

    plt.xlabel("SNR (db)")
    plt.ylabel("mean TSD value")
    plt.title(f"TSD vs SNR (dB) for average TSD value for all lead, for 100 patients")
    plt.grid()
    plt.show()


def Random_phase_shuffling(signal):
    """

    Function that randomly shuffle the phase of your signal

    Args:
        signal (1D numpy array): Signal you want to shuffle

    Returns:
        1D numpy array : Phase shuffled signal
    """
    fft_signal = rfft(signal)
    phase_fs = np.arctan2(fft_signal[2::2], fft_signal[1:-1:2])
    mag = np.sqrt((fft_signal[1:-1:2]) ** 2 + (fft_signal[2::2]) ** 2)
    ##phase shuffler:
    rng_phase = phase_fs.copy()
    np.random.shuffle(rng_phase)
    fsrp = mag[:, np.newaxis] * np.c_[np.cos(rng_phase), np.sin(rng_phase)]
    fsrp = np.r_[fft_signal[0], fsrp.ravel(), fft_signal[-1]]
    tsr = irfft(fsrp)
    return tsr
