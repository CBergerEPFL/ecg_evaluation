import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import rfft, irfft
from numba import njit


def plot_attractor_timevscoord(dico_xyzs, t, name):
    for i in name:
        plt.plot(t, dico_xyzs[i], label=i)
        plt.xlabel("time")
        plt.ylabel("function value")
        plt.legend(loc="best", bbox_to_anchor=(1, 1))
        plt.grid()
        plt.plot()


def Plot_attractors(xyzs_attractor, name_attractor):
    ax = plt.figure().add_subplot(projection="3d")

    ax.plot(*xyzs_attractor.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(f"{name_attractor} Attractor")

    plt.show()


def system_coordinates_reader(Path_to_data, Attractor_name, num_attractor=0):
    path = Path_to_data + f"/{Attractor_name}_attractors"
    df = pd.read_csv(path + ".csv")
    df_n = df.to_numpy()
    xyzs = df_n[:, 1:4]
    t = df_n[:, 0]
    return xyzs, t


def add_observational_noise(sig, SNR):
    Power_sig = (1 / len(sig)) * np.sum(np.abs(sig) ** 2, dtype=np.float64)
    P_db = 10 * np.log10(Power_sig)
    noisedb = P_db - SNR
    sd_db_watts = 10 ** (noisedb / 10)
    # sd_noise = np.sqrt(Power_sig/(SNR))
    noise = np.random.normal(0, np.sqrt(sd_db_watts), len(sig))
    sig_noisy = sig + noise
    return sig_noisy


def add_observational_noise_val(sig, SNR):
    Power_sig = np.mean(np.abs(sig) ** 2)
    P_db = 10 * np.log10(Power_sig)
    noisedb = P_db - SNR
    sd_db_watts = 10 ** (noisedb / 10)
    # sd_noise = np.sqrt(Power_sig/(SNR))
    noise = np.random.normal(0, np.sqrt(sd_db_watts), (len(sig), 3))
    # sig_noisy = sig+noise
    return noise


@njit
def Lm_q(signal1, m, k, fs):
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
def Lq_k(signal, k, fs):
    # calc_L_series = np.frompyfunc(lambda m: Lm_q(signal, m, k, fs), 1, 1)
    calc_L_series = np.zeros(k)
    for m in range(1, k + 1):
        calc_L_series[m - 1] = Lm_q(signal, m, k, fs)
    L_average = np.mean(calc_L_series)
    return L_average


@njit
def TSD_mean_calculator(signal2, segment_length, fs):
    Ds = np.zeros(int(len(signal2) - segment_length) - 1)
    for w in range(1, int(len(signal2) - segment_length)):
        sig_true = signal2[int((w - 1)) : int((w) + segment_length)]
        L1 = Lq_k(sig_true, 1, fs)
        L2 = Lq_k(sig_true, 2, fs)
        Ds[w - 1] = (np.log(L1) - np.log(L2)) / (np.log(2))
        ##Thresholding necessary since we use an approximation of the Higuchi method
        if Ds[w - 1] > 2 or np.isnan(Ds[w - 1]):
            Ds[w - 1] = 2
        elif Ds[w - 1] < 1:
            Ds[w - 1] = 1
    return np.mean(Ds[~np.isnan(Ds)]), np.std(Ds[~np.isnan(Ds)])


@njit
def taux_Mean_fast(signal, taux, hprime, h=0):
    return np.mean((signal[int(h + taux) : int(taux + hprime + h)]))


@njit
def taux_var_fast(signal, taux, hprime, h=0):
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


def get_interval_length_c_val(dico_xyzs, t, name, fs_l):
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
    I1c_r = I1_c[np.logical_and(I1_c > 0, I1_c <= 1)]
    I2c_r = I1_c[np.logical_and(I2_c > 0, I2_c <= 1)]
    c1 = c[np.isclose(I1_cx, [0.01], atol=0.0001)]
    c2 = c[np.isclose(I2_cx, [0.0005], atol=0.0001)]
    cs = np.minimum(np.mean(c1), np.mean(c2))
    print(f"distances (c1,c2) : {np.mean(c1),np.mean(c2)}")
    print(f"smallest c* : {np.minimum(np.mean(c1),np.mean(c2))}")
    print(f"Highest length short time series :{(cs-t[0])*fs_l}")


def Interval_calculator_all(dico_signal, name_signal, fs):
    dic_segment_lead = {}
    for i in name_signal:
        dic_segment_lead[i] = Interval_calculator_lead(dico_signal[i], fs)
    return dic_segment_lead


def is_segment_flatline(sig):
    cond = np.where(np.diff(sig.copy(), 1) != 0.0, False, True)
    if len(cond[cond == True]) < 0.50 * len(sig):
        return False
    return True


def TSD_plot(dico_lead, name_lead, segment_length, fs, t):

    D_lead = {}
    for i in name_lead:
        w = 1
        Ds = np.array([])
        sig = dico_lead[i]
        while (w * segment_length * fs) <= len(sig):
            sig_c = sig[
                int((w - 1) * segment_length * fs) : int((w) * segment_length * fs)
            ]
            L1 = Lq_k(sig_c, 1, fs)
            L2 = Lq_k(sig_c, 2, fs)
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
    Dmean = {name: np.array([]) for name in list_attractor}
    SD_D = {name: np.empty([2, len(noise_level)]) for name in list_attractor}
    for i, n in zip(noise_level, range(len(noise_level))):
        for name in list_attractor:
            mid_Dmean = np.array([])
            coord, _ = system_coordinates_reader(path_to_data, name)
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


def TSDvsObsNoise_plot_100ECG(noise_level, dergrossdataset, name_lead, fs):
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
