import numpy as np
from scipy.signal import periodogram
import pywt


def sdr_score(signals, fs, **kwargs):
    ##SDR coeff:
    """
    Calculate the SDR score for signal quality assessment

    Args:
        signals (Numpy array): Numpy array containing all the signal (expected shape : [num_feature (ex : #lead),signal_length])
        fs (int): Sampling frequency of the signla (if multiple signals, all signal must have the same sampling frequency)

    Raises:
        ValueError: Raised when the score is not between 0.5 and 0.8

    Returns:
        SDR_arr (Numpy array): 1D numpy array containing the index for each lead (shape [num_feature])
    """
    SDR_arr = np.array([])
    for i in range(signals.shape[0]):
        f, PSD = periodogram(signals[i, :], fs)
        QRS_signal_PSD = np.sum(PSD[np.logical_and(f >= 5, f <= 14)])
        ECG_tot = np.sum(PSD[np.logical_and(f >= 5, f <= 50)], dtype=np.float64)
        if ECG_tot == 0:
            ECG_tot = np.sum(np.abs(signals[i, :]) ** 2)
            if ECG_tot == 0:
                ECG_tot = 2**63 - 1
        SDR_val = QRS_signal_PSD / ECG_tot
        if SDR_val > 0.8 or SDR_val < 0.5:
            raise ValueError(
                "Your SDR score is not in the range (normal range : 0.5<SDR<0.8; your SDR :{})".format(
                    SDR_val
                )
            )
        SDR_arr = np.append(SDR_arr, SDR_val)
    return SDR_arr


def wavelet_coef(sig, name, lev):
    """
    Calculate the wavelet coefficient of a signal with the given wavelet and type and decomposition level

    Args:
        sig (Numpy Array): 1D array representing the signal
        name (String): Name of the wavelet type to use (ex : db4,sym4,...)
        lev (int): Decomposition level

    Returns:
        Tuple : Tuple contianing the approximation coefficients and the detail coefficients array (in that order)
    """
    All_coeff = pywt.wavedec(sig, name, level=lev)

    CA = All_coeff[0]
    CD = All_coeff[1 : len(All_coeff)]
    return CA, CD


def energy_l2(coeff):
    """
    Calculate the L2 energy of the signal

    Args:
        coeff (Numpy Array): 1D Numpy array containing the coefficients

    Returns:
        float64 : L2 norm value
    """
    return np.sum(np.abs(coeff) ** 2, dtype=np.float64)


def wpmf_score(signals, **kwargs):
    """
    Calculate the wPMF score for signal quality assessment

    Args:
        signals (Numpy array): Numpy array containing all the signal (expected shape : [num_feature (ex : #lead),signal_length])

    Returns:
        1D numpy array : 1D numpy array containing the index for each lead (shape [num_feature])
    """
    waveletname = "db4"
    level_w = 9
    wPMF_arr = np.array([], dtype=np.float64)
    for i in range(signals.shape[0]):
        CA_w, CD_w = wavelet_coef(signals[i, :], waveletname, level_w)
        CD_w = np.array(CD_w, dtype=object)
        CA_w = np.array(CA_w, dtype=object)
        E = np.array([])
        for CD in range(level_w):
            E = np.append(E, energy_l2(CD_w[-(CD + 1)]))
        E = np.append(E, energy_l2(CA_w[0]))
        Etot = np.sum(E, dtype=np.float64)
        if Etot == 0:
            Etot = energy_l2(signals[i, :][: int((2**level_w) - 1)])
            if Etot == 0:
                Etot = 2**63 - 1
        p = E / Etot
        SQI_ECG = np.sum(p[3:6])
        wPMF_arr = np.append(wPMF_arr, SQI_ECG)
    return wPMF_arr


def snr_index(signals, fs, **kwargs):
    """
    Calculate the SNRECG score for signal quality assessment

    Args:
        signals (Numpy array): Numpy array containing all the signal (expected shape : [num_feature (ex : #lead),signal_length])
        fs (int): Sampling frequency of the signla (if multiple signals, all signal must have the same sampling frequency)

    Raises:
        ValueError: Raised if the Normalization didn't work as expected
        ValueError: Raise if for Normalization, a negative value is obtained

    Returns:
        1D numpy array : 1D numpy array containing the index for each lead (shape [num_feature])
    """
    SNR_arr = np.array([], dtype=np.float64)
    for i in range(signals.shape[0]):
        f, PSD = periodogram(signals[i, :], fs, scaling="spectrum")
        Sig_PSD_tot = np.sum(PSD)
        signal_power = np.sum(PSD[np.logical_and(f >= 2, f <= 40)])
        if kwargs.get("normalization") == True:
            if sum(PSD):
                SNR = signal_power / Sig_PSD_tot
                if SNR > 1:
                    print(SNR)
                    raise ValueError("Non Normalized SNR value!")
                elif SNR < 0:
                    raise ValueError("NEGATIVE VALUE! check : ", SNR)
                elif np.isnan(SNR):
                    SNR_arr = np.append(SNR_arr, 0)
                    continue

            else:
                SNR_arr = np.append(SNR_arr, 0)
                continue
        else:
            if sum(PSD):
                SNR = signal_power / (np.sum(PSD) - signal_power)
                if np.isnan(SNR):
                    SNR_arr = np.append(SNR_arr, 0)
                    continue
            else:
                SNR_arr = np.append(SNR_arr, 0)
                continue
        SNR_arr = np.append(SNR_arr, SNR)
    return SNR_arr
