import scipy.signal
import numpy as np
from ecgdetectors import Detectors
import neurokit2 as nk


"""

This code is an adaption of the ECG ASSESS code found on the following link :https://github.com/LinusKra/ECGAssess
"""
detectors = Detectors(500)

# region set parameters
sampling_frequency = 500  # Hz
nyquist_frequency = sampling_frequency * 0.5  # Hz
max_loss_passband = 0.1  # dB
min_loss_stopband = 20  # dB
SNR_threshold = 0.5
signal_freq_band = [2, 40]  # from .. to .. in Hz
heart_rate_limits = [24, 300]  # from ... to ... in beats per minute
t = 10  # seconds
window_length = 100  # measurements
# endregion


def high_frequency_noise_filter(data):
    order, normal_cutoff = scipy.signal.buttord(
        20, 30, max_loss_passband, min_loss_stopband, fs=sampling_frequency
    )
    iir_b, iir_a = scipy.signal.butter(order, normal_cutoff, fs=sampling_frequency)
    filtered_data = scipy.signal.filtfilt(iir_b, iir_a, data)
    return filtered_data


def baseline_filter(data):
    order, normal_cutoff = scipy.signal.buttord(
        0.5, 8, max_loss_passband, min_loss_stopband, fs=sampling_frequency
    )
    iir_b, iir_a = scipy.signal.butter(order, normal_cutoff, fs=sampling_frequency)
    filtered_data = scipy.signal.filtfilt(iir_b, iir_a, data)
    return filtered_data


def stationary_signal_check(data):
    res = []
    for lead in range(1, data.shape[0] + 1):
        window_matrix = np.lib.stride_tricks.sliding_window_view(
            data[lead - 1, :], window_length
        )[::10]
        for window in window_matrix:
            if np.amax(window) == np.amin(window):
                res.append(1)
                break
        if len(res) != lead:
            res.append(0)
    return res


def heart_rate_check(data):
    res = []
    for lead in range(data.shape[0]):
        beats = detectors.pan_tompkins_detector(data[lead, :])
        if len(beats) > ((heart_rate_limits[1] * t) / 60) or len(beats) < (
            (heart_rate_limits[0] * t) / 60
        ):
            res.append(1)
        else:
            res.append(0)
    return res


def signal_to_noise_ratio_check(data):
    res = []
    for lead in range(data.shape[0]):
        _, pxx_den = scipy.signal.periodogram(
            data[lead, :], fs=sampling_frequency, scaling="spectrum"
        )
        if sum(pxx_den):
            signal_power = sum(
                pxx_den[(signal_freq_band[0] * 10) : (signal_freq_band[1] * 10)]
            )
            SNR = signal_power / (sum(pxx_den) - signal_power)
        else:
            res.append(0)
            continue
        if SNR > SNR_threshold:
            res.append(1)
        else:
            res.append(0)
    return res


def processing(ECG, temp_freq):
    resampled_ECG = []
    if temp_freq != 500:
        for n in range(0, ECG.shape[0]):
            resampled_ECG.append(
                nk.signal_resample(
                    ECG[n, :],
                    sampling_rate=int(temp_freq),
                    desired_sampling_rate=500,
                    method="numpy",
                )
            )
    else:
        resampled_ECG = ECG

    filt_ECG = np.zeros_like(ECG)
    for lead in range(ECG.shape[0]):
        x = high_frequency_noise_filter(ECG[lead, :]) - baseline_filter(ECG[lead, :])
        filt_ECG[lead, :] = x

    # SQM = []  # Signal Quality Matrix
    # SQM.append(stationary_signal_check(ECG, total_leads))
    # SQM.append(heart_rate_check(filt_ECG, total_leads))
    # SQM.append(signal_to_noise_ratio_check(ECG, total_leads))
    SQM = np.empty([3, 12])
    SQM[0, :] = stationary_signal_check(ECG)
    SQM[1, :] = heart_rate_check(filt_ECG)
    SQM[2, :] = signal_to_noise_ratio_check(ECG)
    # SQM = np.vstack([stationary_signal_check(ECG),heart_rate_check(ECG),signal_to_noise_ratio_check(ECG)])
    res = np.array([])
    for i in range(ECG.shape[0]):
        a = np.sum(SQM[:, i])
        if a < 3:
            res = np.append(res, 0)
        else:
            res = np.append(res, 1)

    return res
