import numpy as np
from scipy.stats import kurtosis, pearsonr
from ecgdetectors import Detectors


def get_time_axis(sign_length, fs):
    x = np.linspace(0, int(sign_length / fs), sign_length)
    return x


def kurto_score(arr_signal, **kwargs):
    result = np.array([])
    for i in range(arr_signal.shape[0]):
        K = kurtosis(arr_signal[i, :])
        if np.isnan(K):
            K = 0
        result = np.append(result, K)
    return result


def pqrst_template_extractor(ecg_signal, rpeaks):
    ##Adapted from the Biosspy function _extract_heartbeats
    R = np.sort(rpeaks)
    length = len(ecg_signal)
    templates = []
    newR = []

    for r in R:
        a = r - (np.median(np.diff(rpeaks, 1)) / 2)
        if a < 0:
            continue
        b = r + (np.median(np.diff(rpeaks, 1)) / 2)
        if b > length:
            break

        templates.append(ecg_signal[int(a) : int(b)])
        newR.append(r)

    templates = np.array(templates)
    newR = np.array(newR, dtype="int")

    return templates, newR


def morph_score(signals, fs, **kwargs):
    QRS_arr = np.array([])
    detect = Detectors(fs)
    for i in range(signals.shape[0]):
        r_peaks = detect.pan_tompkins_detector(signals[i])
        if len(r_peaks) <= 2:
            QRS_arr = np.append(QRS_arr, 0)
            continue
        else:
            template, _ = pqrst_template_extractor(signals[i, :], rpeaks=r_peaks)
            empty_index = np.array([], dtype=int)
            for ble in range(template.shape[0]):
                if template[ble].size == 0:
                    empty_index = np.append(empty_index, ble)
            template = np.delete(template, empty_index, 0)
            index_maxima = [
                np.argmax(template[w].copy()) for w in range(template.shape[0])
            ]
            median_index = np.median(index_maxima.copy())
            templates_good = template[
                np.isclose(index_maxima.copy(), median_index, rtol=0.1)
            ].copy()
            if templates_good.shape[0] == 0:
                QRS_arr = np.append(QRS_arr, 0)
                continue

            sig_mean = templates_good[0].copy()
            for j in range(1, templates_good.shape[0]):
                if sig_mean.size != templates_good[j].size:
                    templates_good[j] = templates_good[j][: len(sig_mean)]
                sig_mean = np.add(sig_mean, templates_good[j].copy())

            sig = sig_mean / templates_good.shape[0]
            r_p = np.array([])
            for w in range(templates_good.shape[0]):
                beats = templates_good[w, :].copy()
                r_p = np.append(r_p, np.abs(pearsonr(sig, beats))[0])
            r_v = np.mean(r_p.copy())
            if r_v < 0:
                r_v = 0
            QRS_arr = np.append(QRS_arr, r_v)
    return QRS_arr


def flatline_score(signals, fs, **norm):
    flat_arr = np.array([], dtype=np.float64)
    for i in range(signals.shape[0]):
        cond = np.where(np.diff(signals[i, :].copy()) != 0.0, np.nan, True)
        score = len(cond[cond == True]) / len(signals[i, :].copy())
        if norm.get("normalization") == True:
            flat_arr = np.append(flat_arr, 1 - score)
        else:
            flat_arr = np.append(flat_arr, score)

    return flat_arr


def corr_lead_score(signals, **kwargs):
    results = np.zeros(signals.shape[0])
    M = np.corrcoef(signals)
    if M.size == 1:
        results[0] = 1
        return results
    for j in range(signals.shape[0]):
        # Why abs val ==> We want the correlation between lead. Two leads can
        # be anti correlated because it measures the same event but from a different heart axis
        val = np.mean(np.abs(M[j, :]))

        if np.isnan(val):
            val = 0
        results[j] = val
    return results


def HR_index_calculator(signals, fs, **kwargs):
    mean_RR_interval = np.zeros(signals.shape[0])
    x = get_time_axis(signals.shape[1], fs)
    detect = Detectors(fs)
    for i in range(signals.shape[0]):
        r_peaks = detect.pan_tompkins_detector(signals[i, :])
        r_sec = x[r_peaks]
        r_msec = r_sec * 1000
        if len(r_msec) <= 1:
            mean_RR_interval[i] = 0
        else:
            RR_bpm_interval = (60 / (np.diff(r_msec))) * 1000
            if np.mean(RR_bpm_interval) < 24 or np.mean(RR_bpm_interval) > 425:
                mean_RR_interval[i] = 0
            else:
                # mean_RR_interval[i] = np.std(RR_bpm_interval)/np.mean(RR_bpm_interval)
                mean_RR_interval[i] = 1
    return mean_RR_interval
