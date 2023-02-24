import numpy as np
from numba import njit


@njit
def genhurst(S, q):
    """_summary_

    Args:
        S (1D Numpy array): Signal
        q (int): Decree of the generalized form of the Hurst Exponent

    Returns:
        Float : Hurst exponent
    """
    ##adapted from : https://github.com/PTRRupprecht/GenHurst/blob/master/genhurst.py
    L = len(S)
    # if L < 100:
    #     warnings.warn("Data series very short!")

    H = np.zeros((len(range(5, 20)), 1), dtype=np.float64)
    k = 0

    for Tmax in range(5, 20):

        x = np.arange(1, Tmax + 1, 1)
        mcord = np.zeros((Tmax, 1), dtype=np.float64)

        for tt in range(1, Tmax + 1):
            dV = S[np.arange(tt, L, tt)] - S[np.arange(tt, L, tt) - tt]
            VV = S[np.arange(tt, L + tt, tt) - tt]
            N = len(dV) + 1
            X = np.arange(1, N + 1)
            Y = VV
            mx = np.sum(X, dtype=np.float64) / N
            SSxx = np.sum(X**2, dtype=np.float64) - N * mx**2
            my = np.sum(Y, dtype=np.float64) / N
            SSxy = np.sum(np.multiply(X, Y), dtype=np.float64) - N * mx * my
            cc1 = SSxy / SSxx
            cc2 = my - cc1 * mx
            ddVd = dV - cc1
            VVVd = VV - np.multiply(cc1, np.arange(1, N + 1)) - cc2
            mcord[tt - 1] = np.mean(np.abs(ddVd) ** q) / np.mean(np.abs(VVVd) ** q)

        mx = np.mean(np.log10(x))
        SSxx = np.sum(np.log10(x) ** 2) - Tmax * mx**2
        my = np.mean(np.log10(mcord))
        SSxy = (
            np.sum(np.multiply(np.log10(x), np.transpose(np.log10(mcord))))
            - Tmax * mx * my
        )
        H[k] = SSxy / SSxx
        k = k + 1

    mH = np.mean(H) / q

    return mH


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


def HurstD_index(signals, fs):

    """
    Calculate the Hurst exponent for signal quality assessment

    Args:
        signals (Numpy array): Numpy array containing all the signal (expected shape : [num_feature (ex : #lead),signal_length])
        fs (int): Sampling frequency of the signla (if multiple signals, all signal must have the same sampling frequency)

    Returns:
        _type_: _description_
    """
    H_array = np.array([])
    for i in range(signals.shape[0]):
        if is_segment_flatline(signals[i, :]):
            H_array = np.append(H_array, 2)
        else:
            H = genhurst(signals[i, :], 1)
            H_array = np.append(H_array, 2 - H)
    return H_array
