import numpy as np
from scipy.linalg import eigvals
from numba import njit

## Adaptated code from the code found : https://physionet.org/content/challenge-2011/1.0.0/


@njit
def missing_signal_counts(all_lead, j, upperbnd, lowerbnd, N_signal):
    """
    Clacualte missing singal counts score index used for the MoRE matrix

    Args:
        all_lead (2D Numpy array): Numpy array containing all the signal (expected shape : [num_feature (ex : #lead),signal_length])
        j (int): Current index of the loop (i.e. current lead compared)
        upperbnd (float): Upper bound value
        lowerbnd (float): Lower bound value
        N_signal (int): Signal length

    Returns:
        float : Missing count values
    """
    missingsignalcounts = 0
    for i in range(N_signal):
        if all_lead[j, i] > upperbnd or all_lead[j, i] < lowerbnd:
            missingsignalcounts = missingsignalcounts + 1
    missTest = missingsignalcounts / N_signal
    return missTest


@njit
def Fill_Matrix(Mat, j, test_smth, weigth):
    """
    Updated the given matrix for all indexes except a given one

    Args:
        Mat (2D Numpy array): More Matrix (shape  : n_lead*n_lead)
        j (int): Lead index to avoid
        test_smth (float): Score used for updated
        weigth (flaot): Weigth for the score

    Returns:
        2D Numpy array : Updated MoRE matrix
    """
    for k in range(Mat.shape[1]):
        if k != j:
            Mat[j, k] = Mat[j, k] + test_smth * weigth
            Mat[k, j] = Mat[k, j] + test_smth * weigth

    Mat[j, j] = Mat[j, j] + test_smth
    return Mat


@njit
def Flatsegment_detection(all_lead, j, segNum, segLen, upperbnd, lowerbnd):
    """
    Calcualte the Flatline score as implemented by the MoRE index

    Args:
        all_lead (2D Numpy array): Numpy array containing all the signal (expected shape : [num_feature (ex : #lead),signal_length])
        j (int): Current index of the loop (i.e. current lead compared)
        segNum (int): Number of segment
        segLen (int): Segment length
        upperbnd (float): Upper bound value
        lowerbnd (float): Lower bound value

    Returns:
        Float64 : Flatline segment score value
    """
    flatTest = 0.0
    z = np.zeros(segLen)
    for s in range(segNum):
        for e in range(segLen):
            z[e] = all_lead[j, s * segLen + e]
        minz = z[0]
        maxz = z[0]
        for f in range(segLen):
            if minz > z[f]:
                minz = z[f]
            if maxz < z[f]:
                maxz = z[f]
        rangez = int(maxz - minz)
        if (rangez <= 10) and (z[0] > lowerbnd) and (z[0] < upperbnd):
            flatTest = flatTest + 1

    flatTest = flatTest / segNum
    return flatTest


@njit
def Large_derivative(all_lead, j, N_signal, LDdv_th):
    """

    Calculate index score for large deerivative as used in the MoRE index

    Args:
        all_lead (2D Numpy array): Numpy array containing all the signal (expected shape : [num_feature (ex : #lead),signal_length])
        j (int): Current index of the loop (i.e. current lead compared)
        N_signal (int): Number of lead in the ECG
        LDdv_th (Float): threshold derivative

    Returns:
        Float64 : Large derivative score value
    """
    dy = np.zeros(N_signal - 1)
    for k in range(N_signal - 1):
        dy[k] = all_lead[j, k + 1] - all_lead[j, k]
    ldTestCount = 0
    for kk in range(N_signal - 1):
        if np.abs(dy[kk]) > LDdv_th:
            ldTestCount = ldTestCount + 1

    ldTest = ldTestCount / (N_signal - 1)
    return ldTest


def Matrix_Regularity(
    all_lead,
    N_channels,
    N_signal,
    segLen,
    segNum,
    upperbnd,
    lowerbnd,
    LDdv_th,
    missWeigth,
    flatweigth,
    ldWeigth,
):
    """_summary_

    Args:
        all_lead (2D Numpy array): Numpy array containing all the signal (expected shape : [num_feature (ex : #lead),signal_length])
        N_channels (int): Number of lead
        N_signal (int): Signal length
        segLen (int): Segment length
        segNum (int): Number of segment
        upperbnd (Float): Upper bound value
        lowerbnd (Float): Lower vound value
        LDdv_th (Float): Threshold large derivative value
        missWeigth (Float): Missing signal count wiegth
        flatweigth (Float): Flatline segment count length
        ldWeigth (Float): Large derivative weigth

    Returns:
        2D Numpy array : MoRE Matrix
    """
    Mat = np.zeros([N_channels, N_channels])
    ##In all channels:
    for j in range(N_channels):
        ##First Test : Missing signals
        missTest = missing_signal_counts(all_lead, j, upperbnd, lowerbnd, N_signal)
        Mat = Fill_Matrix(Mat, j, missTest, missWeigth)
        if missTest == 1:
            continue

        ##Flatsegment_detection:

        flatTest = Flatsegment_detection(
            all_lead, j, segNum, segLen, upperbnd, lowerbnd
        )
        Mat = Fill_Matrix(Mat, j, flatTest, flatweigth)
        if flatTest == 1:
            continue

        ##Large derivative :
        ldTest = Large_derivative(all_lead, j, N_signal, LDdv_th)
        Mat = Fill_Matrix(Mat, j, ldTest, ldWeigth)

        ##Lastly : Set to 1 if bigger
    for i in range(N_channels):
        for ii in range(N_channels):
            Mat[i, ii] = np.minimum(Mat[i, ii], 1)
    return Mat


def MoRE_score(signals, fs):
    """

    Calculate the MoRE signal quality index

    Args:
        signals (2D Numpy array): Numpy array containing all the signal (expected shape : [num_feature (ex : #lead),signal_length])
        fs (int): Sampling Frequency

    Returns:
        Float : MoRE score value
    """
    N_channels = signals.shape[0]
    N_signal = signals.shape[1]
    all_lead = np.empty([N_channels, N_signal])
    for i in range(N_channels):
        all_lead[i, :] = signals[i, :].copy() + -400 * i
    LDdv_th = 35
    missWeigth = 1 / 11
    flatweigth = 1 / 11
    ldWeigth = 1 / 11
    segLen = 500
    segNum = N_signal / segLen
    upperbnd = 200
    lowerbnd = -4600
    Mat = Matrix_Regularity(
        all_lead,
        N_channels,
        N_signal,
        segLen,
        segNum,
        upperbnd,
        lowerbnd,
        LDdv_th,
        missWeigth,
        flatweigth,
        ldWeigth,
    )

    results = np.array([])
    for i in range(Mat.shape[0]):
        results = np.append(results, np.sum(Mat[:, i].copy()))
    eigen = eigvals(np.matrix(Mat))
    SR = np.max(np.abs(eigen))

    # numgrade = int( np.round( np.minimum( 9 / (0.21 * SR) - 9 , 10 ) ))
    return SR
