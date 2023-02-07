import numpy as np
from scipy.misc import derivative
from sklearn.metrics import f1_score, matthews_corrcoef


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
    """
    Focal Loss for lightgbm

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    """
    a, g = alpha, gamma
    y_true = dtrain.label

    def fl(x, t):
        p = 1 / (1 + np.exp(-x))
        return (
            -(a * t + (1 - a) * (1 - t))
            * ((1 - (t * p + (1 - t) * (1 - p))) ** g)
            * (t * np.log(p) + (1 - t) * np.log(1 - p))
        )

    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess


def lgb_f1_score(preds, dtrain):
    """
    When using custom losses the row prediction needs to passed through a
    sigmoid to represent a probability

    Parameters:
    -----------
    preds: numpy.ndarray
        array with the predictions
    lgbDataset: lightgbm.Dataset
    """
    preds = sigmoid(preds)
    binary_preds = [int(p > 0.5) for p in preds]
    y_true = dtrain.label
    return "f1", f1_score(y_true, binary_preds), True


def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
    a, g = alpha, gamma
    y_true = dtrain.label
    p = 1 / (1 + np.exp(-y_pred))
    loss = (
        -(a * y_true + (1 - a) * (1 - y_true))
        * ((1 - (y_true * p + (1 - y_true) * (1 - p))) ** g)
        * (y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    )
    # (eval_name, eval_result, is_higher_better)
    return "focal_loss", np.mean(loss), False


def mcc_error(y_pred, dtrain):
    y_true = dtrain.label
    y_pred = sigmoid(y_pred)
    binary_preds = [int(p > 0.5) for p in y_pred]
    return "mcc", matthews_corrcoef(y_true, binary_preds), True


focal_loss = lambda x, y: focal_loss_lgb(x, y, alpha=0.25, gamma=2)
focal_loss_error = lambda y_pred, dtrain: focal_loss_lgb_eval_error(
    y_pred, dtrain, alpha=0.25, gamma=2
)
