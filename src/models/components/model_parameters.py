import os
import sys

FILEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FILEPATH))

from .custom_loss import focal_loss, focal_loss_error, lgb_f1_score

dict_param = {
    "focal_loss": {
        "metric": "focal_loss",
        "fobj": focal_loss,
        "feval": focal_loss_error,
    },
    "auc": {
        "metric": "auc",
        "objective": "binary",
    },
    "mcc": {
        "metric": "mcc",
        "objective": "binary",
    },
    "f1": {
        "metric": "f1",
        "objective": "binary",
        "feval": lgb_f1_score,
    },
}

direction_study_dict = {
    "focal_loss": "minimize",
    "auc": "maximize",
    "mcc": "maximize",
    "f1": "maximize",
}


def get_model_parameters(metric_name):

    param_fixed = {
        "metric": dict_param[metric_name].get("metric"),
        "objective": dict_param[metric_name].get("objective"),
        "verbosity": -1,
    }

    return param_fixed, dict_param[metric_name]
