import os
import pickle as pkl
import sys
import warnings

import numpy as np
import optuna
import optuna.integration.lightgbm as lgb
import pandas as pd
import xarray as xr

FILEPATH = os.path.dirname(os.path.realpath(__file__))
warnings.simplefilter(action="ignore", category=FutureWarning)
sys.path.append(os.path.join(FILEPATH, ".."))
from shared_utils.utils_data import extract_index_label
from shared_utils.utils_path import results_path
from models.components.model_parameters import (
    get_model_parameters,
    direction_study_dict,
)

# metric_name = "binary_logloss"
# metric_name = 'auc'
# metric_name = "mcc"
metric_name = "focal_loss"

direction_study = direction_study_dict[metric_name]

param_fixed, dict_param = get_model_parameters(metric_name)


def objective(trial, X, y):

    d_train = lgb.Dataset(X, label=y)
    param_grid = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 50, 2990, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "max_bin": trial.suggest_int("max_bin", 200, 300),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.05
        ),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.2, 0.95, step=0.05
        ),
    }

    param = param_fixed | param_grid

    lcv = lgb.cv(
        param,
        d_train,
        callbacks=[
            lgb.early_stopping(50),
            # lgb.log_evaluation(0)
        ],
        # num_boost_round =300,
        fobj=dict_param.get("fobj"),
        feval=dict_param.get("feval"),
    )
    trial.set_user_attr("num_iterations", len(lcv[f"{metric_name}-mean"]))
    return lcv[f"{metric_name}-mean"][-1]


def main():
    list_features = [
        "Corr_interlead",
        "Corr_intralead",
        "wPMF",
        "SNRECG",
        "HR",
        "Flatline",
        "TSD",
    ]
    input_data_path = os.path.join(results_path, "quality_metrics.nc")
    ds_metrics = xr.load_dataset(input_data_path)
    df_X_mean, df_y = extract_index_label(
        ds_metrics, list_features, aggregation_method="mean"
    )
    df_X_min, _ = extract_index_label(
        ds_metrics, list_features, aggregation_method="min"
    )
    X = np.concatenate((df_X_mean.values, df_X_min.values), axis=-1)
    metric_name_merged = [f"{x}_mean" for x in df_X_mean.columns] + [
        f"{x}_min" for x in df_X_mean.columns
    ]

    df_X = pd.DataFrame(X, columns=metric_name_merged)
    df_y = pd.DataFrame(df_y.values, columns=["y"])
    study = optuna.create_study(direction=direction_study, study_name="LGBM Classifier")
    func = lambda trial: objective(trial, df_X, df_y)
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(func, n_trials=200)

    print(f"\tBest value focal_loss_error: {study.best_value:.5f}")
    print(f"\tBest params:")

    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")

    dict_hp = (
        study.best_params
        | param_fixed
        | {"num_iterations": study.best_trial.user_attrs["num_iterations"]}
        | {"metric_name": metric_name}
    )

    with open(os.path.join(results_path, "hp_lgbm.pkl"), "wb") as f:
        pkl.dump(dict_hp, f)


if __name__ == "__main__":
    main()
