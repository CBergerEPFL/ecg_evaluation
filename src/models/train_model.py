import os
import pickle as pkl
import sys

import lightgbm as lgbm
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

FILEPATH = os.path.dirname(os.path.realpath(__file__))
ROOTPATH = os.path.dirname(FILEPATH)
sys.path.append(os.path.join(FILEPATH))
sys.path.append(os.path.join(ROOTPATH))

from shared_utils.utils_data import extract_index_label, feature_checker
from shared_utils.utils_evaluation import metrics_cv
from shared_utils.utils_path import results_path
from shared_utils.utils_type import Results_Data

from .components.custom_logit import Logit_binary
from .components.custom_loss import sigmoid
from .components.features_selection import (
    backward_model_selection,
    JMI_score,
    model_selection_L2reg,
    hjmi_selection,
)
from models.components.model_parameters import get_model_parameters

seed = 0


def train_model(
    input_data_path,
    path_results="",
    model_type="lgbm",
    list_features=None,
    feature_selection=None,
    nb_fold=5,
    save_name=None,
):
    """Train a model on the input data and evaluate it

    Args:
        input_data_path (str): Path to the input metric computed from the ECG data
        list_features (list): List of features to be evaluated
    """
    kwargs = {}
    ds_metrics = xr.load_dataset(input_data_path)
    df_X_mean, df_y = extract_index_label(
        ds_metrics, list_features, aggregation_method="mean"
    )
    feature_checker(df_X_mean)
    if (list_features is None) and feature_selection is None:
        raise ValueError("One of list_features or feature_selection must be provided")

    if feature_selection is not None:
        if feature_selection == "backward_selection":
            list_features = backward_model_selection(df_X_mean, df_y)
        elif feature_selection == "JMI_score":
            list_features = JMI_score(df_X_mean, df_y)
        elif feature_selection == "L2_regularization":
            list_features = model_selection_L2reg(df_X_mean, df_y)
        elif feature_selection == "HJMI":
            list_features = hjmi_selection(df_X_mean, df_y)
        else:
            raise ValueError("Feature selection method not recognized")

    if model_type == "logistic":
        if "HR" in list_features and len(list_features) > 1:
            Hindex = list(df_X_mean[list_features].columns.values).index("HR")
        else:
            Hindex = None
        X = df_X_mean[list_features].values
        y = df_y.values.ravel()
        kwargs["Hindex"] = Hindex

    elif model_type == "lgbm":
        path_file = os.path.join(results_path, "hp_lgbm_best.pkl")
        with open(path_file, "rb") as f:
            param_hp = pkl.load(f)

        metric_name = param_hp.pop("metric_name")
        param_fixed, dict_param = get_model_parameters(metric_name)
        kwargs["dict_param"] = dict_param
        kwargs["hp"] = param_fixed | param_hp

        df_X_min, _ = extract_index_label(
            ds_metrics, list_features, aggregation_method="min"
        )
        X = np.concatenate((df_X_mean.values, df_X_min.values), axis=-1)
        metric_name_merged = [f"{x}_mean" for x in df_X_mean.columns] + [
            f"{x}_min" for x in df_X_mean.columns
        ]
        X = pd.DataFrame(X, columns=metric_name_merged)
        y = pd.DataFrame(df_y.values, columns=["y"])

    perform_cv_evaluation(X, y, model_type, nb_fold, save_name, **kwargs)

    return True


def pick_model(model_type, **kwargs):
    if model_type == "extra_tree_classifier":
        model = ExtraTreesClassifier(random_state=seed)
    elif model_type == "random_tree_classifier":
        model = RandomForestClassifier(random_state=seed)
    elif model_type == "logistic":
        if kwargs["Hindex"] is not None:
            model_type = model_type + " Binary"
            model = Logit_binary(random_state=seed, HR_index=kwargs["Hindex"])
        else:
            model = LogisticRegression(random_state=seed)
    elif model_type == "lgbm":
        model = lgbm.LGBMClassifier(random_state=seed, **kwargs)

    else:
        raise ValueError("Model type not recognized")

    return model


def perform_cv_evaluation(X, y, model_type, nb_fold, save_name, **kwargs):
    cv = StratifiedKFold(n_splits=nb_fold, random_state=seed, shuffle=True)
    data_results = Results_Data(model_type)
    if kwargs.get("hp"):
        kwargs["hp"].pop("num_iterations")
    for i, (train, test) in enumerate(cv.split(X, y)):
        print(f"Fold {i}")
        model = pick_model(model_type, **kwargs)
        if model_type == "lgbm":
            d_train = lgbm.Dataset(X.iloc[train, :], label=y.iloc[train, :])
            gbm = lgbm.train(
                kwargs["hp"],
                d_train,
                fobj=kwargs["dict_param"].get("fobj"),
                feval=kwargs["dict_param"].get("feval"),
            )

            pred = gbm.predict(X.iloc[test, :])
            pred = sigmoid(pred)
            label = y.iloc[test, :].values.ravel()
        else:
            model.fit(X[train], y[train])
            pred = model.predict_proba(X[test])[:, 1]
            label = y[test]

        data_results.append_results(pred, label)

    if save_name is not None:
        data_results.dump_to_file(save_name)
        # pkl.dump(model,open(save_name+".sav","wb"))

    metrics_cv(
        data_results.dict_results,
        save_name,
        t_used=None,
    )

    return True
