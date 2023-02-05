import os
import sys
import xarray as xr
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
    StratifiedKFold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

FILEPATH = os.path.dirname(os.path.realpath(__file__))
ROOTPATH = os.path.dirname(FILEPATH)
sys.path.append(os.path.join(FILEPATH))
sys.path.append(os.path.join(ROOTPATH))

from shared_utils.utils_data import feature_checker, extract_index_label
from .components.backward_model_selection import backward_model_selection
from .components.custom_logit import Logit_binary
from shared_utils.utils_type import Results_Data
from shared_utils.utils_evaluation import metrics_cv

seed = 0


def train_model(
    input_data_path,
    model_type="logistic",
    list_features=None,
    nb_fold=5,
    save_name=None,
):
    """Train a model on the input data and evaluate it

    Args:
        input_data_path (str): Path to the input metric computed from the ECG data
        list_features (list): List of features to be evaluated
    """
    ds_metrics = xr.load_dataset(input_data_path)
    df_X, df_y = extract_index_label(ds_metrics, list_features)
    feature_checker(df_X)

    y = df_y.values

    if list_features is None:
        print("Using Backward model selection")
        list_features = backward_model_selection(df_X, df_y)
        if "HR" in list_features and len(list_features) > 1:
            Hindex = list(df_X[list_features].columns.values).index("HR")

        else:
            Hindex = None
        X = df_X[list_features].values
    else:
        if "HR" in list_features and len(list_features) > 1:
            Hindex = list(df_X[list_features].columns.values).index("HR")
        else:
            Hindex = None
        X = df_X[list_features].values

    cv = StratifiedKFold(n_splits=nb_fold, random_state=seed, shuffle=True)
    list_pred = []
    list_label = []
    for i, (train, test) in enumerate(cv.split(X, y.ravel())):
        print(f"Fold {i}")
        model = pick_model(model_type, Hindex=Hindex)
        model.fit(X[train], y[train].ravel())
        list_pred.append(model.predict_proba(X[test])[:, 0])
        list_label.append(y[test].ravel())

    results = Results_Data(list_pred, list_label)
    if save_name is not None:
        results.dump_to_file(save_name)
    breakpoint()
    metrics_cv(
        results,
        save_name,
        t_used=None,
    )
    return True


def pick_model(model_type, Hindex=None):
    if model_type == "extra_tree_classifier":
        model = ExtraTreesClassifier(random_state=seed)
    elif model_type == "random_tree_classifier":
        model = RandomForestClassifier(random_state=seed)
    elif model_type == "logistic":
        if Hindex is not None:
            model_type = model_type + " Binary"
            model = Logit_binary(HR_index=Hindex, random_state=seed)
        else:
            model = LogisticRegression(random_state=seed)
    else:
        raise ValueError("Model type not recognized")

    return model
