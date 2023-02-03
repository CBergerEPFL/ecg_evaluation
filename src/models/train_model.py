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

seed = 0


def train_model(input_data_path, model_type="logistic", list_features=None, nb_fold=5):
    """Evaluate a list of indices

    Args:
        input_data_path (str): Path to the input data with ECG to be analysed
        list_features (list): List of features to be evaluated
    """
    ds_metrics = xr.load_dataset(input_data_path)
    df_X, df_y = extract_index_label(ds_metrics, list_features)
    feature_checker(df_X)

    y = df_y.values

    if list_features is None:
        print("Using Backward model selection")
        cols = backward_model_selection(df_X, df_y)
        if "HR" in cols and len(cols) > 1:
            Hindex = list(df_X[cols].columns.values).index("HR")

        else:
            Hindex = None
        X = df_X[cols].values
    else:
        if "HR" in cols and len(cols) > 1:
            Hindex = list(df_X[cols].columns.values).index("HR")
        else:
            Hindex = None
        X = df_X[cols].values

    cv = StratifiedKFold(n_splits=nb_fold, random_state=seed, shuffle=True)
    for i, (train, test) in enumerate(cv.split(X, y.ravel())):
        print(f"Fold {i}")
        model = pick_model(model_type, Hindex=Hindex)
        model.fit(X[train], y[train].ravel())
        y_pred = model.predict_proba(X[test])
        y_ref = y[test].ravel()
        Hugues_data = np.c_[proba_model[:, 0], proba_model[:, 1], y_pred, y_ref]
        df_m = pd.DataFrame(
            Hugues_data,
            columns=[
                "proba_label_0",
                "proba_label_1",
                "predicted_test_label",
                "ref_test_label",
            ],
        )
        df_m.to_csv(os.path.join(path_model_fold_cv, f"Fold_{i}.csv"))

    return X, y, model, Hindex


def save_model_index(X_data, y_data, save_path, cols, **kwargs):
    """
    Function that save a sklearn model using a a dataset and labels, given a specific set of features you want to create your model around
    Inputs :
        X_data [2D Pandas Dataframe] : Dataframe containing your features [shape : [number_of_patients*number_of_features]. The columns must have the name of your feature]
        y_data [1D Pandas Dataframe] : Dataframe containing your labels for each patient
        cols [String list] : Columns name you want to use. The index of each feature must be the same than X_data columns

    """
    if len(save_path) == 0:
        raise AttributeError("you didn't give a path where to save!")

    ##Create Folder where to stock our data
    model_folder = os.path.join(save_path, "Models")
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    index_folder = os.path.join(save_path, "Indexes")
    if not os.path.exists(index_folder):
        os.mkdir(index_folder)

    ##Now let the fun begin : We do cv on all indexes and store their results into a pandas then csv
    cv = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
    y = y_data.values
    if len(os.listdir(index_folder)) == 0:
        X, reference_col = feature_checker(X_data.values, list(X_data.columns.values))
        for j, c in enumerate(reference_col):
            X_s = X[:, j]

            path_folder_index = os.path.join(index_folder, c)

            if not os.path.exists(path_folder_index):
                os.mkdir(path_folder_index)

            for i, (_, test) in enumerate(cv.split(X, y.ravel())):
                index_val_test = X_s[test]
                lab_test = y[test].ravel()
                der_data = np.c_[1 - index_val_test, index_val_test, lab_test]
                df = pd.DataFrame(
                    der_data,
                    columns=["proba_label_0", "proba_label_1", "ref_test_label"],
                )
                df.to_csv(os.path.join(path_folder_index, f"Test_Fold_{i}.csv"))
    else:
        print(
            "Indexes already present! if you need to add another index, please use the following function : Index_saver"
        )

    if len(cols) != 0:
        ##See if user proposer a model (already trained or not)
        if kwargs.get("model"):
            model = kwargs["model"]
            X, y, _, _ = feature_selector(X_data, y_data, cols, already_model=True)
        else:
            X, y, model, _ = feature_selector(X_data, y_data, cols)

        if kwargs.get("Model_name"):
            name_model = kwargs["Model_name"]
        else:
            print("Auto Generation name model")
            name_model = "Model_"
            for c in cols:
                name_model += c + "_"
            print(f"This will be the name of your model : {name_model}")

        path_folder_model = os.path.join(model_folder, name_model)

        if not os.path.exists(path_folder_model):
            os.mkdir(path_folder_model)

        path_model_fold_cv = os.path.join(path_folder_model, "Fold_CV")
        if not os.path.exists(path_model_fold_cv):
            os.mkdir(path_model_fold_cv)
        X_train, _, y_train, _ = train_test_split(X, y.ravel())

        if not kwargs.get("model"):
            model.fit(X_train, y_train)

        for i, (_, test) in enumerate(cv.split(X, y.ravel())):
            y_pred = model.predict(X[test])
            proba_model = model.predict_proba(X[test])
            y_ref = y[test].ravel()
            Hugues_data = np.c_[proba_model[:, 0], proba_model[:, 1], y_pred, y_ref]
            df_m = pd.DataFrame(
                Hugues_data,
                columns=[
                    "proba_label_0",
                    "proba_label_1",
                    "predicted_test_label",
                    "ref_test_label",
                ],
            )
            df_m.to_csv(os.path.join(path_model_fold_cv, f"Fold_{i}.csv"))

        filename = name_model + ".sav"
        pickle.dump(model, open(os.path.join(path_folder_model, filename), "wb"))
    else:
        print("You didn't give any cols. No model were created")


def feature_selector(X_data, y_data, cols, model_type="Logistic", already_model=False):
    """
    Function that prepare your feature dataset for a specific model you want to create.

    Inputs :
        X_data [2D Pandas Dataframe] : Dataframe with the rows being the patients ECG score and the columns the
            features you evaluated (with their name)
        y_data [1D Pandas Dataframe] : Dataframe containing the label assigned to your patient ECG recording
        cols [String list] : Array of string containing the features name you tested. The name contains must
            be the same as X_data column (Otherwise an error will be thrown)
        model_type [String variable] : The name of the model you want to train. Include : Custom logistic regression,
            Logistic regression, ExtraClassifier, RandomForestClassifier from sklearn
        **kwargs : Any additional arguments
                - model : your custom model (herited from class sklearn)

    Outputs :
        X [2D Numpy Array] : Your prepared feature Dataset of size [number_of_patient*features_selected]
        y [1D Numpy Array] : Your true label assigned to your dataset
        model : The model you selected (or gave)
        Hindex [int] : index where the columns correspond to the Heart Rate in your dataset
    """
    y = y_data.values
    if cols is None:
        print("Using Backward model selection")
        cols = backward_model_selection(X_data, y_data)
        if "HR" in cols and len(cols) > 1:
            Hindex = list(X_data[cols].columns.values).index("HR")

        else:
            Hindex = None
        X = X_data[cols].values
    else:
        if "HR" in cols and len(cols) > 1:
            Hindex = list(X_data[cols].columns.values).index("HR")
        else:
            Hindex = None
        X = X_data[cols].values

    if already_model:
        return X, y, None, Hindex
    else:
        if model_type == "ExtraTreeClassifier":
            model = ExtraTreesClassifier(random_state=seed)
        elif model_type == "RandomTreeClassifier":
            model = RandomForestClassifier(random_state=seed)
        elif model_type == "Logistic" and Hindex is not None:
            model_type = model_type + " Binary"
            model = Logit_binary(HR_index=Hindex, random_state=seed)
        else:
            model = LogisticRegression(random_state=seed)

        return X, y, model, Hindex


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
