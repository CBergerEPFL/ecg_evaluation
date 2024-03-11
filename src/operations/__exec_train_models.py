import argparse
import os
import sys
import xarray as xr

CWD = os.getcwd()
FILEPATH = os.path.dirname(os.path.realpath(__file__))
ROOTPATH = os.path.dirname(FILEPATH)
sys.path.append(os.path.join(ROOTPATH))

# custom imports
from models.train_model import train_model
from shared_utils.utils_path import results_path


def main():
    # parse arguments
    # list_features = ['Corr_interlead', 'SNRECG', 'TSD', 'Corr_intralead']
    # list_features = ["Corr_interlead", "SNRECG", "HR", "Corr_intralead", "wPMF"]
    # list_features = ["Corr_interlead", "HR", "wPMF", "TSD"]
    # list_features = ["Corr_interlead", "TSD", "Corr_intralead", "SNRECG"]

    # print(results_path)
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
    train_model(
        input_data_path,
        path_results=results_path,
        model_type="lgbm",
        # model_type="logistic",
        list_features=list_features,
        # feature_selection="L2_regularization",
        nb_fold=5,
        save_name="lgbm",
        # save_name="L2_reg_logistic",
        aggregation = "min"

    )


if __name__ == "__main__":
    main()
