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

    list_features = ["Corr_interlead", "HR", "SNRECG", "Corr_intralead"]
    input_data_path = os.path.join(results_path, "quality_metrics.nc")
    train_model(
        input_data_path,
        model_type="logistic",
        list_features=list_features,
        nb_fold=5,
        save_name="JMI_MI_selection_method",
    )


if __name__ == "__main__":
    main()
