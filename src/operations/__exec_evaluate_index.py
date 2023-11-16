import argparse
import os
import sys
import xarray as xr

CWD = os.getcwd()
FILEPATH = os.path.dirname(os.path.realpath(__file__))
ROOTPATH = os.path.dirname(FILEPATH)
sys.path.append(os.path.join(ROOTPATH))

# custom imports
from metrics.evaluate import evaluate_list_indices
from shared_utils.utils_path import results_path


def main():
    # parse arguments
    save_path = results_path

    input_data_path = os.path.join(save_path, "quality_metrics.nc")
    list_features = [
        "Corr_interlead",
        "Corr_intralead",
        "wPMF",
        "SNRECG",
        "HR",
        "Kurtosis",
        "Flatline",
        "TSD",
    ]
    evaluate_list_indices(input_data_path, list_features)


if __name__ == "__main__":
    main()
