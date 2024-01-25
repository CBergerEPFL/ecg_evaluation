import argparse
import os
import sys

CWD = os.getcwd()
FILEPATH = os.path.dirname(os.path.realpath(__file__))
ROOTPATH = os.path.dirname(FILEPATH)
sys.path.append(os.path.join(ROOTPATH))

# custom imports
import shared_utils.utils_data as utils_data
from shared_utils.utils_path import results_path
from metrics.compute import save_metrics_to_xarray


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Process files.")
    parser.add_argument(
        "--input_path",
        help="Path to the input data with ECG to be analysed",
        default="/workspaces/ecg_evaluation/data/20221006_physio_quality/set-a/dataParquet",
    )

    args = parser.parse_args()
    input_path = args.input_path
    save_path = results_path
    path_petastorm = f"file:///{input_path}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ds_data = utils_data.format_data_to_xarray(path_petastorm, save_path)

    name_method_required = [
        "Corr_interlead",
        "Corr_intralead",
        "wPMF",
        "SNRECG",
        "HR",
        "Kurtosis",
        "Flatline",
        "TSD",
    ]
    save_metrics_to_xarray(ds_data, name_method_required, save_path, verbose=True)


if __name__ == "__main__":
    main()
