import argparse
import os
import sys
import xarray as xr

CWD = os.getcwd()
FILEPATH = os.path.dirname(os.path.realpath(__file__))
ROOTPATH = os.path.dirname(FILEPATH)
sys.path.append(os.path.join(ROOTPATH))

# custom imports
import shared_utils.utils_data as utils_data
import shared_utils.utils_evaluation as utils_evaluation


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Process files.")
    parser.add_argument(
        "--input_path",
        help="Path to the input data with ECG to be analysed",
        default="/workspaces/ecg_evaluation/data/20221006_physio_quality/set-a/dataParquet",
    )
    parser.add_argument(
        "--save_path",
        help="Path where computed metrics should be saved",
        default="/workspaces/ecg_evaluation/results",
    )

    args = parser.parse_args()
    input_path = args.input_path
    save_path = args.save_path

    ds_metrics = xr.load_dataset(os.path.join(save_path, "quality_metrics.nc"))
    df_X, df_y = utils_data.extract_index_label(ds_metrics, ["Corr_interlead"])
    utils_evaluation.evaluate_index(df_X, df_y)


if __name__ == "__main__":
    main()
