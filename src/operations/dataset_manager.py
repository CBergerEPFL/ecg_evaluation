import numpy as np
import argparse
import os
import sys
import warnings

CWD = os.getcwd()
FILEPATH = os.path.dirname(os.path.realpath(__file__))
ROOTPATH = os.path.dirname(FILEPATH)
sys.path.append(os.path.join(ROOTPATH))

##Custom import
from shared_utils.utils_path import data_path


def Physionet_reader(name_dataset, ignore_inner_folder=True):
    path_to_folder = os.path.join(data_path, name_dataset)
    if os.path.isdir(path_to_folder) == False:
        raise OSError("Please indicate a valid path to your dataset")
    if ignore_inner_folder:
        files = np.array(
            [
                f
                for f in os.listdir(path_to_folder)
                if os.path.isfile(os.path.join(path_to_folder, f))
            ]
        )
