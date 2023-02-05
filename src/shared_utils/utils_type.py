import os
import sys
import numpy as np
import pickle as pkl
from dataclasses import dataclass

FILEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FILEPATH))
from utils_path import results_path


@dataclass
class Results_Data:
    proba_unacceptable: list[np.array]
    label: list[np.array]

    def __post_init__(self):
        self.dict_results = {
            "proba_unacceptable": self.proba_unacceptable,
            "label": self.label,
        }

    def dump_to_file(self, save_name):
        with open(
            os.path.join(results_path, "proba_methods", f"proba_{save_name}.pkl"),
            "wb",
        ) as f:
            pkl.dump(self.dict_results, f)
