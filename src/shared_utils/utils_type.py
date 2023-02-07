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
    model: str
    hp = None

    def __post_init__(self):
        self._proba_unacceptable = []
        self._label = []

    def dump_to_file(self, save_name):
        with open(
            os.path.join(results_path, "proba_methods", f"proba_{save_name}.pkl"),
            "wb",
        ) as f:
            pkl.dump(self.dict_results, f)

    @property
    def dict_results(self):
        dict_results = {
            "model": self.model,
            "proba_unacceptable": self._proba_unacceptable,
            "label": self._label,
        }
        if self.hp:
            dict_results["hp"] = self.hp
        return dict_results

    def append_results(self, proba_unacceptable: np.array, label: np.array):
        self._proba_unacceptable.append(proba_unacceptable)
        self._label.append(label)
