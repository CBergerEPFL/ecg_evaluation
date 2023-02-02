import os
from os.path import join as pj
from os.path import abspath as ap

FILEPATH = os.path.dirname(os.path.realpath(__file__))

src_path = ap(pj(FILEPATH, ".."))
# utils_path = ap(pj(FILEPATH, "..", "shared_utils"))
data_path = ap(pj(FILEPATH, "..", "..", "data"))
results_path = ap(pj(FILEPATH, "..", "..", "results"))
mount_path = "U"


if not os.path.exists(results_path):
    os.mkdir(results_path)

if not os.path.exists(os.path.join(results_path, "evaluation_metrics")):
    os.mkdir(os.path.join(results_path, "evaluation_metrics"))

if not os.path.exists(os.path.join(results_path, "proba_methods")):
    os.mkdir(os.path.join(results_path, "proba_methods"))
