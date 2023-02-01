from .fiducial_metrics import (
    corr_lead_score,
    morph_score,
    kurto_score,
    HR_index_calculator,
    flatline_score,
)
from .non_fiducial_metrics import wpmf_score, snr_index, sdr_score
from .tsd_metrics import tsd_index


method_registry = {
    "Corr_interlead": corr_lead_score,
    "Corr_intralead": morph_score,
    "HR": HR_index_calculator,
    "Kurtosis": kurto_score,
    "Flatline": flatline_score,
    "wPMF": wpmf_score,
    "SNRECG": snr_index,
    "SDR": sdr_score,
    "TSD": tsd_index,
}
