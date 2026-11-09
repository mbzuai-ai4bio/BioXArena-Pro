from __future__ import annotations

import pandas as pd

from tasks.native_manifest_metrics import grade_microscopy


METRIC_INFO = {
    "name": "per_instance_mean_msa",
    "direction": "higher_is_better",
    "normalization": "already_in_0_1",
}


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    return grade_microscopy(submission, answers, **kwargs)
