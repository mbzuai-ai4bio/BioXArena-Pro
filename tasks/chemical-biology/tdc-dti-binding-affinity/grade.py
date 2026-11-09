"""tdc-dti-binding-affinity grader, v5.1 (2026-10-18).

Metric: fixed-weight mean over the three sources of the calibrated regression score
    source_score = 0.5 * clip(Pearson_r, 0, 1) + 0.5 * clip(1 - RMSE / std(y_true), 0, 1)
    score        = 0.4 * kiba + 0.4 * bindingdb + 0.2 * davis
v5 used equal weights (1/3 each). DAVIS is down-weighted because its test set has only 23 drugs and is the
noisiest component. A per-source constant scores 0, the true values score 1. Task-local; shared utils untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tasks.utils import (
    InvalidSubmissionError,
    calibrated_regression_safe,
    numeric_array,
    read_test,
    validate_submission_frame,
    validate_test_alignment,
)

TASK_KEY = "chemical-biology/tdc-dti-binding-affinity"
COL = "affinity"
SOURCE_WEIGHTS = {"kiba": 0.4, "bindingdb": 0.4, "davis": 0.2}
METRIC_INFO = {
    "name": "per_source_weighted_calibrated_regression_kiba0.4_bindingdb0.4_davis0.2",
    "direction": "higher_is_better",
    "normalization": "already_in_0_1",
}


def per_source_scores(test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for s, idx in test.groupby("source", sort=True).indices.items():
        idx = np.asarray(idx, dtype=int)
        out[str(s)] = calibrated_regression_safe(y_true[idx], y_pred[idx])
    return out


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    submission, answers = validate_submission_frame(submission, answers, ["id", COL])
    test = validate_test_alignment(read_test(kwargs.get("public_dir"), kwargs.get("answers_path")), answers)
    if "source" not in test.columns:
        raise InvalidSubmissionError("test.csv is missing source.")
    y_true = numeric_array(answers, COL, "answers")
    y_pred = numeric_array(submission, COL, "submission")
    scores = per_source_scores(test, y_true, y_pred)
    if set(scores) != set(SOURCE_WEIGHTS):
        raise InvalidSubmissionError(f"unexpected sources in test.csv: {sorted(scores)}")
    return float(sum(SOURCE_WEIGHTS[s] * v for s, v in scores.items()))
