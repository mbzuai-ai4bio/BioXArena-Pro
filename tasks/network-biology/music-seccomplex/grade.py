"""music-seccomplex grader, v4 (2026-10-16).

Metric: prevalence-normalised average precision (AUPRC) over all test pairs,
    score = max(0, (AP - p) / (1 - p)),  p = fraction of positive test pairs,
where AP = sklearn.metrics.average_precision_score (step-wise precision-recall area). A constant or random ranking
has AP ~= p and scores ~0; a perfect ranking scores 1. ROC-AUC is no longer scored (task-local; shared utils untouched).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from tasks.utils import InvalidSubmissionError, numeric_array, validate_submission_frame

TASK_KEY = "network-biology/music-seccomplex"
COL = "in_same_corum_complex"
METRIC_INFO = {
    "name": "prevalence_normalised_average_precision",
    "direction": "higher_is_better",
    "normalization": "already_in_0_1",
}


def normalised_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    if not np.isin(y_true, [0.0, 1.0]).all():
        raise InvalidSubmissionError("answers must be binary 0/1.")
    p = float(y_true.mean())
    if p <= 0.0 or p >= 1.0:
        raise InvalidSubmissionError("average precision requires at least one positive and one negative.")
    ap = float(average_precision_score(y_true, np.asarray(y_score, dtype=float)))
    return max(0.0, (ap - p) / (1.0 - p))


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    submission, answers = validate_submission_frame(submission, answers, ["id", COL])
    y_true = numeric_array(answers, COL, "answers")
    y_pred = numeric_array(submission, COL, "submission")
    return normalised_ap(y_true, y_pred)
