"""asep-epitope grader, v5.1 (2026-10-18).

Metric: per-complex chance-normalised average precision, averaged equally over the test complexes.
For each complex c with epitope prevalence p_c (fraction of its scored residues that are epitope),
    s_c = max(0, (AP_c - p_c) / (1 - p_c)),
where AP_c = sklearn.metrics.average_precision_score on the residues of complex c. The score is mean_c s_c.
A random or constant ranking has AP_c ~= p_c and scores ~0; a perfect ranking scores 1. Complexes without
both an epitope and a non-epitope residue are skipped (none in the v5 test set).
v5 scored the mean raw AP_c (random ~0.13); v5.1 scores are not comparable with v5. Task-local; shared utils untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from tasks.utils import (
    InvalidSubmissionError,
    mean_or_raise,
    numeric_array,
    read_test,
    validate_submission_frame,
    validate_test_alignment,
)

TASK_KEY = "structure/asep-epitope"
COL = "epitope"
GROUP_COL = "complex_id"
METRIC_INFO = {
    "name": "per_complex_mean_prevalence_normalised_average_precision",
    "direction": "higher_is_better",
    "normalization": "already_in_0_1",
}


def normalised_ap(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=float)
    p = float(y_true.mean())
    if p <= 0.0 or p >= 1.0:
        return None
    ap = float(average_precision_score(y_true, np.asarray(y_score, dtype=float)))
    return max(0.0, (ap - p) / (1.0 - p))


def per_complex_scores(test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for c, idx in test.groupby(GROUP_COL, sort=True).indices.items():
        idx = np.asarray(idx, dtype=int)
        if len(idx) < 2:
            continue
        s = normalised_ap(y_true[idx], y_pred[idx])
        if s is not None:
            out[c] = s
    return out


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    submission, answers = validate_submission_frame(submission, answers, ["id", COL])
    test = validate_test_alignment(read_test(kwargs.get("public_dir"), kwargs.get("answers_path")), answers)
    if GROUP_COL not in test.columns:
        raise InvalidSubmissionError(f"test.csv is missing {GROUP_COL}.")
    y_true = numeric_array(answers, COL, "answers")
    if not np.isin(y_true, [0.0, 1.0]).all():
        raise InvalidSubmissionError("answers must be binary 0/1.")
    y_pred = numeric_array(submission, COL, "submission")
    return mean_or_raise(list(per_complex_scores(test, y_true, y_pred).values()), "per-complex normalised AP")
