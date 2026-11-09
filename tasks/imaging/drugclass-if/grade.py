"""drugclass-if grader, v5 (2026-10-17), task-local.

Macro-F1 over the three treatment classes. v5: labels outside {paclitaxel, untreated, vorinostat} (exact, lower case)
make the submission invalid instead of being scored as an extra zero-F1 class, and f1_score is computed over the fixed
three labels. Native scale (chance about 1/3; F1 is not a correlation metric).
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score

from tasks.utils import METRIC_INFO_BY_TASK, TASK_CONFIG, InvalidSubmissionError, prepare_standard_task

TASK_KEY = 'imaging/drugclass-if'
METRIC_INFO = METRIC_INFO_BY_TASK[TASK_KEY]
LABELS = ["paclitaxel", "untreated", "vorinostat"]


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    config = TASK_CONFIG[TASK_KEY]
    submission, answers = prepare_standard_task(submission, answers, config)
    pred = submission[config["pred_col"]]
    if pred.isna().any():
        raise InvalidSubmissionError("Submission column 'drug' contains missing labels.")
    pred = pred.astype(str)
    unknown = sorted(set(pred) - set(LABELS))
    if unknown:
        raise InvalidSubmissionError(f"Submission column 'drug' contains unknown labels {unknown[:5]}; allowed: {LABELS}")
    return float(f1_score(answers[config["target_col"]].astype(str), pred, labels=LABELS, average="macro", zero_division=0))
