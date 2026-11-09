"""scgenescope-treatment grader, v5 (2026-10-17): macro-F1 over the 29 classes (shared grade_task), plus a
task-local check that every treatment_id is an integer in 0..28 (v4 silently accepted out-of-range ids)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tasks.utils import METRIC_INFO_BY_TASK, InvalidSubmissionError, grade_task


TASK_KEY = 'perturbation/scgenescope-treatment'
METRIC_INFO = METRIC_INFO_BY_TASK[TASK_KEY]
N_CLASSES = 29


def _check_labels(submission: pd.DataFrame) -> pd.DataFrame:
    if "treatment_id" not in submission.columns:
        raise InvalidSubmissionError("Submission must have a 'treatment_id' column.")
    v = pd.to_numeric(submission["treatment_id"], errors="coerce")
    if v.isna().any():
        raise InvalidSubmissionError("treatment_id must be an integer from 0 through 28 (missing/non-numeric found).")
    arr = v.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(arr)) or np.any(arr != np.round(arr)):
        raise InvalidSubmissionError("treatment_id must be an integer from 0 through 28 (non-integer found).")
    if arr.min() < 0 or arr.max() > N_CLASSES - 1:
        raise InvalidSubmissionError(f"treatment_id must be in 0..{N_CLASSES - 1}; got range {arr.min():g}..{arr.max():g}.")
    submission = submission.copy(); submission["treatment_id"] = arr.astype(np.int64)
    return submission


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    kwargs.pop("task_key", None)
    return grade_task(TASK_KEY, _check_labels(submission), answers, **kwargs)
