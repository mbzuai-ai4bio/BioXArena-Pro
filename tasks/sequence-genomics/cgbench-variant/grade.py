from __future__ import annotations

import numpy as np
import pandas as pd

from tasks.utils import METRIC_INFO_BY_TASK, grade_task


TASK_KEY = 'sequence-genomics/cgbench-variant'
METRIC_INFO = METRIC_INFO_BY_TASK[TASK_KEY]
VALID_LABELS = {0, 1, 2, 3, 4}


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    kwargs.pop("task_key", None)
    # corrected_v3: every prediction must be an integer class 0-4 (description.md); anything else is invalid
    if "assertion" not in submission.columns:
        raise ValueError("submission must have an 'assertion' column")
    vals = pd.to_numeric(submission["assertion"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("assertion contains missing or non-numeric values")
    if not (np.round(vals) == vals).all() or not set(vals.astype(int).tolist()) <= VALID_LABELS:
        raise ValueError("assertion values must be integers in {0,1,2,3,4}")
    return grade_task(TASK_KEY, submission, answers, **kwargs)
