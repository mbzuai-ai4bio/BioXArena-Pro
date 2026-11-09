from __future__ import annotations

import pandas as pd

from tasks.utils import METRIC_INFO_BY_TASK, grade_task


TASK_KEY = 'imaging/cryocrab-quality'
METRIC_INFO = METRIC_INFO_BY_TASK[TASK_KEY]


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    kwargs.pop("task_key", None)
    return grade_task(TASK_KEY, submission, answers, **kwargs)
