from __future__ import annotations

import pandas as pd

from tasks.nb_grading import grade_from_contract, metric_info_from_contract


METRIC_INFO = metric_info_from_contract(__file__)


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    return grade_from_contract(submission, answers, **kwargs)
