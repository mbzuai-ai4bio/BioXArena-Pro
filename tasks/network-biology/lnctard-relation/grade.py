"""LncTarD-XL v5 grader (task-local): mean per-query average precision over the SCORED queries.

answers.csv: id, interaction, scored. Queries whose regulator has no edge in the training positives or the auxiliary
graph (cold start) have scored=0 and are excluded. The submission must still cover every test row (id, interaction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tasks.utils import (InvalidSubmissionError, average_precision_safe, grouped_metric, numeric_array,
                         read_test, validate_submission_frame, validate_test_alignment)

TASK_KEY = "network-biology/lnctard-relation"
METRIC_INFO = {"name": "mean_per_query_average_precision", "direction": "higher_is_better",
               "normalization": "already_in_0_1"}


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    public_dir, answers_path = kwargs.get("public_dir"), kwargs.get("answers_path")
    submission, answers = validate_submission_frame(submission, answers, ["id", "interaction"])
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    y_pred = numeric_array(submission, "interaction", "submission")
    if not np.all(np.isfinite(y_pred)):
        raise InvalidSubmissionError("Submission column 'interaction' must be finite.")
    y_true = numeric_array(answers, "interaction", "answers")
    keep = (answers["scored"].astype(int).values == 1) if "scored" in answers.columns else np.ones(len(answers), bool)
    test = test.loc[keep].reset_index(drop=True)
    return grouped_metric(test, y_true[keep], y_pred[keep], "query_id", average_precision_safe, min_rows=2)
