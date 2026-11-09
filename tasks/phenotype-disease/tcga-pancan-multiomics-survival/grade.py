"""BioXArena-Pro v5 (2026-10-17), task-local grader for phenotype-disease/tcga-pancan-multiomics-survival.

Metric: event-weighted mean of per-cohort Harrell c-indices.
* Inside each cohort of test.csv the Harrell c-index of risk_score against the hidden
  (survival_months, censorship) is computed: a pair (i, j) is comparable when patient i died
  (censorship == 0) and t_j > t_i (strictly); it is concordant when risk_i > risk_j and counts 0.5
  when risk_i == risk_j. Pairs with tied times are not comparable. A cohort without any comparable
  pair scores 0.5.
* The per-cohort values are averaged with weights = number of observed deaths in that cohort
  (v4: equal weights). Only within-cohort ranking matters. Native c-index scale (chance 0.5).
Validation (ids, order, finite floats) uses the shared helpers unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tasks.utils import (InvalidSubmissionError, c_index, numeric_array, read_test, require_columns,
                         validate_submission_frame, validate_test_alignment)

TASK_KEY = "phenotype-disease/tcga-pancan-multiomics-survival"
GROUP_COL = "cohort"
METRIC_INFO = {"name": "event_weighted_per_cohort_c_index", "direction": "higher_is_better", "normalization": "already_in_0_1"}


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    public_dir, answers_path = kwargs.get("public_dir"), kwargs.get("answers_path")
    submission, answers = validate_submission_frame(submission, answers, ["id", "risk_score"])
    require_columns(answers, ["survival_months", "censorship"], "answers")
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    times = numeric_array(answers, "survival_months", "answers")
    events = 1 - numeric_array(answers, "censorship", "answers")
    risk = numeric_array(submission, "risk_score", "submission")
    if not np.all(np.isfinite(risk)):
        raise InvalidSubmissionError("risk_score must be finite for every row.")
    require_columns(test, [GROUP_COL], "test.csv")
    scores, weights = [], []
    for _, positions in test.groupby(GROUP_COL, sort=True).indices.items():
        idx = np.asarray(list(positions), dtype=int)
        scores.append(c_index(times[idx], events[idx], risk[idx]))
        weights.append(float(events[idx].sum()))
    w = np.asarray(weights)
    if w.sum() <= 0:
        raise InvalidSubmissionError("no observed deaths in the answers")
    return float(np.dot(np.asarray(scores), w) / w.sum())
