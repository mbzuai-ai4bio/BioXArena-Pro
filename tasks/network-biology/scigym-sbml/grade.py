"""scigym-sbml grader, v4 (2026-10-16), task-local.

Score = mean exact integer accuracy over all test rows and the three independent targets
    delta_qualitative_species, delta_transitions, delta_species_glyphs.
(v2 scored six columns, but in every complete model #transitions = #inputs = #outputs and #species glyphs =
#text glyphs, so delta_inputs / delta_outputs / delta_text_glyphs were deterministic functions of the other targets
and the partial counts; they are no longer asked for.)
Predictions must be finite non-negative integers; ids must match the answers exactly and in order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tasks.utils import InvalidSubmissionError, integer_accuracy_safe, numeric_array, validate_submission_frame

TASK_KEY = "network-biology/scigym-sbml"
LABELS = ["delta_qualitative_species", "delta_transitions", "delta_species_glyphs"]
METRIC_INFO = {
    "name": "mean_exact_deleted_count_accuracy_3_independent_targets",
    "direction": "higher_is_better",
    "normalization": "already_in_0_1",
}


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    submission, answers = validate_submission_frame(submission, answers, ["id", *LABELS])
    scores = []
    for c in LABELS:
        y_true = numeric_array(answers, c, "answers")
        y_pred = numeric_array(submission, c, "submission")
        if (y_pred < 0).any():
            raise InvalidSubmissionError(f"submission column '{c}' contains negative counts.")
        scores.append(integer_accuracy_safe(y_true, y_pred))
    return float(np.mean(scores))
