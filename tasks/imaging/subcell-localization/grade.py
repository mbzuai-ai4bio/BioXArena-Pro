"""subcell-localization grader, v5 (2026-10-17), task-local.

Metric family unchanged: 0.5 * macro-F1 (threshold 0.5) + 0.5 * macro-AP (native scale). v5: labels with fewer than
5 positive test proteins (protein_group) are excluded from both macro means, because a label resting on 2-3 proteins
is decided by single proteins (mitochondria 2, big_aggregates 3, focal_adhesions 3 positive test proteins; the next
label, centrosome, has 6). The submission must still contain all 17 label columns with finite values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from tasks.utils import METRIC_INFO_BY_TASK, TASK_CONFIG, InvalidSubmissionError, prepare_standard_task

TASK_KEY = 'imaging/subcell-localization'
METRIC_INFO = METRIC_INFO_BY_TASK[TASK_KEY]
EXCLUDED_LABELS = ["big_aggregates", "focal_adhesions", "mitochondria"]


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    config = TASK_CONFIG[TASK_KEY]
    submission, answers = prepare_standard_task(submission, answers, config)
    all_cols = config["label_cols"]
    scores = submission[all_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise InvalidSubmissionError("Multilabel submission contains non-finite values.")
    cols = [c for c in all_cols if c not in EXCLUDED_LABELS]
    y_true = answers[cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=int)
    y_score = submission[cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    macro_f1 = f1_score(y_true, (y_score >= 0.5).astype(int), average="macro", zero_division=0)
    mean_ap = average_precision_score(y_true, y_score, average="macro")
    return float(0.5 * macro_f1 + 0.5 * mean_ap)
