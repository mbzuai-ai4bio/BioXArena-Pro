"""dlpfc-spatial-domain, BioXArena-Pro v5 (2026-10-17). Task-local grader (no longer uses tasks.utils grouped_macro_f1).

Score = mean over test sections of the macro-F1 over the layers that are PRESENT in that section's reference labels
(sklearn f1_score(..., labels=sorted(set(y_true_section)))). A spot predicted as a layer that is absent from its section
still counts as a false negative of its true layer (lowers that layer's recall), but the absent layer is not added as an
extra zero-F1 class. v4 averaged over union(true, pred), so 8 stray L1/L2 spots (0.03 %) in the 4 test sections without
L1/L2 cut a perfect answer to 0.857 (v4 verification, 2026-10-17).
Labels outside {L1..L6, WM} or missing values make the submission invalid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from tasks.utils import (InvalidSubmissionError, prepare_standard_task, read_test, require_columns,
                         validate_test_alignment)

TASK_KEY = "single-cell-spatial/dlpfc-spatial-domain"
METRIC_INFO = {"name": "per_section_mean_macro_f1_over_present_layers", "direction": "higher_is_better",
               "normalization": "already_in_0_1"}
LABELS = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]
CONFIG = {"pred_col": "layer", "target_col": "layer"}


def section_scores(submission: pd.DataFrame, answers: pd.DataFrame, public_dir=None, answers_path=None) -> dict:
    submission, answers = prepare_standard_task(submission, answers, CONFIG)
    raw = submission["layer"]
    if raw.isna().any():
        raise InvalidSubmissionError("layer contains missing values")
    pred = raw.astype(str).str.strip().to_numpy()
    bad = sorted(set(pred) - set(LABELS))
    if bad:
        raise InvalidSubmissionError(f"layer contains labels outside {LABELS}: {bad[:5]}")
    true = answers["layer"].astype(str).str.strip().to_numpy()
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    require_columns(test, ["section"], "test.csv")
    out = {}
    for sec, pos in test.groupby("section", sort=True).indices.items():
        idx = np.asarray(list(pos), dtype=int)
        present = sorted(set(true[idx]))
        out[sec] = float(f1_score(true[idx], pred[idx], labels=present, average="macro", zero_division=0))
    if not out:
        raise InvalidSubmissionError("no test sections")
    return out


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    s = section_scores(submission, answers, kwargs.get("public_dir"), kwargs.get("answers_path"))
    return float(np.mean(list(s.values())))
