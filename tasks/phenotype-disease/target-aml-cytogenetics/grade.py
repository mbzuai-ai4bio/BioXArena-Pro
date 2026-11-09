"""target-aml-cytogenetics, BioXArena-Pro v4 (2026-10-16): 7-class macro-F1.

The v2.1 class `MLL` is split into KMT2A partner risk groups (COG paediatric AML risk stratification):
`KMT2A-MLLT3` [t(9;11)], `KMT2A-high-risk` [t(6;11) or t(10;11)], `KMT2A-other` (any other / unknown partner).
v5 (2026-10-17): the class `KMT2A-high-risk` is renamed `KMT2A-MLLT10/AFDN` (same patients; a neutral partner name
instead of a risk claim). Labels outside the fixed 7-label vocabulary (including the old `MLL`) fail the submission. Task-local copy of the shared
macro-F1 grader (tasks/utils.py is owned by the GRADERS agent)."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score

from tasks.utils import InvalidSubmissionError, validate_submission_frame

TASK_KEY = "phenotype-disease/target-aml-cytogenetics"
LABELS = ["KMT2A-MLLT3", "KMT2A-MLLT10/AFDN", "KMT2A-other", "Normal", "Other", "inv(16)", "t(8;21)"]
METRIC_INFO = {"name": "macro_f1_7class", "direction": "higher_is_better", "normalization": "already_in_0_1"}
COLS = ["id", "cytogenetic_class"]


def per_class(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    sub, ans = validate_submission_frame(submission, answers[COLS], COLS)
    p = sub.cytogenetic_class
    if p.isna().any():
        raise InvalidSubmissionError("cytogenetic_class contains missing labels")
    bad = sorted(set(p.astype(str)) - set(LABELS))
    if bad:
        raise InvalidSubmissionError(f"cytogenetic_class contains labels outside {LABELS}: {bad[:5]}")
    f = f1_score(ans.cytogenetic_class.astype(str), p.astype(str), labels=LABELS, average=None, zero_division=0)
    return dict(zip(LABELS, map(float, f)))


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    pc = per_class(submission, answers)
    return float(sum(pc.values()) / len(LABELS))
