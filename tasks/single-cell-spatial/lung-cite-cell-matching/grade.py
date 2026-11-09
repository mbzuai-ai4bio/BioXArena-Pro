"""lung-cite-cell-matching, BioXArena-Pro v5 (2026-10-17). Task-local grader.

Top-1 matching accuracy (unchanged). v5 adds id validation: every submitted adt_cell_id must be a member of the row's
(candidate_group, protocol) in public adt_index.csv; a missing value, an unknown id or an id from another group/protocol
makes the submission invalid (v4 silently counted such rows as wrong).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tasks.utils import (InvalidSubmissionError, prepare_standard_task, read_test, require_public_dir,
                         validate_test_alignment)

TASK_KEY = "single-cell-spatial/lung-cite-cell-matching"
METRIC_INFO = {"name": "top1_cell_matching_accuracy", "direction": "higher_is_better", "normalization": "already_in_0_1"}
CONFIG = {"pred_col": "adt_cell_id", "target_col": "adt_cell_id"}


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    submission, answers = prepare_standard_task(submission, answers, CONFIG)
    pred = submission["adt_cell_id"]
    if pred.isna().any():
        raise InvalidSubmissionError("adt_cell_id contains missing values")
    public_dir = require_public_dir(kwargs.get("public_dir"), kwargs.get("answers_path"))
    test = validate_test_alignment(read_test(public_dir, kwargs.get("answers_path")), answers)
    idx = pd.read_csv(Path(public_dir) / "adt_index.csv", dtype={"adt_cell_id": str})
    member = set(zip(idx["adt_cell_id"].astype(str), idx["candidate_group"].astype(str), idx["protocol"].astype(str)))
    keys = zip(pred.astype(str), test["candidate_group"].astype(str), test["protocol"].astype(str))
    bad = [i for i, k in enumerate(keys) if k not in member]
    if bad:
        raise InvalidSubmissionError(
            f"{len(bad)} adt_cell_id values are not candidates of their row's candidate_group/protocol "
            f"(first row {bad[0]}: {pred.iloc[bad[0]]!r})")
    return float(np.mean(pred.astype(str).to_numpy() == answers["adt_cell_id"].astype(str).to_numpy()))
