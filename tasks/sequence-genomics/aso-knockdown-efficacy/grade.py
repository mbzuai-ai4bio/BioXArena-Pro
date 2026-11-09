"""Grader for `sequence-genomics/aso-knockdown-efficacy`, data version corrected_v4.

v4 metric change (2026-10-16). Up to corrected_v3 the score was the mean of the Pearson
correlation computed separately inside each `source`. The v4 audit showed that most of that
correlation was a chemistry-class offset handed over by the input: stripping the nucleobases
from the HELM string leaves the sugar/linker (gapmer design) pattern, and predicting the
per-(source, pattern) training mean alone reached a raw per-source mean r of 0.285 against
0.448 for the best agent, with all three agents inside 1.7 points.

v4 therefore centres truth and prediction inside every (source, chem_pattern) group before
pooling, exactly as corrected_v3 already centred per source. What is scored is the ability to
rank sequences *within a fixed chemistry platform* -- the decision an ASO screen actually
makes. The chemistry-pattern lookup now scores exactly 0.

This grader is deliberately self-contained: `tasks/utils.py` is a shared file, so the metric
lives here. Only the submission/alignment validators are reused from it.

The returned value is the RAW correlation in [-1, 1]; `METRIC_INFO` advertises
`map_minus1_1_to_0_1`, which the v4 harness reports as clip(r, 0, 1), so chance and
anti-correlation score 0.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from tasks.utils import (
    InvalidSubmissionError,
    numeric_array,
    pearson_safe,
    read_test,
    require_columns,
    validate_submission_frame,
    validate_test_alignment,
)

TASK_KEY = "sequence-genomics/aso-knockdown-efficacy"

PRED_COL = "inhibition"
TARGET_COL = "inhibition"
GROUP_COL = "source"
CENTER_COL = "chem_pattern"

METRIC_INFO = {
    "name": "per_source_mean_pearson_within_chemistry_pattern",
    "direction": "higher_is_better",
    "normalization": "map_minus1_1_to_0_1",
}

_BASE_TOKEN = re.compile(r"\([ACGTU]\)")


def chem_pattern(helm: str) -> str:
    """The sugar/linker pattern: the HELM polymer with every nucleobase token removed.

    Kept here so the grader can rebuild the centring groups even if `chem_pattern` is
    missing from an older test.csv.
    """
    return _BASE_TOKEN.sub("", str(helm).split("$")[0])


def _centre_within(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Subtract each key's mean. A singleton group becomes exactly 0, which sits at the
    grand mean of the centred vector and so contributes nothing to the correlation."""
    out = np.asarray(values, dtype=float).copy()
    frame = pd.DataFrame({"k": keys, "v": out})
    out -= frame.groupby("k", sort=False).v.transform("mean").to_numpy(dtype=float)
    return out


def grade(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    public_dir: Path | str | None = None,
    answers_path: Path | str | None = None,
    **kwargs,
) -> float:
    kwargs.pop("task_key", None)
    submission, answers = validate_submission_frame(submission, answers, ["id", PRED_COL])
    require_columns(answers, [TARGET_COL], "answers")

    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    require_columns(test, [GROUP_COL], "test.csv")
    if CENTER_COL in test.columns:
        centre_keys = test[CENTER_COL].astype(str).to_numpy()
    elif "helm" in test.columns:
        centre_keys = test["helm"].map(chem_pattern).astype(str).to_numpy()
    else:
        raise InvalidSubmissionError(
            f"test.csv must carry '{CENTER_COL}' or 'helm' so the centring groups can be built."
        )

    y_true = numeric_array(answers, TARGET_COL, "answers")
    y_pred = numeric_array(submission, PRED_COL, "submission")

    groups = test[GROUP_COL].astype(str).to_numpy()
    scores: list[float] = []
    for source in sorted(set(groups)):
        idx = np.flatnonzero(groups == source)
        if len(idx) < 2:
            continue
        keys = np.char.add(source + "\x1f", centre_keys[idx].astype(str))
        value = pearson_safe(
            _centre_within(y_true[idx], keys),
            _centre_within(y_pred[idx], keys),
        )
        if np.isfinite(value):
            scores.append(float(value))
    if not scores:
        raise InvalidSubmissionError("no evaluation group had at least two rows")
    return float(np.mean(scores))
