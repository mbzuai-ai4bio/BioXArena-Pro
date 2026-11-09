"""boom-csd grader, corrected_v4 (2026-10-16).

Per (source x extrapolation tail) calibrated regression, equal-weight mean of the 4 cells.
tail = 'low' if the true value is below the training median of that source, else 'high'
(stored in private answers.csv; the test tails lie entirely outside the training range).
cell score = (clip(Pearson, 0, 1) + clip(1 - RMSE / std(y_true_cell), 0, 1)) / 2.
Self-contained (no shared helpers) so that shared-code edits cannot change it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TASK_KEY = "chemical-biology/boom-csd"
METRIC_INFO = {
    "name": "per_source_per_tail_mean_calibrated_regression_clip",
    "direction": "higher_is_better",
    "normalization": "already_in_0_1",
}
CELLS = [("density", "low"), ("density", "high"), ("hof", "low"), ("hof", "high")]


class InvalidSubmissionError(Exception):
    pass


def _pearson(t: np.ndarray, p: np.ndarray) -> float:
    if np.std(t) == 0 or np.std(p) == 0:
        return 1.0 if np.allclose(t, p) else 0.0
    r = float(np.corrcoef(t, p)[0, 1])
    return r if np.isfinite(r) else 0.0


def cell_score(t: np.ndarray, p: np.ndarray) -> float:
    corr = float(np.clip(_pearson(t, p), 0.0, 1.0))
    sd = float(np.std(t)); rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    cal = (1.0 if rmse <= 1e-12 else 0.0) if sd <= 1e-12 else float(np.clip(1.0 - rmse / sd, 0.0, 1.0))
    return 0.5 * corr + 0.5 * cal


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    if list(submission.columns) != ["id", "value"]:
        raise InvalidSubmissionError(f"Submission columns must be exactly ['id', 'value']; got {list(submission.columns)}")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(f"Submission has {len(submission)} rows; expected {len(answers)}")
    if submission["id"].astype(str).duplicated().any():
        raise InvalidSubmissionError("Duplicate ids in submission")
    if submission["id"].astype(str).tolist() != answers["id"].astype(str).tolist():
        raise InvalidSubmissionError("Submission ids must exactly match sample_submission ids, in order")
    try:
        p = pd.to_numeric(submission["value"], errors="raise").to_numpy(float)
    except Exception as exc:
        raise InvalidSubmissionError(f"value must be numeric: {exc}")
    if not np.isfinite(p).all():
        raise InvalidSubmissionError("value contains missing or non-finite entries")
    t = answers["value"].to_numpy(float)
    src = answers["source"].astype(str).to_numpy(); tail = answers["tail"].astype(str).to_numpy()
    scores = []
    for s, k in CELLS:
        m = (src == s) & (tail == k)
        if m.sum() < 20:
            raise InvalidSubmissionError(f"answers cell {s}/{k} has too few rows")
        scores.append(cell_score(t[m], p[m]))
    return float(np.mean(scores))
