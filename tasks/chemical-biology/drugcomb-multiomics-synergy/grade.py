"""drugcomb-multiomics-synergy grader, corrected_v4 (2026-10-16), self-contained.

Leave-drug-out release. Metric unchanged in definition from corrected_v2: within each tissue, truth and prediction
are projected onto the orthogonal complement of span{cell-line indicators, reference_zip (test.csv)}; the tissue score
is the Pearson correlation of the projected vectors; the final score is the mean over tissues with >= 20 rows, then
clip(., 0, 1) (v4 chance-zero scale). A prediction inside the span (constant, reference, per-cell offsets) scores 0.
v5 (2026-10-17): in-span degeneracy test relative to ||y_pred|| only (scale invariance for any positive factor).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TASK_KEY = "chemical-biology/drugcomb-multiomics-synergy"
METRIC_INFO = {
    "name": "per_tissue_mean_partial_pearson_beyond_additive_reference_clip",
    "direction": "higher_is_better",
    "normalization": "already_in_0_1",
}
MIN_ROWS = 20


class InvalidSubmissionError(Exception):
    pass


def _numeric(df: pd.DataFrame, col: str, label: str) -> np.ndarray:
    try:
        v = pd.to_numeric(df[col], errors="raise").to_numpy(float)
    except Exception as exc:
        raise InvalidSubmissionError(f"{label} column {col!r} must be numeric: {exc}")
    if not np.isfinite(v).all():
        raise InvalidSubmissionError(f"{label} column {col!r} contains missing or non-finite values")
    return v


def raw_score(test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ref = test["reference_zip"].to_numpy(float)
    scores = []
    for _, idx in test.groupby("tissue", sort=True).indices.items():
        idx = np.asarray(idx, dtype=int)
        if len(idx) < MIN_ROWS:
            continue
        codes = pd.factorize(test["cell_line"].astype(str).to_numpy()[idx])[0]
        Z = np.column_stack([np.eye(codes.max() + 1)[codes], ref[idx]])
        Q, Rm = np.linalg.qr(Z)
        Q = Q[:, np.abs(np.diag(Rm)) > 1e-10 * max(1.0, np.abs(Rm).max())]
        a = y_true[idx] - Q @ (Q.T @ y_true[idx])
        b = y_pred[idx] - Q @ (Q.T @ y_pred[idx])
        # v5: the in-span test is relative to the prediction's own norm only, so a positive rescaling of a
        # prediction never changes its score (v4 compared against ||y_pred|| + ||ref|| and zeroed y_true * 1e-8).
        scale = np.linalg.norm(y_pred[idx])
        if np.linalg.norm(b) <= 1e-9 * scale + 1e-300 or np.linalg.norm(a) == 0:
            scores.append(0.0)
        else:
            scores.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    if not scores:
        raise InvalidSubmissionError("no tissue with enough rows")
    return float(np.mean(scores))


def grade(submission: pd.DataFrame, answers: pd.DataFrame, public_dir=None, answers_path=None, **kwargs) -> float:
    if list(submission.columns) != ["id", "synergy_zip"]:
        raise InvalidSubmissionError(f"Submission columns must be exactly ['id', 'synergy_zip']; got {list(submission.columns)}")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(f"Submission has {len(submission)} rows; expected {len(answers)}")
    sid = submission["id"].astype(str)
    if sid.duplicated().any():
        raise InvalidSubmissionError("Duplicate ids in submission")
    if sid.tolist() != answers["id"].astype(str).tolist():
        raise InvalidSubmissionError("Submission ids must exactly match sample_submission ids, in order")
    if public_dir is None:
        raise InvalidSubmissionError("public_dir is required (test.csv holds tissue, cell_line, reference_zip)")
    test = pd.read_csv(Path(public_dir) / "test.csv", usecols=["id", "cell_line", "tissue", "reference_zip"])
    if test["id"].astype(str).tolist() != answers["id"].astype(str).tolist():
        raise InvalidSubmissionError("test.csv ids do not match answers")
    y_pred = _numeric(submission, "synergy_zip", "submission")
    y_true = _numeric(answers, "synergy_zip", "answers")
    return float(np.clip(raw_score(test, y_true, y_pred), 0.0, 1.0))
