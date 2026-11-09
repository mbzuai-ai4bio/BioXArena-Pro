"""Grader for structure/skempi-ppi-ddg (BioXArena-Pro v5, 2026-10-18).

Primary metric: mean over test complexes of the Spearman rank correlation between predicted and measured ddG
inside the complex. A complex enters the mean when it has >= MIN_ROWS test mutations and a nonzero SD of the
measured ddG (fixed by the hidden labels, not by the submission). A prediction that is constant inside a complex
scores 0 for that complex. The score is a correlation in [-1, 1]; the BioXArena-Pro normaliser maps it to
clip(r, 0, 1) (normalization "map_minus1_1_to_0_1").

Secondary metrics (returned by `grade_details`): pooled Pearson, pooled Spearman, RMSE, n complexes evaluated.
Self-contained (no shared helpers) so that shared-code edits cannot change it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


class InvalidSubmissionError(Exception):
    pass


TASK_KEY = "structure/skempi-ppi-ddg"
METRIC_INFO = {
    "name": "per_complex_mean_spearman_ddg",
    "direction": "higher_is_better",
    "normalization": "map_minus1_1_to_0_1",
}
ID_COL, PRED_COL, GROUP_COL = "id", "ddg", "complex_id"
MIN_ROWS = 5


def _validate(submission: pd.DataFrame, answers: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if list(submission.columns) != [ID_COL, PRED_COL]:
        raise InvalidSubmissionError(f"Submission columns must be exactly ['{ID_COL}', '{PRED_COL}'], got {list(submission.columns)}.")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(f"Submission has {len(submission)} rows, expected {len(answers)}.")
    sid = submission[ID_COL].astype(str).to_numpy(); aid = answers[ID_COL].astype(str).to_numpy()
    if pd.Series(sid).duplicated().any():
        raise InvalidSubmissionError("Duplicated ids in submission.")
    if not np.array_equal(sid, aid):
        missing = set(aid) - set(sid)
        raise InvalidSubmissionError(
            f"Submission ids must match sample_submission.csv in the same order ({len(missing)} missing/unknown)."
            if missing else "Submission ids are not in the order of sample_submission.csv.")
    raw = submission[PRED_COL]
    pred = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
    if np.isnan(pred).any() or not np.isfinite(pred).all():
        raise InvalidSubmissionError("Every ddg must be a finite number (no empty, non-numeric, nan or inf values).")
    true = pd.to_numeric(answers[PRED_COL], errors="raise").to_numpy(dtype=float)
    groups = answers[GROUP_COL].astype(str).to_numpy()
    return pred, true, groups


def _spearman(t: np.ndarray, p: np.ndarray) -> float:
    if np.std(p) == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = spearmanr(t, p).statistic
    return float(r) if np.isfinite(r) else 0.0


def grade_details(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    pred, true, groups = _validate(submission, answers)
    per = {}
    for g in sorted(set(groups)):
        m = groups == g
        if m.sum() < MIN_ROWS or np.std(true[m]) == 0:
            continue
        per[g] = _spearman(true[m], pred[m])
    if not per:
        raise InvalidSubmissionError("No evaluable complex (answers malformed).")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = float(pearsonr(true, pred).statistic) if np.std(pred) > 0 else 0.0
        ps = float(spearmanr(true, pred).statistic) if np.std(pred) > 0 else 0.0
    return {
        "per_complex_mean_spearman": float(np.mean(list(per.values()))),
        "n_complexes_evaluated": len(per),
        "n_complexes_excluded": len(set(groups)) - len(per),
        "pooled_pearson": pp,
        "pooled_spearman": ps,
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "per_complex": per,
    }


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    return grade_details(submission, answers)["per_complex_mean_spearman"]
