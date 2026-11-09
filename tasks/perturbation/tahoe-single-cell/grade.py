"""tahoe-single-cell v5.2 (2026-10-20): the reference shift is now plate-specific. Release check: 95 of the 631 test
treatments were measured on 2+ plates, and the all-plate treatment_shift left the plate-specific (same-well) part of the
response in the residual; the held-out line shares that part with its training well-mates, so copying same-plate training
responses scored r = 0.54 on those treatments without any line-specific modelling. v5.2 subtracts
treatment_plate_shift[treatment|plate] (public; mean over the training lines on that plate of effect minus context
baseline) instead. For single-plate treatments the two shifts are identical, so their scores are unchanged.

Earlier notes (v5, 2026-10-17; metric identical to corrected_v4 except the shift): cold cell line.
v5: prediction arrays must be 1-D real vectors (2-D arrays are rejected instead of flattened).
v5b (2026-10-17): zero-offset correction. s0 = the same score for the all-zero prediction on the same answers
(deterministic, private answers only); the grader returns (s - |s0|) / (1 - |s0|) and the evaluator clips it to [0, 1].
Pearson is scale-invariant, so every drug-blind amplitude guess c x shift has raw score +s0 (c < 1) or -s0 (c > 1);
subtracting |s0| makes all of them (zeros included) score exactly 0, keeps chance-level predictions at 0 when s0 < 0,
and leaves a perfect prediction at 1.

Test rows are units of 10 cell lines that have no perturbation data in training. Their context baseline (mean training
effect of the plate x line) therefore cannot be published, so the grader removes each line's generic response on both
sides itself:

    r(row)      = y(row) - treatment_plate_shift[treatment(row)|plate(row)]   (public, training lines only; v5.2)
    r_c(row)    = r(row) - mean of r over the test rows of the same context (plate x cell line)

computed separately for the prediction and for the truth. Per test treatment, Pearson r between the flattened truth and
prediction residual matrices (held-out lines x 2,000 genes); the score is the mean over treatments, reported on the v4
chance-zero scale clip(r, 0, 1) by the evaluator. Copying the training-line treatment mean, adding any per-context
offset gives a constant residual and scores exactly 0. Predicting zeros does not (the line-wide response amplitude
survives the context centring), hence the v5b zero-offset correction above.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tasks.utils import (
    METRIC_INFO_BY_TASK, InvalidSubmissionError, load_npz_y, mean_or_raise, pearson_safe, read_test,
    require_columns, require_output_dir, require_private_dir, require_public_dir, resolve_relative_path,
    validate_submission_frame, validate_test_alignment,
)

TASK_KEY = "perturbation/tahoe-single-cell"
METRIC_INFO = METRIC_INFO_BY_TASK[TASK_KEY]
FILE_COL, GROUP_COL, CONTEXT_COL, PLATE_COL = "response_file", "treatment_id", "context_id", "batch_id"
SHIFT_FILE = "treatment_plate_shift.npz"


def shift_keys(test: pd.DataFrame) -> list:
    """v5.2: the reference shift of a unit is keyed by treatment and plate."""
    return [f"{t}|{b}" for t, b in zip(test[GROUP_COL].astype(str), test[PLATE_COL].astype(str))]
_PRED_REL_TOL = 1e-6   # residual spread <= this x RMS of the raw prediction rows -> constant prediction (r = 0)


def residual_matrices(test: pd.DataFrame, Y: np.ndarray, shifts: dict) -> np.ndarray:
    S = np.stack([shifts[k] for k in shift_keys(test)])
    Rm = Y - S
    out = np.empty_like(Rm)
    for _, pos in test.groupby(CONTEXT_COL, sort=True).indices.items():
        out[pos] = Rm[pos] - Rm[pos].mean(0, keepdims=True)
    return out


def score_matrices(test: pd.DataFrame, P: np.ndarray, T: np.ndarray, shifts: dict) -> dict:
    rp, rt = residual_matrices(test, P, shifts), residual_matrices(test, T, shifts)
    scores = {}
    for g, pos in test.groupby(GROUP_COL, sort=True).indices.items():
        a, b = rt[pos].ravel(), rp[pos].ravel()
        ref = float(np.sqrt(np.mean(np.square(P[pos])))) + 1e-12
        scores[g] = 0.0 if float(np.std(b)) <= _PRED_REL_TOL * ref else pearson_safe(a, b)
    return scores


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    public_dir, answers_path = kwargs.get("public_dir"), kwargs.get("answers_path")
    submission, answers = validate_submission_frame(submission, answers)
    public_dir = require_public_dir(public_dir, answers_path)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    require_columns(test, [GROUP_COL, CONTEXT_COL, PLATE_COL], "test.csv")
    private_dir = require_private_dir(answers_path)
    output_dir = require_output_dir(kwargs.get("output_dir"))
    shift_path = public_dir / SHIFT_FILE
    if not shift_path.exists():
        raise InvalidSubmissionError(f"Missing treatment x plate shift archive: {shift_path}")
    with np.load(shift_path, allow_pickle=False) as z:
        shifts = {k: np.asarray(z[k], dtype=np.float64) for k in z.files}
    missing = set(shift_keys(test)) - set(shifts)
    if missing:
        raise InvalidSubmissionError(f"Missing treatment x plate shift for {len(missing)} test treatment-plate pairs.")
    G = len(next(iter(shifts.values())))
    P = np.empty((len(test), G)); T = np.empty((len(test), G))
    for i in range(len(test)):
        p = np.asarray(load_npz_y(resolve_relative_path(output_dir, submission.at[i, FILE_COL], "prediction")))
        if p.ndim != 1:  # v5: a (1, 2000) / (2000, 1) / (40, 50) array used to be flattened silently
            raise InvalidSubmissionError(f"Prediction for row {i} must be a 1-D vector of length {G}; got shape {p.shape}.")
        if not np.issubdtype(p.dtype, np.number) or np.issubdtype(p.dtype, np.complexfloating):
            raise InvalidSubmissionError(f"Prediction for row {i} must be real numeric; got dtype {p.dtype}.")
        p = p.astype(np.float64)
        t = np.asarray(load_npz_y(resolve_relative_path(private_dir, answers.at[i, FILE_COL], "answer")), dtype=np.float64).reshape(-1)
        if p.shape != (G,) or t.shape != (G,):
            raise InvalidSubmissionError(f"Response vector shape mismatch for row {i}: pred {p.shape}, true {t.shape}, expected ({G},).")
        if not np.isfinite(p).all():
            raise InvalidSubmissionError(f"Prediction for row {i} contains NaN or inf.")
        P[i], T[i] = p, t
    s = mean_or_raise(list(score_matrices(test, P, T, shifts).values()), "per-treatment cold-line residual Pearson")
    s0 = zero_score(test, T, shifts)
    return float((s - s0) / (1.0 - s0))


def zero_score(test: pd.DataFrame, T: np.ndarray, shifts: dict) -> float:
    """v5b: |raw score| of the all-zero prediction on these answers (deterministic)."""
    s0 = mean_or_raise(list(score_matrices(test, np.zeros_like(T), T, shifts).values()), "all-zero reference score")
    return abs(float(s0))
