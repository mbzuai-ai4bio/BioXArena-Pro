"""sciplex-distribution grader, corrected_v4 revision b (2026-10-16).

Conditions are (cell line, drug, dose, exposure) with the two biological replicates pooled; only responsive test
conditions are graded (>= 50 of the 1,000 genes at BH-FDR < 0.05 vs pooled vehicle; this is a property of the test set,
fixed at build time). Prediction and answer are (3, 1000) arrays: row 0 mean shift, row 1 log SD ratio, row 2 median
shift, all treated minus pooled vehicle of the same cell line and exposure.

Per test condition, after subtracting the training-only baseline of its (cell_line, dose_nM, exposure_hours) context
(the mean target of the RESPONSIVE training conditions of that context, i.e. the same population as the test set):
  0.50 * Pearson(mean-shift residual) on the condition's top-50 DEGs (ranked by |Welch t| of the reference vs vehicle,
         stored privately with the answer)
  0.25 * Pearson(log-SD-ratio residual) over the 1,000 genes
  0.25 * Pearson(median-shift residual) over the 1,000 genes
A residual that is constant (prediction equal to the baseline up to float noise) scores 0 for that component.
The raw score s is the mean over test conditions.
v5b (2026-10-17): zero-offset correction. s0 = the same score for the all-zero prediction (no response vs vehicle) on the
same answers (deterministic, private answers only); the grader returns (s - |s0|) / (1 - |s0|), clipped to [0, 1] by the
evaluator. Pearson is scale-invariant, so any drug-blind amplitude guess c x baseline (c != 1) has raw score +s0 (c < 1)
or -s0 (c > 1); subtracting |s0| makes all of them (zeros included) score exactly 0 and leaves a perfect prediction at 1.
Task-local implementation; it only imports unchanged shared helpers.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from tasks.utils import (
    InvalidSubmissionError,
    load_npz_y,
    matrix_axis_mean_pearson,
    read_test,
    require_output_dir,
    require_private_dir,
    require_public_dir,
    resolve_relative_path,
    validate_submission_frame,
    validate_test_alignment,
)

TASK_KEY = "perturbation/sciplex-distribution"
METRIC_INFO = {
    "name": "mean_context_residual_deg_meanshift_logsd_median_pearson",
    "direction": "higher_is_better",
    "normalization": "map_minus1_1_to_0_1",  # v4: evaluator reports clip(r, 0, 1)
}
SHAPE = (3, 1000)
N_DEG = 50
WEIGHTS = (0.5, 0.25, 0.25)
CONTEXT_COLS = ["cell_line", "dose_nM", "exposure_hours"]
FILE_COL = "prediction_file"
TRAIN_FILE_COL = "distribution_file"


def _baselines(public_dir) -> dict:
    train = pd.read_csv(public_dir / "train.csv")
    need = set(CONTEXT_COLS + [TRAIN_FILE_COL, "responsive"])
    if not need.issubset(train.columns):
        raise InvalidSubmissionError("train.csv lacks the context / responsive / distribution_file columns.")
    resp = train["responsive"].astype(str).str.lower().isin(["true", "1"])
    sums, counts = {}, {}
    for _, row in train[resp.values].iterrows():
        key = tuple(row[c] for c in CONTEXT_COLS)
        y = np.asarray(load_npz_y(resolve_relative_path(public_dir, row[TRAIN_FILE_COL], "training target")), dtype=np.float64)
        if y.shape != SHAPE or not np.isfinite(y).all():
            raise InvalidSubmissionError("Training target matrix is invalid.")
        sums[key] = y if key not in sums else sums[key] + y
        counts[key] = counts.get(key, 0) + 1
    return {k: v / counts[k] for k, v in sums.items()}


def score_condition(pred: np.ndarray, true: np.ndarray, t: np.ndarray, base: np.ndarray) -> float:
    pr, tr = pred - base, true - base
    deg = np.argsort(-np.abs(t), kind="stable")[:N_DEG]
    c_mean = matrix_axis_mean_pearson(tr[0:1, deg], pr[0:1, deg], axis=1, pred_reference=pred[0:1, deg])
    c_sd = matrix_axis_mean_pearson(tr[1:2], pr[1:2], axis=1, pred_reference=pred[1:2])
    c_med = matrix_axis_mean_pearson(tr[2:3], pr[2:3], axis=1, pred_reference=pred[2:3])
    return WEIGHTS[0] * c_mean + WEIGHTS[1] * c_sd + WEIGHTS[2] * c_med


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    answers_path = kwargs.get("answers_path")
    submission, answers = validate_submission_frame(submission, answers, ["id", FILE_COL])
    test = validate_test_alignment(read_test(kwargs.get("public_dir"), answers_path), answers)
    public_dir = require_public_dir(kwargs.get("public_dir"), answers_path)
    private_dir = require_private_dir(answers_path)
    output_dir = require_output_dir(kwargs.get("output_dir"))
    if not set(CONTEXT_COLS).issubset(test.columns):
        raise InvalidSubmissionError("test.csv lacks the context columns.")
    baselines = _baselines(public_dir)
    scores, zero_scores = [], []
    for i in range(len(answers)):
        key = tuple(test.at[i, c] for c in CONTEXT_COLS)
        if key not in baselines:
            raise InvalidSubmissionError(f"No training baseline for test context {key!r}.")
        pred = np.asarray(load_npz_y(resolve_relative_path(output_dir, submission.at[i, FILE_COL], "prediction")))
        if pred.shape != SHAPE or not np.issubdtype(pred.dtype, np.number) or np.issubdtype(pred.dtype, np.complexfloating):
            raise InvalidSubmissionError(f"Prediction for row {i} must be a real numeric array of shape {SHAPE}, got {pred.shape} {pred.dtype}.")
        pred = pred.astype(np.float64)
        if not np.isfinite(pred).all():
            raise InvalidSubmissionError(f"Prediction matrix for row {i} contains NaN or inf.")
        answer_file = resolve_relative_path(private_dir, answers.at[i, FILE_COL], "answer")
        try:
            with np.load(answer_file, allow_pickle=False) as data:
                true = np.asarray(data["y"], dtype=np.float64)
                t = np.asarray(data["t"], dtype=np.float64)
        except Exception as exc:
            raise InvalidSubmissionError(f"Unable to read answer NPZ {answer_file}: {exc}")
        s = score_condition(pred, true, t, baselines[key])
        if not np.isfinite(s):
            raise InvalidSubmissionError(f"Non-finite score for row {i}.")
        scores.append(float(s))
        zero_scores.append(float(score_condition(np.zeros(SHAPE), true, t, baselines[key])))
    if not scores:
        raise InvalidSubmissionError("No test conditions to score.")
    s, s0 = float(np.mean(scores)), abs(float(np.mean(zero_scores)))
    return float((s - s0) / (1.0 - s0))
