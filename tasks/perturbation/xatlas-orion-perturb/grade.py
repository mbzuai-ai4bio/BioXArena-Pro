"""xatlas-orion-perturb grader, v5 (2026-10-17). Task-local; imports only unchanged shared helpers.

Per test row (scoring genes of its line minus the private cis/target exclusions):
  a = truth - baseline, b = prediction - baseline, both projected onto the orthogonal complement of
  span{1, baseline, axis}, where axis = the line's dominant shared residual direction (first right-singular vector
  of the unit-normalised, {1, baseline}-projected effects of the line's reproducible TRAINING knock-downs on the
  scoring genes; shipped publicly in residual_axis.npz, key <source>). row score = Pearson(a, b).
Row scores are averaged per line, then over lines; the evaluator reports clip(r, 0, 1).
v5 change vs v4: the axis was added to the projected span (v4 projected out span{1, baseline} only), so the score no
longer rewards getting the sign along one generic program right, and constant vectors score ~0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tasks.utils import (
    METRIC_INFO_BY_TASK,
    InvalidSubmissionError,
    load_npz_y,
    read_test,
    require_columns,
    require_output_dir,
    require_private_dir,
    require_public_dir,
    resolve_relative_path,
    strict_group_mean_from_row_scores,
    validate_submission_frame,
    validate_test_alignment,
)

TASK_KEY = "perturbation/xatlas-orion-perturb"
METRIC_INFO = METRIC_INFO_BY_TASK[TASK_KEY]
FILE_COL = "response_file"
GROUP_COL = "source"
CONTEXT_COL = "context_id"
MAX_ABS = 1e3
N_GENES = 5000


def _project_out(vec, basis):
    return vec - basis @ (basis.T @ vec)


def grade(submission: pd.DataFrame, answers: pd.DataFrame, *, public_dir=None, answers_path=None, output_dir=None,
          **_) -> float:
    submission, answers = validate_submission_frame(submission, answers)
    public_dir = require_public_dir(public_dir, answers_path)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    require_columns(test, [GROUP_COL, CONTEXT_COL], "test.csv")
    private_dir = require_private_dir(answers_path)
    output_dir = require_output_dir(output_dir)
    paths = {k: public_dir / f for k, f in [("baseline", "context_baseline.npz"), ("idx", "score_gene_indices.npz"),
                                             ("axis", "residual_axis.npz")]}
    paths["excl"] = private_dir / "score_exclusions.npz"
    for p in paths.values():
        if not Path(p).exists():
            raise InvalidSubmissionError(f"Missing grading archive: {p}")
    excl = np.load(paths["excl"], allow_pickle=False)
    row_scores = []
    with np.load(paths["baseline"], allow_pickle=False) as baselines, \
            np.load(paths["idx"], allow_pickle=False) as score_indices, \
            np.load(paths["axis"], allow_pickle=False) as axes:
        for i in range(len(answers)):
            rid = str(test.at[i, "id"]); source = str(test.at[i, GROUP_COL]); context_id = str(test.at[i, CONTEXT_COL])
            key = f"{source}_indices"
            if context_id not in baselines or key not in score_indices or source not in axes:
                raise InvalidSubmissionError(f"Missing residual metadata for source={source!r}.")
            baseline = np.asarray(baselines[context_id], dtype=np.float64)
            axis = np.asarray(axes[source], dtype=np.float64)
            indices = np.asarray(score_indices[key], dtype=np.int64)
            if rid in excl:
                indices = np.setdiff1d(indices, np.asarray(excl[rid], dtype=np.int64))
            pred = np.asarray(load_npz_y(resolve_relative_path(output_dir, submission.at[i, FILE_COL], "prediction")),
                              dtype=np.float64)
            true = np.asarray(load_npz_y(resolve_relative_path(private_dir, answers.at[i, FILE_COL], "answer")),
                              dtype=np.float64)
            if pred.ndim != 1 or pred.shape != (N_GENES,) or true.shape != pred.shape or baseline.shape != pred.shape:
                raise InvalidSubmissionError(f"Response vector for row {i} must have shape ({N_GENES},); got {pred.shape}.")
            if not np.isfinite(pred).all():
                raise InvalidSubmissionError(f"Prediction for row {i} contains non-finite values.")
            if float(np.max(np.abs(pred))) > MAX_ABS:
                raise InvalidSubmissionError(f"Prediction for row {i} has |y| > {MAX_ABS:g}; use the scale of train_y.")
            if len(indices) < 4:
                raise InvalidSubmissionError(f"Too few scoring genes for row {i}.")
            Q, Rm = np.linalg.qr(np.stack([np.ones(len(indices)), baseline[indices], axis[indices]], axis=1))
            Q = Q[:, np.abs(np.diag(Rm)) > 1e-10 * max(1.0, np.abs(Rm).max())]
            a = _project_out(true[indices] - baseline[indices], Q)
            b = _project_out(pred[indices] - baseline[indices], Q)
            # float32 rounding of a prediction inside the span leaves ~1e-7 relative noise -> counts as inside (0)
            scale = np.linalg.norm(pred[indices]) + np.linalg.norm(baseline[indices])
            if np.linalg.norm(b) <= 1e-4 * scale + 1e-12 or np.linalg.norm(a) == 0:
                row_scores.append(0.0)
                continue
            with np.errstate(all="ignore"):
                v = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            row_scores.append(float(np.clip(v, -1.0, 1.0)) if np.isfinite(v) else 0.0)
    return strict_group_mean_from_row_scores(test, row_scores, GROUP_COL)
