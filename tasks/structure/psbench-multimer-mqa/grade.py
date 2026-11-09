"""psbench-multimer-mqa, data version corrected_v4 (2026-10-16).

Composite, self-contained grader (no shared-helper change):

    score = ( clip(A, 0, 1) + clip(B, 0, 1) ) / 2

    A = mean over the 193 test pools of Pearson(predicted, true TM-score) across the pool's 15 models
        -- "which of the 15 candidate models of this complex should I use?"
    B = mean over the pipeline groups with a true 5-model TM spread >= 0.02 of Pearson(predicted, true)
        across that pipeline's 5 models
        -- "which of my own predictor's 5 models should I use?"

Both components use the chance-zero scale of fix round v4: clip(r, 0, 1), so chance and anti-correlation
score 0.  A group whose prediction is constant contributes r = 0; nothing is skipped on the submission side.
Grouping comes from the hidden answers (columns `pool`, `pipeline`), which are also public in test.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tasks.utils import InvalidSubmissionError

TASK_KEY = "structure/psbench-multimer-mqa"
PRED_COL = "tmscore_mmalign"
MIN_GROUP_SPREAD = 0.02
METRIC_INFO = {
    "name": "mean_pool_pearson_and_within_pipeline_pearson",
    "direction": "higher_is_better",
    "normalization": "already_in_0_1",
}


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_true) <= 0.0 or np.std(y_pred) <= 0.0:
        return 0.0
    r = float(np.corrcoef(y_pred, y_true)[0, 1])
    return 0.0 if not np.isfinite(r) else r


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    for frame, label, cols in ((submission, "submission", ["id", PRED_COL]),
                               (answers, "answers", ["id", "pool", "pipeline", PRED_COL])):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise InvalidSubmissionError(f"{label} is missing column(s) {missing}")
    if submission["id"].duplicated().any():
        raise InvalidSubmissionError("submission contains duplicate ids")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(f"Submission has {len(submission)} rows; expected {len(answers)}")
    if set(submission["id"]) != set(answers["id"]):
        raise InvalidSubmissionError("submission ids do not match the expected ids")
    pred = pd.to_numeric(submission[PRED_COL], errors="coerce")
    if not np.isfinite(pred.to_numpy(dtype=float)).all():
        raise InvalidSubmissionError(f"submission column '{PRED_COL}' must be finite for every row")

    df = answers.merge(submission[["id", PRED_COL]].rename(columns={PRED_COL: "pred"}), on="id", validate="one_to_one")
    y = df[PRED_COL].astype(float).to_numpy()
    p = df["pred"].astype(float).to_numpy()

    pool_r = [_pearson(y[idx], p[idx]) for idx in
              (np.asarray(list(v), dtype=int) for v in df.groupby("pool", sort=True).indices.values())]
    grp_r = []
    for _, v in df.groupby(["pool", "pipeline"], sort=True).indices.items():
        idx = np.asarray(list(v), dtype=int)
        if y[idx].max() - y[idx].min() < MIN_GROUP_SPREAD:
            continue
        grp_r.append(_pearson(y[idx], p[idx]))
    if not pool_r or not grp_r:
        raise RuntimeError("no scorable groups in the answers")
    a = float(np.clip(np.mean(pool_r), 0.0, 1.0))
    b = float(np.clip(np.mean(grp_r), 0.0, 1.0))
    return float((a + b) / 2.0)
