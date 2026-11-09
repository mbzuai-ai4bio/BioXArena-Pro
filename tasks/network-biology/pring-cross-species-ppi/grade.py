from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tasks.utils import (
    InvalidSubmissionError,
    numeric_array,
    read_test,
    validate_submission_frame,
    validate_test_alignment,
)

# v5 (2026-10-17) used IPF node-balanced pair weights in the private `weight` column. Since v5.3 the test set gives every
# protein exactly 3 negatives per positive, all weights are 1, and protein offsets are removed by per-protein centring.
# v5.3 (2026-10-20): every protein of the test file has exactly three negative pairs per positive pair, so a per-protein
# offset in a submission carries no label information. Before the AUC, the submission of each species is rank-transformed
# to logits and two-way centred on the test pair graph (the per-protein additive offsets are removed, as in
# ppi-cross-species), so the score depends only on within-protein pair preferences and is the same whether or not a
# submission removes protein offsets itself.
TASK_KEY = 'network-biology/pring-cross-species-ppi'
METRIC_INFO = {
    "name": "per_source_mean_protein_centred_roc_auc",
    "direction": "higher_is_better",
    "normalization": "already_in_0_1",
}
SUB_COLS = ["id", "interaction"]


def per_protein_center(scores, a_ids, b_ids, iterations=50):
    from scipy.stats import rankdata
    s = np.asarray(scores, dtype=float)
    r = rankdata(s) / (len(s) + 1.0)
    s = np.log(r / (1 - r))
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(s)
    s = (s - s.mean()) / sd
    ids = pd.Index(np.unique(np.r_[a_ids, b_ids]))
    ia, ib = ids.get_indexer(a_ids), ids.get_indexer(b_ids)
    cnt = np.bincount(ia, minlength=len(ids)) + np.bincount(ib, minlength=len(ids))
    bias = np.zeros(len(ids))
    for _ in range(iterations):
        res = s - bias[ia] - bias[ib]
        bias += 0.5 * (np.bincount(ia, res, len(ids)) + np.bincount(ib, res, len(ids))) / cnt
    return s - bias[ia] - bias[ib]


def grade(submission: pd.DataFrame, answers: pd.DataFrame, public_dir=None, answers_path=None, **kwargs) -> float:
    if "weight" not in answers.columns:
        raise InvalidSubmissionError("answers.csv lacks the node-balancing `weight` column")
    submission, ans = validate_submission_frame(submission, answers[SUB_COLS], SUB_COLS)
    w_by_id = answers.set_index(answers.columns[0])["weight"]
    test = validate_test_alignment(read_test(public_dir, answers_path), ans)
    y = numeric_array(ans, "interaction", "answers")
    p = numeric_array(submission, "interaction", "submission")
    w = w_by_id.reindex(ans[ans.columns[0]].values).to_numpy(dtype=float)
    if not np.isfinite(w).all():
        raise InvalidSubmissionError("weight column misaligned with answers")
    scores = []
    for _, idx in test.groupby("source", sort=True).indices.items():
        idx = np.asarray(idx)
        keep = w[idx] > 0
        yy, pp, ww = y[idx][keep], p[idx][keep], w[idx][keep]
        pp = per_protein_center(pp, test["protein_1"].astype(str).values[idx][keep], test["protein_2"].astype(str).values[idx][keep])
        if len(np.unique(yy)) < 2:
            continue
        scores.append(float(roc_auc_score(yy, pp, sample_weight=ww)))
    if not scores:
        raise InvalidSubmissionError("no scorable species group")
    return float(np.mean(scores))
