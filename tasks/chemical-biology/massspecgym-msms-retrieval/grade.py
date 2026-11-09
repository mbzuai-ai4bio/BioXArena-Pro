from __future__ import annotations

"""massspecgym-msms-retrieval grader (corrected_v3, 2026-10-15).

Replaces chemcot-xl-molecule-counting. Each hidden query (one unknown compound, 1+ MS/MS spectra)
has a list of candidate structures; exactly one is correct. The submission gives a finite score per
candidate row (higher = more likely). Per query the reciprocal rank of the correct candidate is
computed with ties broken uniformly at random (expected value, so constant scores give exactly the
chance level). Score = mean reciprocal rank over queries (equal weight per query), in [0, 1].

Contract: columns id, score; ids unique and identical to the hidden candidate ids (no missing or
extra rows); every score finite. Violations raise (scored 0 by the evaluator).
"""

from typing import Any

import numpy as np
import pandas as pd


METRIC_INFO = {
    "name": "mean_reciprocal_rank",
    "direction": "higher_is_better",
    "normalization": "already_in_0_1",
}


class InvalidSubmissionError(ValueError):
    pass


def _expected_rr(greater: np.ndarray, ties: np.ndarray) -> np.ndarray:
    # correct candidate occupies a uniformly random position among ranks greater+1 .. greater+ties+1
    out = np.empty(len(greater), dtype=float)
    for i, (g, t) in enumerate(zip(greater, ties)):
        r = np.arange(g + 1, g + t + 2, dtype=float)
        out[i] = float(np.mean(1.0 / r))
    return out


def per_query_rr(submission: pd.DataFrame, answers: pd.DataFrame) -> pd.Series:
    if not {"id", "score"}.issubset(submission.columns):
        raise InvalidSubmissionError("submission must have columns: id, score")
    sub = submission[["id", "score"]].copy()
    sub["id"] = sub["id"].astype(str)
    if sub["id"].duplicated().any():
        raise InvalidSubmissionError("duplicate ids in submission")
    ans = answers[["id", "query_id", "label"]].copy()
    ans["id"] = ans["id"].astype(str)
    if len(sub) != len(ans) or set(sub["id"]) != set(ans["id"]):
        missing = len(set(ans["id"]) - set(sub["id"]))
        extra = len(set(sub["id"]) - set(ans["id"]))
        raise InvalidSubmissionError(f"submission ids must match the test ids exactly (missing {missing}, extra {extra})")
    score = pd.to_numeric(sub["score"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(score).all():
        raise InvalidSubmissionError("all scores must be finite numbers")
    sub["score"] = score
    m = ans.merge(sub, on="id", how="left", validate="one_to_one")
    lab = m["label"].astype(int).to_numpy()
    if (m.groupby("query_id")["label"].sum() != 1).any():
        raise RuntimeError("hidden answers must hold exactly one correct candidate per query")
    true_score = m.loc[lab == 1].set_index("query_id")["score"]
    ts = m["query_id"].map(true_score).to_numpy(dtype=float)
    m["gt"] = (m["score"].to_numpy() > ts).astype(int)
    m["eq"] = ((m["score"].to_numpy() == ts) & (lab == 0)).astype(int)
    agg = m.groupby("query_id")[["gt", "eq"]].sum()
    return pd.Series(_expected_rr(agg["gt"].to_numpy(), agg["eq"].to_numpy()), index=agg.index)


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs: Any) -> float:
    return float(per_query_rr(submission, answers).mean())
