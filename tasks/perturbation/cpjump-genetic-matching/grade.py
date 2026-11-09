from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tasks.utils import METRIC_INFO_BY_TASK, grade_task


TASK_KEY = "perturbation/cpjump-genetic-matching"
METRIC_INFO = METRIC_INFO_BY_TASK[TASK_KEY]


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    """Primary score (unchanged): mean per-compound ROC-AUC over the 65 candidates."""
    kwargs.pop("task_key", None)
    return grade_task(TASK_KEY, submission, answers, **kwargs)


def secondary_metrics(submission: pd.DataFrame, answers: pd.DataFrame, public_dir: Path | str) -> dict[str, float]:
    """v5 diagnostics reported next to the primary score (they do not enter it): Recall@1 (target ranked first,
    ties counted pessimistically) and mean reciprocal rank of the annotated target, per compound then averaged."""
    test = pd.read_csv(Path(public_dir) / "test.csv")
    df = test[["id", "query_id"]].merge(answers[["id", "interaction"]], on="id").merge(
        submission[["id", "interaction"]].rename(columns={"interaction": "score"}), on="id")
    top1, rr = [], []
    for _, g in df.groupby("query_id"):
        s = g.score.to_numpy(float); t = s[g.interaction.to_numpy() == 1][0]
        rank = int((s >= t).sum())  # pessimistic: ties ranked ahead of the target
        top1.append(rank == 1); rr.append(1.0 / rank)
    return {"recall_at_1": float(np.mean(top1)), "mrr": float(np.mean(rr))}
