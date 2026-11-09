from __future__ import annotations

import pandas as pd

from tasks.utils import METRIC_INFO_BY_TASK, InvalidSubmissionError, grade_task


TASK_KEY = 'sequence-genomics/sirna-knockdown-efficacy'
METRIC_INFO = METRIC_INFO_BY_TASK[TASK_KEY]

PRED_COL = "inhibition"


def _check_range(submission: pd.DataFrame) -> None:
    """v5: enforce the documented output contract `inhibition` in [0, 1].

    The shared numeric path only required finite values, so submissions with out-of-range
    predictions (e.g. -1 or 2) were silently graded even though description.md promises the
    range. Pearson correlation is scale invariant, so this does not change the score of any
    in-range submission; it only rejects submissions that break the stated contract.
    """
    if PRED_COL not in submission.columns:
        return  # the shared path raises the canonical column/order error
    values = pd.to_numeric(submission[PRED_COL], errors="coerce")
    if values.isna().any():
        return  # missing / non-numeric values are reported by the shared path
    below, above = float(values.min()), float(values.max())
    if below < 0.0 or above > 1.0:
        raise InvalidSubmissionError(
            f"column '{PRED_COL}' must stay within [0, 1] as documented in description.md; "
            f"observed range [{below:.6g}, {above:.6g}]"
        )


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    kwargs.pop("task_key", None)
    _check_range(submission)
    return grade_task(TASK_KEY, submission, answers, **kwargs)
