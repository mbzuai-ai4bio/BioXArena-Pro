"""mammovqa-external grader, v5 (2026-10-17), task-local wrapper.

Metric unchanged (mean over question_topic of macro-F1; native scale). v5 adds per-row validation: every answer, after
stripping surrounding whitespace, must be one of the options listed for that row in test.csv (`options`, separated by
" | "). NaN, blank or non-option answers make the submission invalid instead of being scored as wrong.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tasks.utils import METRIC_INFO_BY_TASK, InvalidSubmissionError, grade_task

TASK_KEY = "imaging/mammovqa-external"
METRIC_INFO = METRIC_INFO_BY_TASK[TASK_KEY]


def _validate_options(submission: pd.DataFrame, public_dir) -> None:
    if public_dir is None:
        raise InvalidSubmissionError("public_dir is required (test.csv holds the per-row options)")
    test = pd.read_csv(Path(public_dir) / "test.csv", usecols=["id", "options"])
    if "answer" not in submission.columns or "id" not in submission.columns:
        return  # column errors are reported by the shared validator
    sub = submission[["id", "answer"]].copy()
    if sub["answer"].isna().any():
        raise InvalidSubmissionError("answer contains missing values")
    m = test.merge(sub.assign(id=sub["id"].astype(str)), on="id", how="left")
    bad = []
    for qid, opts, ans in zip(m["id"], m["options"], m["answer"]):
        if pd.isna(ans):
            continue  # missing ids are reported by the shared validator
        choices = {o.strip() for o in str(opts).split("|")}
        if str(ans).strip() not in choices:
            bad.append(qid)
    if bad:
        raise InvalidSubmissionError(f"{len(bad)} answers are not one of their row's options (e.g. {bad[:3]})")


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    kwargs.pop("task_key", None)
    _validate_options(submission, kwargs.get("public_dir"))
    return grade_task(TASK_KEY, submission, answers, **kwargs)
