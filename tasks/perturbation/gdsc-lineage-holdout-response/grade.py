from __future__ import annotations

import pandas as pd

from tasks.utils import (METRIC_INFO_BY_TASK, TASK_CONFIG, mean_or_raise, numeric_array, pearson_safe,
                         prepare_standard_task, read_test, validate_test_alignment)


TASK_KEY = 'perturbation/gdsc-lineage-holdout-response'
METRIC_INFO = METRIC_INFO_BY_TASK[TASK_KEY]
# v5: task-local nested grouped Pearson that enforces the documented minimum group size (>= 10 cell lines per
# (lineage, compound) group); the shared helper used a hard-coded minimum of 2.
MIN_ROWS = 10


def grade(submission: pd.DataFrame, answers: pd.DataFrame, *, public_dir=None, answers_path=None, **kwargs) -> float:
    config = TASK_CONFIG[TASK_KEY]
    submission, answers = prepare_standard_task(submission, answers, config)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    df = test[[config["outer_group_col"], config["inner_group_col"]]].copy()
    df["true"] = numeric_array(answers, config["target_col"], "answers")
    df["pred"] = numeric_array(submission, config["pred_col"], "submission")
    outer = []
    for _, og in df.groupby(config["outer_group_col"], sort=True):
        inner = [pearson_safe(g["true"].to_numpy(), g["pred"].to_numpy())
                 for _, g in og.groupby(config["inner_group_col"], sort=True) if len(g) >= MIN_ROWS]
        outer.append(mean_or_raise(inner, "nested target Pearson"))
    return mean_or_raise(outer, "nested source Pearson")
