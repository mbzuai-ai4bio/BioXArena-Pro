from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tasks import nb_grading as nb
from tasks.nb_grading import metric_info_from_contract

# v5 (2026-10-17): task-local copy of nb_grading.grade_from_contract (File-Manifest branch) with a WEIGHTED reduction
# over instances (weights in contract.INSTANCE_PLAN: Swiss-Prot MF/BP/CC 0.8/3 each, neXtProt MF/BP 0.1 each).
# Each instance is still validated/converted by the shared nb._reduce_primary on a one-item plan.
METRIC_INFO = metric_info_from_contract(__file__)


def grade(submission: pd.DataFrame, answers: pd.DataFrame, *, grade_path, answers_path, public_dir, **_: Any) -> float:
    task_dir = Path(grade_path).resolve().parent
    contract = nb._load_contract(task_dir)
    assert not contract.ROWWISE
    ids = nb._validate_ids(submission, answers)
    private_dir = Path(answers_path).resolve().parent
    nb._validate_answer_paths(answers, ids, private_dir)
    with tempfile.TemporaryDirectory(prefix="bioxarena_pro_nb_") as temp_dir:
        working_dir = Path(temp_dir); native_output = working_dir / "output"
        expected_columns = ["id", "prediction_file"]
        if submission.columns.tolist() != expected_columns:
            raise nb.InvalidNBSubmissionError("File-Manifest submission.csv columns must exactly match: " + ", ".join(expected_columns))
        for instance, artifact in zip(ids, submission["prediction_file"].astype(str)):
            nb._safe_extract_zip(Path(artifact), native_output / instance)
        env = dict(os.environ, OUTPUT_DIR=str(native_output), GROUND_TRUTH_DIR=str(private_dir / "answers"),
                   DATA_DIR=str(Path(public_dir).resolve() / "data"))
        proc = subprocess.run([sys.executable, str(task_dir / "evaluator.py")], cwd=working_dir, env=env,
                              capture_output=True, text=True, timeout=7200, check=False)
        score_path = working_dir / "score.json"
        if proc.returncode != 0 or not score_path.is_file():
            raise RuntimeError(f"Original NB evaluator failed with exit code {proc.returncode}. "
                               f"stdout tail: {proc.stdout[-1000:]} stderr tail: {proc.stderr[-1000:]}")
        results = json.loads(score_path.read_text(encoding="utf-8"))
    plan = list(contract.INSTANCE_PLAN)
    vals = [nb._reduce_primary(results, [item]) for item in plan]
    w = np.array([float(item["weight"]) for item in plan])
    assert abs(w.sum() - 1.0) < 1e-9
    return float(np.dot(w, vals))
