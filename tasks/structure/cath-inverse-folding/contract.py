"""Self-contained evaluator for BioXArena-Pro task `structure/cath-inverse-folding`.

    from evaluate import evaluate
    score = evaluate(my_submission, ground_truth, output_dir="...")

This task came from NatureBench and is scored by the adjacent `evaluator.py`. Hidden
answers are supplied from 2_BioXArena-Pro-Data-Private at evaluation time. The adapter uses
the primary-metric contract consolidated in the public `description.md`: each instance's
primary metric is mapped to [0, 1], then instances are averaged with equal weight. The
scientific metric itself remains in the original evaluator so its implementation cannot
drift from the benchmark scorer.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TASK_KEY = "structure/cath-inverse-folding"
METRIC = {'name': 'per_instance_mean_recovery_rate', 'direction': 'higher_is_better', 'normalization': 'already_in_0_1'}
# v5: cath_4_2 and cath_4_3 test sets merged into one sequence-deduplicated `cath` instance (they shared 170 identical
# sequences, so CATH was counted twice); task score = mean(cath, pdb2022).
INSTANCE_PLAN = [{'instance': 'cath', 'metric': 'recovery_rate', 'transform': 'percent'}, {'instance': 'pdb2022', 'metric': 'recovery_rate', 'transform': 'percent'}]
KIND = "nb_evaluator_dirs"
ROWWISE = None

_HERE = Path(__file__).resolve().parent
EVALUATOR = _HERE / "evaluator.py"


def _to_unit(value: Any, transform: str) -> float:
    if value is None:
        raise RuntimeError("NB evaluator omitted the selected primary metric")
    value = float(value)
    if not np.isfinite(value):
        raise RuntimeError(f"NB evaluator returned a non-finite primary metric: {value!r}")
    if transform == "map_minus1_1":
        # v4 chance-zero scale: clip(r, 0, 1) instead of (r + 1) / 2
        return float(np.clip(value, 0.0, 1.0))
    if transform == "percent":
        return float(np.clip(value / 100.0, 0.0, 1.0))
    return float(np.clip(value, 0.0, 1.0))


def _reduce(results: dict) -> float:
    if not isinstance(results, dict):
        raise RuntimeError("NB evaluator score.json must contain a JSON object")
    scores = []
    for item in INSTANCE_PLAN:
        instance = str(item["instance"])
        entry = results.get(instance)
        if not isinstance(entry, dict):
            raise RuntimeError(f"NB evaluator omitted or malformed result for {instance!r}")
        error = entry.get("error")
        if error:
            error_text = str(error)
            if error_text.startswith("Validation:"):
                raise ValueError(
                    f"Invalid submission for {instance!r}: "
                    f"{error_text.removeprefix('Validation:').strip()}"
                )
            raise RuntimeError(f"NB evaluator failed internally for {instance!r}: {error_text}")
        value = entry.get(item["metric"])
        if value is None:
            lookup = {str(k).lower().replace(" ", ""): v for k, v in entry.items()}
            value = lookup.get(item["metric"].lower().replace(" ", ""))
        if value is None:
            raise RuntimeError(
                f"NB evaluator omitted primary metric {item['metric']!r} for {instance!r}"
            )
        scores.append(_to_unit(value, item["transform"]))
    if not scores:
        raise RuntimeError(f"no instances resolved for {TASK_KEY}")
    return float(np.mean(scores))


def _run(out_dir: Path, cwd: Path) -> dict:
    if not EVALUATOR.is_file():
        raise FileNotFoundError(f"missing NB evaluator: {EVALUATOR}")
    env = dict(os.environ, OUTPUT_DIR=str(out_dir))
    proc = subprocess.run([sys.executable, str(EVALUATOR)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=7200)
    score_path = cwd / "score.json"
    if not score_path.is_file():
        raise RuntimeError("evaluator produced no score.json\n"
                           f"stdout:\n{proc.stdout[-800:]}\nstderr:\n{proc.stderr[-800:]}")
    return json.loads(score_path.read_text())


def _as_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, (str, Path)):
        return pd.read_csv(value, low_memory=False)
    raise TypeError(f"expected a DataFrame or a path to a CSV, got {type(value).__name__}")


def evaluate(estimated: Any, ground_truth: Any, output_dir: Any = None, **kwargs: Any) -> float:
    """Score `estimated` against `ground_truth`.

    estimated: the submission frame (or CSV path).  Directory-based tasks carry a
    `prediction_dir` column; `output_dir` says what those relative paths resolve against.
    """
    submission = _as_frame(estimated)
    answers = _as_frame(ground_truth)
    if submission["id"].astype(str).tolist() != answers["id"].astype(str).tolist():
        raise ValueError("submission ids must match answers ids exactly, including order")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        if ROWWISE:
            frame = submission.copy()
            frame.columns = [ROWWISE["id_col"], *ROWWISE["pred_cols"]]
            inst = INSTANCE_PLAN[0]["instance"]
            (out / inst).mkdir(parents=True, exist_ok=True)
            frame.to_csv(out / inst / ROWWISE["output_file"], index=False)
        else:
            if "prediction_dir" not in submission.columns:
                raise ValueError("submission must carry a 'prediction_dir' column")
            root = Path(output_dir) if output_dir is not None else Path.cwd()
            for instance, rel in zip(answers["id"].astype(str), submission["prediction_dir"].astype(str)):
                src = (root / rel).resolve()
                if not src.is_dir():
                    raise FileNotFoundError(f"prediction_dir not found for {instance}: {src}")
                shutil.copytree(src, out / instance, dirs_exist_ok=True)
        return _reduce(_run(out, Path(tmp)))


def normalized_score(estimated: Any, ground_truth: Any, **kwargs: Any) -> float:
    """Already in [0, 1]; provided so every task exposes the same two entry points."""
    return max(0.0, min(1.0, evaluate(estimated, ground_truth, **kwargs)))
