from __future__ import annotations

"""kano-molecular-property-prediction grader (corrected_v2, round v2b, 2026-10-14; v5 2026-10-17).

Same per-instance metrics as before (AUPRC gain over prevalence for classification, clipped R^2
for regression, equal-instance mean), but only targets listed with scored=1 in the private
scored_targets.csv count. A classification target is scored only when the test set holds at
least 20 labelled molecules of each class (v5; was 10); MUV and (v5) ClinTox are no longer part of
the task (12 instances). Prediction files must still contain every target column (all values
finite; probabilities in [0, 1]). v5 (2026-10-17): qm7 rows with inconsistent labels or radical
SMILES were removed from the data; the grading code itself is unchanged.
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tasks.native_manifest_metrics import (
    CLASSIFICATION_INSTANCES,
    _finite_numeric_frame,
    _mean_target_auprc_gain,
    _mean_target_r2,
    _prediction_roots,
)
from tasks.nb_grading import InvalidNBSubmissionError


METRIC_INFO = {
    "name": "per_instance_mean_auprc_gain_or_r2_over_scored_targets",
    "direction": "higher_is_better",
    "normalization": "already_in_0_1",
}


def _scored_targets(answers_path: str | Path) -> dict[str, list[str]]:
    path = Path(answers_path).resolve().parent / "scored_targets.csv"
    if not path.is_file():
        raise RuntimeError(f"Hidden scored-target list is missing: {path}")
    table = pd.read_csv(path)
    table = table[table["scored"].astype(int) == 1]
    return {inst: grp["target"].astype(str).tolist() for inst, grp in table.groupby("instance", sort=False)}


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs: Any) -> float:
    answers_path = kwargs.get("answers_path")
    if answers_path is None:
        raise RuntimeError("answers_path is required")
    scored = _scored_targets(answers_path)
    with tempfile.TemporaryDirectory(prefix="bioxarena_pro_kano_") as tmp:
        rows = _prediction_roots(submission, answers, answers_path=answers_path, temp_root=Path(tmp))
        scores: list[float] = []
        for instance, prediction_dir, answer_dir in rows:
            truth_path = answer_dir / "y_ref.csv"
            if not truth_path.is_file():
                raise RuntimeError(f"Hidden answer is missing for {instance}: y_ref.csv")
            truth = pd.read_csv(truth_path, low_memory=False)
            prediction = _finite_numeric_frame(
                prediction_dir / "predictions.csv", list(truth.columns), len(truth)
            )
            cols = [c for c in truth.columns if c in set(scored.get(instance, []))]
            if not cols:
                raise RuntimeError(f"No scored target for instance {instance}")
            if instance in CLASSIFICATION_INSTANCES:
                values = prediction.to_numpy(dtype=float)
                if (values < 0.0).any() or (values > 1.0).any():
                    raise InvalidNBSubmissionError(
                        f"Classification predictions for {instance} must be probabilities in [0, 1]"
                    )
                score = _mean_target_auprc_gain(prediction[cols], truth[cols])
            else:
                score = _mean_target_r2(prediction[cols], truth[cols])
            scores.append(float(np.clip(score, 0.0, 1.0)))
        if not scores:
            raise RuntimeError("KANO contract contains no instances")
        return float(np.mean(scores))
