"""gastrulation-kinetic-state, BioXArena-Pro v4 (2026-10-16). Data unchanged from corrected_v1 (held-out atlas embryos).

Score = 8/42 * stage_score + 34/42 * fate_macro_F1.
* Labels must come from the fixed vocabularies below (8 stages, 34 fates); any other string fails the submission.
* Every test embryo (`embryo_id` in test.csv) has exactly one developmental stage. Stage is scored per embryo: the
  embryo's predicted stage is the most frequent per-cell `developmental_stage` among its cells (ties -> earliest stage).
  Per-embryo credit is exact match only (1 for the right stage, 0 otherwise; no partial credit, v4 lead decision
  2026-10-16); stage_score = fraction of the test samples staged correctly (13 since v5.2, 2026-10-20).
* fate_macro_F1 = per-cell macro-F1 over the fates of the 34-label vocabulary that occur in the answers (all 34 do);
  predicting a vocabulary fate that does not occur still costs precision.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from tasks.utils import InvalidSubmissionError, validate_submission_frame

TASK_KEY = "single-cell-spatial/gastrulation-kinetic-state"
METRIC_INFO = {"name": "weighted_embryo_stage_credit_and_cell_fate_macro_f1", "direction": "higher_is_better",
               "normalization": "already_in_0_1"}
STAGES = ["E6.5", "E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"]
FATES = ['Allantois', 'Anterior Primitive Streak', 'Blood progenitors 1', 'Blood progenitors 2', 'Cardiomyocytes',
         'Caudal Mesoderm', 'Caudal epiblast', 'Caudal neurectoderm', 'Def. endoderm', 'Endothelium', 'Epiblast',
         'Erythroid1', 'Erythroid2', 'Erythroid3', 'ExE mesoderm', 'Forebrain/Midbrain/Hindbrain', 'Gut',
         'Haematoendothelial progenitors', 'Intermediate mesoderm', 'Mesenchyme', 'Mixed mesoderm', 'NMP',
         'Nascent mesoderm', 'Neural crest', 'Notochord', 'PGC', 'Paraxial mesoderm', 'Pharyngeal mesoderm',
         'Primitive Streak', 'Rostral neurectoderm', 'Somitic mesoderm', 'Spinal cord', 'Surface ectoderm',
         'Visceral endoderm']
W_STAGE, W_FATE = 8 / 42, 34 / 42
COLS = ["id", "developmental_stage", "cell_fate"]


def _days(s: pd.Series) -> np.ndarray:
    return s.str.slice(1).astype(float).to_numpy()


def _check_labels(sub: pd.DataFrame) -> None:
    for col, vocab in (("developmental_stage", STAGES), ("cell_fate", FATES)):
        v = sub[col]
        if v.isna().any():
            raise InvalidSubmissionError(f"column '{col}' contains missing labels")
        bad = sorted(set(v.astype(str)) - set(vocab))
        if bad:
            raise InvalidSubmissionError(f"column '{col}' contains labels outside the allowed vocabulary: {bad[:5]}")


def _embryo_mode(stage: pd.Series, embryo: pd.Series) -> pd.Series:
    order = {s: i for i, s in enumerate(STAGES)}
    df = pd.DataFrame({"e": embryo.to_numpy(), "s": stage.astype(str).to_numpy()})
    cnt = df.groupby(["e", "s"]).size().reset_index(name="n")
    cnt["o"] = cnt.s.map(order)
    cnt = cnt.sort_values(["e", "n", "o"], ascending=[True, False, True])
    return cnt.drop_duplicates("e").set_index("e").s


def components(submission: pd.DataFrame, answers: pd.DataFrame, embryo: pd.Series) -> dict:
    sub, ans = validate_submission_frame(submission, answers[COLS], COLS)
    _check_labels(sub)
    present = [f for f in FATES if f in set(ans.cell_fate.astype(str))]
    fate = float(f1_score(ans.cell_fate.astype(str), sub.cell_fate.astype(str), labels=present, average="macro",
                          zero_division=0))
    emb = embryo.reset_index(drop=True)
    true_e = _embryo_mode(ans.developmental_stage, emb)
    if (ans.groupby(emb.to_numpy()).developmental_stage.nunique() != 1).any():
        raise RuntimeError("answers: an embryo with more than one stage")
    pred_e = _embryo_mode(sub.developmental_stage, emb).reindex(true_e.index)
    err = np.abs(_days(pred_e) - _days(true_e))
    credit = (err == 0).astype(float)
    return {"score": W_STAGE * float(credit.mean()) + W_FATE * fate, "stage_score": float(credit.mean()),
            "stage_exact": float(np.mean(err == 0)), "stage_mae_days": float(err.mean()), "fate_macro_f1": fate,
            "n_embryos": int(len(true_e))}


GROUP_COL = "embryo_id"


def _embryo_ids(answers: pd.DataFrame, public_dir) -> pd.Series:
    if public_dir is None:
        raise RuntimeError("public_dir is required (test.csv carries embryo_id)")
    test = pd.read_csv(Path(public_dir) / "test.csv", usecols=["id", GROUP_COL])
    if test["id"].astype(str).tolist() != answers["id"].astype(str).tolist():
        raise RuntimeError("test.csv ids do not match answers ids")
    return test[GROUP_COL].astype(str)


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    try:
        return components(submission, answers, _embryo_ids(answers, kwargs.get("public_dir")))["score"]
    except ValueError as e:  # pandas/sklearn parsing problems in a submission
        raise InvalidSubmissionError(str(e)) from e
