"""tabula-sapiens-ood-annotation, BioXArena-Pro v5 (2026-10-17). Task-local grader.

v5: 11 child cell types were collapsed to their Cell Ontology parent, which is also a class
(monocyte <- classical / intermediate / non-classical monocyte; CD4-positive, alpha-beta T cell <- naive CD4, regulatory,
activated CD4; CD8-positive, alpha-beta T cell <- naive CD8, activated CD8; macrophage <- tissue-resident macrophage;
granulocyte <- neutrophil, basophil). 19 classes remain. Answers and train labels were collapsed in the v5 overlay.
Submitted child labels are NOT mapped: every label must be one of the 19 VOCAB strings, else the submission is invalid.
Metric unchanged: macro-F1 within each donor x organ block (labels present in the block's truth or predictions), mean over
blocks within a donor, then equal-weight mean over the 20 donors.
"""
from __future__ import annotations

import pandas as pd

from tasks.utils import InvalidSubmissionError, grade_nested_grouped_macro_f1_task

TASK_KEY = "single-cell-spatial/tabula-sapiens-ood-annotation"
METRIC_INFO = {"name": "per_donor_mean_of_per_tissue_macro_f1", "direction": "higher_is_better",
               "normalization": "already_in_0_1"}
VOCAB = ['B cell', 'CD4-positive, alpha-beta T cell', 'CD4-positive, alpha-beta thymocyte',
         'CD8-positive, alpha-beta T cell', 'common myeloid progenitor', 'erythrocyte', 'erythroid progenitor cell',
         'gamma-delta T cell', 'granulocyte', 'hematopoietic precursor cell', 'hematopoietic stem cell',
         'innate lymphoid cell', 'macrophage', 'mast cell', 'mature NK T cell', 'monocyte', 'mononuclear phagocyte',
         'natural killer cell', 'plasma cell']
CONFIG = {"kind": "nested_grouped_macro_f1", "pred_col": "cell_type", "target_col": "cell_type",
          "reject_missing_labels": True, "outer_group_col": "donor_group", "inner_group_col": "tissue",
          "metric": METRIC_INFO}


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    if "cell_type" in submission.columns:
        v = submission["cell_type"]
        if v.isna().any() or v.astype(str).str.strip().eq("").any():
            raise InvalidSubmissionError("Submission contains missing labels.")
        bad = sorted(set(v.astype(str)) - set(VOCAB))
        if bad:
            raise InvalidSubmissionError(f"cell_type contains labels outside the 19-class vocabulary: {bad[:5]}")
    return grade_nested_grouped_macro_f1_task(submission, answers, CONFIG, kwargs.get("public_dir"),
                                              kwargs.get("answers_path"))
