"""cellverse-cta corrected_v4 (2026-10-16): healthy-control reference -> MS cohort.
Macro-F1 over the 21 allowed labels (20 reference classes + Novel_unassigned for nuclei of a type absent from the
reference). Task-local config: the shared helper is reused with the v4 label set."""
from __future__ import annotations

import copy

import pandas as pd

from tasks.utils import METRIC_INFO_BY_TASK, TASK_CONFIG, grade_macro_f1_task

TASK_KEY = "perturbation/cellverse-cta"
METRIC_INFO = METRIC_INFO_BY_TASK[TASK_KEY]
ALLOWED = ['Astrocytes', 'Astrocytes2', 'COPs', 'Endothelial_cells1', 'Endothelial_cells2', 'ImOlGs', 'Microglia_Macrophages',
           'Neuron1', 'Neuron2', 'Neuron3', 'Neuron4', 'Neuron5', 'OPCs', 'Oligo1', 'Oligo2', 'Oligo3', 'Oligo4', 'Oligo5',
           'Oligo6', 'Pericytes_VSMC', 'Novel_unassigned']
CONFIG = copy.deepcopy(TASK_CONFIG["single-cell-perturbation/cellverse-cta"])
CONFIG["allowed_labels"] = ALLOWED


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    return grade_macro_f1_task(submission, answers, CONFIG)
