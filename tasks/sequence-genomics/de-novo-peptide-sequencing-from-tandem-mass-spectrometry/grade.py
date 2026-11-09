from __future__ import annotations

import pandas as pd

from tasks.native_manifest_metrics import grade_peptide_v2


METRIC_INFO = {
    "name": "per_instance_mean_half_peptide_accuracy_half_aa_f1",
    "direction": "higher_is_better",
    "normalization": "already_in_0_1",
}


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    """corrected_v2: mean over the 4 instances of 0.5*peptide accuracy + 0.5*amino-acid F1
    (Casanovo mass-based residue matching, 0.5 Da cumulative / 0.1 Da residue tolerance)."""
    return grade_peptide_v2(submission, answers, **kwargs)
