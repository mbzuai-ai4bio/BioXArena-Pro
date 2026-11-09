"""structure/denoise-cryo, v5 (2026-10-17; grading logic identical to corrected_v4): backbone atoms + residue identity.

Per test volume:
  backbone = mean over the five atom types (CA, CB, C, O, N) of the tolerant one-to-one F1 (unchanged from v2)
  residue  = tolerant one-to-one F1 of (CA position, amino-acid type): a predicted residue voxel with code k (1..20,
             order ACDEFGHIKLMNPQRSTVWY) matches a reference CA of type k within one voxel along every axis
             (maximum bipartite matching per type); F1 = 2*matches / (n_pred + n_ref).
             Reference CAs whose residue type is unknown (code 0; 105 of 139,915 = 0.075 % of the v5 test CAs) are excluded, and predicted
             residues within one voxel of such a CA are ignored, so the reference itself scores exactly 1.
  volume score = 0.5 * backbone + 0.5 * residue
Task score = mean over sequence clusters of the mean volume score inside each cluster. Already in [0, 1].
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tasks.utils import (
    InvalidSubmissionError,
    group_mean_from_row_scores,
    load_npz_y,
    read_mrc_volume,
    read_test,
    require_output_dir,
    require_private_dir,
    resolve_relative_path,
    tolerant_atom_f1_per_label,
    validate_submission_frame,
    validate_test_alignment,
)

TASK_KEY = "structure/denoise-cryo"
METRIC_INFO = {
    "name": "per_cluster_mean_of_backbone_atom_f1_and_residue_identity_f1_tol1voxel",
    "direction": "higher_is_better",
    "normalization": "already_in_0_1",
}
ATOM_LABELS = (1, 2, 3, 4, 5)
N_AA = 20
TOL = 1
RADIUS = float(np.sqrt(3.0) * TOL) + 1e-3


def _check_int_volume(pred: np.ndarray, shape: tuple, max_label: int, i: int, what: str) -> np.ndarray:
    if pred.ndim != 3 or tuple(pred.shape) != tuple(shape):
        raise InvalidSubmissionError(f"{what} shape mismatch for row {i}: pred {pred.shape}, expected {tuple(shape)}.")
    if pred.dtype.kind not in "iub":
        if pred.dtype.kind != "f":
            raise InvalidSubmissionError(f"{what} for row {i} must be an integer array.")
        if not np.isfinite(pred).all():
            raise InvalidSubmissionError(f"{what} for row {i} contains non-finite values.")
        if not np.array_equal(pred, np.round(pred)):
            raise InvalidSubmissionError(f"{what} for row {i} must hold integer labels.")
    if pred.size and (int(pred.min()) < 0 or int(pred.max()) > max_label):
        raise InvalidSubmissionError(f"{what} must use labels 0..{max_label} (row {i}).")
    return np.asarray(pred).astype(np.int16, copy=False)


def residue_identity_f1(pred: np.ndarray, ref_zyx: np.ndarray, ref_aa: np.ndarray) -> float:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    from scipy.spatial import cKDTree

    pz = np.argwhere(pred > 0)
    pa = pred[tuple(pz.T)].astype(int) if len(pz) else np.zeros(0, int)
    known = ref_aa > 0
    unk = ref_zyx[~known]
    ref_zyx, ref_aa = ref_zyx[known], ref_aa[known].astype(int)
    if len(unk) and len(pz):
        d, _ = cKDTree(unk.astype(np.float32)).query(pz.astype(np.float32), distance_upper_bound=RADIUS)
        keep = np.isinf(d)
        pz, pa = pz[keep], pa[keep]
    n_pred, n_ref = len(pz), len(ref_zyx)
    if n_pred == 0 and n_ref == 0:
        return 1.0
    if n_pred == 0 or n_ref == 0:
        return 0.0
    matches = 0
    for k in range(1, N_AA + 1):
        a = pz[pa == k].astype(np.float32)
        b = ref_zyx[ref_aa == k].astype(np.float32)
        if len(a) == 0 or len(b) == 0:
            continue
        tb = cKDTree(b)
        rows, cols = [], []
        for s in range(0, len(a), 400_000):
            pr = cKDTree(a[s:s + 400_000]).sparse_distance_matrix(tb, RADIUS, output_type="ndarray")
            if len(pr):
                rows.append(pr["i"] + s)
                cols.append(pr["j"])
        if not rows:
            continue
        i = np.concatenate(rows)
        j = np.concatenate(cols)
        g = csr_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(len(a), len(b)))
        matches += int((maximum_bipartite_matching(g, perm_type="column") >= 0).sum())
    return 2.0 * matches / (n_pred + n_ref)


def _load_ref_residues(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise InvalidSubmissionError(f"Missing residue answer file: {path}")
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["zyx"]).astype(np.int64), np.asarray(z["aa"]).astype(np.int16)


def score_components(submission: pd.DataFrame, answers: pd.DataFrame, public_dir, answers_path, output_dir):
    submission, answers = validate_submission_frame(submission, answers)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    private_dir = require_private_dir(answers_path)
    output_dir = require_output_dir(output_dir)

    def score_row(i: int) -> tuple[float, float]:
        true = np.asarray(read_mrc_volume(resolve_relative_path(private_dir, answers.at[i, "type_label_file"], "answer"))).astype(np.int16)
        pred_t = _check_int_volume(load_npz_y(resolve_relative_path(output_dir, submission.at[i, "type_label_file"], "prediction")),
                                   true.shape, max(ATOM_LABELS), i, "type_label_file")
        pred_r = _check_int_volume(load_npz_y(resolve_relative_path(output_dir, submission.at[i, "residue_label_file"], "prediction")),
                                   true.shape, N_AA, i, "residue_label_file")
        backbone = float(np.mean(tolerant_atom_f1_per_label(pred_t, true, ATOM_LABELS, TOL)))
        del pred_t, true
        zyx, aa = _load_ref_residues(resolve_relative_path(private_dir, answers.at[i, "residue_label_file"], "answer"))
        residue = residue_identity_f1(pred_r, zyx, aa)
        return backbone, residue

    with ThreadPoolExecutor(max_workers=min(4, len(answers))) as pool:
        comps = list(pool.map(score_row, range(len(answers))))
    return test, comps


def grade(submission: pd.DataFrame, answers: pd.DataFrame, **kwargs) -> float:
    test, comps = score_components(submission, answers, kwargs.get("public_dir"), kwargs.get("answers_path"), kwargs.get("output_dir"))
    row_scores = [0.5 * b + 0.5 * r for b, r in comps]
    return group_mean_from_row_scores(test, row_scores, "sequence_cluster")
