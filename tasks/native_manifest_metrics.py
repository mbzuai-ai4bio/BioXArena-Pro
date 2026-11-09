"""Primary-metric scorers for native ZIP file-manifest tasks in BioXArena-Pro."""
from __future__ import annotations

import itertools
import math
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score

from tasks.nb_grading import (
    InvalidNBSubmissionError,
    _safe_extract_zip,
    _validate_answer_paths,
    _validate_ids,
)


CLASSIFICATION_INSTANCES = {
    "bace", "bbp", "bbbp", "clintox", "hiv", "muv", "sider", "tox21", "toxcast"
}


def _validate_native_manifest(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    *,
    answers_path: str | Path,
) -> list[str]:
    if submission.columns.tolist() != ["id", "prediction_file"]:
        raise InvalidNBSubmissionError(
            "File-Manifest submission.csv columns must exactly match: id, prediction_file"
        )
    if answers.columns.tolist() != ["id", "answer_path"]:
        raise InvalidNBSubmissionError(
            "File-Manifest answers.csv columns must exactly match: id, answer_path"
        )
    ids = _validate_ids(submission, answers)
    _validate_answer_paths(answers, ids, Path(answers_path).resolve().parent)
    return ids


def _prediction_roots(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    *,
    answers_path: str | Path,
    temp_root: Path,
) -> list[tuple[str, Path, Path]]:
    ids = _validate_native_manifest(submission, answers, answers_path=answers_path)
    rows: list[tuple[str, Path, Path]] = []
    for instance, artifact, answer_dir in zip(
        ids,
        submission["prediction_file"].astype(str),
        answers["answer_path"].astype(str),
    ):
        prediction_dir = temp_root / instance
        _safe_extract_zip(Path(artifact), prediction_dir)
        rows.append((instance, prediction_dir, Path(answer_dir)))
    return rows


def _finite_numeric_frame(path: Path, expected_columns: list[str], expected_rows: int) -> pd.DataFrame:
    if not path.is_file():
        raise InvalidNBSubmissionError(f"Required prediction file is missing: {path.name}")
    frame = pd.read_csv(path, low_memory=False)
    if frame.columns.tolist() != expected_columns:
        raise InvalidNBSubmissionError(
            f"{path.name} columns must exactly match hidden targets, including order"
        )
    if len(frame) != expected_rows:
        raise InvalidNBSubmissionError(
            f"{path.name} must have {expected_rows} rows, got {len(frame)}"
        )
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise InvalidNBSubmissionError(f"{path.name} contains missing or non-finite predictions")
    return numeric


def _mean_target_auc(prediction: pd.DataFrame, truth: pd.DataFrame) -> float:
    scores: list[float] = []
    for column in truth.columns:
        y = pd.to_numeric(truth[column], errors="coerce").to_numpy(dtype=float)
        p = prediction[column].to_numpy(dtype=float)
        valid = np.isfinite(y)
        labels = y[valid]
        if len(labels) == 0 or len(np.unique(labels)) < 2:
            continue
        if not np.isin(labels, [0.0, 1.0]).all():
            raise RuntimeError(f"Hidden classification target {column!r} is not binary")
        scores.append(float(roc_auc_score(labels, p[valid])))
    if not scores:
        raise RuntimeError("No evaluable binary target with both classes")
    return float(np.mean(scores))


def _mean_target_pearson(prediction: pd.DataFrame, truth: pd.DataFrame) -> float:
    scores: list[float] = []
    for column in truth.columns:
        y = pd.to_numeric(truth[column], errors="coerce").to_numpy(dtype=float)
        p = prediction[column].to_numpy(dtype=float)
        valid = np.isfinite(y)
        y = y[valid]
        p = p[valid]
        if len(y) < 2:
            continue
        if np.array_equal(p, y):
            value = 1.0
        elif np.std(p) == 0.0 or np.std(y) == 0.0:
            value = 0.0
        else:
            value = float(np.corrcoef(p, y)[0, 1])
        if not math.isfinite(value):
            value = 0.0
        scores.append(value)
    if not scores:
        raise RuntimeError("No evaluable regression target")
    raw = float(np.mean(scores))
    # v4 chance-zero scale: clip(r, 0, 1) instead of (r + 1) / 2
    return float(np.clip(raw, 0.0, 1.0))


def _mean_target_auprc_gain(prediction: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Mean over targets of (AUPRC - prevalence) / (1 - prevalence), clipped to [0, 1].

    A ranking that carries no information scores about 0 and a perfect one scores 1.
    """
    scores: list[float] = []
    for column in truth.columns:
        y = pd.to_numeric(truth[column], errors="coerce").to_numpy(dtype=float)
        p = prediction[column].to_numpy(dtype=float)
        valid = np.isfinite(y)
        labels = y[valid]
        if len(labels) == 0 or len(np.unique(labels)) < 2:
            continue
        if not np.isin(labels, [0.0, 1.0]).all():
            raise RuntimeError(f"Hidden classification target {column!r} is not binary")
        prevalence = float(labels.mean())
        auprc = float(average_precision_score(labels, p[valid]))
        scores.append(float(np.clip((auprc - prevalence) / (1.0 - prevalence), 0.0, 1.0)))
    if not scores:
        raise RuntimeError("No evaluable binary target with both classes")
    return float(np.mean(scores))


def _mean_target_r2(prediction: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Mean over targets of the coefficient of determination, clipped to [0, 1]."""
    scores: list[float] = []
    for column in truth.columns:
        y = pd.to_numeric(truth[column], errors="coerce").to_numpy(dtype=float)
        p = prediction[column].to_numpy(dtype=float)
        valid = np.isfinite(y)
        y = y[valid]
        p = p[valid]
        if len(y) < 2:
            continue
        variance = float(np.var(y))
        if variance == 0.0:
            continue
        value = 1.0 - float(np.mean((y - p) ** 2)) / variance
        if not math.isfinite(value):
            value = 0.0
        scores.append(float(np.clip(value, 0.0, 1.0)))
    if not scores:
        raise RuntimeError("No evaluable regression target")
    return float(np.mean(scores))


def grade_kano(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    *,
    answers_path: str | Path,
    **_: Any,
) -> float:
    """Equal-instance mean of AUPRC gain over prevalence (classification) or clipped R^2 (regression).

    Until 2026-10-05 this was ROC-AUC and (Pearson + 1) / 2; see audits/pro_fix_20261005/report.md.
    """
    with tempfile.TemporaryDirectory(prefix="bioxarena_pro_kano_") as tmp:
        rows = _prediction_roots(
            submission, answers, answers_path=answers_path, temp_root=Path(tmp)
        )
        scores: list[float] = []
        for instance, prediction_dir, answer_dir in rows:
            truth_path = answer_dir / "y_ref.csv"
            if not truth_path.is_file():
                raise RuntimeError(f"Hidden answer is missing for {instance}: y_ref.csv")
            truth = pd.read_csv(truth_path, low_memory=False)
            prediction = _finite_numeric_frame(
                prediction_dir / "predictions.csv", list(truth.columns), len(truth)
            )
            if instance in CLASSIFICATION_INSTANCES:
                values = prediction.to_numpy(dtype=float)
                if (values < 0.0).any() or (values > 1.0).any():
                    raise InvalidNBSubmissionError(
                        f"Classification predictions for {instance} must be probabilities in [0, 1]"
                    )
                score = _mean_target_auprc_gain(prediction, truth)
            else:
                score = _mean_target_r2(prediction, truth)
            scores.append(float(np.clip(score, 0.0, 1.0)))
        if not scores:
            raise RuntimeError("KANO contract contains no instances")
        return float(np.mean(scores))


def _load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError("tifffile is required for microscopy grading") from exc
        array = tifffile.imread(path)
    else:
        raise RuntimeError(f"Unsupported hidden mask format: {path.name}")
    return np.asarray(array)


def _validate_label_mask(mask: np.ndarray, name: str) -> np.ndarray:
    if mask.ndim != 2:
        raise InvalidNBSubmissionError(f"Mask {name!r} must be a 2D array")
    if not np.issubdtype(mask.dtype, np.integer):
        if not np.isfinite(mask).all() or not np.equal(mask, np.floor(mask)).all():
            raise InvalidNBSubmissionError(f"Mask {name!r} must contain integer labels")
    result = np.asarray(mask, dtype=np.int64)
    if (result < 0).any():
        raise InvalidNBSubmissionError(f"Mask {name!r} contains negative labels")
    return result


def _instance_iou_matrix(truth: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, int, int]:
    if truth.shape != prediction.shape:
        raise InvalidNBSubmissionError(
            f"Prediction shape {prediction.shape} does not match truth shape {truth.shape}"
        )
    truth_labels, truth_inverse = np.unique(truth, return_inverse=True)
    pred_labels, pred_inverse = np.unique(prediction, return_inverse=True)
    truth_keep = truth_labels != 0
    pred_keep = pred_labels != 0
    n_truth = int(truth_keep.sum())
    n_pred = int(pred_keep.sum())
    if n_truth == 0 or n_pred == 0:
        return np.zeros((n_truth, n_pred), dtype=float), n_truth, n_pred
    pair = truth_inverse.astype(np.int64) * len(pred_labels) + pred_inverse
    intersections = np.bincount(
        pair, minlength=len(truth_labels) * len(pred_labels)
    ).reshape(len(truth_labels), len(pred_labels))
    truth_sizes = np.bincount(truth_inverse, minlength=len(truth_labels))
    pred_sizes = np.bincount(pred_inverse, minlength=len(pred_labels))
    intersections = intersections[np.ix_(truth_keep, pred_keep)].astype(float)
    unions = (
        truth_sizes[truth_keep, None].astype(float)
        + pred_sizes[None, pred_keep].astype(float)
        - intersections
    )
    return np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0), n_truth, n_pred


_MSA_THRESHOLDS = np.round(np.arange(0.50, 1.00, 0.05), 2)  # v3: exact 0.50, 0.55, ..., 0.95 (no float drift)


def _mean_segmentation_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    """mSA = mean over IoU thresholds t of TP / (TP + FP + FN).

    v3 (2026-10-15): thresholds are rounded (np.arange gave 0.6000000000000001 etc., so IoU == 0.60 failed
    t = 0.60), and matching is a maximum bipartite matching on {IoU >= t} for each threshold separately instead
    of a single Hungarian assignment on total IoU."""
    iou, n_truth, n_pred = _instance_iou_matrix(truth, prediction)
    if n_truth == 0 and n_pred == 0:
        return 1.0
    if n_truth == 0 or n_pred == 0:
        return 0.0
    scores = []
    eps = 1e-12
    for threshold in _MSA_THRESHOLDS:
        hit = iou >= threshold - eps
        rows = np.flatnonzero(hit.any(axis=1))
        cols = np.flatnonzero(hit.any(axis=0))
        if len(rows) == 0:
            tp = 0
        else:
            sub = hit[np.ix_(rows, cols)]
            if int(sub.sum(axis=1).max()) == 1 and int(sub.sum(axis=0).max()) == 1:
                tp = int(sub.sum())  # already one-to-one (always the case for t > 0.5)
            else:
                r_idx, c_idx = linear_sum_assignment(-sub.astype(np.float64))
                tp = int(np.count_nonzero(sub[r_idx, c_idx]))
        denominator = tp + (n_pred - tp) + (n_truth - tp)
        scores.append(tp / denominator if denominator else 1.0)
    return float(np.mean(scores))


def grade_microscopy(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    *,
    answers_path: str | Path,
    **_: Any,
) -> float:
    """Equal-instance mean of per-image mean Segmentation Accuracy (mSA)."""
    with tempfile.TemporaryDirectory(prefix="bioxarena_pro_microscopy_") as tmp:
        rows = _prediction_roots(
            submission, answers, answers_path=answers_path, temp_root=Path(tmp)
        )
        instance_scores: list[float] = []
        for instance, prediction_dir, answer_dir in rows:
            archive_path = prediction_dir / "predictions.npz"
            if not archive_path.is_file():
                raise InvalidNBSubmissionError(
                    f"Required prediction file is missing for {instance}: predictions.npz"
                )
            truth_files = sorted(
                path for path in answer_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".npy", ".tif", ".tiff"}
            )
            expected_keys = [path.stem for path in truth_files]
            if not expected_keys:
                raise RuntimeError(f"No hidden masks found for {instance}")
            with np.load(archive_path, allow_pickle=False) as predictions:
                if sorted(predictions.files) != sorted(expected_keys):
                    raise InvalidNBSubmissionError(
                        f"predictions.npz keys for {instance} must exactly match hidden image stems"
                    )
                image_scores = []
                for truth_path in truth_files:
                    truth = _validate_label_mask(_load_mask(truth_path), truth_path.stem)
                    prediction = _validate_label_mask(
                        np.asarray(predictions[truth_path.stem]), truth_path.stem
                    )
                    image_scores.append(_mean_segmentation_accuracy(truth, prediction))
            instance_scores.append(float(np.mean(image_scores)))
        if not instance_scores:
            raise RuntimeError("Microscopy contract contains no instances")
        return float(np.mean(instance_scores))


_RESIDUE_MASS = {
    "A": 71.03711, "R": 156.10111, "N": 114.04293, "D": 115.02694,
    "C": 103.00919, "E": 129.04259, "Q": 128.05858, "G": 57.02146,
    "H": 137.05891, "I": 113.08406, "L": 113.08406, "K": 128.09496,
    "M": 131.04049, "F": 147.06841, "P": 97.05276, "S": 87.03203,
    "T": 101.04768, "W": 186.07931, "Y": 163.06333, "V": 99.06841,
}
_MOD_ALIASES = {
    "ox": 15.9949,
    "oxidation": 15.9949,
    "cam": 57.0215,
    "carbamidomethyl": 57.0215,
    "carbamidomethylation": 57.0215,
    "deamidation": 0.9840,
    "deamidated": 0.9840,
}


def _mod_delta(text: str) -> float:
    value = text.strip().lower().replace("da", "")
    if value in _MOD_ALIASES:
        return _MOD_ALIASES[value]
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if not match:
        raise InvalidNBSubmissionError(f"Unsupported peptide modification: {text!r}")
    return float(match.group(0))


def _peptide_masses(sequence: Any) -> list[float]:
    if sequence is None or (isinstance(sequence, float) and math.isnan(sequence)):
        return []
    text = str(sequence).strip().strip(".")
    masses: list[float] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char in ".-_ ":
            index += 1
            continue
        residue = char.upper()
        if residue not in _RESIDUE_MASS:
            raise InvalidNBSubmissionError(f"Unsupported peptide residue: {char!r}")
        mass = _RESIDUE_MASS[residue]
        index += 1
        # A residue can carry more than one consecutive annotation in the
        # benchmark truth (for example ``C(+57.02)[-17.0265]``).  Sum every
        # adjacent modification before advancing to the next residue.
        while index < len(text) and text[index] in "([":
            opening = text[index]
            closing = ")" if opening == "(" else "]"
            end = text.find(closing, index + 1)
            if end < 0:
                raise InvalidNBSubmissionError("Unclosed peptide modification")
            mass += _mod_delta(text[index + 1:end])
            index = end + 1
        masses.append(mass)
    return masses


def _peptide_match(prediction: Any, truth: Any) -> bool:
    pred = _peptide_masses(prediction)
    ref = _peptide_masses(truth)
    if len(pred) != len(ref) or not ref:
        return False
    cumulative_pred = 0.0
    cumulative_ref = 0.0
    for pred_mass, ref_mass in zip(pred, ref):
        cumulative_pred += pred_mass
        cumulative_ref += ref_mass
        if abs(pred_mass - ref_mass) > 0.1:
            return False
        if abs(cumulative_pred - cumulative_ref) > 0.5:
            return False
    return True


def _iter_peptide_truth(path: Path, batch_size: int = 100_000):
    """Yield hidden modified sequences in bounded-memory Arrow batches."""
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=["modified_sequence"]
        ):
            yield from batch.column(0).to_pylist()
        return
    if path.suffix.lower() == ".ipc":
        import pyarrow.ipc as ipc
        with path.open("rb") as handle:
            try:
                reader = ipc.open_file(handle)
                for index in range(reader.num_record_batches):
                    batch = reader.get_batch(index).select(["modified_sequence"])
                    for offset in range(0, batch.num_rows, batch_size):
                        yield from batch.slice(offset, batch_size).column(0).to_pylist()
            except Exception:
                handle.seek(0)
                reader = ipc.open_stream(handle)
                for batch in reader:
                    batch = batch.select(["modified_sequence"])
                    for offset in range(0, batch.num_rows, batch_size):
                        yield from batch.slice(offset, batch_size).column(0).to_pylist()
        return
    raise RuntimeError(f"Unsupported peptide truth format: {path.name}")


def _read_peptide_truth(path: Path) -> list[Any]:
    """Compatibility helper for small tests; production grading streams batches."""
    return list(_iter_peptide_truth(path))


def grade_peptide(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    *,
    answers_path: str | Path,
    **_: Any,
) -> float:
    """Equal-instance Peptide Accuracy using the published Novor mass tolerances."""
    with tempfile.TemporaryDirectory(prefix="bioxarena_pro_peptide_") as tmp:
        rows = _prediction_roots(
            submission, answers, answers_path=answers_path, temp_root=Path(tmp)
        )
        instance_scores: list[float] = []
        for instance, prediction_dir, answer_dir in rows:
            truth_files = list(answer_dir.glob("ground_truth.*"))
            if len(truth_files) != 1:
                raise RuntimeError(f"Expected one hidden ground_truth file for {instance}")
            truth_iterator = iter(_iter_peptide_truth(truth_files[0]))
            prediction_path = prediction_dir / "predictions.csv"
            if not prediction_path.is_file():
                raise InvalidNBSubmissionError(
                    f"Required prediction file is missing for {instance}: predictions.csv"
                )
            correct = 0
            row_count = 0
            chunks = pd.read_csv(
                prediction_path, keep_default_na=False, chunksize=100_000
            )
            for prediction in chunks:
                if prediction.columns.tolist() != ["index", "sequence"]:
                    raise InvalidNBSubmissionError(
                        "predictions.csv columns must exactly match: index, sequence"
                    )
                indices = pd.to_numeric(
                    prediction["index"], errors="coerce"
                ).to_numpy()
                expected_indices = np.arange(row_count, row_count + len(prediction))
                if not np.array_equal(indices, expected_indices):
                    raise InvalidNBSubmissionError(
                        f"{instance} index must be the exact 0-based row sequence"
                    )
                truth_chunk = list(itertools.islice(truth_iterator, len(prediction)))
                if len(truth_chunk) != len(prediction):
                    raise InvalidNBSubmissionError(
                        f"{instance} predictions contain more rows than hidden truth"
                    )
                correct += sum(
                    _peptide_match(predicted, expected)
                    for predicted, expected in zip(prediction["sequence"], truth_chunk)
                )
                row_count += len(prediction)
            if next(truth_iterator, None) is not None:
                raise InvalidNBSubmissionError(
                    f"{instance} predictions contain fewer rows than hidden truth"
                )
            instance_scores.append(correct / row_count if row_count else 0.0)
        if not instance_scores:
            raise RuntimeError("Peptide contract contains no instances")
        return float(np.mean(instance_scores))


# === BEGIN peptide_v2 block (owned by audits/pro_fix_v2_20261013/build/peptide/patch_metrics.py) ===
# De novo peptide sequencing, corrected_v2 grader.  Everything below is additive: the v1
# grade_peptide above is untouched.  Residues are compared by mass (Casanovo / DeepNovo
# convention): cysteine is always carbamidomethylated (160.03065 Da), I/L are isobaric, an
# explicit +57.02 on C is absorbed, an annotation before the first residue is an N-terminal
# modification.  Instance score = 0.5 * peptide accuracy + 0.5 * amino-acid F1, where
# peptide accuracy = exact mass-matched peptides / all test spectra, AA precision = matched
# residues / predicted residues, AA recall = matched residues / reference residues (all
# spectra; an empty prediction contributes 0 matched, 0 predicted residues).
_PEPTIDE_V2_CARBAMIDOMETHYL = 57.02146
_PEPTIDE_V2_RESIDUE_MASS = dict(_RESIDUE_MASS)
_PEPTIDE_V2_RESIDUE_MASS["C"] = _RESIDUE_MASS["C"] + _PEPTIDE_V2_CARBAMIDOMETHYL
_PEPTIDE_V2_MOD_ALIASES = {
    **_MOD_ALIASES,
    "acetyl": 42.0106, "acetylation": 42.0106, "ac": 42.0106,
    "carbamyl": 43.0058, "carbamylation": 43.0058,
    "ammonia-loss": -17.0265, "ammonia_loss": -17.0265, "nh3-loss": -17.0265, "nh3 loss": -17.0265,
    "pyro-glu": -17.0265, "gln->pyro-glu": -17.0265, "glu->pyro-glu": -18.0106,
    "deamidation": 0.9840, "deamidated": 0.9840, "deam": 0.9840,
    "ox": 15.9949, "oxidation": 15.9949, "oxidised": 15.9949, "oxidized": 15.9949,
    "cam": _PEPTIDE_V2_CARBAMIDOMETHYL, "carbamidomethyl": _PEPTIDE_V2_CARBAMIDOMETHYL,
    "carbamidomethylation": _PEPTIDE_V2_CARBAMIDOMETHYL,
}
_PEPTIDE_V2_CUM_TOL = 0.5
_PEPTIDE_V2_IND_TOL = 0.1
_PEPTIDE_V2_MAX_RESIDUES = 60


def _peptide_v2_mod_delta(text: str) -> float:
    value = text.strip().lower()
    if value.endswith("da"):
        value = value[:-2].strip()
    if value in _PEPTIDE_V2_MOD_ALIASES:
        return _PEPTIDE_V2_MOD_ALIASES[value]
    match = re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
    if not match:
        raise InvalidNBSubmissionError(f"Unsupported peptide modification: {text!r}")
    return float(match.group(0))


def _peptide_v2_masses(sequence: Any, max_residues: int | None = _PEPTIDE_V2_MAX_RESIDUES) -> list[float]:
    """Residue mass list of an annotated peptide string (empty list for an empty prediction)."""
    if sequence is None or (isinstance(sequence, float) and math.isnan(sequence)):
        return []
    text = str(sequence).strip()
    masses: list[float] = []
    pending_nterm = 0.0
    index = 0
    while index < len(text):
        char = text[index]
        if char in ".-_ ":
            index += 1
            continue
        if char in "([":
            # annotation before the first residue: N-terminal modification
            if masses:
                raise InvalidNBSubmissionError(f"Dangling modification in peptide {text!r}")
            closing = ")" if char == "(" else "]"
            end = text.find(closing, index + 1)
            if end < 0:
                raise InvalidNBSubmissionError("Unclosed peptide modification")
            pending_nterm += _peptide_v2_mod_delta(text[index + 1:end])
            index = end + 1
            continue
        residue = char  # upper-case one-letter codes only; a literal "nan" is rejected
        if residue not in _PEPTIDE_V2_RESIDUE_MASS:
            raise InvalidNBSubmissionError(f"Unsupported peptide residue: {char!r}")
        mass = _PEPTIDE_V2_RESIDUE_MASS[residue]
        index += 1
        while index < len(text) and text[index] in "([":
            opening = text[index]
            closing = ")" if opening == "(" else "]"
            end = text.find(closing, index + 1)
            if end < 0:
                raise InvalidNBSubmissionError("Unclosed peptide modification")
            delta = _peptide_v2_mod_delta(text[index + 1:end])
            if residue == "C" and abs(delta - _PEPTIDE_V2_CARBAMIDOMETHYL) < 0.05:
                delta = 0.0  # explicit carbamidomethyl on C: already fixed
            mass += delta
            index = end + 1
        if not masses:
            mass += pending_nterm
            pending_nterm = 0.0
        masses.append(mass)
    if pending_nterm and not masses:
        raise InvalidNBSubmissionError(f"Modification without residue in peptide {text!r}")
    if max_residues is not None and len(masses) > max_residues:
        raise InvalidNBSubmissionError(
            f"Peptide prediction longer than {_PEPTIDE_V2_MAX_RESIDUES} residues"
        )
    return masses


def _peptide_v2_aa_match(pred: list[float], ref: list[float]) -> tuple[int, bool]:
    """Casanovo aa_match_prefix_suffix on residue-mass lists.

    Returns (number of mass-matched residues, whole-peptide match).  A residue matches when
    the cumulative masses agree within 0.5 Da and the residue masses within 0.1 Da; the
    alignment runs from the N-terminus and, if the peptides do not fully match, again from
    the C-terminus.  Whole-peptide match requires every position to match (same length).
    """
    n = max(len(pred), len(ref))
    if n == 0:
        return 0, False
    if not pred or not ref:
        return 0, False
    matches = [False] * n
    i1 = i2 = 0
    c1 = c2 = 0.0
    while i1 < len(pred) and i2 < len(ref):
        m1, m2 = pred[i1], ref[i2]
        if abs(c1 + m1 - c2 - m2) < _PEPTIDE_V2_CUM_TOL:
            matches[max(i1, i2)] = abs(m1 - m2) < _PEPTIDE_V2_IND_TOL
            i1 += 1
            i2 += 1
            c1 += m1
            c2 += m2
        elif c2 + m2 > c1 + m1:
            i1 += 1
            c1 += m1
        else:
            i2 += 1
            c2 += m2
    if all(matches):
        return n, True
    i_stop = matches.index(False)
    i1, i2 = len(pred) - 1, len(ref) - 1
    c1 = c2 = 0.0
    while i1 >= i_stop and i2 >= i_stop:
        m1, m2 = pred[i1], ref[i2]
        if abs(c1 + m1 - c2 - m2) < _PEPTIDE_V2_CUM_TOL:
            matches[max(i1, i2)] = abs(m1 - m2) < _PEPTIDE_V2_IND_TOL
            i1 -= 1
            i2 -= 1
            c1 += m1
            c2 += m2
        elif c2 + m2 > c1 + m1:
            i1 -= 1
            c1 += m1
        else:
            i2 -= 1
            c2 += m2
    matched = sum(matches)
    return matched, matched == n


def peptide_v2_instance_metrics(predictions: "list[Any]", references: "list[Any]") -> dict:
    """All v2 components for one instance; predictions[i] corresponds to references[i]."""
    if len(predictions) != len(references):
        raise RuntimeError("peptide_v2_instance_metrics: length mismatch")
    n_spectra = len(references)
    n_pep_correct = 0
    n_nonempty = 0
    n_aa_matched = 0
    n_aa_pred = 0
    n_aa_ref = 0
    ref_cache: dict[Any, list[float]] = {}
    for predicted, expected in zip(predictions, references):
        ref = ref_cache.get(expected)
        if ref is None:
            ref = _peptide_v2_masses(expected, max_residues=None)
            ref_cache[expected] = ref
        pred = _peptide_v2_masses(predicted)
        n_aa_ref += len(ref)
        n_aa_pred += len(pred)
        if pred:
            n_nonempty += 1
        matched, pep_ok = _peptide_v2_aa_match(pred, ref)
        n_aa_matched += matched
        n_pep_correct += int(pep_ok)
    pep_acc = n_pep_correct / n_spectra if n_spectra else 0.0
    aa_prec = n_aa_matched / n_aa_pred if n_aa_pred else 0.0
    aa_rec = n_aa_matched / n_aa_ref if n_aa_ref else 0.0
    aa_f1 = 2 * aa_prec * aa_rec / (aa_prec + aa_rec) if (aa_prec + aa_rec) > 0 else 0.0
    return {
        "n_spectra": n_spectra,
        "n_nonempty": n_nonempty,
        "peptide_accuracy": pep_acc,
        "peptide_precision": n_pep_correct / n_nonempty if n_nonempty else 0.0,
        "aa_precision": aa_prec,
        "aa_recall": aa_rec,
        "aa_f1": aa_f1,
        "score": 0.5 * pep_acc + 0.5 * aa_f1,
    }


def _peptide_v2_read_truth(answer_dir: Path) -> pd.DataFrame:
    truth_path = answer_dir / "ground_truth.parquet"
    if not truth_path.is_file():
        raise RuntimeError(f"Hidden ground_truth.parquet missing in {answer_dir}")
    import pyarrow.parquet as pq
    truth = pq.read_table(truth_path, columns=["spectrum_id", "modified_sequence"]).to_pandas()
    if truth["spectrum_id"].duplicated().any():
        raise RuntimeError(f"Hidden truth has duplicate spectrum ids in {answer_dir}")
    return truth


def grade_peptide_v2(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    *,
    answers_path: str | Path,
    return_details: bool = False,
    **_: Any,
) -> float | tuple[float, dict]:
    """corrected_v2 de novo peptide grader: mean over instances of 0.5*PepAcc + 0.5*AA-F1."""
    details: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="bioxarena_pro_peptide_v2_") as tmp:
        rows = _prediction_roots(
            submission, answers, answers_path=answers_path, temp_root=Path(tmp)
        )
        instance_scores: list[float] = []
        for instance, prediction_dir, answer_dir in rows:
            truth = _peptide_v2_read_truth(answer_dir)
            prediction_path = prediction_dir / "predictions.csv"
            if not prediction_path.is_file():
                raise InvalidNBSubmissionError(
                    f"Required prediction file is missing for {instance}: predictions.csv"
                )
            prediction = pd.read_csv(
                prediction_path, keep_default_na=False, dtype=str, na_filter=False
            )
            if prediction.columns.tolist() != ["spectrum_id", "sequence"]:
                raise InvalidNBSubmissionError(
                    f"{instance} predictions.csv columns must exactly match: spectrum_id, sequence"
                )
            ids = prediction["spectrum_id"].astype(str).str.strip()
            if ids.duplicated().any():
                raise InvalidNBSubmissionError(f"{instance} predictions contain duplicate spectrum ids")
            truth_ids = truth["spectrum_id"].astype(str)
            if len(ids) != len(truth_ids) or set(ids) != set(truth_ids):
                missing = len(set(truth_ids) - set(ids))
                unknown = len(set(ids) - set(truth_ids))
                raise InvalidNBSubmissionError(
                    f"{instance} predictions must cover exactly the test spectrum ids "
                    f"({missing} missing, {unknown} unknown)"
                )
            lookup = pd.Series(prediction["sequence"].values, index=ids.values)
            predicted = lookup.reindex(truth_ids.values).tolist()
            metrics = peptide_v2_instance_metrics(predicted, truth["modified_sequence"].tolist())
            details[instance] = metrics
            instance_scores.append(float(metrics["score"]))
        if not instance_scores:
            raise RuntimeError("Peptide contract contains no instances")
        score = float(np.mean(instance_scores))
    if return_details:
        return score, details
    return score
# === END peptide_v2 block ===
