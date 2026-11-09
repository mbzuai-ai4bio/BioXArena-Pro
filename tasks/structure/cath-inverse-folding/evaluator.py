"""Inverse folding (merged CATH 4.2/4.3 test set, PDB2022) — complete-coverage evaluator (v2 2026-10-13; instances v5 2026-10-17).

Changes against the v1 evaluator (see audits/pro_fix_v2_20261013/report_cath.md):

1. The submission must cover EXACTLY the reference chain set of the instance.  A missing
   or unknown chain name is a validation error.  v1 silently skipped every reference
   chain that was absent from the submission and took the median over whatever remained,
   so a submission holding a single well-chosen chain scored that chain's recovery
   (0.2655 overall with constant sequences, against 0.3074 for a full agent run).
2. A predicted sequence whose length differs from the reference is a validation error as
   well; v1 dropped those chains from the median too.  The length constraint is part of
   the published output format.
3. The reported metric is unchanged: median per-chain recovery rate (percent) over the
   full reference set, and mean NSSR(BLOSUM62) as a secondary metric.
"""

import os
import json
import numpy as np
import torch

# Setup Paths
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.environ.get("GROUND_TRUTH_DIR", os.path.join(EVAL_DIR, "ground_truth"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
if not OUTPUT_DIR:
    raise RuntimeError("OUTPUT_DIR environment variable is required")

# Output specification (must match README Section 5)
OUTPUT_FILE = "predictions.json"
# v5: merged, sequence-deduplicated `cath` instance (was cath_4_2 + cath_4_3); ts50 was dropped in v4.
INSTANCES = ["cath", "pdb2022"]
METRIC_NAMES = ["recovery_rate", "nssr_blosum62"]

# Standard amino acid alphabet (matches BLOSUM matrix order)
AA_ORDER = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
VALID_AAS = set(AA_ORDER)


class ValidationError(Exception):
    """Raised when output file fails validation."""
    pass


def error_result(error_msg):
    """Return a result dict with null metrics and error info."""
    result = {name: None for name in METRIC_NAMES}
    result["error"] = str(error_msg)
    return result


def load_ground_truth(instance_name):
    """Load ground truth sequences for an instance.

    Returns:
        dict: Mapping from protein name to amino acid sequence string.
    """
    gt_dir = os.path.join(GROUND_TRUTH_DIR, instance_name)

    # Try JSONL format first (CATH instances)
    jsonl_path = os.path.join(gt_dir, "y_ref.jsonl")
    json_path = os.path.join(gt_dir, "y_ref.json")

    gt_dict = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                gt_dict[entry['name']] = entry['seq']
    elif os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        for entry in data:
            gt_dict[entry['name']] = entry['seq']
    else:
        raise FileNotFoundError(
            f"No ground truth file found for instance {instance_name}"
        )

    return gt_dict


def load_blosum_matrix(instance_name, blosum_name="BLOSUM62"):
    """Load a BLOSUM substitution matrix from the ground truth directory.

    Returns:
        numpy.ndarray: 20x20 BLOSUM matrix.
    """
    blosum_path = os.path.join(GROUND_TRUTH_DIR, instance_name, "eval_blosums.pth")
    if not os.path.exists(blosum_path):
        raise FileNotFoundError(
            f"BLOSUM matrices not found: {blosum_path}"
        )
    blosum_mats = torch.load(blosum_path, map_location='cpu', weights_only=False)
    mat = blosum_mats[blosum_name]
    if isinstance(mat, torch.Tensor):
        mat = mat.numpy()
    return mat


def load_and_validate(instance_name):
    """Load the output file for an instance and validate.

    Returns:
        dict: Mapping from protein name to predicted amino acid sequence string.

    Raises:
        ValidationError: If the output file is missing or has wrong format.
    """
    pred_file = os.path.join(OUTPUT_DIR, instance_name, OUTPUT_FILE)

    if not os.path.exists(pred_file):
        raise ValidationError(f"Output file not found: {pred_file}")

    try:
        with open(pred_file, 'r') as f:
            predictions = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in {pred_file}: {e}")

    if not isinstance(predictions, list):
        raise ValidationError(
            f"Expected JSON array, got {type(predictions).__name__}"
        )

    pred_dict = {}
    for i, entry in enumerate(predictions):
        if not isinstance(entry, dict):
            raise ValidationError(
                f"Entry {i} is not a JSON object"
            )
        if 'name' not in entry or 'seq' not in entry:
            raise ValidationError(
                f"Entry {i} missing required fields 'name' and/or 'seq'"
            )

        name = entry['name']
        seq = entry['seq']

        if not isinstance(name, str) or not isinstance(seq, str):
            raise ValidationError(
                f"Entry {i}: 'name' and 'seq' must be strings"
            )

        if len(seq) == 0:
            raise ValidationError(
                f"Entry '{name}': empty sequence"
            )

        # Validate amino acid characters
        invalid_chars = set(seq) - VALID_AAS
        if invalid_chars:
            raise ValidationError(
                f"Entry '{name}': invalid amino acid characters: {invalid_chars}"
            )

        if name in pred_dict:
            raise ValidationError(
                f"Duplicate prediction for protein '{name}'"
            )

        pred_dict[name] = seq

    return pred_dict


def seq_to_indices(seq):
    """Convert amino acid sequence string to integer index array."""
    return np.array([AA_TO_IDX[aa] for aa in seq], dtype=np.int64)


def calculate_recovery_rate(pred_seq_str, ref_seq_str):
    """Calculate per-protein recovery rate.

    Recovery rate is the proportion of positions where the predicted
    amino acid exactly matches the native amino acid.
    """
    assert len(pred_seq_str) == len(ref_seq_str)
    matches = sum(1 for p, r in zip(pred_seq_str, ref_seq_str) if p == r)
    return matches / len(ref_seq_str)


def calculate_nssr(pred_seq_str, ref_seq_str, blosum_mat):
    """Calculate per-protein NSSR using BLOSUM matrix.

    A residue position is a positive match if the BLOSUM score
    between predicted and native amino acid is greater than zero.
    """
    assert len(pred_seq_str) == len(ref_seq_str)
    pred_idx = seq_to_indices(pred_seq_str)
    ref_idx = seq_to_indices(ref_seq_str)
    scores = blosum_mat[pred_idx, ref_idx]
    positive_matches = (scores > 0).sum()
    return positive_matches / len(ref_seq_str)


def calculate_metrics(pred_dict, gt_dict, blosum_mat):
    """Calculate evaluation metrics over the complete reference set of an instance.

    Every reference chain must carry a prediction of the correct length; the submission
    may not contain any other chain.  Returns {metric_name: metric_value}.

    Raises:
        ValidationError: if the submitted chain set is not exactly the reference set, or
            if any predicted sequence has the wrong length.
    """
    gt_names = set(gt_dict)
    pred_names = set(pred_dict)

    missing = sorted(gt_names - pred_names)
    unknown = sorted(pred_names - gt_names)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(
                f"{len(missing)} of {len(gt_names)} reference chains have no prediction "
                f"(e.g. {missing[:5]})"
            )
        if unknown:
            parts.append(
                f"{len(unknown)} predicted chains are not in the reference set "
                f"(e.g. {unknown[:5]})"
            )
        raise ValidationError(
            "predictions must cover exactly the reference chain set: " + "; ".join(parts)
        )

    bad_length = [
        (name, len(pred_dict[name]), len(ref))
        for name, ref in gt_dict.items()
        if len(pred_dict[name]) != len(ref)
    ]
    if bad_length:
        raise ValidationError(
            f"{len(bad_length)} predicted sequences have the wrong length "
            f"(e.g. {bad_length[:5]}; each sequence must have one residue per backbone position)"
        )

    recovery_rates = []
    nssr_scores = []
    for name, ref_seq in gt_dict.items():
        pred_seq = pred_dict[name]
        recovery_rates.append(calculate_recovery_rate(pred_seq, ref_seq))
        nssr_scores.append(calculate_nssr(pred_seq, ref_seq, blosum_mat))

    # Median recovery rate (the benchmark reports the median for CATH) and mean NSSR,
    # both over the complete reference set.
    return {
        "recovery_rate": float(np.median(recovery_rates)) * 100.0,
        "nssr_blosum62": float(np.mean(nssr_scores)) * 100.0,
        "n_scored": len(recovery_rates),
    }


def run_evaluation():
    """Run the complete evaluation pipeline."""
    results = {}

    for instance_name in INSTANCES:
        print(f"\n{'='*60}")
        print(f"Evaluating instance: {instance_name}")
        print(f"{'='*60}")

        try:
            # Load and validate output
            pred_dict = load_and_validate(instance_name)
            print(f"  Loaded {len(pred_dict)} predictions")

            # Load ground truth
            gt_dict = load_ground_truth(instance_name)
            print(f"  Loaded {len(gt_dict)} ground truth entries")

            # Load BLOSUM matrix
            blosum_mat = load_blosum_matrix(instance_name, "BLOSUM62")

            # Calculate metrics
            scores = calculate_metrics(pred_dict, gt_dict, blosum_mat)
            results[instance_name] = scores
            print(f"  Results: {scores}")

        except ValidationError as e:
            print(f"  [Validation Error] {instance_name}: {e}")
            results[instance_name] = error_result(f"Validation: {e}")

        except Exception as e:
            print(f"  [Error] {instance_name}: {e}")
            import traceback
            traceback.print_exc()
            results[instance_name] = error_result(e)

    return results


if __name__ == "__main__":
    metrics = run_evaluation()
    print("\n=== Final Results ===")
    print(json.dumps(metrics, indent=2))
    with open("score.json", "w") as f:
        json.dump(metrics, f, indent=2)
