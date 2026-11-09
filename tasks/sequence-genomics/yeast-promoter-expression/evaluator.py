import os
import json
import csv
import numpy as np
from collections import OrderedDict
from scipy.stats import pearsonr, spearmanr

# Setup Paths
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.environ.get("GROUND_TRUTH_DIR", os.path.join(EVAL_DIR, "ground_truth"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
if not OUTPUT_DIR:
    raise RuntimeError("OUTPUT_DIR environment variable is required")

OUTPUT_FILE = "predictions.npy"
INSTANCES = ["yeast_promoter"]
METRIC_NAMES = ["Pearson Score", "DREAM r2 Score", "Spearman Score"]

# Subset weights as defined in the evaluation protocol (Table 1)
SUBSET_WEIGHTS = {
    "all": 1.0,
    "high_exp": 0.3,
    "low_exp": 0.3,
    "yeast": 0.3,
    "random": 0.3,
    "challenging": 0.5,
    "SNVs": 1.25,
    "motif_perturbation": 0.3,
    "motif_tiling": 0.4,
}
TOTAL_WEIGHT = sum(SUBSET_WEIGHTS.values())  # 4.65

# Single-index subsets (correlation on raw values)
SINGLE_SUBSETS = {
    "all": None,  # all indices (after public LB removal)
    "high_exp": "high_exp_seqs.csv",
    "low_exp": "low_exp_seqs.csv",
    "yeast": "yeast_seqs.csv",
    "random": "all_random_seqs.csv",
    "challenging": "challenging_seqs.csv",
}

# Paired subsets (correlation on expression differences)
PAIRED_SUBSETS = {
    "SNVs": "all_SNVs_seqs.csv",
    "motif_perturbation": "motif_perturbation_seqs.csv",
    "motif_tiling": "motif_tiling_seqs.csv",
}

# Public leaderboard index files (to be discarded)
PUBLIC_LB_SINGLE_FILES = {
    "high_exp": "high_exp_indices.json",
    "low_exp": "low_exp_indices.json",
    "yeast": "yeast_exp_indices.json",
    "random": "random_exp_indices.json",
    "challenging": "challenging_exp_indices.json",
}
PUBLIC_LB_PAIRED_FILES = {
    "SNVs": "SNVs_exp_indices.json",
    "motif_perturbation": "motif_perturbation_exp_indices.json",
    "motif_tiling": "motif_tiling_exp_indices.json",
}


class ValidationError(Exception):
    pass


def error_result(error_msg):
    result = {name: None for name in METRIC_NAMES}
    result["error"] = str(error_msg)
    return result


def load_ground_truth(gt_dir):
    """Load expression values from ground truth CSV."""
    expr_path = os.path.join(gt_dir, "expression.csv")
    expressions = {}
    with open(expr_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expressions[int(row["idx"])] = float(row["expression"])
    n_samples = len(expressions)
    gt_array = np.array([expressions[i] for i in range(n_samples)])
    return gt_array


def load_subset_indices(gt_dir, filename):
    """Load single-index subset (pos column)."""
    path = os.path.join(gt_dir, "test_subset_ids", filename)
    indices = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            indices.append(int(row["pos"]))
    return np.unique(np.array(indices))


def load_paired_indices(gt_dir, filename):
    """Load paired-index subset (alt_pos, ref_pos columns)."""
    path = os.path.join(gt_dir, "test_subset_ids", filename)
    pairs = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((int(row["alt_pos"]), int(row["ref_pos"])))
    return list(set(pairs))


def load_public_lb_single(gt_dir, filename):
    """Load public leaderboard single indices from JSON."""
    path = os.path.join(gt_dir, "public_leaderboard_ids", filename)
    with open(path, "r") as f:
        data = json.load(f)
    return [int(k) for k in data.keys()]


def load_public_lb_paired(gt_dir, filename):
    """Load public leaderboard paired indices from JSON."""
    path = os.path.join(gt_dir, "public_leaderboard_ids", filename)
    with open(path, "r") as f:
        data = json.load(f)
    return [(int(k.split(",")[0]), int(k.split(",")[1])) for k in data.keys()]


def load_and_validate(instance_name):
    """Load and validate predictions."""
    pred_file = os.path.join(OUTPUT_DIR, instance_name, OUTPUT_FILE)

    if not os.path.exists(pred_file):
        raise ValidationError(f"Output file not found: {pred_file}")

    predictions = np.load(pred_file)

    if predictions.ndim != 1:
        raise ValidationError(
            f"Expected 1D array, got shape {predictions.shape}"
        )

    if predictions.shape[0] != 71103:
        raise ValidationError(
            f"Expected 71103 predictions, got {predictions.shape[0]}"
        )

    if not np.issubdtype(predictions.dtype, np.floating):
        raise ValidationError(
            f"Expected float dtype, got {predictions.dtype}"
        )

    if np.any(np.isnan(predictions)):
        raise ValidationError("Predictions contain NaN values")

    if np.any(np.isinf(predictions)):
        raise ValidationError("Predictions contain Inf values")

    return predictions


def _safe_corr(gt, pred):
    """v2b: a constant prediction vector has no correlation -> 0 (chance), not NaN."""
    if np.ptp(pred) == 0 or np.ptp(gt) == 0:
        return 0.0, 0.0
    r, _ = pearsonr(gt, pred)
    rho, _ = spearmanr(gt, pred)
    r = 0.0 if not np.isfinite(r) else float(r)
    rho = 0.0 if not np.isfinite(rho) else float(rho)
    return r, rho


def calculate_metrics(predictions, gt_dir):
    """Calculate Pearson Score and Spearman Score."""
    ground_truth = load_ground_truth(gt_dir)

    # Collect all public leaderboard indices to discard
    public_single_indices = []
    for subset_name, filename in PUBLIC_LB_SINGLE_FILES.items():
        public_single_indices.extend(load_public_lb_single(gt_dir, filename))

    public_paired_indices = []
    for subset_name, filename in PUBLIC_LB_PAIRED_FILES.items():
        public_paired_indices.extend(load_public_lb_paired(gt_dir, filename))

    # Build set of all public indices (both from single and paired)
    public_indices_set = set(public_single_indices)
    for alt, ref in public_paired_indices:
        public_indices_set.add(alt)
        public_indices_set.add(ref)

    public_paired_set = set(public_paired_indices)

    # Compute per-subset correlations
    pearson_components = {}
    spearman_components = {}

    # Single-index subsets
    for subset_name, filename in SINGLE_SUBSETS.items():
        if subset_name == "all":
            # All non-public indices
            indices = [
                i for i in range(len(ground_truth)) if i not in public_indices_set
            ]
        else:
            raw_indices = load_subset_indices(gt_dir, filename)
            indices = [i for i in raw_indices if i not in public_indices_set]

        gt_sub = np.array([ground_truth[i] for i in indices])
        pred_sub = np.array([predictions[i] for i in indices])

        r, rho = _safe_corr(gt_sub, pred_sub)

        pearson_components[subset_name] = r
        spearman_components[subset_name] = rho

    # Paired subsets (difference correlations)
    for subset_name, filename in PAIRED_SUBSETS.items():
        raw_pairs = load_paired_indices(gt_dir, filename)
        # Filter out public leaderboard pairs
        pairs = [p for p in raw_pairs if p not in public_paired_set]

        gt_diffs = np.array(
            [ground_truth[alt] - ground_truth[ref] for alt, ref in pairs]
        )
        pred_diffs = np.array(
            [predictions[alt] - predictions[ref] for alt, ref in pairs]
        )

        r, rho = _safe_corr(gt_diffs, pred_diffs)

        pearson_components[subset_name] = r
        spearman_components[subset_name] = rho

    # Compute weighted scores
    # v5: the primary Pearson Score uses max(r, 0)^2 so that predictions anti-correlated with
    # the true expression / mutation effect score 0 instead of 1 (plain r^2 is direction-blind:
    # a perfectly negated submission used to score 1.0).  The original DREAM-2022 r^2 is kept
    # as a secondary diagnostic under "DREAM r2 Score".
    pearson_score = sum(
        SUBSET_WEIGHTS[s] * max(pearson_components[s], 0.0) ** 2
        for s in SUBSET_WEIGHTS
    ) / TOTAL_WEIGHT

    dream_r2_score = sum(
        SUBSET_WEIGHTS[s] * pearson_components[s] ** 2
        for s in SUBSET_WEIGHTS
    ) / TOTAL_WEIGHT

    # Spearman Score uses rho directly
    spearman_score = sum(
        SUBSET_WEIGHTS[s] * spearman_components[s]
        for s in SUBSET_WEIGHTS
    ) / TOTAL_WEIGHT

    return {
        "Pearson Score": float(pearson_score),
        "DREAM r2 Score": float(dream_r2_score),
        "Spearman Score": float(spearman_score),
    }


def run_evaluation():
    """Run the complete evaluation pipeline."""
    results = {}

    for instance_name in INSTANCES:
        print(f"\n{'='*60}")
        print(f"Evaluating instance: {instance_name}")
        print(f"{'='*60}")

        try:
            predictions = load_and_validate(instance_name)
            gt_dir = os.path.join(GROUND_TRUTH_DIR, instance_name)
            scores = calculate_metrics(predictions, gt_dir)
            results[instance_name] = scores
            print(f"Results: {scores}")

        except ValidationError as e:
            print(f"[Validation Error] {instance_name}: {e}")
            results[instance_name] = error_result(f"Validation: {e}")

        except Exception as e:
            print(f"[Error] {instance_name}: {e}")
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
