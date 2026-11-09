import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Setup Paths
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.environ.get("GROUND_TRUTH_DIR", os.path.join(EVAL_DIR, "ground_truth"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
if not OUTPUT_DIR:
    raise RuntimeError("OUTPUT_DIR environment variable is required")

OUTPUT_FILE = "predictions.csv"

PPI_INSTANCES = [
    "cross_species_fly",
    "cross_species_worm",
    "cross_species_yeast",
    "cross_species_ecoli",
    "human_heldout",
    "virus_human",
]

MUTATION_INSTANCES = [
    "mutation_effect",
]

INSTANCES = PPI_INSTANCES + MUTATION_INSTANCES

PPI_METRICS = ["AUPR"]
MUTATION_METRICS = ["AUPR", "AUROC"]


class ValidationError(Exception):
    """Raised when output file fails validation."""
    pass


def error_result(error_msg, metric_names):
    """Return a result dict with null metrics and error info."""
    result = {name: None for name in metric_names}
    result["error"] = str(error_msg)
    return result


def get_metrics_for_instance(instance_name):
    if instance_name in MUTATION_INSTANCES:
        return MUTATION_METRICS
    return PPI_METRICS


def load_and_validate(instance_name):
    """Load and validate predictions for an instance."""
    pred_file = os.path.join(OUTPUT_DIR, instance_name, OUTPUT_FILE)

    if not os.path.exists(pred_file):
        raise ValidationError(f"Output file not found: {pred_file}")

    try:
        df = pd.read_csv(pred_file)
    except Exception as e:
        raise ValidationError(f"Failed to read CSV: {e}")

    # v5: the file must have exactly one column named prediction_score (an extra column or a duplicated
    # header, which pandas renames to 'prediction_score.1', is rejected)
    if list(df.columns) != ["prediction_score"]:
        raise ValidationError(
            f"predictions.csv must have exactly one column 'prediction_score'. Found columns: {list(df.columns)}"
        )

    try:
        predictions = pd.to_numeric(df["prediction_score"], errors="raise").values.astype(float)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Predictions contain non-numeric values: {e}")

    if np.any(np.isnan(predictions)):
        raise ValidationError("Predictions contain NaN values")

    if np.any(np.isinf(predictions)):
        raise ValidationError("Predictions contain infinite values")

    if np.any(predictions < 0) or np.any(predictions > 1):
        raise ValidationError(
            f"Predictions must be in [0, 1]. Found range: [{predictions.min()}, {predictions.max()}]"
        )

    # Check row count against ground truth
    gt_file = os.path.join(GROUND_TRUTH_DIR, instance_name, "y_ref.csv")
    gt_df = pd.read_csv(gt_file)
    expected_rows = len(gt_df)
    if len(predictions) != expected_rows:
        raise ValidationError(
            f"Row count mismatch: predictions has {len(predictions)} rows, "
            f"expected {expected_rows}"
        )

    return predictions


def load_ground_truth(instance_name):
    """Load ground truth labels for an instance."""
    gt_file = os.path.join(GROUND_TRUTH_DIR, instance_name, "y_ref.csv")
    df = pd.read_csv(gt_file)
    return df["label"].values.astype(int)


DATA_DIR = os.environ.get("DATA_DIR")


def per_protein_center(predictions, a_ids, b_ids, iterations=50):
    """v5: remove an additive per-protein offset from the logit scores (two-way centering on the test pair graph).

    Every protein of a pair test file has exactly three negatives per positive, so a per-protein offset carries no
    label information; centering makes a submission's score independent of such offsets (it neutralises label-free
    per-protein recalibration) and keeps only within-protein pair preferences.
    """
    p = np.clip(np.asarray(predictions, dtype=float), 1e-6, 1 - 1e-6)
    s = np.log(p / (1 - p))
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(s)
    s = (s - s.mean()) / sd
    ids = pd.Index(np.unique(np.r_[a_ids, b_ids]))
    ia, ib = ids.get_indexer(a_ids), ids.get_indexer(b_ids)
    cnt = np.bincount(ia, minlength=len(ids)) + np.bincount(ib, minlength=len(ids))
    bias = np.zeros(len(ids))
    for _ in range(iterations):
        r = s - bias[ia] - bias[ib]
        bias += 0.5 * (np.bincount(ia, r, len(ids)) + np.bincount(ib, r, len(ids))) / cnt
    return s - bias[ia] - bias[ib]


def center_for_instance(predictions, instance_name):
    if instance_name not in PPI_INSTANCES:
        return predictions
    if not DATA_DIR:
        raise RuntimeError("DATA_DIR is required for per-protein centering")
    x = pd.read_csv(os.path.join(DATA_DIR, instance_name, "x_test.csv"))
    if len(x) != len(predictions):
        raise RuntimeError("x_test.csv row count does not match predictions")
    return per_protein_center(predictions, x["protein_a_id"].astype(str).values, x["protein_b_id"].astype(str).values)


def calculate_metrics(predictions, ground_truth, instance_name):
    """Calculate evaluation metrics (v5: pair instances are scored after per-protein centering)."""
    metrics = {}
    predictions = center_for_instance(predictions, instance_name)

    # AUPR (all instances)
    metrics["AUPR"] = float(average_precision_score(ground_truth, predictions))

    # AUROC (mutation instances only)
    if instance_name in MUTATION_INSTANCES:
        metrics["AUROC"] = float(roc_auc_score(ground_truth, predictions))

    return metrics


def run_evaluation():
    """Run the complete evaluation pipeline."""
    results = {}

    for instance_name in INSTANCES:
        print(f"\n{'='*60}")
        print(f"Evaluating instance: {instance_name}")
        print(f"{'='*60}")

        metric_names = get_metrics_for_instance(instance_name)

        try:
            predictions = load_and_validate(instance_name)
            ground_truth = load_ground_truth(instance_name)
            scores = calculate_metrics(predictions, ground_truth, instance_name)
            results[instance_name] = scores
            print(f"Results: {scores}")

        except ValidationError as e:
            print(f"[Validation Error] {instance_name}: {e}")
            results[instance_name] = error_result(f"Validation: {e}", metric_names)

        except Exception as e:
            print(f"[Error] {instance_name}: {e}")
            import traceback
            traceback.print_exc()
            results[instance_name] = error_result(e, metric_names)

    return results


if __name__ == "__main__":
    metrics = run_evaluation()
    print("\n=== Final Results ===")
    print(json.dumps(metrics, indent=2))
    with open("score.json", "w") as f:
        json.dump(metrics, f, indent=2)
