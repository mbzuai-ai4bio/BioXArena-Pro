import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# Setup Paths
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.environ.get("GROUND_TRUTH_DIR", os.path.join(EVAL_DIR, "ground_truth"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
if not OUTPUT_DIR:
    raise RuntimeError("OUTPUT_DIR environment variable is required")

# Output specification (must match README Section 5)
OUTPUT_FILE = "predictions.npy"
INSTANCES = [
    "cm_rare_variant",
    "arm_rare_variant",
    "clinvar_cm",
    "clinvar_arm",
    "mfass_splicing",
    "clinvar_splicing",
]
METRIC_NAMES = ["AUPR", "AUPR_norm", "AUC"]


class ValidationError(Exception):
    """Raised when output file fails validation."""
    pass


def error_result(error_msg):
    """Return a result dict with null metrics and error info."""
    result = {name: None for name in METRIC_NAMES}
    result["error"] = str(error_msg)
    return result


def load_and_validate(instance_name):
    """
    Load the output file for an instance and validate.

    Args:
        instance_name: Name of the task instance.

    Returns:
        The loaded and validated predictions as a 1D numpy array.

    Raises:
        ValidationError: If the output file is missing or has wrong
            format, shape, dtype, or values out of range.
    """
    pred_file = os.path.join(OUTPUT_DIR, instance_name, OUTPUT_FILE)

    # Check file exists
    if not os.path.exists(pred_file):
        raise ValidationError(f"Output file not found: {pred_file}")

    # Load predictions
    try:
        predictions = np.load(pred_file)
    except Exception as e:
        raise ValidationError(f"Failed to load {pred_file}: {e}")

    # Validate dimensionality
    if predictions.ndim != 1:
        raise ValidationError(
            f"Expected 1D array, got {predictions.ndim}D with shape {predictions.shape}"
        )

    # Load ground truth to check shape alignment
    gt_file = os.path.join(GROUND_TRUTH_DIR, instance_name, "y_ref.csv")
    gt = pd.read_csv(gt_file)
    expected_n = len(gt)

    if len(predictions) != expected_n:
        raise ValidationError(
            f"Shape mismatch: predictions has {len(predictions)} samples, "
            f"expected {expected_n}"
        )

    # Validate dtype is numeric
    if not np.issubdtype(predictions.dtype, np.floating) and not np.issubdtype(
        predictions.dtype, np.integer
    ):
        raise ValidationError(
            f"Expected numeric dtype, got {predictions.dtype}"
        )

    # Check for NaN/Inf
    if np.any(np.isnan(predictions)):
        raise ValidationError("Predictions contain NaN values")
    if np.any(np.isinf(predictions)):
        raise ValidationError("Predictions contain Inf values")

    return predictions.astype(np.float64)


def calculate_metrics(predictions, ground_truth):
    """
    Calculate evaluation metrics.

    Args:
        predictions: Validated predictions (1D array of continuous scores).
        ground_truth: Binary labels (1D array, 0=benign, 1=pathogenic).

    Returns:
        dict: {metric_name: metric_value} matching README metric names exactly.
    """
    # AUC: Area Under the ROC Curve
    auc = float(roc_auc_score(ground_truth, predictions))

    # AUPR: Area Under the Precision-Recall Curve
    aupr = float(average_precision_score(ground_truth, predictions))

    # AUPR_norm (BioXArena-Pro v2, 2026-10-13): AUPR above the instance's positive
    # prevalence, rescaled so that a random or constant ranking scores ~0 and a perfect
    # ranking scores 1 on every instance regardless of its class balance:
    #     AUPR_norm = (AUPR - prevalence) / (1 - prevalence)
    prevalence = float(np.mean(ground_truth))
    aupr_norm = float((aupr - prevalence) / (1.0 - prevalence))

    return {"AUPR": aupr, "AUPR_norm": aupr_norm, "AUC": auc}


def run_evaluation():
    """Run the complete evaluation pipeline."""
    results = {}

    for instance_name in INSTANCES:
        print(f"\n{'='*60}")
        print(f"Evaluating instance: {instance_name}")
        print(f"{'='*60}")

        try:
            # Load and validate output
            predictions = load_and_validate(instance_name)

            # Load ground truth
            gt_file = os.path.join(GROUND_TRUTH_DIR, instance_name, "y_ref.csv")
            gt = pd.read_csv(gt_file)
            ground_truth = gt["labels"].values

            # Calculate metrics
            scores = calculate_metrics(predictions, ground_truth)
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
