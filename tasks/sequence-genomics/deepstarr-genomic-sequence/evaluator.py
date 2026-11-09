import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from scipy.stats import pearsonr

# Setup Paths
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.environ.get("GROUND_TRUTH_DIR", os.path.join(EVAL_DIR, "ground_truth"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
if not OUTPUT_DIR:
    raise RuntimeError("OUTPUT_DIR environment variable is required")

# Output specification (must match README Section 5)
OUTPUT_FILE = "predictions.csv"

# corrected_v4 (2026-10-16): five instances that a kept member of the same family subsumes
# were dropped -- `enhancers` (a bit-identical binarisation of `enhancers_types`, same 29,803
# train and 2,943 test sequences), `promoter_no_tata` / `promoter_tata` (sub-samples of the
# `promoter_all` problem, 744/1,363 and 109/212 test sequences shared with it) and
# `splice_sites_acceptors` / `splice_sites_donors` (both solved by the 3-class
# `splice_sites_all`). 19 instances -> 14 distinct problems.
CLASSIFICATION_INSTANCES = [
    "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2",
    "H3K4me3", "H3K9ac", "H3K9me3", "H4K20me1",
    "enhancers_types", "promoter_all", "splice_sites_all"
]

REGRESSION_INSTANCES = ["deepstarr"]

INSTANCES = CLASSIFICATION_INSTANCES + REGRESSION_INSTANCES

# Valid label sets for classification tasks
BINARY_TASKS = [
    "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2",
    "H3K4me3", "H3K9ac", "H3K9me3", "H4K20me1", "promoter_all"
]
MULTICLASS_TASKS = ["enhancers_types", "splice_sites_all"]

# Expected test sample counts. corrected_v4: `deepstarr` ships one orientation per region
# (20,593 instead of 41,186 reverse-complement-paired rows carrying identical labels).
# v5: the 13 classification test sets are deduplicated by sequence (335 duplicate rows
# removed in total), so each one shrank; the counts below are the v5 row counts.
EXPECTED_COUNTS = {
    "H2AFZ": 2960, "H3K27ac": 1605, "H3K27me3": 2947, "H3K36me3": 2960,
    "H3K4me1": 2967, "H3K4me2": 2114, "H3K4me3": 773, "H3K9ac": 1000,
    "H3K9me3": 845, "H4K20me1": 2242,
    "enhancers_types": 2943, "promoter_all": 1571, "splice_sites_all": 2976,
    "deepstarr": 20593
}


class ValidationError(Exception):
    """Raised when output file fails validation."""
    pass


def error_result(error_msg, instance_name):
    """Return a result dict with null metrics and error info."""
    if instance_name in CLASSIFICATION_INSTANCES:
        result = {"MCC": None}
    else:
        result = {"Pearson_Dev": None, "Pearson_Hk": None}
    result["error"] = str(error_msg)
    return result


def load_and_validate_classification(instance_name):
    """Load and validate classification output.

    Output is a single-column CSV with header 'label', row-aligned with test.parquet.
    """
    pred_file = os.path.join(OUTPUT_DIR, instance_name, OUTPUT_FILE)

    if not os.path.exists(pred_file):
        raise ValidationError(f"Output file not found: {pred_file}")

    try:
        pred_df = pd.read_csv(pred_file)
    except Exception as e:
        raise ValidationError(f"Failed to read CSV: {e}")

    # Check required column
    if "label" not in pred_df.columns:
        raise ValidationError("Missing required column 'label' in predictions.csv")

    # Check sample count
    expected = EXPECTED_COUNTS[instance_name]
    if len(pred_df) != expected:
        raise ValidationError(
            f"Expected {expected} predictions, got {len(pred_df)}"
        )

    # Check label values
    unique_labels = set(pred_df["label"].unique())
    if instance_name in BINARY_TASKS:
        valid_labels = {0, 1}
    else:
        valid_labels = {0, 1, 2}
    if not unique_labels.issubset(valid_labels):
        raise ValidationError(
            f"Invalid labels found: {unique_labels - valid_labels}. "
            f"Expected subset of {valid_labels}"
        )

    # Check for integer type
    if not np.issubdtype(pred_df["label"].dtype, np.integer):
        # Try to convert
        try:
            pred_df["label"] = pred_df["label"].astype(int)
        except (ValueError, TypeError):
            raise ValidationError("Column 'label' must contain integer values")

    return pred_df


def load_and_validate_regression(instance_name):
    """Load and validate regression output for deepstarr."""
    pred_file = os.path.join(OUTPUT_DIR, instance_name, OUTPUT_FILE)

    if not os.path.exists(pred_file):
        raise ValidationError(f"Output file not found: {pred_file}")

    try:
        pred_df = pd.read_csv(pred_file)
    except Exception as e:
        raise ValidationError(f"Failed to read CSV: {e}")

    # Check required columns (v3: an `id` column matching the opaque FASTA ids is required)
    if "id" not in pred_df.columns:
        raise ValidationError("Missing required column 'id' in predictions.csv (corrected_v3 format)")
    required_cols = ["Dev_log2_enrichment", "Hk_log2_enrichment"]
    for col in required_cols:
        if col not in pred_df.columns:
            raise ValidationError(
                f"Missing required column '{col}' in predictions.csv"
            )

    # Check sample count
    expected = EXPECTED_COUNTS[instance_name]
    if len(pred_df) != expected:
        raise ValidationError(
            f"Expected {expected} predictions, got {len(pred_df)}"
        )

    ids = pred_df["id"].astype(str)
    if ids.duplicated().any():
        raise ValidationError("Duplicate ids in deepstarr predictions.csv")
    # Check for numeric values
    for col in required_cols:
        if not np.issubdtype(pred_df[col].dtype, np.number):
            raise ValidationError(f"Column '{col}' must contain numeric values")
        if pred_df[col].isna().any():
            raise ValidationError(f"Column '{col}' contains NaN values")
        if np.isinf(pred_df[col].values).any():
            raise ValidationError(f"Column '{col}' contains infinite values")

    return pred_df


def calculate_classification_metrics(predictions_df, ground_truth_df):
    """Calculate MCC for a classification instance.

    Uses positional (row-order) alignment.
    """
    y_true = ground_truth_df["label"].values
    y_pred = predictions_df["label"].values

    mcc = matthews_corrcoef(y_true, y_pred)

    return {"MCC": float(mcc)}


def calculate_regression_metrics(predictions_df, ground_truth_df):
    """Calculate Pearson correlations for the deepstarr instance (v3: aligned by opaque id)."""
    gt_ids = ground_truth_df["id"].astype(str)
    pr_ids = predictions_df["id"].astype(str)
    if set(gt_ids) != set(pr_ids):
        raise ValidationError("deepstarr prediction ids must equal the Sequences_Test.fa ids exactly")
    predictions_df = predictions_df.assign(id=pr_ids).set_index("id").loc[gt_ids.values].reset_index()
    pred_dev = predictions_df["Dev_log2_enrichment"].values
    pred_hk = predictions_df["Hk_log2_enrichment"].values
    true_dev = ground_truth_df["Dev_log2_enrichment"].values
    true_hk = ground_truth_df["Hk_log2_enrichment"].values

    pearson_dev, _ = pearsonr(pred_dev, true_dev)
    pearson_hk, _ = pearsonr(pred_hk, true_hk)

    # v3: a constant prediction column has undefined Pearson r; score it as 0 (chance) instead of erroring
    pearson_dev = 0.0 if not np.isfinite(pearson_dev) else pearson_dev
    pearson_hk = 0.0 if not np.isfinite(pearson_hk) else pearson_hk
    return {
        "Pearson_Dev": float(pearson_dev),
        "Pearson_Hk": float(pearson_hk)
    }


def load_ground_truth_classification(instance_name):
    """Load ground truth for a classification instance."""
    gt_file = os.path.join(GROUND_TRUTH_DIR, instance_name, "labels.parquet")
    return pd.read_parquet(gt_file)


def load_ground_truth_regression(instance_name):
    """Load ground truth for the deepstarr regression instance."""
    gt_file = os.path.join(
        GROUND_TRUTH_DIR, instance_name, "Sequences_activity_Test.txt"
    )
    return pd.read_csv(gt_file, sep="\t")


def run_evaluation():
    """Run the complete evaluation pipeline."""
    results = {}

    for instance_name in INSTANCES:
        print(f"\n{'='*60}")
        print(f"Evaluating instance: {instance_name}")
        print(f"{'='*60}")

        try:
            if instance_name in CLASSIFICATION_INSTANCES:
                predictions = load_and_validate_classification(instance_name)
                ground_truth = load_ground_truth_classification(instance_name)
                scores = calculate_classification_metrics(predictions, ground_truth)
            else:
                predictions = load_and_validate_regression(instance_name)
                ground_truth = load_ground_truth_regression(instance_name)
                scores = calculate_regression_metrics(predictions, ground_truth)

            results[instance_name] = scores
            print(f"Results: {scores}")

        except ValidationError as e:
            print(f"[Validation Error] {instance_name}: {e}")
            results[instance_name] = error_result(
                f"Validation: {e}", instance_name
            )

        except Exception as e:
            print(f"[Error] {instance_name}: {e}")
            import traceback
            traceback.print_exc()
            results[instance_name] = error_result(e, instance_name)

    return results


if __name__ == "__main__":
    metrics = run_evaluation()
    print("\n=== Final Results ===")
    print(json.dumps(metrics, indent=2))
    with open("score.json", "w") as f:
        json.dump(metrics, f, indent=2)
