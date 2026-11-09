import os
import json
import numpy as np
import pandas as pd

# Setup Paths
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.environ.get("GROUND_TRUTH_DIR", os.path.join(EVAL_DIR, "ground_truth"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
if not OUTPUT_DIR:
    raise RuntimeError("OUTPUT_DIR environment variable is required")

OUTPUT_FILE = "output.json"
# BioXArena-Pro corrected_v2 (2026-10-13): one instance, four forecast origins (50 states each).
# Every metric is computed per origin (y_ref column `week_id`) and averaged over origins;
# Accuracy_origin_mean = mean over origins of (Accuracy_t1 + Accuracy_t3) / 2 is the primary metric.
INSTANCES = ["multi_origin"]

# 5-class ordinal HTC labels
CLASSES = [
    "SUBSTANTIAL INCREASING",
    "MODERATE INCREASING",
    "STABLE",
    "MODERATE DECREASING",
    "SUBSTANTIAL DECREASING",
]
# Ordinal numeric value mapping (used for MSE, WMSE, RPS)
CLASS_VALUES = {
    "SUBSTANTIAL DECREASING": 0,
    "MODERATE DECREASING": 1,
    "STABLE": 2,
    "MODERATE INCREASING": 3,
    "SUBSTANTIAL INCREASING": 4,
}
NUM_CLASSES = len(CLASSES)

# v5 (2026-10-17): primary metric = RPS skill against the training climatology.
# CLIMATOLOGY = class frequencies of t1 / t3 over the public train_indices rows (weeks 0-63) of
# train_val_data.pkl; fixed here so that the score does not depend on the submission.
CLIMATOLOGY = {"t1": {"SUBSTANTIAL DECREASING": 0.2946875, "MODERATE DECREASING": 0.1778125, "STABLE": 0.213125, "MODERATE INCREASING": 0.118125, "SUBSTANTIAL INCREASING": 0.19625}, "t3": {"SUBSTANTIAL DECREASING": 0.328125, "MODERATE DECREASING": 0.17375, "STABLE": 0.175625, "MODERATE INCREASING": 0.101875, "SUBSTANTIAL INCREASING": 0.220625}}

METRIC_NAMES = [
    "RPS_skill",
    "Accuracy_origin_mean",
    "Accuracy_t1", "Accuracy_t3",
    "MSE_t1", "MSE_t3",
    "WMSE_t1", "WMSE_t3",
    "BrierScore_t1", "BrierScore_t3",
    "RPS_t1", "RPS_t3",
]


class ValidationError(Exception):
    """Raised when output file fails validation."""
    pass


def error_result(error_msg):
    """Return a result dict with null metrics and error info."""
    result = {name: None for name in METRIC_NAMES}
    result["error"] = str(error_msg)
    return result


def load_and_validate(instance_name, n_expected):
    """
    Load and validate the output file for an instance.

    Args:
        instance_name: Name of the task instance (v1, v2, v3).
        n_expected: Expected number of test samples.

    Returns:
        Tuple (t1_probs, t3_probs) where each is a list of dicts
        mapping class string -> float probability.

    Raises:
        ValidationError: If the output file is missing or invalid.
    """
    pred_file = os.path.join(OUTPUT_DIR, instance_name, OUTPUT_FILE)

    if not os.path.exists(pred_file):
        raise ValidationError(f"Output file not found: {pred_file}")

    try:
        with open(pred_file, "r") as f:
            output = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in output file: {e}")

    if not isinstance(output, dict):
        raise ValidationError(f"Output must be a JSON object, got {type(output).__name__}")

    for key in ("t1", "t3"):
        if key not in output:
            raise ValidationError(f"Output missing required key '{key}'")
        preds = output[key]
        if not isinstance(preds, list):
            raise ValidationError(f"output['{key}'] must be a list, got {type(preds).__name__}")
        if len(preds) != n_expected:
            raise ValidationError(
                f"output['{key}'] has {len(preds)} entries, expected {n_expected}"
            )
        for i, dist in enumerate(preds):
            if not isinstance(dist, dict):
                raise ValidationError(
                    f"output['{key}'][{i}] must be a dict, got {type(dist).__name__}"
                )
            keys = set(dist.keys())
            if keys != set(CLASSES):
                raise ValidationError(
                    f"output['{key}'][{i}] must have exactly the five class keys; "
                    f"missing {sorted(set(CLASSES) - keys)}, unknown {sorted(keys - set(CLASSES))}"
                )
            for cls in CLASSES:
                val = dist[cls]
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ValidationError(
                        f"output['{key}'][{i}]['{cls}'] must be numeric, got {type(val).__name__}"
                    )
                if not np.isfinite(val):
                    raise ValidationError(f"output['{key}'][{i}]['{cls}'] is not finite")
                if val < 0 or val > 1.000001:
                    raise ValidationError(
                        f"output['{key}'][{i}]['{cls}'] = {val} is outside [0, 1]"
                    )
            total = float(sum(dist[c] for c in CLASSES))
            if abs(total - 1.0) > 1e-3:
                raise ValidationError(
                    f"output['{key}'][{i}] probabilities sum to {total:.6f}, expected 1 (tolerance 1e-3)"
                )

    return output["t1"], output["t3"]


def _argmax(dist):
    """v5: deterministic argmax; ties go to the first class in the fixed CLASSES order."""
    return max(CLASSES, key=lambda c: (dist[c], -CLASSES.index(c)))


def calc_accuracy(y_true_labels, pred_dists):
    """
    Accuracy: fraction of samples where argmax(predicted distribution) == true label.

    Args:
        y_true_labels: array-like of string class labels.
        pred_dists: list of dicts {class_str: prob}.

    Returns:
        float accuracy in [0, 1].
    """
    correct = 0
    for true_label, dist in zip(y_true_labels, pred_dists):
        pred_label = _argmax(dist)
        if pred_label == true_label:
            correct += 1
    return correct / len(y_true_labels)


def calc_mse(y_true_labels, pred_dists):
    """
    MSE on ordinal class values: compares argmax prediction vs. true class numeric value.

    Args:
        y_true_labels: array-like of string class labels.
        pred_dists: list of dicts {class_str: prob}.

    Returns:
        float MSE.
    """
    errors = []
    for true_label, dist in zip(y_true_labels, pred_dists):
        true_val = CLASS_VALUES[true_label]
        pred_label = _argmax(dist)
        pred_val = CLASS_VALUES[pred_label]
        errors.append((pred_val - true_val) ** 2)
    return float(np.mean(errors))


def calc_wmse(y_true_labels, pred_dists):
    """
    Weighted MSE using predicted probability distribution.
    WMSE = (1/N) sum_i sum_k P(y_hat_i = k) * (k - y_true_i)^2

    Args:
        y_true_labels: array-like of string class labels.
        pred_dists: list of dicts {class_str: prob}.

    Returns:
        float WMSE.
    """
    errors = []
    for true_label, dist in zip(y_true_labels, pred_dists):
        true_val = CLASS_VALUES[true_label]
        sample_error = 0.0
        for cls, prob in dist.items():
            k = CLASS_VALUES[cls]
            sample_error += prob * (k - true_val) ** 2
        errors.append(sample_error)
    return float(np.mean(errors))


def calc_brier_score(y_true_labels, pred_dists):
    """
    Brier Score: mean squared difference between predicted probabilities and one-hot ground truth.
    BS = (1/N) sum_i sum_k (P(y_hat_i = k) - o_k^(i))^2

    Args:
        y_true_labels: array-like of string class labels.
        pred_dists: list of dicts {class_str: prob}.

    Returns:
        float Brier Score.
    """
    errors = []
    for true_label, dist in zip(y_true_labels, pred_dists):
        sample_error = 0.0
        for cls in CLASSES:
            prob = dist.get(cls, 0.0)
            one_hot = 1.0 if cls == true_label else 0.0
            sample_error += (prob - one_hot) ** 2
        errors.append(sample_error)
    return float(np.mean(errors))


def calc_rps(y_true_labels, pred_dists):
    """
    Ranked Probability Score over cumulative probability distributions.
    RPS = (1/(N*(K-1))) sum_i sum_k (q_hat_k^(i) - q_k^(i))^2

    where q_hat_k = cumulative predicted probability up to class k,
    q_k = step function at true class (0 before, 1 at and after true class in sorted order).

    Class order (ascending): SUBSTANTIAL DECREASING(0), MODERATE DECREASING(1),
    STABLE(2), MODERATE INCREASING(3), SUBSTANTIAL INCREASING(4).

    Args:
        y_true_labels: array-like of string class labels.
        pred_dists: list of dicts {class_str: prob}.

    Returns:
        float RPS.
    """
    # Class order by ascending ordinal value
    ordered_classes = [
        "SUBSTANTIAL DECREASING",
        "MODERATE DECREASING",
        "STABLE",
        "MODERATE INCREASING",
        "SUBSTANTIAL INCREASING",
    ]

    errors = []
    for true_label, dist in zip(y_true_labels, pred_dists):
        true_val = CLASS_VALUES[true_label]
        sample_rps = 0.0
        cum_pred = 0.0
        cum_true = 0.0
        # Accumulate over K-1 thresholds (classes 0 through K-2)
        for cls in ordered_classes[:-1]:
            cls_val = CLASS_VALUES[cls]
            cum_pred += dist.get(cls, 0.0)
            cum_true = 1.0 if cls_val >= true_val else 0.0
            sample_rps += (cum_pred - cum_true) ** 2
        errors.append(sample_rps)

    # Per the paper formula: 1/(N-1) * sum_i sum_k ... but re-reading,
    # the denominator is N samples and K-1 threshold steps per sample.
    # The paper uses RPS = (1/(N-1)) sum_i sum_k ..., which seems to be
    # a per-sample average divided by (K-1). Verified against author code:
    # author averages over samples (res.sum()/len(res)), without dividing by K-1.
    # We follow the same convention: average over samples only.
    return float(np.mean(errors))


def calculate_metrics(t1_preds, t3_preds, y_ref_df):
    """
    Calculate all evaluation metrics for both prediction horizons.

    corrected_v2: if `y_ref_df` carries a `week_id` column, every metric is computed
    separately for each forecast origin (week_id) and the per-origin values are averaged
    with equal weight; `Accuracy_origin_mean` is the mean of the averaged t1 and t3 accuracies.

    Args:
        t1_preds: list of dicts, predicted distributions for 1-week ahead.
        t3_preds: list of dicts, predicted distributions for 3-week ahead.
        y_ref_df: DataFrame with columns 't1' and 't3' (ground truth labels) and optionally 'week_id'.

    Returns:
        dict of {metric_name: float}.
    """
    if "week_id" in y_ref_df.columns:
        origins = sorted(y_ref_df["week_id"].unique())
        per_origin = []
        for origin in origins:
            pos = np.where(y_ref_df["week_id"].values == origin)[0]
            sub = y_ref_df.iloc[pos].drop(columns=["week_id"])
            per_origin.append(_calculate_metrics_single(
                [t1_preds[i] for i in pos], [t3_preds[i] for i in pos], sub))
        results = {k: float(np.mean([m[k] for m in per_origin])) for k in per_origin[0]}
        results["Accuracy_origin_mean"] = 0.5 * (results["Accuracy_t1"] + results["Accuracy_t3"])
        results["RPS_skill"] = _rps_skill(results)
        results["per_origin"] = {str(int(o)): {"RPS_skill": _rps_skill(m), "Accuracy": 0.5 * (m["Accuracy_t1"] + m["Accuracy_t3"])}
                                 for o, m in zip(origins, per_origin)}
        results["n_origins"] = len(origins)
        return results
    results = _calculate_metrics_single(t1_preds, t3_preds, y_ref_df)
    results["Accuracy_origin_mean"] = 0.5 * (results["Accuracy_t1"] + results["Accuracy_t3"])
    results["RPS_skill"] = _rps_skill(results)
    return results


def _rps_skill(m):
    """1 - (RPS_t1 + RPS_t3) / (RPS_clim_t1 + RPS_clim_t3); origin-averaged RPS values when called on the means."""
    return float(1.0 - (m["RPS_t1"] + m["RPS_t3"]) / (m["RPS_clim_t1"] + m["RPS_clim_t3"]))


def _calculate_metrics_single(t1_preds, t3_preds, y_ref_df):
    y_t1 = y_ref_df["t1"].tolist()
    y_t3 = y_ref_df["t3"].tolist()

    results = {}
    results["Accuracy_t1"] = calc_accuracy(y_t1, t1_preds)
    results["Accuracy_t3"] = calc_accuracy(y_t3, t3_preds)
    results["MSE_t1"] = calc_mse(y_t1, t1_preds)
    results["MSE_t3"] = calc_mse(y_t3, t3_preds)
    results["WMSE_t1"] = calc_wmse(y_t1, t1_preds)
    results["WMSE_t3"] = calc_wmse(y_t3, t3_preds)
    results["BrierScore_t1"] = calc_brier_score(y_t1, t1_preds)
    results["BrierScore_t3"] = calc_brier_score(y_t3, t3_preds)
    results["RPS_t1"] = calc_rps(y_t1, t1_preds)
    results["RPS_t3"] = calc_rps(y_t3, t3_preds)
    results["RPS_clim_t1"] = calc_rps(y_t1, [CLIMATOLOGY["t1"]] * len(y_t1))
    results["RPS_clim_t3"] = calc_rps(y_t3, [CLIMATOLOGY["t3"]] * len(y_t3))

    return results


def run_evaluation():
    """Run the complete evaluation pipeline."""
    results = {}

    for instance_name in INSTANCES:
        print(f"\n{'='*60}")
        print(f"Evaluating instance: {instance_name}")
        print(f"{'='*60}")

        try:
            # Load ground truth
            gt_file = os.path.join(GROUND_TRUTH_DIR, instance_name, "y_ref.pkl")
            y_ref = pd.read_pickle(gt_file)
            n_expected = len(y_ref)

            # Load and validate output
            t1_preds, t3_preds = load_and_validate(instance_name, n_expected)

            # Calculate metrics
            scores = calculate_metrics(t1_preds, t3_preds, y_ref)
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
