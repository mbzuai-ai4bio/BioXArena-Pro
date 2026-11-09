"""Evaluator for structure/ppi-interface-residues, data version corrected_v2 (2026-10-13).

Primary metric: mean per-chain AUROC.  Each of the 800 test chains (400 pairs x 2 sides)
is scored separately on its valid (non-nan) residues; chains with fewer than MIN_POS
interface residues or fewer than MIN_NEG non-interface residues are excluded from the
average (the exclusion is fixed by the hidden labels, not by the submission).  Because
every residue is ranked only against residues of the same chain, protein length and the
global interface fraction carry no information (the v1 pooled AUROC gave 0.624 to
-length alone).

Validation is strict: exactly 400 rows, the exact pair_id set in the reference order,
no duplicates, one score per residue, every score finite and in [0, 1].  Any violation
is a ValidationError -> Status=Fail, Score=null.  No partial submission is scored.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
)

# Setup Paths
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.environ.get("GROUND_TRUTH_DIR", os.path.join(EVAL_DIR, "ground_truth"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
if not OUTPUT_DIR:
    raise RuntimeError("OUTPUT_DIR environment variable is required")

OUTPUT_FILE = "predictions.csv"
INSTANCES = ["ppi_interface_benchmark"]
N_PAIRS = 400
MIN_POS = 5   # a chain enters the average only with >= MIN_POS interface residues
MIN_NEG = 5   # ... and >= MIN_NEG non-interface residues (valid, non-nan labels only)
PRIMARY = "AUROC_per_chain_mean"
METRIC_NAMES = [
    PRIMARY, "AUPR_per_chain_mean", "AUROC_pooled", "AUPR_pooled",
    "F1-score", "MCC", "Precision", "Recall", "Accuracy",
    "n_chains_evaluated", "n_chains_excluded",
]


class ValidationError(Exception):
    """Raised when output file fails validation."""
    pass


def error_result(error_msg):
    """Return a result dict with null metrics and error info."""
    result = {name: None for name in METRIC_NAMES}
    result["error"] = str(error_msg)
    return result


def parse_scores(score_str, where):
    """Parse a semicolon-separated score string; every token must be a finite float."""
    if score_str is None or (isinstance(score_str, float) and np.isnan(score_str)):
        raise ValidationError(f"{where}: empty score field")
    tokens = str(score_str).split(";")
    try:
        arr = np.array([float(t) for t in tokens], dtype=float)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"{where}: cannot parse scores: {e}")
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValidationError(f"{where}: scores contain nan/inf or empty tokens")
    return arr


def parse_labels(label_str):
    """Parse semicolon-separated label string, returning array with nan for missing."""
    tokens = str(label_str).split(";")
    result = []
    for t in tokens:
        t = t.strip()
        if t.lower() == "nan" or t == "":
            result.append(np.nan)
        else:
            result.append(float(t))
    return np.array(result)


def load_and_validate(instance_name):
    """Load and validate prediction file."""
    pred_file = os.path.join(OUTPUT_DIR, instance_name, OUTPUT_FILE)

    if not os.path.exists(pred_file):
        raise ValidationError(f"Output file not found: {pred_file}")

    pred_df = pd.read_csv(pred_file, dtype=str, keep_default_na=False)

    required_cols = ["pair_id", "scores1", "scores2"]
    if list(pred_df.columns) != required_cols:
        raise ValidationError(
            f"Columns must be exactly {required_cols}, got {list(pred_df.columns)}"
        )

    if len(pred_df) != N_PAIRS:
        raise ValidationError(f"Expected {N_PAIRS} rows, got {len(pred_df)}")

    if pred_df["pair_id"].duplicated().any():
        dup = pred_df["pair_id"][pred_df["pair_id"].duplicated()].iloc[0]
        raise ValidationError(f"Duplicate pair_id in submission: {dup!r}")

    return pred_df


def load_ground_truth(instance_name):
    """Load ground truth labels."""
    gt_file = os.path.join(GROUND_TRUTH_DIR, instance_name, "y_ref.csv")
    return pd.read_csv(gt_file, dtype=str, keep_default_na=False)


def calculate_metrics(pred_df, gt_df):
    """Per-chain AUROC averaged over evaluable chains, plus pooled secondary metrics."""
    gt_ids = gt_df["pair_id"].astype(str).tolist()
    pred_ids = pred_df["pair_id"].astype(str).tolist()
    if set(pred_ids) != set(gt_ids):
        missing = sorted(set(gt_ids) - set(pred_ids))[:3]
        unknown = sorted(set(pred_ids) - set(gt_ids))[:3]
        raise ValidationError(
            f"pair_id set mismatch: missing {missing}..., unknown {unknown}..."
        )
    if pred_ids != gt_ids:
        raise ValidationError("pair_id order must match the reference order (as in sample_submission.csv)")

    chain_auroc = []
    chain_aupr = []
    n_excluded = 0
    all_labels = []
    all_scores = []

    for idx in range(len(gt_df)):
        pid = gt_ids[idx]
        for side in ["1", "2"]:
            labels = parse_labels(gt_df.iloc[idx][f"labels{side}"])
            scores = parse_scores(pred_df.iloc[idx][f"scores{side}"], f"{pid} protein{side}")

            if len(scores) != len(labels):
                raise ValidationError(
                    f"{pid} protein{side}: score length ({len(scores)}) != "
                    f"sequence length ({len(labels)})"
                )
            if np.any(scores < 0) or np.any(scores > 1):
                raise ValidationError(f"{pid} protein{side}: scores out of [0,1] range")

            valid_mask = ~np.isnan(labels)
            y = labels[valid_mask].astype(int)
            s = scores[valid_mask]
            all_labels.append(y)
            all_scores.append(s)

            n_pos = int(y.sum())
            n_neg = int(len(y) - n_pos)
            if n_pos < MIN_POS or n_neg < MIN_NEG:
                n_excluded += 1
                continue
            chain_auroc.append(float(roc_auc_score(y, s)))
            chain_aupr.append(float(average_precision_score(y, s)))

    if not chain_auroc:
        raise RuntimeError("no evaluable chain in the reference set")

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    all_preds_binary = (all_scores >= 0.5).astype(int)

    return {
        PRIMARY: round(float(np.mean(chain_auroc)), 6),
        "AUPR_per_chain_mean": round(float(np.mean(chain_aupr)), 6),
        "AUROC_pooled": round(float(roc_auc_score(all_labels, all_scores)), 6),
        "AUPR_pooled": round(float(average_precision_score(all_labels, all_scores)), 6),
        "F1-score": round(float(f1_score(all_labels, all_preds_binary)), 6),
        "MCC": round(float(matthews_corrcoef(all_labels, all_preds_binary)), 6),
        "Precision": round(float(precision_score(all_labels, all_preds_binary, zero_division=0)), 6),
        "Recall": round(float(recall_score(all_labels, all_preds_binary, zero_division=0)), 6),
        "Accuracy": round(float(accuracy_score(all_labels, all_preds_binary)), 6),
        "n_chains_evaluated": len(chain_auroc),
        "n_chains_excluded": n_excluded,
    }


def run_evaluation():
    """Run the complete evaluation pipeline."""
    results = {}

    for instance_name in INSTANCES:
        print(f"\n{'='*60}")
        print(f"Evaluating instance: {instance_name}")
        print(f"{'='*60}")

        try:
            pred_df = load_and_validate(instance_name)
            gt_df = load_ground_truth(instance_name)
            scores = calculate_metrics(pred_df, gt_df)
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
