"""Evaluator for imaging/calcium-neuron-segmentation, v5.1 (2026-10-18).

v5.1: second instance `real` (Allen Brain Observatory two-photon crops, STNeuroNet expert masks).
    Primary metric of `real` = plain neuron mask F1 (same one-to-one greedy IoU>0.5 matching), pooled over all
    real test crops.  Ignored reference neurons of `real` are flagged in the hidden masks file (key "ignore", (K,)
    uint8: border rule <50 % of the neuron inside the crop, or no transient in the window) and, as for
    `simulated`, references with < MIN_AREA pixels inside the crop are ignored too.  traces_<id>.mat is OPTIONAL
    for `real` (not scored); if it is supplied it is validated exactly as for `simulated`.
    The `simulated` instance and its metric are unchanged from v5.  The task score is the equal-weight mean of
    the two clipped instance scores (contract.py).

v5: if <GROUND_TRUTH_DIR>/<instance>/video_groups.csv (id,video_id,...) exists, the result also reports a
video-cluster bootstrap SD / 95% interval of the primary metric (2,000 resamples of whole test videos, seed 0).
These are diagnostics only; the primary metric is not changed.

Primary metric: trace-weighted neuron F1.
    matches  = one-to-one IoU>0.5 matching between predicted and reference masks (greedy by IoU);
    F1_w     = 2 * sum_{matched pairs} max(0, Pearson(pred_trace, ref_trace)) / (n_ref + n_pred)
pooled over every test substack (micro average).  Perfect masks + perfect traces = 1; perfect masks with
random, constant or anti-correlated traces ~ 0; a predicted neuron that matches nothing is a false positive.

Ignored reference neurons (v4): (a) border slivers with fewer than MIN_AREA pixels and (b) silent neurons
(constant reference trace: no activity, so neither detectable from the activity nor scorable as a trace;
until v3 they received a free trace credit of 1).  Ignored neurons are removed from n_ref, and a prediction
whose best IoU>0.5 partner is an ignored neuron is removed from n_pred (not counted as a false positive).
The trace credit max(0, Pearson) is already on the chance-zero scale (clip(r, 0, 1)).

Submission contract (validated, any violation fails the whole submission):
    masks_<id>.mat  key "masks"  shape (64, 64, K), 0 <= K <= 15, binarised at 0.5, every mask non-empty
    traces_<id>.mat key "traces" shape (K, 1000), finite values  (required for `simulated`, optional for `real`)
for every id listed in <GROUND_TRUTH_DIR>/<instance>/index.csv; no required file may be missing.
"""
import csv
import json
import os

import numpy as np
import scipy.io as sio

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.environ.get("GROUND_TRUTH_DIR", os.path.join(EVAL_DIR, "ground_truth"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
if not OUTPUT_DIR:
    raise RuntimeError("OUTPUT_DIR environment variable is required")

INSTANCES = ["real", "simulated"]
TRACES_REQUIRED = {"real": False, "simulated": True}
PRIMARY = {"real": "F1 Score", "simulated": "Trace-weighted F1"}
METRIC_NAMES = ["Trace-weighted F1", "F1 Score", "Precision", "Recall", "Pearson Correlation"]
IOU_THRESHOLD = 0.5
MAX_NEURONS = 15
MIN_AREA = 10          # reference neurons smaller than this are ignored (see module docstring)
N_FRAMES = 1000
H = W = 64


class ValidationError(Exception):
    pass


def error_result(error_msg):
    result = {name: None for name in METRIC_NAMES}
    result["error"] = str(error_msg)
    return result


def compute_iou(mask_a, mask_b):
    intersection = np.sum((mask_a > 0) & (mask_b > 0))
    union = np.sum((mask_a > 0) | (mask_b > 0))
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def match_neurons(gt_masks, pred_masks, iou_threshold=IOU_THRESHOLD, min_area=MIN_AREA, gt_traces=None, gt_ignore_flags=None):
    """One-to-one greedy IoU matching.

    Returns (matches, n_gt, n_pred): matches = list of (gt_idx, pred_idx, iou) over *valid* reference
    neurons (area >= min_area); n_gt = number of valid reference neurons; n_pred = number of predicted
    neurons minus those whose best IoU>threshold partner is an ignored (small) reference neuron.
    """
    n_gt_all = gt_masks.shape[2]
    n_pred = pred_masks.shape[2]
    areas = [int(np.sum(gt_masks[:, :, i] > 0)) for i in range(n_gt_all)]
    silent = set()
    if gt_traces is not None:
        silent = {i for i in range(n_gt_all) if float(np.std(np.asarray(gt_traces[i], dtype=np.float64))) < 1e-12}
    if gt_ignore_flags is not None:  # `real`: precomputed border / inactivity flags
        silent |= {i for i in range(n_gt_all) if int(gt_ignore_flags[i])}
    gt_valid = [i for i in range(n_gt_all) if areas[i] >= min_area and i not in silent]
    gt_ignore = [i for i in range(n_gt_all) if areas[i] > 0 and (areas[i] < min_area or i in silent)]
    pred_valid = [i for i in range(n_pred) if np.sum(pred_masks[:, :, i]) > 0]
    if len(pred_valid) == 0:
        return [], len(gt_valid), 0

    iou_matrix = np.zeros((len(gt_valid), len(pred_valid)))
    for gi, g_idx in enumerate(gt_valid):
        for pi, p_idx in enumerate(pred_valid):
            iou_matrix[gi, pi] = compute_iou(gt_masks[:, :, g_idx], pred_masks[:, :, p_idx])
    candidates = [(iou_matrix[gi, pi], gi, pi)
                  for gi in range(len(gt_valid)) for pi in range(len(pred_valid))
                  if iou_matrix[gi, pi] > iou_threshold]
    candidates.sort(reverse=True)
    matches, used_gt, used_pred = [], set(), set()
    for iou_val, gi, pi in candidates:
        if gi not in used_gt and pi not in used_pred:
            matches.append((gt_valid[gi], pred_valid[pi], iou_val))
            used_gt.add(gi)
            used_pred.add(pi)

    # unmatched predictions that sit on an ignored (tiny) reference neuron are dropped from n_pred
    n_dropped = 0
    if gt_ignore:
        for pi, p_idx in enumerate(pred_valid):
            if pi in used_pred:
                continue
            best_valid = iou_matrix[:, pi].max() if len(gt_valid) else 0.0
            best_ignore = max(compute_iou(gt_masks[:, :, g], pred_masks[:, :, p_idx]) for g in gt_ignore)
            if best_ignore > iou_threshold and best_ignore >= best_valid:
                n_dropped += 1
    return matches, len(gt_valid), len(pred_valid) - n_dropped


def compute_pearson(trace_a, trace_b):
    a = trace_a.astype(np.float64)
    b = trace_b.astype(np.float64)
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denominator = np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2))
    if denominator < 1e-12:
        return 0.0
    return float(np.sum(a_centered * b_centered) / denominator)


def read_index(gt_dir):
    path = os.path.join(gt_dir, "index.csv")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or not ids:
        raise RuntimeError("ground-truth index.csv is empty or has duplicate ids")
    return ids


def load_and_validate(instance_name, ids):
    output_dir = os.path.join(OUTPUT_DIR, instance_name)
    if not os.path.isdir(output_dir):
        raise ValidationError(f"Output directory not found: {output_dir}")
    predictions, missing, errors = {}, [], []
    for sid in ids:
        mask_file = os.path.join(output_dir, f"masks_{sid}.mat")
        trace_file = os.path.join(output_dir, f"traces_{sid}.mat")
        if not os.path.exists(mask_file):
            missing.append(f"masks_{sid}.mat")
            continue
        has_traces = os.path.exists(trace_file)
        if not has_traces and TRACES_REQUIRED[instance_name]:
            missing.append(f"traces_{sid}.mat")
            continue
        try:
            mask_data = sio.loadmat(mask_file)
            if "masks" not in mask_data:
                errors.append(f"masks_{sid}.mat: missing 'masks' key")
                continue
            pred_masks = np.asarray(mask_data["masks"])
            if has_traces:
                trace_data = sio.loadmat(trace_file)
                if "traces" not in trace_data:
                    errors.append(f"traces_{sid}.mat: missing 'traces' key")
                    continue
                pred_traces = np.asarray(trace_data["traces"])
            else:  # optional for `real`: placeholder, never scored
                pred_traces = np.zeros((pred_masks.shape[2] if pred_masks.ndim == 3 else 0, N_FRAMES))
            if pred_masks.ndim == 2 and pred_masks.size == 0:
                pred_masks = pred_masks.reshape(H, W, 0)
            if pred_traces.ndim == 2 and pred_traces.size == 0:
                pred_traces = pred_traces.reshape(0, N_FRAMES)
            if pred_masks.ndim != 3 or pred_masks.shape[0] != H or pred_masks.shape[1] != W:
                errors.append(f"masks_{sid}.mat: expected shape (64, 64, K), got {pred_masks.shape}")
                continue
            if not np.issubdtype(pred_masks.dtype, np.number) and pred_masks.dtype != np.bool_:
                errors.append(f"masks_{sid}.mat: non-numeric dtype {pred_masks.dtype}")
                continue
            k = pred_masks.shape[2]
            if k > MAX_NEURONS:
                errors.append(f"masks_{sid}.mat: K={k} exceeds the limit of {MAX_NEURONS} neurons")
                continue
            if pred_traces.ndim != 2 or pred_traces.shape[1] != N_FRAMES:
                errors.append(f"traces_{sid}.mat: expected shape (K, {N_FRAMES}), got {pred_traces.shape}")
                continue
            if pred_traces.shape[0] != k:
                errors.append(f"Substack {sid}: mask count ({k}) != trace count ({pred_traces.shape[0]})")
                continue
            pred_masks = pred_masks.astype(np.float64)
            if not np.all(np.isfinite(pred_masks)):
                errors.append(f"masks_{sid}.mat: non-finite values")
                continue
            pred_masks = (pred_masks > 0.5).astype(np.uint8)
            empty = [i for i in range(k) if pred_masks[:, :, i].sum() == 0]
            if empty:
                errors.append(f"masks_{sid}.mat: {len(empty)} empty mask(s) (first index {empty[0]})")
                continue
            pred_traces = pred_traces.astype(np.float64)
            if not np.all(np.isfinite(pred_traces)):
                errors.append(f"traces_{sid}.mat: non-finite values")
                continue
            predictions[sid] = (pred_masks, pred_traces)
        except Exception as e:  # noqa: BLE001
            errors.append(f"Substack {sid}: loading error: {e}")
    if missing:
        raise ValidationError(f"Missing {len(missing)} output files (first 5): {missing[:5]}")
    if errors:
        raise ValidationError(f"{len(errors)} validation errors (first 5): {errors[:5]}")
    return predictions


def pair_credit(ref_trace, pred_trace):
    """Trace credit of one matched pair: max(0, Pearson) (silent reference neurons are never matched, see above)."""
    return max(0.0, compute_pearson(ref_trace, pred_trace))


def calculate_metrics_real(predictions, gt_dir, ids):
    """`real`: plain mask F1 pooled over crops (no trace credit)."""
    tp = n_gt_tot = n_pred_tot = 0
    ious, per_sid = [], {}
    for sid in ids:
        gt = sio.loadmat(os.path.join(gt_dir, f"masks_{sid}.mat"))
        gt_masks = np.asarray(gt["masks"])
        if gt_masks.ndim == 2:
            gt_masks = gt_masks.reshape(H, W, -1)
        flags = np.asarray(gt["ignore"]).ravel() if "ignore" in gt else np.zeros(gt_masks.shape[2])
        if flags.size != gt_masks.shape[2]:
            raise RuntimeError(f"ground truth {sid}: ignore flags do not match the mask count")
        pred_masks, _ = predictions[sid]
        matches, n_gt, n_pred = match_neurons(gt_masks, pred_masks, gt_ignore_flags=flags)
        tp += len(matches); n_gt_tot += n_gt; n_pred_tot += n_pred
        ious += [m[2] for m in matches]
        per_sid[sid] = (float(len(matches)), n_gt, n_pred)
    denom = float(n_gt_tot + n_pred_tot)
    out = {
        "F1 Score": round(2.0 * tp / denom, 6) if denom > 0 else 0.0,
        "Precision": round(tp / n_pred_tot, 6) if n_pred_tot > 0 else 0.0,
        "Recall": round(tp / n_gt_tot, 6) if n_gt_tot > 0 else 0.0,
        "Trace-weighted F1": None, "Pearson Correlation": None,
        "Mean matched IoU": round(float(np.mean(ious)), 6) if ious else 0.0,
        "n_ref": int(n_gt_tot), "n_pred": int(n_pred_tot), "n_tp": int(tp),
    }
    out.update(video_bootstrap(per_sid, gt_dir))
    return out


def calculate_metrics(predictions, gt_dir, ids):
    all_tp = 0
    all_n_gt = 0
    all_n_pred = 0
    all_pearson = []
    all_credit = []
    per_sid = {}
    for sid in ids:
        gt_masks = sio.loadmat(os.path.join(gt_dir, f"masks_{sid}.mat"))["masks"]
        gt_traces = sio.loadmat(os.path.join(gt_dir, f"traces_{sid}.mat"))["traces"]
        if gt_masks.ndim == 2 and gt_masks.size == 0:
            gt_masks = gt_masks.reshape(H, W, 0)
        if gt_traces.ndim == 2 and gt_traces.size == 0:
            gt_traces = gt_traces.reshape(0, N_FRAMES)
        pred_masks, pred_traces = predictions[sid]
        matches, n_gt, n_pred = match_neurons(gt_masks, pred_masks, gt_traces=gt_traces)
        sid_credit = 0.0
        all_tp += len(matches)
        all_n_gt += n_gt
        all_n_pred += n_pred
        for gt_idx, pred_idx, _ in matches:
            all_pearson.append(compute_pearson(gt_traces[gt_idx, :], pred_traces[pred_idx, :]))
            all_credit.append(pair_credit(gt_traces[gt_idx, :], pred_traces[pred_idx, :]))
            sid_credit += all_credit[-1]
        per_sid[sid] = (sid_credit, n_gt, n_pred)
    pearson = np.asarray(all_pearson, dtype=np.float64)
    denom = float(all_n_gt + all_n_pred)
    precision = float(all_tp) / float(all_n_pred) if all_n_pred > 0 else 0.0
    recall = float(all_tp) / float(all_n_gt) if all_n_gt > 0 else 0.0
    f1 = 2.0 * all_tp / denom if denom > 0 else 0.0
    f1_w = 2.0 * float(np.sum(all_credit)) / denom if denom > 0 else 0.0
    out = {
        "Trace-weighted F1": round(f1_w, 6),
        "F1 Score": round(f1, 6),
        "Precision": round(precision, 6),
        "Recall": round(recall, 6),
        "Pearson Correlation": round(float(pearson.mean()), 6) if len(pearson) else 0.0,
        "n_ref": int(all_n_gt), "n_pred": int(all_n_pred), "n_tp": int(all_tp),
    }
    out.update(video_bootstrap(per_sid, gt_dir))
    return out


def video_bootstrap(per_sid, gt_dir, n_boot=2000, seed=0):
    """Video-cluster bootstrap of the pooled primary metric (diagnostic; needs video_groups.csv).
    per_sid values are (credit, n_ref, n_pred); credit = trace credit (`simulated`) or n_tp (`real`)."""
    path = os.path.join(gt_dir, "video_groups.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as fh:
        vid = {r["id"]: r["video_id"] for r in csv.DictReader(fh)}
    agg = {}
    for sid, (c, g, p) in per_sid.items():
        a = agg.setdefault(vid[sid], [0.0, 0, 0]); a[0] += c; a[1] += g; a[2] += p
    arr = np.array(list(agg.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    s = arr[idx].sum(axis=1)
    den = s[:, 1] + s[:, 2]
    boot = np.where(den > 0, 2.0 * s[:, 0] / np.maximum(den, 1e-12), 0.0)
    return {"n_videos": int(len(arr)), "video_bootstrap_sd": round(float(boot.std()), 6),
            "video_bootstrap_ci95": [round(float(np.percentile(boot, 2.5)), 6), round(float(np.percentile(boot, 97.5)), 6)]}


def run_evaluation():
    results = {}
    for instance_name in INSTANCES:
        print(f"\n{'=' * 60}\nEvaluating instance: {instance_name}\n{'=' * 60}")
        try:
            gt_dir = os.path.join(GROUND_TRUTH_DIR, instance_name)
            if not os.path.isdir(gt_dir):
                raise RuntimeError(f"Ground truth directory not found: {gt_dir}")
            ids = read_index(gt_dir)
            predictions = load_and_validate(instance_name, ids)
            scores = (calculate_metrics_real if instance_name == "real" else calculate_metrics)(predictions, gt_dir, ids)
            results[instance_name] = scores
            print(f"Results: {scores}")
        except ValidationError as e:
            print(f"[Validation Error] {instance_name}: {e}")
            results[instance_name] = error_result(f"Validation: {e}")
        except Exception as e:  # noqa: BLE001
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
