"""USPTO-480k reaction product prediction — strict set-equality evaluator (v2, 2026-10-13; v5 parsing fix 2026-10-17).

v5 (2026-10-17): records are physical lines (empty fields / empty lines are positional misses, the 5-candidate
cap counts raw fields, exactly one record per reaction), heavy-atom fragment filter, fragment cap before de-duplication.

Changes against the v1 evaluator (see audits/pro_fix_v2_20261013/report_uspto.md):

1. A candidate matches only when its cleaned product fragment set is **equal** to the
   cleaned reference product fragment set.  v1 accepted any intersection, so a single
   candidate holding many '.'-joined molecules matched whenever one of them was right.
2. The reactants are no longer part of the matching.  v1 unioned the reactant fragments
   into every prediction, so a reaction whose product fragment also occurs among the
   reactants matched any string at all (315 of the 40,000 test reactions; a garbage file
   scored 0.0079).  This evaluator never reads x_test.txt.
3. Top-k uses candidate positions: Top-1 is decided by the FIRST tab-separated candidate
   only.  At most MAX_CANDIDATES candidates per line are read; a line carrying more is a
   validation error, so the top-5 list cannot be packed into slot 1.
4. A candidate with more than MAX_FRAGMENTS heavy fragments cannot match (it is counted
   and reported, not silently intersected).

Cleaning, identical for prediction and reference: split on '.', drop fragments with a
single heavy atom, remove atom-map numbers, canonicalise with RDKit, collect as a set.
An unparseable candidate simply never matches.
"""

import os
import json
from multiprocessing import Pool

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

# Setup Paths
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.environ.get("GROUND_TRUTH_DIR", os.path.join(EVAL_DIR, "ground_truth"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
if not OUTPUT_DIR:
    raise RuntimeError("OUTPUT_DIR environment variable is required")

# Output specification (must match description.md Section 5)
OUTPUT_FILE = "predictions.txt"
INSTANCES = ["uspto_480k"]
METRIC_NAMES = [
    "Top-1 Exact Match Accuracy",
    "Top-2 Exact Match Accuracy",
    "Top-3 Exact Match Accuracy",
    "Top-5 Exact Match Accuracy",
]
KS = [1, 2, 3, 5]

# Published caps (description.md): at most 5 ranked candidates per line, and a single
# candidate may describe at most 5 heavy fragments.
MAX_CANDIDATES = 5
MAX_FRAGMENTS = 5

NPROC = max(1, min(8, (os.cpu_count() or 1)))


# Sentinel returned by clean_smiles_set when a candidate carries more than MAX_FRAGMENTS
# heavy fragments (it can never match).
TOO_MANY = frozenset({"__too_many_fragments__"})


class ValidationError(Exception):
    """Raised when output file fails validation."""
    pass


def error_result(error_msg):
    """Return a result dict with null metrics and error info."""
    result = {name: None for name in METRIC_NAMES}
    result["error"] = str(error_msg)
    return result


def clean_smiles_set(smi, cap=True):
    """Canonical fragment set of a product SMILES.

    Single-heavy-atom fragments are dropped, atom-map numbers are removed, and the
    remaining molecules are canonicalised.  Returns a frozenset of canonical SMILES
    (empty when nothing parses or nothing survives the single-atom filter).
    """
    if smi is None:
        return frozenset()
    smi = smi.strip()
    if not smi:
        return frozenset()
    kept = []
    for frag in smi.split('.'):
        frag = frag.strip()
        if not frag:
            continue
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            return frozenset()          # an unparseable fragment invalidates the candidate
        if mol.GetNumHeavyAtoms() > 1:
            kept.append(mol)
    if cap and len(kept) > MAX_FRAGMENTS:
        # v5: the fragment cap counts heavy fragments BEFORE de-duplication, so k copies of
        # the same molecule cannot hide behind set equality.
        return TOO_MANY
    out = set()
    for mol in kept:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        out.add(Chem.MolToSmiles(mol))
    return frozenset(out)


def clean_product(product_smi):
    """Reference-side cleaning (same rule as the prediction side, without the candidate cap)."""
    return clean_smiles_set(product_smi, cap=False)


def exact_match_rank(product_set, predictions):
    """1-based rank of the first candidate whose cleaned fragment set equals the
    reference set, or -1.  Only the first MAX_CANDIDATES candidates are considered and a
    candidate with more than MAX_FRAGMENTS fragments can never match."""
    if not product_set:
        return -1
    for k, pred in enumerate(predictions[:MAX_CANDIDATES]):
        pred_set = clean_smiles_set(pred)
        if not pred_set or pred_set == TOO_MANY or len(pred_set) > MAX_FRAGMENTS:
            continue
        if pred_set == product_set:
            return k + 1
    return -1


def _rank_worker(item):
    product_set, preds = item
    return exact_match_rank(product_set, preds)


def load_ground_truth(instance_name):
    """Load the reference products as cleaned canonical fragment sets.

    The reactants are deliberately NOT loaded: matching never consults them.
    """
    y_ref_path = os.path.join(GROUND_TRUTH_DIR, instance_name, "y_ref.txt")
    if not os.path.exists(y_ref_path):
        raise FileNotFoundError(f"Ground truth not found: {y_ref_path}")

    raw = []
    with open(y_ref_path, 'r') as f:
        for line in f:
            raw.append(line.strip())

    while raw and raw[-1] == "":
        raw.pop()

    with Pool(NPROC) as pool:
        products = pool.map(clean_product, raw, chunksize=200)
    return products


def load_and_validate(instance_name, expected_count):
    """Load and validate predictions from the output file (v5, 2026-10-17).

    The file is read as physical records: one record per line, only the line terminator
    (LF, optionally preceded by CR) is removed.  A single final line terminator does not
    start an extra record.  Fields are split on TAB and kept by position, so an empty field
    is a missing candidate at that rank (a miss), never a gap that shifts later candidates
    forward.  The MAX_CANDIDATES cap counts raw fields.  An empty record is a valid line
    with no candidates (counted as a miss).  The record count must equal expected_count
    exactly; blank lines are records like any other.
    """
    pred_file = os.path.join(OUTPUT_DIR, instance_name, OUTPUT_FILE)

    if not os.path.exists(pred_file):
        raise ValidationError(f"Output file not found: {pred_file}")

    with open(pred_file, 'r', newline='') as f:
        text = f.read()
    if text == "":
        raise ValidationError("Predictions file is empty")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()                     # a final terminator does not start a new record

    predictions = []
    for line_num, line in enumerate(lines):
        if line.endswith("\r"):
            line = line[:-1]
        fields = line.split('\t') if line != "" else []
        if len(fields) > MAX_CANDIDATES:
            raise ValidationError(
                f"Line {line_num + 1} carries {len(fields)} tab-separated fields; at most "
                f"{MAX_CANDIDATES} ranked candidates per reaction are allowed"
            )
        predictions.append([p.strip() for p in fields])

    if len(predictions) != expected_count:
        raise ValidationError(
            f"Expected {expected_count} prediction lines, got {len(predictions)}"
        )

    return predictions


def calculate_metrics(predictions, products):
    """Top-k exact-match accuracy over all reference reactions.

    Every reference line is scored; a missing or unmatched line counts as a miss.
    """
    n = len(products)
    if len(predictions) != n:
        raise ValidationError(
            f"Expected {n} prediction lines, got {len(predictions)}"
        )

    with Pool(NPROC) as pool:
        match_ranks = pool.map(_rank_worker, zip(products, predictions), chunksize=200)

    results = {}
    for k, metric_name in zip(KS, METRIC_NAMES):
        count = sum(1 for r in match_ranks if 0 < r <= k)
        results[metric_name] = round(count / n, 6)

    empty_refs = sum(1 for p in products if not p)
    if empty_refs:
        results["warning"] = (
            f"{empty_refs} reference products are empty after the single-atom filter "
            f"and can never be matched"
        )
    return results


def run_evaluation():
    """Run the complete evaluation pipeline."""
    results = {}

    for instance_name in INSTANCES:
        print(f"\n{'=' * 60}")
        print(f"Evaluating instance: {instance_name}")
        print(f"{'=' * 60}")

        try:
            products = load_ground_truth(instance_name)
            predictions = load_and_validate(instance_name, len(products))
            scores = calculate_metrics(predictions, products)
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
