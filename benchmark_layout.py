#!/usr/bin/env python3
"""Canonical domain, task, and four-part layout for BioXArena-Pro."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable


DOMAIN_TASKS: dict[str, tuple[str, ...]] = {
    "chemical-biology": (
        "boom-csd",
        "massspecgym-msms-retrieval",
        "chemixhub-conductivity",
        "drugcomb-multiomics-synergy",
        "kano-molecular-property-prediction",
        "solubility",
        "tdc-dti-binding-affinity",
        "uspto-reaction-product",
    ),
    "imaging": (
        "calcium-neuron-segmentation",
        "cryocrab-quality",
        "drugclass-if",
        "mammovqa-external",
        "medmnist3d-decathlon",
        "microscopy-instance-segmentation",
        "rocov2-retrieval",
        "subcell-localization",
    ),
    "network-biology": (
        "causalbench-grn",
        "lnctard-relation",
        "music-seccomplex",
        "ppi-cross-species",
        "pring-cross-species-ppi",
        "scigym-sbml",
        "swissprot-protein-function",
        "synlethdb-sl",
    ),
    "perturbation": (
        "cellverse-cta",
        "cpjump-genetic-matching",
        "gdsc-lineage-holdout-response",
        "perturbench-response",
        "scgenescope-treatment",
        "sciplex-distribution",
        "tahoe-single-cell",
        "xatlas-orion-perturb",
    ),
    "phenotype-disease": (
        "gutcrc-cross-cohort-detection",
        "clinvar-disease-variant-effect",
        "covid-hospitalization-forecast",
        "cptac-pan-protein-imputation",
        "msk-met-tumor-origin",
        "survpath-multimodal",
        "tcga-pancan-multiomics-survival",
        "target-aml-cytogenetics",
    ),
    "sequence-genomics": (
        "aso-knockdown-efficacy",
        "cgbench-variant",
        "de-novo-peptide-sequencing-from-tandem-mass-spectrometry",
        "deepstarr-genomic-sequence",
        "escape-amp-multilabel",
        "proteingym-fitness",
        "sirna-knockdown-efficacy",
        "yeast-promoter-expression",
    ),
    "single-cell-spatial": (
        "dlpfc-spatial-domain",
        "gastrulation-kinetic-state",
        "hest-coad-spatial-expression",
        "lung-cite-cell-matching",
        "openproblems-rna-atac",
        "openproblems-rna-protein",
        "seqfish-deconvolution",
        "tabula-sapiens-ood-annotation",
    ),
    "structure": (
        "asep-epitope",
        "cath-inverse-folding",
        "cpsea-cyclic-peptide-affinity",
        "decoydb-pose-classify",
        "denoise-cryo",
        "ppi-interface-residues",
        "psbench-multimer-mqa",
        # v5.1 (2026-10-18): replaces davis-complete-xl-dti (archived).
        "skempi-ppi-ddg",
    ),
}

PARTITION_DOMAINS: dict[int, tuple[str, str]] = {
    1: ("chemical-biology", "imaging"),
    2: ("network-biology", "perturbation"),
    3: ("phenotype-disease", "sequence-genomics"),
    4: ("single-cell-spatial", "structure"),
}

TASK_KEYS: tuple[str, ...] = tuple(
    f"{domain}/{task}"
    for domain, tasks in DOMAIN_TASKS.items()
    for task in tasks
)
TASK_KEY_SET = frozenset(TASK_KEYS)

# Old keys remain useful for intentionally locating historical output runs.  New
# training and evaluation always write and report the canonical key on the right.
LEGACY_TO_CANONICAL_TASK_KEY: dict[str, str] = {
    "perturbation-dynamics/drugcomb-multiomics-synergy": (
        "chemical-biology/drugcomb-multiomics-synergy"
    ),
    "perturbation-dynamics/cellverse-cta": "perturbation/cellverse-cta",
    "perturbation-dynamics/cpjump-genetic-matching": (
        "perturbation/cpjump-genetic-matching"
    ),
    "perturbation-dynamics/gdsc-lineage-holdout-response": (
        "perturbation/gdsc-lineage-holdout-response"
    ),
    "perturbation-dynamics/perturbench-response": (
        "perturbation/perturbench-response"
    ),
    "perturbation-dynamics/sciplex-distribution": (
        "perturbation/sciplex-distribution"
    ),
    "single-cell-multiomics/scgenescope-treatment": (
        "perturbation/scgenescope-treatment"
    ),
    "perturbation-dynamics/tahoe-single-cell": (
        "perturbation/tahoe-single-cell"
    ),
    "perturbation-dynamics/xatlas-orion-perturb": (
        "perturbation/xatlas-orion-perturb"
    ),
    "spatial-tissue-organization/dlpfc-spatial-domain": (
        "single-cell-spatial/dlpfc-spatial-domain"
    ),
    "single-cell-multiomics/gastrulation-kinetic-state": (
        "single-cell-spatial/gastrulation-kinetic-state"
    ),
    "spatial-tissue-organization/hest-coad-spatial-expression": (
        "single-cell-spatial/hest-coad-spatial-expression"
    ),
    "single-cell-multiomics/lung-cite-cell-matching": (
        "single-cell-spatial/lung-cite-cell-matching"
    ),
    "single-cell-multiomics/openproblems-rna-atac": (
        "single-cell-spatial/openproblems-rna-atac"
    ),
    "single-cell-multiomics/openproblems-rna-protein": (
        "single-cell-spatial/openproblems-rna-protein"
    ),
    "spatial-tissue-organization/seqfish-deconvolution": (
        "single-cell-spatial/seqfish-deconvolution"
    ),
    "single-cell-multiomics/tabula-sapiens-ood-annotation": (
        "single-cell-spatial/tabula-sapiens-ood-annotation"
    ),
}
CANONICAL_TO_LEGACY_TASK_KEY = {
    canonical: legacy for legacy, canonical in LEGACY_TO_CANONICAL_TASK_KEY.items()
}


# v4 (2026-10-16): data-version precedence for a task package. The first version whose
# <public_root>/<version>/<domain>/<task>/public directory exists is the task's current version;
# "original" (no version sub-directory) is the fallback.
DATA_VERSION_PRECEDENCE: tuple[str, ...] = (
    "corrected_v4",
    "corrected_v3",
    "corrected_v2",
    "corrected",
)


def task_data_version(task_key: str, public_root, versions: Iterable[str] = DATA_VERSION_PRECEDENCE) -> str:
    """Return the latest data version of ``task_key`` under ``public_root`` (or "original")."""
    from pathlib import Path

    root = Path(public_root)
    for version in versions:
        if (root / version / task_key / "public").is_dir():
            return version
    return "original"


def partition_task_keys(part: int) -> tuple[str, ...]:
    """Return the stable 16-task partition used by every agent backend."""
    try:
        domains = PARTITION_DOMAINS[part]
    except KeyError as exc:
        raise ValueError(f"partition must be one of {sorted(PARTITION_DOMAINS)}") from exc
    return tuple(
        f"{domain}/{task}"
        for domain in domains
        for task in DOMAIN_TASKS[domain]
    )


def canonicalize_task_key(task_key: str) -> str:
    """Translate one legacy key while leaving canonical/unknown keys unchanged."""
    return LEGACY_TO_CANONICAL_TASK_KEY.get(task_key, task_key)


def layout_issues(task_keys: Iterable[str], label: str) -> list[str]:
    """Describe duplicates and deviations from the canonical 64-task set."""
    values = tuple(task_keys)
    counts = Counter(values)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    actual = set(values)
    issues: list[str] = []
    if duplicates:
        issues.append(f"{label} contains duplicate task keys: {duplicates}")
    if actual != TASK_KEY_SET:
        issues.append(
            f"{label} differs from canonical layout: "
            f"missing={sorted(TASK_KEY_SET-actual)}, extra={sorted(actual-TASK_KEY_SET)}"
        )
    return issues


def require_canonical_layout(task_keys: Iterable[str], label: str) -> None:
    """Raise a single actionable error when a discovered layout has drifted."""
    issues = layout_issues(task_keys, label)
    if issues:
        raise ValueError("; ".join(issues))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=int, choices=sorted(PARTITION_DOMAINS))
    args = parser.parse_args()
    keys = partition_task_keys(args.part) if args.part else TASK_KEYS
    print("\n".join(keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
