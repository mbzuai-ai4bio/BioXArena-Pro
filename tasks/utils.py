"""Shared grading utilities for BioXArena-Pro tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import warnings

import numpy as np
import numpy.lib.format as _np_format

# Compatibility shim for code expecting the old/non-standard name.
if not hasattr(_np_format, "MAGIC") and hasattr(_np_format, "MAGIC_PREFIX"):
    _np_format.MAGIC = _np_format.MAGIC_PREFIX
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


class InvalidSubmissionError(Exception):
    """Raised when a submission cannot be graded."""


TASK_CONFIG: dict[str, dict[str, Any]] = {
    "chemical-biology/boom-csd": {
        "kind": "grouped_calibrated_regression",
        "pred_col": "value",
        "target_col": "value",
        "group_col": "source",
        "metric": {
            "name": "per_source_mean_calibrated_regression",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "chemical-biology/chemcot-xl-molecule-counting": {
        # corrected_v2 (v2b, 2026-10-14): counts binned per source (fg_count: <=1, 2, >=3; ring_count: <=2, 3, >=4)
        # before macro-F1, because exact-count classes with 1-2 test rows made the score noisy (bootstrap SD 0.079).
        "kind": "grouped_binned_integer_macro_f1",
        "pred_col": "count",
        "target_col": "count",
        "group_col": "source",
        "count_bins": {"fg_count": [1, 3], "ring_count": [2, 4]},
        "metric": {
            "name": "per_source_macro_f1_over_binned_counts",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "chemical-biology/chemixhub-conductivity": {
        # corrected_v2 (v2b, 2026-10-14): scored on the residual after the published conditions-only reference
        # (test.csv column reference_ln_conductivity), so temperature / composition-fraction physics alone scores ~chance.
        "kind": "grouped_calibrated_regression",
        "pred_col": "ln_conductivity_S_m",
        "target_col": "ln_conductivity_S_m",
        "group_col": "source",
        "reference_col": "reference_ln_conductivity",
        "metric": {
            "name": "per_source_mean_calibrated_regression_on_residual_vs_conditions_reference",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "chemical-biology/solubility": {
        "kind": "calibrated_regression",
        "pred_col": "log_solubility",
        "target_col": "log_solubility",
        "metric": {
            "name": "calibrated_regression",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "chemical-biology/tdc-dti-binding-affinity": {
        "kind": "grouped_calibrated_regression",
        "pred_col": "affinity",
        "target_col": "affinity",
        "group_col": "source",
        "metric": {
            "name": "per_source_mean_calibrated_regression",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "chemical-biology/drugcomb-multiomics-synergy": {
        # corrected_v2 (v2b, 2026-10-14): only the pair-specific synergy is scored. Within each tissue,
        # truth and prediction are projected off span{cell-line indicators, public train-derived additive
        # reference (test.csv reference_zip)}; drug/cell identity priors then score ~0.
        "kind": "reference_projected_grouped_pearson",
        "pred_col": "synergy_zip",
        "target_col": "synergy_zip",
        "group_col": "tissue",
        "reference_col": "reference_zip",
        "indicator_col": "cell_line",
        "min_rows": 20,
        "metric": {
            "name": "per_tissue_mean_partial_pearson_beyond_additive_reference",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "imaging/cryocrab-quality": {
        "kind": "grouped_pearson",
        "pred_col": "ctf_fit_resolution",
        "target_col": "ctf_fit_resolution",
        "group_col": "empiar_id",
        "min_rows": 6,
        "metric": {
            "name": "per_empiar_entry_mean_pearson",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "imaging/drugclass-if": {
        "kind": "macro_f1",
        "pred_col": "drug",
        "target_col": "drug",
        "metric": {
            "name": "macro_f1",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "imaging/subcell-localization": {
        "kind": "multilabel_combined",
        "label_cols": [
            "big_aggregates",
            "cell_contact",
            "centrosome",
            "chromatin",
            "cytoplasmic",
            "cytoskeleton",
            "er",
            "focal_adhesions",
            "golgi",
            "membrane",
            "mitochondria",
            "nuclear_membrane",
            "nuclear_punctae",
            "nucleolus_fc_dfc",
            "nucleolus_gc",
            "nucleoplasm",
            "vesicles",
        ],
        "metric": {
            "name": "combined_macro_f1_map",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "imaging/rocov2-retrieval": {
        "kind": "grouped_average_precision",
        "pred_col": "match",
        "target_col": "match",
        "group_col": "query_id",
        "metric": {
            "name": "mean_reciprocal_rank_via_per_query_average_precision",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "imaging/mammovqa-external": {
        "kind": "grouped_macro_f1",
        "pred_col": "answer",
        "target_col": "answer",
        "group_col": "question_topic",
        "strip_labels": True,
        "reject_missing_labels": True,  # corrected_v2b (2026-10-14): NaN / empty answers are invalid
        # v3 (2026-10-15): 3 BMCD image pairs in test are pixel-identical but carry contradictory ACR / Bi-Rads
        # labels (source linkage error). These 12 rows are still required in the submission but not scored.
        "exclude_ids": [
            "q_054d5f8240b9a3f9", "q_1c5280807f95bd33", "q_1f20b1923b881dcf", "q_3161cecb8a37a991",
            "q_39b990398095f8a9", "q_5c019506e099a96f", "q_6fbbdfc02a0201ba", "q_72b4c80effde034c",
            "q_7bd7e83112f2183a", "q_b7a523c22b52ec44", "q_e97a1a0dc4985561", "q_fcdffaa959df4754",
        ],
        "metric": {
            "name": "per_question_topic_mean_macro_f1",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "imaging/medmnist3d-decathlon": {
        "kind": "grouped_macro_f1",
        "pred_col": "diagnosis",
        "target_col": "diagnosis",
        "group_col": "source",
        "reject_unknown_labels": True,
        "metric": {
            "name": "per_source_mean_macro_f1",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "phenotype-disease/tcga-pancan-multiomics-survival": {
        "kind": "grouped_c_index",
        "submission_columns": ["id", "risk_score"],
        "time_col": "survival_months",
        "censor_col": "censorship",
        "risk_col": "risk_score",
        "group_col": "cohort",
        "metric": {
            "name": "per_cohort_mean_c_index",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "phenotype-disease/survpath-multimodal": {
        "kind": "grouped_c_index",
        "submission_columns": ["id", "risk_score"],
        "time_col": "survival_months",
        "censor_col": "censorship",
        "risk_col": "risk_score",
        "group_col": "source",
        "metric": {
            "name": "per_cancer_mean_c_index",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "phenotype-disease/survpath-xl-rna": {
        "kind": "grouped_c_index",
        "submission_columns": ["id", "risk_score"],
        "time_col": "survival_months",
        "censor_col": "censorship",
        "risk_col": "risk_score",
        "group_col": "source",
        "metric": {
            "name": "per_cancer_mean_c_index",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "phenotype-disease/aomic-xl-fluid-intelligence": {
        "kind": "pearson",
        "pred_col": "fluid_intelligence",
        "target_col": "fluid_intelligence",
        "metric": {
            "name": "pearson",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "phenotype-disease/target-aml-cytogenetics": {
        "kind": "macro_f1",
        "allowed_labels": ["MLL", "Normal", "Other", "inv(16)", "t(8;21)"],
        "pred_col": "cytogenetic_class",
        "target_col": "cytogenetic_class",
        "metric": {
            "name": "macro_f1",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "phenotype-disease/cptac-pan-protein-imputation": {
        "kind": "masked_matrix_pearson_files",
        "file_col": "prediction_file",
        "metric": {
            "name": "masked_mean_patient_and_protein_pearson",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "network-biology/lnctard-relation": {
        "kind": "grouped_average_precision",
        "pred_col": "interaction",
        "target_col": "interaction",
        "group_col": "query_id",
        "metric": {
            "name": "mean_reciprocal_rank_via_per_query_average_precision",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "network-biology/synlethdb-sl": {
        "kind": "grouped_average_precision",
        "pred_col": "synthetic_lethal",
        "target_col": "synthetic_lethal",
        "group_col": "query_id",
        "metric": {
            "name": "mean_reciprocal_rank_via_per_query_average_precision",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "phenotype-disease/msk-met-tumor-origin": {
        "kind": "macro_f1",
        "pred_col": "cancer_type",
        "target_col": "cancer_type",
        "metric": {
            "name": "macro_f1",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "network-biology/causalbench-grn": {
        # corrected_v2 (2026-10-13): AUROC inside each (cell line, downstream gene) over strength-matched pairs
        # (private answers column `scored`); every test row must still be predicted.
        "kind": "masked_grouped_roc_auc",
        "pred_col": "edge",
        "target_col": "edge",
        "mask_col": "scored",
        "group_col": "eval_group",
        "min_rows": 2,
        "metric": {
            "name": "per_downstream_gene_strength_matched_mean_roc_auc",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "network-biology/music-seccomplex": {
        "kind": "roc_auc",
        "pred_col": "in_same_corum_complex",
        "target_col": "in_same_corum_complex",
        "metric": {
            "name": "roc_auc",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "network-biology/pring-cross-species-ppi": {
        "kind": "grouped_roc_auc",
        "pred_col": "interaction",
        "target_col": "interaction",
        "group_col": "source",
        "metric": {
            "name": "per_source_mean_roc_auc",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "network-biology/scigym-sbml": {
        "kind": "multioutput_integer_accuracy",
        "label_cols": [
            "delta_qualitative_species", "delta_transitions", "delta_inputs",
            "delta_outputs", "delta_species_glyphs", "delta_text_glyphs",
        ],
        "metric": {
            "name": "mean_exact_missing_qual_and_layout_element_count_accuracy",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "sequence-genomics/aso-knockdown-efficacy": {
        "kind": "grouped_pearson",
        "pred_col": "inhibition",
        "target_col": "inhibition",
        "group_col": "source",
        "metric": {
            "name": "per_source_mean_pearson",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "sequence-genomics/cgbench-variant": {
        "kind": "macro_f1",
        "pred_col": "assertion",
        "target_col": "assertion",
        "metric": {
            "name": "macro_f1",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "sequence-genomics/escape-amp-multilabel": {
        "kind": "multilabel_prevalence_corrected",
        "label_cols": [
            "antibacterial",
            "antifungal",
            "antiviral",
            "antiparasitic",
            "antimicrobial",
        ],
        "threshold": 0.5,
        "min_label_count": 30,
        "metric": {
            "name": "prevalence_corrected_macro_ap_f1_gain",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "sequence-genomics/sirna-knockdown-efficacy": {
        # v2 (2026-10-13): two evaluation groups instead of four sources. `shrna_sortseq` is
        # sherwood; `sirna_duplex` pools huesken + ichihara + shmushkovich, whose per-source test
        # sets (549 / 102 / 81 oligos) are far too small to carry a quarter of the score each.
        # Inside `sirna_duplex` truth and prediction are centred per source, so the pooled Pearson
        # cannot be earned by predicting the per-assay offset.
        "kind": "grouped_pearson",
        "pred_col": "inhibition",
        "target_col": "inhibition",
        "group_col": "eval_group",
        "center_col": "source",
        "metric": {
            "name": "per_group_mean_pearson_within_source",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "sequence-genomics/proteingym-fitness": {
        "kind": "grouped_spearman",
        "pred_col": "fitness",
        "target_col": "fitness",
        "group_col": "assay_id",
        "metric": {
            "name": "per_assay_mean_spearman",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "single-cell/openproblems-rna-protein": {
        "kind": "matrix_pearson_files",
        "file_col": "prediction_file",
        "metric": {
            "name": "mean_cell_and_feature_pearson",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "single-cell/openproblems-rna-atac": {
        "kind": "matrix_pearson_files",
        "file_col": "prediction_file",
        "metric": {
            "name": "mean_cell_and_feature_pearson",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "single-cell/dlpfc-spatial-domain": {
        "kind": "grouped_macro_f1",
        "pred_col": "layer",
        "target_col": "layer",
        "group_col": "section",
        "reject_missing_labels": True,
        "metric": {
            "name": "per_section_mean_macro_f1",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "single-cell/tabula-sapiens-ood-annotation": {
        "kind": "nested_grouped_macro_f1",
        "pred_col": "cell_type",
        "target_col": "cell_type",
        "reject_missing_labels": True,  # corrected_v2b tabula (2026-10-14): NaN / empty predictions are invalid
        "outer_group_col": "donor_group",
        "inner_group_col": "tissue",
        "metric": {
            "name": "per_donor_mean_of_per_tissue_macro_f1",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "single-cell/hest-coad-spatial-expression": {
        # corrected_v4 (2026-10-16): the per-spot (across genes) Pearson is computed after centring every gene by
        # its mean over the section's spots (truth and prediction each by their own mean), so the average expression
        # profile alone (r = 0.142 before), a constant and a section-wide offset all score 0. Centring by the
        # training mean was tried first and rejected: the test section is shifted against the training mean, so a
        # constant scored 0.19 and spot-shuffled truth 0.30. The per-gene (across spots) Pearson is unchanged.
        "kind": "matrix_pearson_files",
        "file_col": "prediction_file",
        "row_centre_columns": True,
        "metric": {
            "name": "mean_gene_centred_spot_and_gene_pearson",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "single-cell/seqfish-deconvolution": {
        "kind": "multioutput_r2",
        "label_cols": ["Astrocyte", "CA1", "CA3", "Cajal_Retzius", "Denate", "Endothelial_Stalk", "Endothelial_Tip", "Entorihinal",
                       "Ependymal", "Interneuron", "Microglia_Macrophages", "Mural", "Neurogenesis", "Neuron.Slc17a6",
                       "Oligodendrocyte", "Polydendrocyte"],
        "metric": {
            "name": "mean_cell_type_proportion_r2",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "single-cell/lung-cite-cell-matching": {
        "kind": "accuracy",
        "pred_col": "adt_cell_id",
        "target_col": "adt_cell_id",
        "metric": {
            "name": "top1_cell_matching_accuracy",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "single-cell/gastrulation-kinetic-state": {
        "kind": "multioutput_macro_f1",
        "label_cols": ["developmental_stage", "cell_fate"],
        # Weight targets by their label cardinality (8 stages, 34 retained fates)
        # so the easy coarse-stage head cannot dominate the finer state task.
        "label_weights": [8, 34],
        "metric": {
            "name": "class_cardinality_weighted_stage_and_cell_fate_macro_f1",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "single-cell-perturbation/sciplex-distribution": {
        "kind": "context_residual_matrix_pearson_files",
        "file_col": "prediction_file",
        "train_file_col": "distribution_file",
        "context_cols": ["cell_line", "dose_nM", "replicate"],
        "metric": {
            "name": "mean_context_residual_distribution_statistic_and_gene_pearson",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "single-cell-perturbation/cpjump-genetic-matching": {
        # v2 (2026-10-13): one query per compound, 65 candidate genes; mean per-compound ROC-AUC
        "kind": "grouped_roc_auc",
        "pred_col": "interaction",
        "target_col": "interaction",
        "group_col": "query_id",
        "metric": {
            "name": "mean_per_compound_roc_auc",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "single-cell-perturbation/cellverse-cta": {
        "kind": "macro_f1",
        "pred_col": "answer",
        "target_col": "answer",
        "allowed_labels": ['Astrocytes', 'Astrocytes2', 'COPs', 'Endothelial_cells1', 'Endothelial_cells2', 'ImOlGs', 'Immune_cells', 'Macrophages', 'Microglia_Macrophages', 'Neuron1', 'Neuron2', 'Neuron3', 'Neuron4', 'Neuron5', 'OPCs', 'Oligo1', 'Oligo2', 'Oligo3', 'Oligo4', 'Oligo5', 'Oligo6', 'Pericytes', 'Vasc_smooth_muscle'],  # v2b 2026-10-14: reject unknown labels
        "metric": {
            "name": "macro_f1",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "single-cell-perturbation/perturbench-response": {
        # corrected_v2 (2026-10-13): context x treatment interaction (reference = train-only treatment mean)
        "kind": "projected_residual_pearson_files",
        "file_col": "response_file",
        "group_col": "source",
        "context_col": "context_id",
        "reference_file": "treatment_reference.npz",
        "metric": {
            "name": "per_source_mean_interaction_pearson",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "single-cell-perturbation/scgenescope-treatment": {
        "kind": "macro_f1",
        "pred_col": "treatment_id",
        "target_col": "treatment_id",
        "metric": {
            "name": "macro_f1",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "single-cell-perturbation/tahoe-single-cell": {
        # v2 (2026-10-13): held-out (compound, cell line) response transfer; one effect vector per test row.
        # Scored on the line-specific part: reference = context baseline + public train-only treatment shift,
        # so copying the treatment's mean response from the training lines scores 0.
        "kind": "grouped_context_residual_pearson_files",
        "file_col": "response_file",
        "group_col": "treatment_id",
        "context_col": "context_id",
        "treatment_shift_file": "treatment_shift.npz",
        "aggregate": "group_matrix",
        "metric": {
            "name": "per_treatment_line_specific_response_matrix_pearson",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "single-cell-perturbation/xatlas-orion-perturb": {
        # corrected_v2 (2026-10-13): target gene and cis neighbours (+-200 kb) excluded (private), baseline
        # direction projected out
        "kind": "projected_residual_pearson_files",
        "file_col": "response_file",
        "group_col": "source",
        "context_col": "context_id",
        "exclusion_file": "score_exclusions.npz",
        "metric": {
            "name": "per_source_mean_projected_residual_effect_pearson",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "structure/asep-epitope": {
        "kind": "grouped_average_precision",
        "pred_col": "epitope",
        "target_col": "epitope",
        "group_col": "complex_id",
        "metric": {
            "name": "per_complex_mean_average_precision",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "structure/cpsea-cyclic-peptide-affinity": {
        "kind": "multioutput_calibrated_regression",
        "label_cols": ["rosetta_dG", "vina_score"],
        "metric": {
            "name": "mean_calibrated_rosetta_vina_regression",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "perturbation/gdsc-lineage-holdout-response": {
        "kind": "nested_grouped_pearson",
        "pred_col": "selective_response",
        "target_col": "selective_response",
        "outer_group_col": "lineage",
        "inner_group_col": "drug_id",
        "metric": {
            "name": "per_lineage_per_drug_mean_pearson",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "phenotype-disease/gutcrc-cross-cohort-detection": {
        "kind": "grouped_sensitivity_at_specificity",
        "pred_col": "crc",
        "target_col": "crc",
        "group_col": "cohort",
        "min_rows": 20,
        "metric": {
            "name": "per_cohort_mean_early_stage_sensitivity_at_90_specificity",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "structure/davis-complete-xl-dti": {
        "kind": "grouped_pearson",
        "pred_col": "pKd",
        "target_col": "pKd",
        "group_col": "drug_id",
        "min_rows": 20,
        "metric": {
            "name": "per_drug_mean_pearson_across_kinases",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "structure/decoydb-pose-classify": {
        # v2c (2026-10-14): crystal pose removed; rank AutoDock Vina poses by RMSD to the crystal
        # ligand. Spearman(predicted RMSD, true RMSD) inside each complex, mean over complexes.
        # (v2 top-k near-native config: kind "decoy_topk", pred/target "near_native".)
        "kind": "grouped_spearman",
        "pred_col": "rmsd",
        "target_col": "rmsd",
        "group_col": "complex_id",
        "metric": {
            "name": "per_complex_mean_spearman_rmsd",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
    "structure/denoise-cryo": {
        # v2 (2026-10-13): tolerant per-atom F1 instead of per-voxel Dice; test volumes are
        # averaged inside each sequence cluster first, then across clusters.
        "kind": "denoise_atom_f1",
        "file_col": "type_label_file",
        "aux_file_cols": ["bin_label_file", "reg_label_file"],
        "group_col": "sequence_cluster",
        "labels": (1, 2, 3, 4, 5),
        "tolerance_voxels": 1,
        "metric": {
            "name": "per_cluster_macro_f1_atom_types_1_to_5_tol1voxel",
            "direction": "higher_is_better",
            "normalization": "already_in_0_1",
        },
    },
    "structure/psbench-multimer-mqa": {
        "kind": "nested_grouped_pearson",
        "pred_col": "tmscore_mmalign",
        "target_col": "tmscore_mmalign",
        "outer_group_col": "source",
        "inner_group_col": "target",
        "metric": {
            "name": "per_source_per_target_mean_pearson",
            "direction": "higher_is_better",
            "normalization": "map_minus1_1_to_0_1",
        },
    },
}

# Legacy Pro keys from the former taxonomy still resolve to the validated
# upstream metric implementations so historical results remain interpretable.
_PRO_TASK_ALIASES = {
    "perturbation-dynamics/drugcomb-multiomics-synergy": "chemical-biology/drugcomb-multiomics-synergy",
    "perturbation-dynamics/cellverse-cta": "single-cell-perturbation/cellverse-cta",
    "perturbation-dynamics/cpjump-genetic-matching": "single-cell-perturbation/cpjump-genetic-matching",
    "perturbation-dynamics/perturbench-response": "single-cell-perturbation/perturbench-response",
    "perturbation-dynamics/sciplex-distribution": "single-cell-perturbation/sciplex-distribution",
    "single-cell-multiomics/scgenescope-treatment": "single-cell-perturbation/scgenescope-treatment",
    "perturbation-dynamics/tahoe-single-cell": "single-cell-perturbation/tahoe-single-cell",
    "perturbation-dynamics/xatlas-orion-perturb": "single-cell-perturbation/xatlas-orion-perturb",
    "spatial-tissue-organization/dlpfc-spatial-domain": "single-cell/dlpfc-spatial-domain",
    "single-cell-multiomics/gastrulation-kinetic-state": "single-cell/gastrulation-kinetic-state",
    "spatial-tissue-organization/hest-coad-spatial-expression": "single-cell/hest-coad-spatial-expression",
    "single-cell-multiomics/lung-cite-cell-matching": "single-cell/lung-cite-cell-matching",
    "single-cell-multiomics/openproblems-rna-atac": "single-cell/openproblems-rna-atac",
    "single-cell-multiomics/openproblems-rna-protein": "single-cell/openproblems-rna-protein",
    "spatial-tissue-organization/seqfish-deconvolution": "single-cell/seqfish-deconvolution",
    "single-cell-multiomics/tabula-sapiens-ood-annotation": "single-cell/tabula-sapiens-ood-annotation",
    "chemical-biology/davis-complete-xl-dti": "structure/davis-complete-xl-dti",
}

# Canonical BioXArena-Pro data and grader trees use eight domains with eight
# tasks each. These are the only keys emitted by current training/evaluation.
_PRO_TASK_ALIASES.update({
    "perturbation/cellverse-cta": "single-cell-perturbation/cellverse-cta",
    "perturbation/cpjump-genetic-matching": "single-cell-perturbation/cpjump-genetic-matching",
    "perturbation/perturbench-response": "single-cell-perturbation/perturbench-response",
    "perturbation/sciplex-distribution": "single-cell-perturbation/sciplex-distribution",
    "perturbation/scgenescope-treatment": "single-cell-perturbation/scgenescope-treatment",
    "perturbation/tahoe-single-cell": "single-cell-perturbation/tahoe-single-cell",
    "perturbation/xatlas-orion-perturb": "single-cell-perturbation/xatlas-orion-perturb",
    "single-cell-spatial/dlpfc-spatial-domain": "single-cell/dlpfc-spatial-domain",
    "single-cell-spatial/gastrulation-kinetic-state": "single-cell/gastrulation-kinetic-state",
    "single-cell-spatial/hest-coad-spatial-expression": "single-cell/hest-coad-spatial-expression",
    "single-cell-spatial/lung-cite-cell-matching": "single-cell/lung-cite-cell-matching",
    "single-cell-spatial/openproblems-rna-atac": "single-cell/openproblems-rna-atac",
    "single-cell-spatial/openproblems-rna-protein": "single-cell/openproblems-rna-protein",
    "single-cell-spatial/seqfish-deconvolution": "single-cell/seqfish-deconvolution",
    "single-cell-spatial/tabula-sapiens-ood-annotation": "single-cell/tabula-sapiens-ood-annotation",
})

for _pro_key, _source_key in _PRO_TASK_ALIASES.items():
    TASK_CONFIG[_pro_key] = TASK_CONFIG[_source_key]

METRIC_INFO_BY_TASK = {
    task_key: config["metric"] for task_key, config in TASK_CONFIG.items()
}


def validate_submission_frame(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    expected_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate exact submission columns and first-column ID order."""
    if expected_columns is None:
        expected_columns = list(answers.columns)
    if list(submission.columns) != expected_columns:
        raise InvalidSubmissionError(
            f"Submission columns must be exactly {expected_columns}; "
            f"got {list(submission.columns)}."
        )
    if submission.empty and not answers.empty:
        raise InvalidSubmissionError("Submission is empty but answers are not.")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission has {len(submission)} rows; expected {len(answers)}."
        )
    submission_id_col = submission.columns[0]
    answer_id_col = answers.columns[0]
    if submission_id_col != answer_id_col:
        raise InvalidSubmissionError(
            f"Submission first column '{submission_id_col}' does not match "
            f"answers first column '{answer_id_col}'."
        )
    if submission[submission_id_col].duplicated().any():
        raise InvalidSubmissionError("Submission contains duplicate IDs.")
    if answers[answer_id_col].duplicated().any():
        raise InvalidSubmissionError("Answers contain duplicate IDs.")
    if submission[submission_id_col].tolist() != answers[answer_id_col].tolist():
        raise InvalidSubmissionError(
            "Submission IDs must exactly match answers IDs, including order."
        )
    return submission.reset_index(drop=True), answers.reset_index(drop=True)


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise InvalidSubmissionError(f"{label} is missing columns: {missing}")


def numeric_array(df: pd.DataFrame, column: str, label: str) -> np.ndarray:
    try:
        values = pd.to_numeric(df[column], errors="raise").to_numpy(dtype=float)
    except Exception as exc:
        raise InvalidSubmissionError(f"{label} column '{column}' must be numeric: {exc}")
    if not np.isfinite(values).all():
        raise InvalidSubmissionError(f"{label} column '{column}' contains non-finite values.")
    return values


def pearson_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        raise InvalidSubmissionError("Pearson correlation requires at least two rows.")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        value = pearsonr(y_true, y_pred).statistic
    return 0.0 if not np.isfinite(value) else float(value)


def calibrated_regression_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Reward both ranking and calibrated numerical accuracy on a 0--1 scale."""
    # v4 chance-zero scale: clip(r, 0, 1) instead of (r + 1) / 2 for the correlation component
    correlation_score = float(np.clip(pearson_safe(y_true, y_pred), 0.0, 1.0))
    scale = float(np.std(y_true))
    rmse = float(np.sqrt(np.mean(np.square(y_pred - y_true))))
    if scale <= 1e-12:
        calibration_score = 1.0 if rmse <= 1e-12 else 0.0
    else:
        calibration_score = float(np.clip(1.0 - rmse / scale, 0.0, 1.0))
    return float(0.5 * correlation_score + 0.5 * calibration_score)


def integer_accuracy_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rounded = np.rint(y_pred)
    if not np.allclose(y_pred, rounded, atol=1e-8, rtol=0.0):
        raise InvalidSubmissionError("Count predictions must be integers.")
    return float(np.mean(np.rint(y_true).astype(np.int64) == rounded.astype(np.int64)))


def integer_macro_f1_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rounded = np.rint(y_pred)
    if not np.allclose(y_pred, rounded, atol=1e-8, rtol=0.0):
        raise InvalidSubmissionError("Count predictions must be integers.")
    return float(f1_score(
        np.rint(y_true).astype(np.int64),
        rounded.astype(np.int64),
        average="macro",
        zero_division=0,
    ))


def spearman_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        raise InvalidSubmissionError("Spearman correlation requires at least two rows.")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        value = spearmanr(y_true, y_pred).statistic
    return 0.0 if not np.isfinite(value) else float(value)


def roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    unique = np.unique(y_true)
    if len(unique) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def average_precision_safe(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.sum(y_true == 1) == 0:
        return None
    return float(average_precision_score(y_true, y_score))


def sensitivity_at_specificity_safe(y_true: np.ndarray, y_score: np.ndarray, specificity: float = 0.9) -> float | None:
    """Sensitivity at a fixed specificity. Labels: 1 = scored positive, 0 = negative, -1 = not scored.

    The ROC curve is interpolated at FPR = 1 - specificity, so a constant score gets 1 - specificity.
    """
    from sklearn.metrics import roc_curve

    keep = y_true >= 0
    y, s = y_true[keep], y_score[keep]
    if int((y == 1).sum()) < 5 or int((y == 0).sum()) < 5:
        return None
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(1.0 - specificity, fpr, tpr))


def mean_or_raise(scores: list[float], context: str) -> float:
    scores = [float(score) for score in scores if np.isfinite(score)]
    if not scores:
        raise InvalidSubmissionError(f"No valid scores available for {context}.")
    return float(np.mean(scores))


def infer_public_dir(answers_path: Path | str | None) -> Path | None:
    if answers_path is None:
        return None
    answers_path = Path(answers_path).resolve()
    try:
        private_dir = answers_path.parent
        task = private_dir.parent.name
        domain = private_dir.parent.parent.name
        private_root = private_dir.parent.parent.parent
        workspace_root = private_root.parent
    except IndexError:
        return None
    return workspace_root / "2_BioXArena-Pro-Data-Public" / domain / task / "public"


def require_public_dir(public_dir: Path | str | None, answers_path: Path | str | None) -> Path:
    if public_dir is None:
        public_dir = infer_public_dir(answers_path)
    if public_dir is None:
        raise InvalidSubmissionError("public_dir is required for this metric.")
    public_dir = Path(public_dir)
    if not public_dir.exists():
        raise InvalidSubmissionError(f"public_dir does not exist: {public_dir}")
    return public_dir


def require_private_dir(answers_path: Path | str | None) -> Path:
    if answers_path is None:
        raise InvalidSubmissionError("answers_path is required for file-based grading.")
    private_dir = Path(answers_path).resolve().parent
    if not private_dir.exists():
        raise InvalidSubmissionError(f"Private directory does not exist: {private_dir}")
    return private_dir


def require_output_dir(output_dir: Path | str | None) -> Path:
    if output_dir is None:
        raise InvalidSubmissionError("output_dir is required for file-based grading.")
    output_dir = Path(output_dir).resolve()
    if not output_dir.exists():
        raise InvalidSubmissionError(f"Output directory does not exist: {output_dir}")
    return output_dir


def read_test(public_dir: Path | str | None, answers_path: Path | str | None) -> pd.DataFrame:
    public_dir = require_public_dir(public_dir, answers_path)
    test_path = public_dir / "test.csv"
    if not test_path.exists():
        raise InvalidSubmissionError(f"Missing public test.csv: {test_path}")
    return pd.read_csv(test_path)


def validate_test_alignment(test: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    require_columns(test, ["id"], "test.csv")
    if len(test) != len(answers):
        raise InvalidSubmissionError(
            f"test.csv has {len(test)} rows; answers has {len(answers)} rows."
        )
    if test["id"].tolist() != answers["id"].tolist():
        raise InvalidSubmissionError("test.csv IDs must exactly match answers IDs and order.")
    return test.reset_index(drop=True)


def apply_test_subset(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    test: pd.DataFrame,
    subset: tuple[str, str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if subset is None:
        return submission, answers, test
    column, value = subset
    require_columns(test, [column], "test.csv")
    mask = test[column].astype(str).str.lower() == str(value).lower()
    if not mask.any():
        raise InvalidSubmissionError(f"No test rows matched subset {column}={value!r}.")
    return (
        submission.loc[mask].reset_index(drop=True),
        answers.loc[mask].reset_index(drop=True),
        test.loc[mask].reset_index(drop=True),
    )


def grouped_metric(
    test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_col: str,
    metric_fn: Callable[[np.ndarray, np.ndarray], float | None],
    min_rows: int = 2,
) -> float:
    require_columns(test, [group_col], "test.csv")
    scores: list[float] = []
    for _, positions in test.groupby(group_col, sort=True).indices.items():
        idx = np.asarray(list(positions), dtype=int)
        if len(idx) < min_rows:
            continue
        score = metric_fn(y_true[idx], y_pred[idx])
        if score is not None and np.isfinite(score):
            scores.append(float(score))
    return mean_or_raise(scores, f"grouped metric by {group_col}")


def prepare_standard_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_columns = config.get("submission_columns") or list(answers.columns)
    submission, answers = validate_submission_frame(submission, answers, expected_columns)
    target_cols = [config[col] for col in ("pred_col", "target_col") if col in config]
    if "label_cols" in config:
        target_cols.extend(config["label_cols"])
    require_columns(submission, [col for col in target_cols if col in submission.columns], "submission")
    require_columns(answers, [col for col in target_cols if col in answers.columns], "answers")
    return submission, answers


def grade_pearson_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    submission, answers = prepare_standard_task(submission, answers, config)
    y_true = numeric_array(answers, config["target_col"], "answers")
    y_pred = numeric_array(submission, config["pred_col"], "submission")
    return pearson_safe(y_true, y_pred)


def grade_roc_auc_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    submission, answers = prepare_standard_task(submission, answers, config)
    y_true = numeric_array(answers, config["target_col"], "answers")
    y_pred = numeric_array(submission, config["pred_col"], "submission")
    score = roc_auc_safe(y_true, y_pred)
    if score is None:
        raise InvalidSubmissionError("ROC-AUC requires at least one positive and one negative.")
    return score


def grade_calibrated_regression_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    submission, answers = prepare_standard_task(submission, answers, config)
    return calibrated_regression_safe(
        numeric_array(answers, config["target_col"], "answers"),
        numeric_array(submission, config["pred_col"], "submission"),
    )


def grade_grouped_numeric_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
    metric_fn: Callable[[np.ndarray, np.ndarray], float | None],
) -> float:
    submission, answers = prepare_standard_task(submission, answers, config)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    submission, answers, test = apply_test_subset(
        submission, answers, test, config.get("subset")
    )
    y_true = numeric_array(answers, config["target_col"], "answers")
    y_pred = numeric_array(submission, config["pred_col"], "submission")
    reference_col = config.get("reference_col")
    if reference_col:
        # Score the deviation from a published per-row reference (test.csv column) for truth and prediction alike.
        require_columns(test, [reference_col], "test.csv")
        reference = numeric_array(test, reference_col, "test.csv")
        y_true = y_true.astype(float) - reference
        y_pred = y_pred.astype(float) - reference
    center_col = config.get("center_col")
    if center_col:
        # Remove each sub-group's mean from truth and prediction before the group metric, so
        # that pooling several assays into one evaluation group cannot be scored by predicting
        # the per-assay offset (within-sub-group correlation).
        require_columns(test, [center_col], "test.csv")
        y_true = y_true.astype(float).copy()
        y_pred = y_pred.astype(float).copy()
        for _, positions in test.groupby(center_col, sort=True).indices.items():
            idx = np.asarray(list(positions), dtype=int)
            y_true[idx] -= float(np.mean(y_true[idx]))
            y_pred[idx] -= float(np.mean(y_pred[idx]))
    return grouped_metric(
        test,
        y_true,
        y_pred,
        config["group_col"],
        metric_fn,
        min_rows=int(config.get("min_rows", 2)),
    )


def grade_macro_f1_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    submission, answers = prepare_standard_task(submission, answers, config)
    if submission[config["pred_col"]].isna().any():
        raise InvalidSubmissionError(
            f"Submission column '{config['pred_col']}' contains missing class labels."
        )
    allowed = config.get("allowed_labels")
    if allowed is not None:
        unknown = sorted(set(submission[config["pred_col"]].astype(str)) - set(allowed))
        if unknown:
            raise InvalidSubmissionError(
                f"Submission column '{config['pred_col']}' contains unknown labels: {unknown[:5]}"
            )
    return float(
        f1_score(
            answers[config["target_col"]],
            submission[config["pred_col"]],
            average="macro",
            zero_division=0,
        )
    )


def grade_accuracy_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    submission, answers = prepare_standard_task(submission, answers, config)
    pred = submission[config["pred_col"]]
    if pred.isna().any():
        raise InvalidSubmissionError(
            f"Submission column '{config['pred_col']}' contains missing labels."
        )
    return float(np.mean(pred.astype(str).to_numpy() == answers[config["target_col"]].astype(str).to_numpy()))


def grade_grouped_macro_f1_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
) -> float:
    submission, answers = prepare_standard_task(submission, answers, config)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    require_columns(test, [config["group_col"]], "test.csv")
    if config.get("reject_missing_labels") and submission[config["pred_col"]].isna().any():
        raise InvalidSubmissionError(
            f"Submission column '{config['pred_col']}' contains missing labels."
        )
    pred_series = submission[config["pred_col"]].astype(str)
    true_series = answers[config["target_col"]].astype(str)
    if config.get("strip_labels"):
        pred_series = pred_series.str.strip()
        true_series = true_series.str.strip()
    if config.get("reject_unknown_labels"):
        # corrected_v2 (2026-10-13): missing / NaN / unknown labels are an invalid submission, not a wrong answer
        raw = submission[config["pred_col"]]
        if raw.isna().any():
            raise InvalidSubmissionError(f"{config['pred_col']} contains missing values")
        allowed = set(true_series.unique())
        bad = sorted(set(pred_series.unique()) - allowed)
        if bad:
            raise InvalidSubmissionError(
                f"{config['pred_col']} contains labels outside the class set: {bad[:5]}"
            )
    pred = pred_series.to_numpy()
    true = true_series.to_numpy()
    if config.get("exclude_ids"):
        # v3: rows with unresolvable label conflicts are required in the submission but not scored
        keep = ~answers["id"].astype(str).isin(set(config["exclude_ids"])).to_numpy()
        test = test.loc[keep].reset_index(drop=True)
        pred, true = pred[keep], true[keep]
    scores: list[float] = []
    for _, positions in test.groupby(config["group_col"], sort=True).indices.items():
        idx = np.asarray(list(positions), dtype=int)
        scores.append(
            float(f1_score(true[idx], pred[idx], average="macro", zero_division=0))
        )
    return mean_or_raise(scores, f"macro-F1 by {config['group_col']}")


def grade_nested_grouped_macro_f1_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
) -> float:
    """Balance macro-F1 within inner strata, then equally across outer units."""
    submission, answers = prepare_standard_task(submission, answers, config)
    if config.get("reject_missing_labels") and (
        submission[config["pred_col"]].isna().any()
        or submission[config["pred_col"]].astype(str).str.strip().eq("").any()
    ):
        raise InvalidSubmissionError("Submission contains missing labels.")
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    outer_col = config["outer_group_col"]
    inner_col = config["inner_group_col"]
    require_columns(test, [outer_col, inner_col], "test.csv")
    frame = test[[outer_col, inner_col]].copy()
    frame["pred"] = submission[config["pred_col"]].astype(str).to_numpy()
    frame["true"] = answers[config["target_col"]].astype(str).to_numpy()
    outer_scores: list[float] = []
    for _, outer_group in frame.groupby(outer_col, sort=True):
        inner_scores = [
            float(f1_score(group["true"], group["pred"], average="macro", zero_division=0))
            for _, group in outer_group.groupby(inner_col, sort=True)
        ]
        outer_scores.append(mean_or_raise(inner_scores, f"macro-F1 by {inner_col}"))
    return mean_or_raise(outer_scores, f"nested macro-F1 by {outer_col}")


def grade_multilabel_combined_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    submission, answers = prepare_standard_task(submission, answers, config)
    label_cols = config["label_cols"]
    y_true = answers[label_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=int)
    y_score = submission[label_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y_score).all():
        raise InvalidSubmissionError("Multilabel submission contains non-finite values.")
    y_pred = (y_score >= 0.5).astype(int)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mean_ap = average_precision_score(y_true, y_score, average="macro")
    return float(0.5 * macro_f1 + 0.5 * mean_ap)


def grade_multilabel_prevalence_corrected_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    """Prevalence-corrected multilabel score in [0, 1].

    Per evaluated label j with prevalence p_j in the hidden test set:
        ap_gain_j = clip((AP_j - p_j) / (1 - p_j), 0, 1)
        f1_gain_j = clip((F1_j - 2 p_j / (1 + p_j)) / (1 - 2 p_j / (1 + p_j)), 0, 1)
    where AP_j is the average precision of the submitted score column and F1_j the
    binary F1 of (score >= threshold).  2 p_j / (1 + p_j) is the best F1 any
    label-independent constant rule can reach (predict-all-positive), so an
    uninformative submission scores 0 on both components and the reference scores 1.

        score = 0.5 * mean_j(ap_gain_j) + 0.5 * mean_j(f1_gain_j)

    A label with fewer than config["min_label_count"] positives or negatives in the
    test set is excluded from both means (its gain is statistically meaningless).
    """
    submission, answers = prepare_standard_task(submission, answers, config)
    label_cols = config["label_cols"]
    threshold = float(config.get("threshold", 0.5))
    min_label_count = int(config.get("min_label_count", 30))
    y_true = answers[label_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    y_score = submission[label_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y_score).all():
        raise InvalidSubmissionError("Multilabel submission contains non-finite values.")
    if float(y_score.min()) < 0.0 or float(y_score.max()) > 1.0:
        raise InvalidSubmissionError(
            "Multilabel submission values must be probabilities in [0, 1]."
        )
    if not np.isin(y_true, (0.0, 1.0)).all():
        raise InvalidSubmissionError("Answer labels must be binary 0/1.")
    y_pred = (y_score >= threshold).astype(int)
    ap_gains: list[float] = []
    f1_gains: list[float] = []
    for index in range(len(label_cols)):
        column = y_true[:, index]
        positives = int(column.sum())
        negatives = int(len(column) - positives)
        if positives < min_label_count or negatives < min_label_count:
            continue
        prevalence = positives / float(len(column))
        average_precision = float(average_precision_score(column, y_score[:, index]))
        ap_gains.append(
            float(np.clip((average_precision - prevalence) / (1.0 - prevalence), 0.0, 1.0))
        )
        f1_floor = 2.0 * prevalence / (1.0 + prevalence)
        label_f1 = float(f1_score(column, y_pred[:, index], zero_division=0))
        f1_gains.append(float(np.clip((label_f1 - f1_floor) / (1.0 - f1_floor), 0.0, 1.0)))
    if not ap_gains:
        raise InvalidSubmissionError(
            "No label has enough positives and negatives to be evaluated."
        )
    return float(0.5 * float(np.mean(ap_gains)) + 0.5 * float(np.mean(f1_gains)))


def grade_multioutput_macro_f1_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    submission, answers = prepare_standard_task(submission, answers, config)
    scores = []
    for column in config["label_cols"]:
        if submission[column].isna().any():
            raise InvalidSubmissionError(
                f"Submission column '{column}' contains missing class labels."
            )
        scores.append(
            f1_score(
                answers[column],
                submission[column],
                average="macro",
                zero_division=0,
            )
        )
    weights = config.get("label_weights")
    if weights is None:
        return mean_or_raise(scores, "multi-output macro-F1")
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.shape != (len(scores),) or not np.isfinite(weight_array).all() or np.any(weight_array <= 0):
        raise InvalidSubmissionError(
            "label_weights must contain one positive finite value per label column."
        )
    return float(np.average(np.asarray(scores, dtype=float), weights=weight_array))


def grade_multioutput_r2_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    """Mean over output columns of the coefficient of determination, clipped to [0, 1]."""
    submission, answers = validate_submission_frame(
        submission, answers, ["id", *config["label_cols"]]
    )
    scores: list[float] = []
    for column in config["label_cols"]:
        y_true = numeric_array(answers, column, "answers")
        y_pred = numeric_array(submission, column, "submission")
        variance = float(np.var(y_true))
        if variance == 0.0:
            continue
        value = 1.0 - float(np.mean((y_true - y_pred) ** 2)) / variance
        if not np.isfinite(value):
            value = 0.0
        scores.append(float(np.clip(value, 0.0, 1.0)))
    return mean_or_raise(scores, "multi-output R^2")


def grade_multioutput_pearson_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    submission, answers = validate_submission_frame(
        submission, answers, ["id", *config["label_cols"]]
    )
    scores: list[float] = []
    for column in config["label_cols"]:
        scores.append(
            pearson_safe(
                numeric_array(answers, column, "answers"),
                numeric_array(submission, column, "submission"),
            )
        )
    return mean_or_raise(scores, "multi-output Pearson")


def grade_multioutput_calibrated_regression_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    submission, answers = validate_submission_frame(
        submission, answers, ["id", *config["label_cols"]]
    )
    scores = [
        calibrated_regression_safe(
            numeric_array(answers, column, "answers"),
            numeric_array(submission, column, "submission"),
        )
        for column in config["label_cols"]
    ]
    return mean_or_raise(scores, "multi-output calibrated regression")


def grade_multioutput_integer_accuracy_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    submission, answers = validate_submission_frame(
        submission, answers, ["id", *config["label_cols"]]
    )
    scores = [
        integer_accuracy_safe(
            numeric_array(answers, column, "answers"),
            numeric_array(submission, column, "submission"),
        )
        for column in config["label_cols"]
    ]
    return mean_or_raise(scores, "multi-output exact count accuracy")


def c_index(times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0.0
    n = len(times)
    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if times[j] <= times[i]:
                continue
            comparable += 1.0
            if risk_scores[i] > risk_scores[j]:
                concordant += 1.0
            elif risk_scores[i] == risk_scores[j]:
                concordant += 0.5
    if comparable == 0:
        return 0.5
    return float(concordant / comparable)


def grade_grouped_c_index_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
) -> float:
    submission, answers = validate_submission_frame(
        submission, answers, config["submission_columns"]
    )
    require_columns(answers, [config["time_col"], config["censor_col"]], "answers")
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    times = numeric_array(answers, config["time_col"], "answers")
    censor = numeric_array(answers, config["censor_col"], "answers")
    events = 1 - censor
    risk = numeric_array(submission, config["risk_col"], "submission")
    require_columns(test, [config["group_col"]], "test.csv")
    scores: list[float] = []
    for _, positions in test.groupby(config["group_col"], sort=True).indices.items():
        idx = np.asarray(list(positions), dtype=int)
        scores.append(c_index(times[idx], events[idx], risk[idx]))
    return mean_or_raise(scores, f"grouped C-index by {config['group_col']}")


def resolve_relative_path(root: Path, value: Any, label: str) -> Path:
    if pd.isna(value):
        raise InvalidSubmissionError(f"{label} path is missing.")
    candidate = Path(str(value))
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        if any(part == ".." for part in candidate.parts):
            raise InvalidSubmissionError(f"{label} path may not contain '..': {value!r}")
        resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise InvalidSubmissionError(f"{label} path escapes its root: {value!r}")
    return resolved


def load_npz_y(path: Path) -> np.ndarray:
    if not path.exists():
        raise InvalidSubmissionError(f"Missing prediction/label NPZ file: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            if "y" not in data:
                raise InvalidSubmissionError(f"NPZ file lacks key 'y': {path}")
            return np.asarray(data["y"])
    except InvalidSubmissionError:
        raise
    except Exception as exc:
        raise InvalidSubmissionError(f"Unable to read NPZ file {path}: {exc}")


def read_mrc_volume(path: Path) -> np.ndarray:
    if not path.exists():
        raise InvalidSubmissionError(f"Missing MRC label file: {path}")
    mode_to_dtype = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16}
    try:
        with path.open("rb") as handle:
            header = np.fromfile(handle, dtype="<i4", count=4)
            if len(header) != 4:
                raise InvalidSubmissionError(f"Invalid MRC header: {path}")
            nx, ny, nz, mode = [int(x) for x in header]
            if mode not in mode_to_dtype:
                raise InvalidSubmissionError(f"Unsupported MRC mode {mode} in {path}")
            handle.seek(1024)
            data = np.fromfile(handle, dtype=mode_to_dtype[mode], count=nx * ny * nz)
        if data.size != nx * ny * nz:
            raise InvalidSubmissionError(f"MRC body size does not match header: {path}")
        return data.reshape((nz, ny, nx))
    except InvalidSubmissionError:
        raise
    except Exception as exc:
        raise InvalidSubmissionError(f"Unable to read MRC file {path}: {exc}")


def dice_binary(pred: np.ndarray, true: np.ndarray) -> float:
    pred_mask = pred.astype(bool)
    true_mask = true.astype(bool)
    denom = pred_mask.sum() + true_mask.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred_mask, true_mask).sum() / denom)


def dice_label(pred: np.ndarray, true: np.ndarray, label: int) -> float:
    pred_mask = pred == label
    true_mask = true == label
    denom = pred_mask.sum() + true_mask.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred_mask, true_mask).sum() / denom)


def mean_dice_labels_chunked(
    pred: np.ndarray,
    true: np.ndarray,
    labels: tuple[int, ...],
    block_size: int = 8_000_000,
) -> float:
    """Compute multiclass Dice in one chunked pass over very large volumes."""
    pred_flat = np.asarray(pred).reshape(-1)
    true_flat = np.asarray(true).reshape(-1)
    if pred_flat.size != true_flat.size:
        raise InvalidSubmissionError(
            f"Dice arrays have different sizes: {pred_flat.size} and {true_flat.size}."
        )
    max_label = max(labels)
    n_classes = max_label + 1
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for start in range(0, pred_flat.size, block_size):
        stop = min(start + block_size, pred_flat.size)
        pred_block = np.asarray(pred_flat[start:stop], dtype=np.int16)
        true_block = np.asarray(true_flat[start:stop], dtype=np.int16)
        if pred_block.size and (pred_block.min() < 0 or pred_block.max() > max_label):
            raise InvalidSubmissionError(
                f"DenoiseCryo predictions must use labels 0..{max_label}."
            )
        if true_block.size and (true_block.min() < 0 or true_block.max() > max_label):
            raise InvalidSubmissionError(
                f"DenoiseCryo answers must use labels 0..{max_label}."
            )
        joint = true_block * n_classes + pred_block
        confusion += np.bincount(joint, minlength=n_classes * n_classes).reshape(
            n_classes, n_classes
        )
    true_count = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    intersection = np.diag(confusion)
    scores = []
    for label in labels:
        denominator = pred_count[label] + true_count[label]
        scores.append(1.0 if denominator == 0 else 2.0 * intersection[label] / denominator)
    return float(np.mean(scores))


def group_mean_from_row_scores(test: pd.DataFrame, row_scores: list[float], group_col: str) -> float:
    require_columns(test, [group_col], "test.csv")
    df = pd.DataFrame({group_col: test[group_col].to_numpy(), "score": row_scores})
    return mean_or_raise(df.groupby(group_col, sort=True)["score"].mean().tolist(), group_col)


def grade_cellsam_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
    output_dir: Path | str | None,
) -> float:
    submission, answers = validate_submission_frame(submission, answers)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    private_dir = require_private_dir(answers_path)
    output_dir = require_output_dir(output_dir)
    row_scores: list[float] = []
    for i in range(len(answers)):
        pred = load_npz_y(resolve_relative_path(output_dir, submission.at[i, config["file_col"]], "prediction"))
        true = load_npz_y(resolve_relative_path(private_dir, answers.at[i, config["file_col"]], "answer"))
        if pred.shape != true.shape or pred.shape[-1] != 2:
            raise InvalidSubmissionError(
                f"CellSAM shape mismatch for row {i}: pred {pred.shape}, true {true.shape}."
            )
        pred = pred > 0
        true = true > 0
        row_scores.append(
            float(np.mean([dice_binary(pred[:, :, 0], true[:, :, 0]), dice_binary(pred[:, :, 1], true[:, :, 1])]))
        )
    return group_mean_from_row_scores(test, row_scores, config["group_col"])


def load_control_mean(public_dir: Path, source: str) -> np.ndarray:
    control_path = public_dir / "control_mean.npz"
    if not control_path.exists():
        raise InvalidSubmissionError(f"Missing control_mean.npz: {control_path}")
    key = f"{source}_control"
    with np.load(control_path, allow_pickle=False) as data:
        if key not in data:
            raise InvalidSubmissionError(f"control_mean.npz lacks key {key!r}.")
        return np.asarray(data[key], dtype=float)


def grade_delta_pearson_files_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
    output_dir: Path | str | None,
) -> float:
    submission, answers = validate_submission_frame(submission, answers)
    public_dir = require_public_dir(public_dir, answers_path)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    private_dir = require_private_dir(answers_path)
    output_dir = require_output_dir(output_dir)
    row_scores: list[float] = []
    for i in range(len(answers)):
        source = str(test.at[i, config["group_col"]])
        control = load_control_mean(public_dir, source)
        pred = np.asarray(
            load_npz_y(resolve_relative_path(output_dir, submission.at[i, config["file_col"]], "prediction")),
            dtype=float,
        )
        true = np.asarray(
            load_npz_y(resolve_relative_path(private_dir, answers.at[i, config["file_col"]], "answer")),
            dtype=float,
        )
        if pred.shape != true.shape or pred.shape != control.shape:
            raise InvalidSubmissionError(
                f"Response vector shape mismatch for row {i}: "
                f"pred {pred.shape}, true {true.shape}, control {control.shape}."
            )
        row_scores.append(pearson_safe(true - control, pred - control))
    return group_mean_from_row_scores(test, row_scores, config["group_col"])


def grade_context_residual_pearson_files_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
    output_dir: Path | str | None,
) -> float:
    """Score treatment-specific response after removing a train-only context mean."""
    submission, answers = validate_submission_frame(submission, answers)
    public_dir = require_public_dir(public_dir, answers_path)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    require_columns(test, [config["group_col"], config["context_col"]], "test.csv")
    private_dir = require_private_dir(answers_path)
    output_dir = require_output_dir(output_dir)
    baseline_path = public_dir / "context_baseline.npz"
    score_index_path = public_dir / "score_gene_indices.npz"
    if not baseline_path.exists() or not score_index_path.exists():
        raise InvalidSubmissionError("Missing context baseline or score-gene archive.")

    row_scores: list[float] = []
    with np.load(baseline_path, allow_pickle=False) as baselines, np.load(
        score_index_path, allow_pickle=False
    ) as score_indices:
        for i in range(len(answers)):
            source = str(test.at[i, config["group_col"]])
            context_id = str(test.at[i, config["context_col"]])
            index_key = f"{source}_indices"
            if context_id not in baselines or index_key not in score_indices:
                raise InvalidSubmissionError(
                    f"Missing residual metadata for source={source!r}, context={context_id!r}."
                )
            baseline = np.asarray(baselines[context_id], dtype=float)
            indices = np.asarray(score_indices[index_key], dtype=np.int64)
            pred = np.asarray(
                load_npz_y(
                    resolve_relative_path(
                        output_dir, submission.at[i, config["file_col"]], "prediction"
                    )
                ),
                dtype=float,
            )
            true = np.asarray(
                load_npz_y(
                    resolve_relative_path(
                        private_dir, answers.at[i, config["file_col"]], "answer"
                    )
                ),
                dtype=float,
            )
            if pred.shape != true.shape or pred.shape != baseline.shape:
                raise InvalidSubmissionError(
                    f"Response vector shape mismatch for row {i}: "
                    f"pred {pred.shape}, true {true.shape}, baseline {baseline.shape}."
                )
            if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(true)):
                raise InvalidSubmissionError(f"Invalid score-gene indices for source {source!r}.")
            row_scores.append(
                pearson_safe(true[indices] - baseline[indices], pred[indices] - baseline[indices])
            )
    return group_mean_from_row_scores(test, row_scores, config["group_col"])


def grade_grouped_context_residual_pearson_files_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
    output_dir: Path | str | None,
) -> float:
    """Per test group (e.g. one treatment over its held-out cell lines): Pearson between the flattened
    (prediction - context baseline) and (truth - context baseline) matrices over the group's rows and all
    genes; the task score is the mean over groups. The context baseline (public context_baseline.npz, keyed by
    the row's context id) is the mean training effect in that context, so a treatment-blind prediction scores
    zero, and because rows of one group are correlated jointly, the relative response size across cell lines
    counts as well as its direction. With config "aggregate": "unit" the per-row correlations are averaged
    within the group instead."""
    submission, answers = validate_submission_frame(submission, answers)
    public_dir = require_public_dir(public_dir, answers_path)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    group_col, context_col, file_col = config["group_col"], config["context_col"], config["file_col"]
    require_columns(test, [group_col, context_col], "test.csv")
    require_columns(submission, [file_col], "submission")
    require_columns(answers, [file_col], "answers")
    private_dir = require_private_dir(answers_path)
    output_dir = require_output_dir(output_dir)
    baseline_path = public_dir / "context_baseline.npz"
    if not baseline_path.exists():
        raise InvalidSubmissionError(f"Missing context baseline archive: {baseline_path}")
    aggregate = config.get("aggregate", "group_matrix")
    shift_file = config.get("treatment_shift_file")
    shifts: dict[str, np.ndarray] = {}
    if shift_file:
        shift_path = public_dir / shift_file
        if not shift_path.exists():
            raise InvalidSubmissionError(f"Missing treatment shift archive: {shift_path}")
        with np.load(shift_path, allow_pickle=False) as data:
            shifts = {key: np.asarray(data[key], dtype=float) for key in data.files}
    group_scores: list[float] = []
    with np.load(baseline_path, allow_pickle=False) as baselines:
        for group_value, positions in test.groupby(group_col, sort=True).indices.items():
            true_rows: list[np.ndarray] = []
            pred_rows: list[np.ndarray] = []
            for i in sorted(int(p) for p in positions):
                context_id = str(test.at[i, context_col])
                if context_id not in baselines:
                    raise InvalidSubmissionError(f"Missing context baseline for context {context_id!r}.")
                baseline = np.asarray(baselines[context_id], dtype=float)
                if shift_file:
                    if str(group_value) not in shifts:
                        raise InvalidSubmissionError(f"Missing treatment shift for group {group_value!r}.")
                    baseline = baseline + shifts[str(group_value)]
                pred = np.asarray(
                    load_npz_y(resolve_relative_path(output_dir, submission.at[i, file_col], "prediction")),
                    dtype=float,
                ).reshape(-1)
                true = np.asarray(
                    load_npz_y(resolve_relative_path(private_dir, answers.at[i, file_col], "answer")),
                    dtype=float,
                ).reshape(-1)
                if pred.shape != true.shape or pred.shape != baseline.shape:
                    raise InvalidSubmissionError(
                        f"Response vector shape mismatch for row {i}: "
                        f"pred {pred.shape}, true {true.shape}, baseline {baseline.shape}."
                    )
                if not np.isfinite(pred).all():
                    raise InvalidSubmissionError(f"Prediction for row {i} contains NaN or inf.")
                true_rows.append(true - baseline)
                pred_rows.append(pred - baseline)
            if aggregate == "unit":
                group_scores.append(
                    float(np.mean([pearson_safe(t, p) for t, p in zip(true_rows, pred_rows)]))
                )
            else:
                group_scores.append(
                    pearson_safe(np.concatenate(true_rows), np.concatenate(pred_rows))
                )
    return mean_or_raise(group_scores, f"grouped context-residual Pearson by {group_col}")


def grade_response_pearson_files_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
    output_dir: Path | str | None,
) -> float:
    submission, answers = validate_submission_frame(submission, answers)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    private_dir = require_private_dir(answers_path)
    output_dir = require_output_dir(output_dir)
    row_scores: list[float] = []
    for i in range(len(answers)):
        pred = np.asarray(
            load_npz_y(resolve_relative_path(output_dir, submission.at[i, config["file_col"]], "prediction")),
            dtype=float,
        )
        true = np.asarray(
            load_npz_y(resolve_relative_path(private_dir, answers.at[i, config["file_col"]], "answer")),
            dtype=float,
        )
        if pred.shape != true.shape:
            raise InvalidSubmissionError(
                f"Response vector shape mismatch for row {i}: pred {pred.shape}, true {true.shape}."
            )
        row_scores.append(pearson_safe(true, pred))
    return group_mean_from_row_scores(test, row_scores, config["group_col"])


# v3 (2026-10-15): the scored set of rows/columns is fixed by the ground truth only. A target whose truth has
# non-degenerate variance is always in the denominator; a constant (or numerically constant) prediction for it
# scores r = 0 instead of being dropped. The old code used `denominator > 0`, which included the prediction
# variance, so a submission could abstain on hard targets by predicting a constant and raise its mean.
_TRUTH_REL_TOL = 1e-12   # truth std <= this x truth RMS along the axis -> target is not scored (degenerate truth)
_PRED_REL_TOL = 1e-6     # pred std  <= this x reference RMS            -> prediction counts as constant (r = 0)


def _axis_pearson_truth_scored(
    true: np.ndarray,
    pred: np.ndarray,
    axis: int,
    mask: np.ndarray | None = None,
    pred_reference: np.ndarray | None = None,
) -> float:
    """Mean Pearson over rows (axis=1) / columns (axis=0); scored set decided by the truth alone.

    pred_reference: optional array of the same shape whose per-axis RMS is the scale against which the
    prediction's spread is judged (used for residual matrices, so a prediction equal to the subtracted
    baseline up to float noise counts as constant). Defaults to the prediction itself."""
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if mask is None:
        mask = np.ones(true.shape, dtype=bool)
    ref = pred if pred_reference is None else np.asarray(pred_reference, dtype=np.float64)
    count = np.sum(mask, axis=axis)
    safe_count = np.maximum(count, 1)
    true_mean = np.sum(np.where(mask, true, 0.0), axis=axis) / safe_count
    pred_mean = np.sum(np.where(mask, pred, 0.0), axis=axis) / safe_count
    tc = np.where(mask, true - np.expand_dims(true_mean, axis), 0.0)
    pc = np.where(mask, pred - np.expand_dims(pred_mean, axis), 0.0)
    t_ss = np.sum(tc * tc, axis=axis)
    p_ss = np.sum(pc * pc, axis=axis)
    num = np.sum(tc * pc, axis=axis)
    t_rms = np.sqrt(np.sum(np.where(mask, true * true, 0.0), axis=axis) / safe_count)
    r_rms = np.sqrt(np.sum(np.where(mask, ref * ref, 0.0), axis=axis) / safe_count)
    t_std = np.sqrt(t_ss / safe_count)
    p_std = np.sqrt(p_ss / safe_count)
    scored = (count >= 2) & (t_ss > 0) & (t_std > _TRUTH_REL_TOL * t_rms)
    if not np.any(scored):
        return 0.0
    pred_ok = scored & (p_ss > 0) & (p_std > _PRED_REL_TOL * r_rms)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = num / np.sqrt(t_ss * p_ss)
    r = np.where(pred_ok & np.isfinite(vals), np.clip(vals, -1.0, 1.0), 0.0)
    return float(np.mean(r[scored]))


def matrix_axis_mean_pearson(
    true: np.ndarray,
    pred: np.ndarray,
    axis: int,
    pred_reference: np.ndarray | None = None,
) -> float:
    """Mean Pearson correlation over rows (axis=1) or columns (axis=0); see _axis_pearson_truth_scored."""
    return _axis_pearson_truth_scored(true, pred, axis, None, pred_reference)


def masked_matrix_axis_mean_pearson(
    true: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    axis: int,
) -> float:
    """Mean Pearson over rows/columns, using only entries selected by a boolean mask (truth-determined set)."""
    return _axis_pearson_truth_scored(true, pred, axis, np.asarray(mask, dtype=bool), None)


def grade_masked_matrix_pearson_files_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
    output_dir: Path | str | None,
) -> float:
    submission, answers = validate_submission_frame(submission, answers, ["id", config["file_col"]])
    validate_test_alignment(read_test(public_dir, answers_path), answers)
    private_dir = require_private_dir(answers_path)
    output_dir = require_output_dir(output_dir)
    scores: list[float] = []
    for i in range(len(answers)):
        pred = np.asarray(load_npz_y(resolve_relative_path(
            output_dir, submission.at[i, config["file_col"]], "prediction")), dtype=np.float64)
        answer_file = resolve_relative_path(private_dir, answers.at[i, config["file_col"]], "answer")
        try:
            with np.load(answer_file, allow_pickle=False) as data:
                true = np.asarray(data["y"], dtype=np.float64)
                mask = np.asarray(data["mask"], dtype=bool)
        except Exception as exc:
            raise InvalidSubmissionError(f"Unable to read masked answer NPZ {answer_file}: {exc}")
        if pred.shape != true.shape or mask.shape != true.shape:
            raise InvalidSubmissionError(
                f"Masked matrix shape mismatch for row {i}: pred {pred.shape}, true {true.shape}, mask {mask.shape}."
            )
        if not np.isfinite(pred).all():
            raise InvalidSubmissionError(f"Prediction matrix for row {i} contains NaN or inf.")
        patient = masked_matrix_axis_mean_pearson(true, pred, mask, axis=1)
        protein = masked_matrix_axis_mean_pearson(true, pred, mask, axis=0)
        scores.append(0.5 * patient + 0.5 * protein)
    return mean_or_raise(scores, "masked matrix prediction cohorts")


def grade_matrix_pearson_files_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
    output_dir: Path | str | None,
) -> float:
    submission, answers = validate_submission_frame(
        submission, answers, ["id", config["file_col"]]
    )
    validate_test_alignment(read_test(public_dir, answers_path), answers)
    private_dir = require_private_dir(answers_path)
    output_dir = require_output_dir(output_dir)
    row_centre = bool(config.get("row_centre_columns", False))
    group_scores: list[float] = []
    for i in range(len(answers)):
        pred = np.asarray(
            load_npz_y(
                resolve_relative_path(
                    output_dir, submission.at[i, config["file_col"]], "prediction"
                )
            ),
            dtype=np.float64,
        )
        true = np.asarray(
            load_npz_y(
                resolve_relative_path(
                    private_dir, answers.at[i, config["file_col"]], "answer"
                )
            ),
            dtype=np.float64,
        )
        if pred.ndim != 2 or pred.shape != true.shape:
            raise InvalidSubmissionError(
                f"Matrix shape mismatch for row {i}: pred {pred.shape}, true {true.shape}."
            )
        if not np.isfinite(pred).all():
            raise InvalidSubmissionError(f"Prediction matrix for row {i} contains NaN or inf.")
        if row_centre:
            # v4: row-wise (per cell/spot, across features) Pearson on deviations from each feature's mean over the
            # rows of this group (truth and prediction centred separately). The uncentred prediction is the scale
            # reference, so a prediction that is constant across rows (up to float noise) counts as constant (r = 0).
            cell_score = matrix_axis_mean_pearson(
                true - true.mean(axis=0, keepdims=True), pred - pred.mean(axis=0, keepdims=True), axis=1, pred_reference=pred
            )
        else:
            cell_score = matrix_axis_mean_pearson(true, pred, axis=1)
        feature_score = matrix_axis_mean_pearson(true, pred, axis=0)
        group_scores.append(0.5 * cell_score + 0.5 * feature_score)
    return mean_or_raise(group_scores, "matrix prediction groups")


def grade_context_residual_matrix_pearson_files_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
    output_dir: Path | str | None,
) -> float:
    """Grade matrix predictions after removing a training-only context mean.

    This prevents a treatment-blind model from scoring well merely by learning
    generic response patterns associated with cell line, dose, or replicate.
    The baseline for every test context is computed only from public training
    targets, then subtracted from both the prediction and private truth.
    """
    submission, answers = validate_submission_frame(
        submission, answers, ["id", config["file_col"]]
    )
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    public_dir = require_public_dir(public_dir, answers_path)
    private_dir = require_private_dir(answers_path)
    output_dir = require_output_dir(output_dir)
    train = pd.read_csv(public_dir / "train.csv")
    context_cols = list(config["context_cols"])
    required_train = set(context_cols + [config["train_file_col"]])
    required_test = set(context_cols)
    if not required_train.issubset(train.columns) or not required_test.issubset(test.columns):
        raise InvalidSubmissionError(
            "Context-residual matrix task is missing configured train/test context columns."
        )

    sums: dict[tuple[Any, ...], np.ndarray] = {}
    counts: dict[tuple[Any, ...], int] = {}
    for _, row in train.iterrows():
        key = tuple(row[col] for col in context_cols)
        value = np.asarray(
            load_npz_y(
                resolve_relative_path(
                    public_dir, row[config["train_file_col"]], "training target"
                )
            ),
            dtype=np.float64,
        )
        if value.ndim != 2 or not np.isfinite(value).all():
            raise InvalidSubmissionError("Training context matrix is invalid.")
        sums[key] = value.copy() if key not in sums else sums[key] + value
        counts[key] = counts.get(key, 0) + 1
    baselines = {key: value / counts[key] for key, value in sums.items()}

    group_scores: list[float] = []
    for i in range(len(answers)):
        key = tuple(test.at[i, col] for col in context_cols)
        if key not in baselines:
            raise InvalidSubmissionError(f"No training baseline for test context {key!r}.")
        pred = np.asarray(
            load_npz_y(
                resolve_relative_path(
                    output_dir, submission.at[i, config["file_col"]], "prediction"
                )
            ),
            dtype=np.float64,
        )
        true = np.asarray(
            load_npz_y(
                resolve_relative_path(
                    private_dir, answers.at[i, config["file_col"]], "answer"
                )
            ),
            dtype=np.float64,
        )
        baseline = baselines[key]
        if pred.ndim != 2 or pred.shape != true.shape or baseline.shape != true.shape:
            raise InvalidSubmissionError(
                f"Context-residual matrix shape mismatch for row {i}: "
                f"pred {pred.shape}, true {true.shape}, baseline {baseline.shape}."
            )
        if not np.isfinite(pred).all():
            raise InvalidSubmissionError(f"Prediction matrix for row {i} contains NaN or inf.")
        pred_residual = pred - baseline
        true_residual = true - baseline
        # v3: the prediction's spread is judged against the raw prediction, so pred == baseline*(1 +- 1e-9)
        # counts as "no residual" (r = 0) exactly like pred == baseline.
        statistic_score = matrix_axis_mean_pearson(true_residual, pred_residual, axis=1, pred_reference=pred)
        gene_score = matrix_axis_mean_pearson(true_residual, pred_residual, axis=0, pred_reference=pred)
        group_scores.append(0.5 * statistic_score + 0.5 * gene_score)
    return mean_or_raise(group_scores, "context-residual matrix prediction groups")


def grade_denoise_type_dice_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
    output_dir: Path | str | None,
) -> float:
    submission, answers = validate_submission_frame(submission, answers)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    private_dir = require_private_dir(answers_path)
    output_dir = require_output_dir(output_dir)
    def score_row(i: int) -> float:
        pred = load_npz_y(resolve_relative_path(output_dir, submission.at[i, config["file_col"]], "prediction"))
        true = read_mrc_volume(resolve_relative_path(private_dir, answers.at[i, config["file_col"]], "answer"))
        if pred.shape != true.shape:
            raise InvalidSubmissionError(
                f"DenoiseCryo shape mismatch for row {i}: pred {pred.shape}, true {true.shape}."
            )
        pred = np.asarray(pred).astype(np.int16, copy=False)
        true = np.asarray(true).astype(np.int16, copy=False)
        # Source encoding uses 0 for background and 1..5 for the five atom
        # types.  Background is not an atom type and must not contribute an
        # easy near-perfect Dice term; include nitrogen (label 5), which the
        # previous range(5) implementation accidentally omitted.
        return mean_dice_labels_chunked(pred, true, tuple(range(1, 6)))

    # Rows are independent PDB volumes. NumPy decompression/counting releases
    # the GIL, so bounded row-level threads substantially reduce wall time
    # without changing any per-volume or macro-averaging semantics.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(8, len(answers))) as pool:
        row_scores = list(pool.map(score_row, range(len(answers))))
    return group_mean_from_row_scores(test, row_scores, config["group_col"])


def tolerant_atom_f1_per_label(
    pred: np.ndarray,
    true: np.ndarray,
    labels: tuple[int, ...],
    tolerance_voxels: int = 1,
) -> list[float]:
    """Per-label F1 between predicted and reference atom voxels with a positional tolerance.

    Every non-background reference voxel is one atom.  For each label the predicted voxels of
    that label are matched one-to-one to reference voxels of the same label that lie within
    `tolerance_voxels` along every axis (Chebyshev distance; 1 voxel = the 26-neighbourhood).
    The number of matches is the maximum bipartite matching, so the result does not depend on
    processing order and a dense blob of predictions around one atom earns one match only.
    F1 = 2 * matches / (n_pred + n_true); 1.0 when both sets are empty, 0.0 when one is empty.
    """
    from scipy import ndimage
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    from scipy.spatial import cKDTree

    t = int(tolerance_voxels)
    structure = np.ones((2 * t + 1,) * 3, dtype=bool)
    radius = float(np.sqrt(3.0) * t) + 1e-3
    scores: list[float] = []
    for label in labels:
        true_mask = true == label
        pred_mask = pred == label
        n_true = int(true_mask.sum())
        n_pred = int(pred_mask.sum())
        if n_true == 0 and n_pred == 0:
            scores.append(1.0)
            continue
        if n_true == 0 or n_pred == 0:
            scores.append(0.0)
            continue
        # Only predictions inside the dilated reference mask can be matched; this bounds the
        # graph size at (2t+1)^3 * n_true even for degenerate dense submissions.
        reachable = ndimage.binary_dilation(true_mask, structure=structure)
        pred_xyz = np.argwhere(pred_mask & reachable).astype(np.float32)
        matches = 0
        if len(pred_xyz):
            true_xyz = np.argwhere(true_mask).astype(np.float32)
            true_tree = cKDTree(true_xyz)
            rows, cols = [], []
            chunk = 400_000
            for start in range(0, len(pred_xyz), chunk):
                part = pred_xyz[start:start + chunk]
                pairs = cKDTree(part).sparse_distance_matrix(true_tree, radius, output_type="ndarray")
                if len(pairs):
                    rows.append(pairs["i"] + start)
                    cols.append(pairs["j"])
            if rows:
                i = np.concatenate(rows)
                j = np.concatenate(cols)
                graph = csr_matrix(
                    (np.ones(len(i), dtype=np.int8), (i, j)), shape=(len(pred_xyz), len(true_xyz))
                )
                matches = int((maximum_bipartite_matching(graph, perm_type="column") >= 0).sum())
        scores.append(2.0 * matches / (n_pred + n_true))
    return scores


def grade_denoise_atom_f1_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
    output_dir: Path | str | None,
) -> float:
    submission, answers = validate_submission_frame(submission, answers)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    private_dir = require_private_dir(answers_path)
    output_dir = require_output_dir(output_dir)
    labels = tuple(int(x) for x in config["labels"])
    max_label = max(labels)
    tol = int(config.get("tolerance_voxels", 1))
    file_col = config["file_col"]
    aux_cols = list(config.get("aux_file_cols", []))

    def score_row(i: int) -> float:
        for col in aux_cols:
            aux_path = resolve_relative_path(output_dir, submission.at[i, col], col)
            if not aux_path.exists():
                raise InvalidSubmissionError(f"Missing prediction file for row {i} ({col}): {aux_path}")
        pred = load_npz_y(resolve_relative_path(output_dir, submission.at[i, file_col], "prediction"))
        true = read_mrc_volume(resolve_relative_path(private_dir, answers.at[i, file_col], "answer"))
        if pred.ndim != 3 or pred.shape != true.shape:
            raise InvalidSubmissionError(
                f"DenoiseCryo shape mismatch for row {i}: pred {pred.shape}, true {true.shape}."
            )
        if pred.dtype.kind not in "iub":
            if pred.dtype.kind != "f":
                raise InvalidSubmissionError(f"DenoiseCryo prediction for row {i} must be an integer array.")
            if not np.isfinite(pred).all():
                raise InvalidSubmissionError(f"DenoiseCryo prediction for row {i} contains non-finite values.")
            if not np.array_equal(pred, np.round(pred)):
                raise InvalidSubmissionError(f"DenoiseCryo prediction for row {i} must hold integer labels.")
        pmin, pmax = int(pred.min()), int(pred.max())
        if pmin < 0 or pmax > max_label:
            raise InvalidSubmissionError(f"DenoiseCryo predictions must use labels 0..{max_label} (row {i}).")
        pred = np.asarray(pred).astype(np.int16, copy=False)
        true = np.asarray(true).astype(np.int16, copy=False)
        if int(true.min()) < 0 or int(true.max()) > max_label:
            raise InvalidSubmissionError(f"DenoiseCryo answers must use labels 0..{max_label} (row {i}).")
        return float(np.mean(tolerant_atom_f1_per_label(pred, true, labels, tol)))

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(4, len(answers))) as pool:
        row_scores = list(pool.map(score_row, range(len(answers))))
    return group_mean_from_row_scores(test, row_scores, config["group_col"])


def grade_decoy_topk_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
) -> float:
    submission, answers = prepare_standard_task(submission, answers, config)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    pred = numeric_array(submission, config["pred_col"], "submission")
    true = numeric_array(answers, config["target_col"], "answers")
    df = pd.DataFrame({config["group_col"]: test[config["group_col"]], "pred": pred, "true": true})
    top1_scores: list[float] = []
    top5_scores: list[float] = []
    for _, group in df.groupby(config["group_col"], sort=True):
        # v2 (2026-10-13): ties are broken pessimistically (tied near-native poses rank
        # last), so constant or heavily tied submissions cannot collect top-k credit.
        ranked = group.sort_values(["pred", "true"], ascending=[False, True], kind="mergesort")
        top1_scores.append(float((ranked.head(1)["true"] > 0).any()))
        top5_scores.append(float((ranked.head(5)["true"] > 0).any()))
    # The description names top-k and explicitly checks both top-1 and top-5.
    return float(0.5 * np.mean(top1_scores) + 0.5 * np.mean(top5_scores))


def grade_nested_grouped_pearson_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
) -> float:
    submission, answers = prepare_standard_task(submission, answers, config)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    y_true = numeric_array(answers, config["target_col"], "answers")
    y_pred = numeric_array(submission, config["pred_col"], "submission")
    df = test[[config["outer_group_col"], config["inner_group_col"]]].copy()
    df["true"] = y_true
    df["pred"] = y_pred
    outer_scores: list[float] = []
    for _, outer_group in df.groupby(config["outer_group_col"], sort=True):
        inner_scores: list[float] = []
        for _, inner_group in outer_group.groupby(config["inner_group_col"], sort=True):
            if len(inner_group) < 2:
                continue
            inner_scores.append(
                pearson_safe(inner_group["true"].to_numpy(), inner_group["pred"].to_numpy())
            )
        outer_scores.append(mean_or_raise(inner_scores, "nested target Pearson"))
    return mean_or_raise(outer_scores, "nested source Pearson")



# ---- tier-4 fixes 2026-10-13 (causalbench, perturbench, xatlas) -------------------------------------------
def grade_masked_grouped_roc_auc_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
) -> float:
    """Grouped ROC-AUC over the rows flagged by a private answers column (e.g. strength-matched pairs).

    The submission must cover every test row (id, pred_col only); the mask lives in answers.csv only."""
    mask_col = config["mask_col"]
    expected = ["id", config["pred_col"]]
    if list(submission.columns) != expected:
        raise InvalidSubmissionError(f"Submission columns must be exactly {expected}; got {list(submission.columns)}.")
    require_columns(answers, ["id", config["target_col"], mask_col], "answers")
    submission, _ = validate_submission_frame(submission, answers[["id", config["target_col"]]].rename(
        columns={config["target_col"]: config["pred_col"]}), expected)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    y_true = numeric_array(answers, config["target_col"], "answers")
    y_pred = numeric_array(submission, config["pred_col"], "submission")
    mask = numeric_array(answers, mask_col, "answers") > 0
    test = test.loc[mask].reset_index(drop=True)
    return grouped_metric(test, y_true[mask], y_pred[mask], config["group_col"], roc_auc_safe,
                          min_rows=int(config.get("min_rows", 2)))


PROJECTED_RESIDUAL_MAX_ABS = 1e3


def strict_group_mean_from_row_scores(test: pd.DataFrame, row_scores: list[float], group_col: str) -> float:
    """Equal-weight mean of per-group mean row scores; every row and every group counts (no NaN skipping)."""
    require_columns(test, [group_col], "test.csv")
    scores = np.asarray(row_scores, dtype=np.float64)
    if len(scores) != len(test):
        raise RuntimeError("Row-score count does not match test rows.")
    scores = np.where(np.isfinite(scores), scores, 0.0)
    df = pd.DataFrame({group_col: test[group_col].to_numpy(), "score": scores})
    means = df.groupby(group_col, sort=True)["score"].mean()
    if len(means) == 0:
        raise InvalidSubmissionError("No rows to score.")
    if not np.isfinite(means.to_numpy()).all():
        raise RuntimeError(f"Non-finite per-{group_col} mean.")
    return float(means.mean())


def _project_out(vec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return vec - basis @ (basis.T @ vec)


def grade_projected_residual_pearson_files_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
    output_dir: Path | str | None,
) -> float:
    """Pearson of the response that no public reference vector explains.

    residual = y - context_baseline[context] - reference[id] (reference optional, public, train-derived);
    on the scoring genes (minus private per-row exclusions) both residuals are projected onto the orthogonal
    complement of span{1, baseline, reference}. Any prediction inside that span scores exactly 0."""
    submission, answers = validate_submission_frame(submission, answers)
    public_dir = require_public_dir(public_dir, answers_path)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    require_columns(test, [config["group_col"], config["context_col"]], "test.csv")
    private_dir = require_private_dir(answers_path)
    output_dir = require_output_dir(output_dir)
    baseline_path = public_dir / "context_baseline.npz"
    score_index_path = public_dir / "score_gene_indices.npz"
    ref_path = public_dir / config["reference_file"] if config.get("reference_file") else None
    excl_path = private_dir / config["exclusion_file"] if config.get("exclusion_file") else None
    for path in [baseline_path, score_index_path, ref_path, excl_path]:
        if path is not None and not path.exists():
            raise InvalidSubmissionError(f"Missing grading archive: {path}")
    refs = np.load(ref_path, allow_pickle=False) if ref_path is not None else None
    excl = np.load(excl_path, allow_pickle=False) if excl_path is not None else None
    row_scores: list[float] = []
    with np.load(baseline_path, allow_pickle=False) as baselines, np.load(score_index_path, allow_pickle=False) as score_indices:
        for i in range(len(answers)):
            rid = str(test.at[i, "id"])
            source = str(test.at[i, config["group_col"]])
            context_id = str(test.at[i, config["context_col"]])
            index_key = f"{source}_indices"
            if context_id not in baselines or index_key not in score_indices:
                raise InvalidSubmissionError(f"Missing residual metadata for source={source!r}, context={context_id!r}.")
            baseline = np.asarray(baselines[context_id], dtype=np.float64)
            indices = np.asarray(score_indices[index_key], dtype=np.int64)
            if excl is not None and rid in excl:
                indices = np.setdiff1d(indices, np.asarray(excl[rid], dtype=np.int64))
            ref = np.asarray(refs[rid], dtype=np.float64) if refs is not None else np.zeros_like(baseline)
            pred = np.asarray(load_npz_y(resolve_relative_path(output_dir, submission.at[i, config["file_col"]], "prediction")), dtype=np.float64)
            true = np.asarray(load_npz_y(resolve_relative_path(private_dir, answers.at[i, config["file_col"]], "answer")), dtype=np.float64)
            if pred.shape != true.shape or pred.shape != baseline.shape or ref.shape != baseline.shape:
                raise InvalidSubmissionError(
                    f"Response vector shape mismatch for row {i}: pred {pred.shape}, true {true.shape}, baseline {baseline.shape}.")
            if not np.isfinite(pred).all():
                raise InvalidSubmissionError(f"Prediction for row {i} contains non-finite values.")
            # v3 (2026-10-15): huge finite values (~1e308) overflowed the projection to NaN and the row was
            # silently skipped; predictions must be on the scale of train_y.
            max_abs = float(np.max(np.abs(pred))) if pred.size else 0.0
            if max_abs > PROJECTED_RESIDUAL_MAX_ABS:
                raise InvalidSubmissionError(
                    f"Prediction for row {i} has |y| = {max_abs:.3g} > {PROJECTED_RESIDUAL_MAX_ABS:g}; "
                    "predictions must be on the scale of train_y.")
            if len(indices) < 3:
                raise InvalidSubmissionError(f"Too few scoring genes for row {i}.")
            dirs = [np.ones(len(indices)), baseline[indices]]
            if refs is not None:
                dirs.append(ref[indices])
            Q, Rm = np.linalg.qr(np.stack(dirs, axis=1))
            Q = Q[:, np.abs(np.diag(Rm)) > 1e-10 * max(1.0, np.abs(Rm).max())]
            a = _project_out(true[indices] - baseline[indices] - ref[indices], Q)
            b_raw = pred[indices] - baseline[indices] - ref[indices]
            b = _project_out(b_raw, Q)
            # float32 rounding of a prediction inside span{1, baseline, reference} leaves ~1e-7 relative noise;
            # treat anything below 1e-4 of the prediction's own scale as "inside the span" (scores 0).
            scale = np.linalg.norm(pred[indices]) + np.linalg.norm(baseline[indices]) + np.linalg.norm(ref[indices])
            if np.linalg.norm(b) <= 1e-4 * scale + 1e-12 or np.linalg.norm(a) == 0:
                row_scores.append(0.0)
            else:
                with np.errstate(all="ignore"):
                    value = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                # v3: a non-finite row score counts 0 (never skipped)
                row_scores.append(float(np.clip(value, -1.0, 1.0)) if np.isfinite(value) else 0.0)
    return strict_group_mean_from_row_scores(test, row_scores, config["group_col"])


def grade_grouped_binned_integer_macro_f1_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
) -> float:
    """Per-group macro-F1 after clipping integer counts into [lo, hi] per group (hi = 'hi or more')."""
    submission, answers = prepare_standard_task(submission, answers, config)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    require_columns(test, [config["group_col"]], "test.csv")
    y_true = numeric_array(answers, config["target_col"], "answers")
    y_pred = numeric_array(submission, config["pred_col"], "submission")
    rounded = np.rint(y_pred)
    if not np.allclose(y_pred, rounded, atol=1e-8, rtol=0.0):
        raise InvalidSubmissionError("Count predictions must be integers.")
    if (rounded < 0).any():
        raise InvalidSubmissionError("Count predictions must be non-negative.")
    y_true = np.rint(y_true).astype(np.int64)
    y_pred = rounded.astype(np.int64)
    bins = config["count_bins"]
    scores: list[float] = []
    for group, positions in test.groupby(config["group_col"], sort=True).indices.items():
        if str(group) not in bins:
            raise InvalidSubmissionError(f"No count bins configured for group {group!r}.")
        lo, hi = bins[str(group)]
        idx = np.asarray(list(positions), dtype=int)
        t = np.clip(y_true[idx], lo, hi)
        p = np.clip(y_pred[idx], lo, hi)
        scores.append(float(f1_score(t, p, labels=list(range(lo, hi + 1)), average="macro", zero_division=0)))
    return mean_or_raise(scores, f"binned macro-F1 by {config['group_col']}")


def grade_reference_projected_grouped_pearson_task(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    config: dict[str, Any],
    public_dir: Path | str | None,
    answers_path: Path | str | None,
) -> float:
    """Per-group Pearson after projecting truth and prediction off span{indicator dummies, reference}.

    The reference is a public, train-derived prediction (column `reference_col` of test.csv); indicator dummies
    (e.g. cell line) remove per-context offsets. A prediction inside that span scores exactly 0."""
    submission, answers = prepare_standard_task(submission, answers, config)
    test = validate_test_alignment(read_test(public_dir, answers_path), answers)
    group_col, ref_col, ind_col = config["group_col"], config["reference_col"], config.get("indicator_col")
    require_columns(test, [group_col, ref_col] + ([ind_col] if ind_col else []), "test.csv")
    y_true = numeric_array(answers, config["target_col"], "answers")
    y_pred = numeric_array(submission, config["pred_col"], "submission")
    ref = numeric_array(test, ref_col, "test.csv")
    scores: list[float] = []
    for _, positions in test.groupby(group_col, sort=True).indices.items():
        idx = np.asarray(list(positions), dtype=int)
        if len(idx) < int(config.get("min_rows", 2)):
            continue
        if ind_col:
            codes = pd.factorize(test[ind_col].iloc[idx].astype(str))[0]
            dummies = np.eye(codes.max() + 1)[codes]
        else:
            dummies = np.ones((len(idx), 1))
        Q, Rm = np.linalg.qr(np.column_stack([dummies, ref[idx]]))
        Q = Q[:, np.abs(np.diag(Rm)) > 1e-10 * max(1.0, np.abs(Rm).max())]
        a = _project_out(y_true[idx].astype(float), Q)
        b = _project_out(y_pred[idx].astype(float), Q)
        scale = np.linalg.norm(y_pred[idx]) + np.linalg.norm(ref[idx])
        if np.linalg.norm(b) <= 1e-6 * scale + 1e-12 or np.linalg.norm(a) == 0:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    return mean_or_raise(scores, f"reference-projected grouped Pearson by {group_col}")


def grade_task(
    task_key: str,
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    *,
    public_dir: Path | str | None = None,
    answers_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    **_: Any,
) -> float:
    if task_key not in TASK_CONFIG:
        raise InvalidSubmissionError(f"Unknown BioXArena-Pro task: {task_key}")
    config = TASK_CONFIG[task_key]
    kind = config["kind"]

    if kind == "pearson":
        return grade_pearson_task(submission, answers, config)
    if kind == "calibrated_regression":
        return grade_calibrated_regression_task(submission, answers, config)
    if kind == "roc_auc":
        return grade_roc_auc_task(submission, answers, config)
    if kind == "grouped_pearson":
        return grade_grouped_numeric_task(
            submission, answers, config, public_dir, answers_path, pearson_safe
        )
    if kind == "grouped_calibrated_regression":
        return grade_grouped_numeric_task(
            submission, answers, config, public_dir, answers_path,
            calibrated_regression_safe,
        )
    if kind == "grouped_integer_accuracy":
        return grade_grouped_numeric_task(
            submission, answers, config, public_dir, answers_path,
            integer_accuracy_safe,
        )
    if kind == "grouped_binned_integer_macro_f1":
        return grade_grouped_binned_integer_macro_f1_task(
            submission, answers, config, public_dir, answers_path
        )
    if kind == "grouped_integer_macro_f1":
        return grade_grouped_numeric_task(
            submission, answers, config, public_dir, answers_path,
            integer_macro_f1_safe,
        )
    if kind == "grouped_spearman":
        return grade_grouped_numeric_task(
            submission, answers, config, public_dir, answers_path, spearman_safe
        )
    if kind == "masked_grouped_roc_auc":
        return grade_masked_grouped_roc_auc_task(submission, answers, config, public_dir, answers_path)
    if kind == "reference_projected_grouped_pearson":
        return grade_reference_projected_grouped_pearson_task(submission, answers, config, public_dir, answers_path)
    if kind == "projected_residual_pearson_files":
        return grade_projected_residual_pearson_files_task(
            submission, answers, config, public_dir, answers_path, output_dir
        )
    if kind == "grouped_roc_auc":
        return grade_grouped_numeric_task(
            submission, answers, config, public_dir, answers_path, roc_auc_safe
        )
    if kind == "grouped_sensitivity_at_specificity":
        return grade_grouped_numeric_task(
            submission, answers, config, public_dir, answers_path,
            sensitivity_at_specificity_safe,
        )
    if kind == "grouped_average_precision":
        return grade_grouped_numeric_task(
            submission, answers, config, public_dir, answers_path, average_precision_safe
        )
    if kind == "macro_f1":
        return grade_macro_f1_task(submission, answers, config)
    if kind == "accuracy":
        return grade_accuracy_task(submission, answers, config)
    if kind == "grouped_macro_f1":
        return grade_grouped_macro_f1_task(
            submission, answers, config, public_dir, answers_path
        )
    if kind == "nested_grouped_macro_f1":
        return grade_nested_grouped_macro_f1_task(
            submission, answers, config, public_dir, answers_path
        )
    if kind == "multilabel_prevalence_corrected":
        return grade_multilabel_prevalence_corrected_task(submission, answers, config)
    if kind == "multilabel_combined":
        return grade_multilabel_combined_task(submission, answers, config)
    if kind == "multioutput_macro_f1":
        return grade_multioutput_macro_f1_task(submission, answers, config)
    if kind == "multioutput_r2":
        return grade_multioutput_r2_task(submission, answers, config)
    if kind == "multioutput_pearson":
        return grade_multioutput_pearson_task(submission, answers, config)
    if kind == "multioutput_calibrated_regression":
        return grade_multioutput_calibrated_regression_task(submission, answers, config)
    if kind == "multioutput_integer_accuracy":
        return grade_multioutput_integer_accuracy_task(submission, answers, config)
    if kind == "grouped_c_index":
        return grade_grouped_c_index_task(submission, answers, config, public_dir, answers_path)
    if kind == "cellsam_dice":
        return grade_cellsam_task(submission, answers, config, public_dir, answers_path, output_dir)
    if kind == "delta_pearson_files":
        return grade_delta_pearson_files_task(
            submission, answers, config, public_dir, answers_path, output_dir
        )
    if kind == "context_residual_pearson_files":
        return grade_context_residual_pearson_files_task(
            submission, answers, config, public_dir, answers_path, output_dir
        )
    if kind == "grouped_context_residual_pearson_files":
        return grade_grouped_context_residual_pearson_files_task(
            submission, answers, config, public_dir, answers_path, output_dir
        )
    if kind == "response_pearson_files":
        return grade_response_pearson_files_task(
            submission, answers, config, public_dir, answers_path, output_dir
        )
    if kind == "matrix_pearson_files":
        return grade_matrix_pearson_files_task(
            submission, answers, config, public_dir, answers_path, output_dir
        )
    if kind == "context_residual_matrix_pearson_files":
        return grade_context_residual_matrix_pearson_files_task(
            submission, answers, config, public_dir, answers_path, output_dir
        )
    if kind == "masked_matrix_pearson_files":
        return grade_masked_matrix_pearson_files_task(
            submission, answers, config, public_dir, answers_path, output_dir
        )
    if kind == "denoise_type_dice":
        return grade_denoise_type_dice_task(
            submission, answers, config, public_dir, answers_path, output_dir
        )
    if kind == "denoise_atom_f1":
        return grade_denoise_atom_f1_task(
            submission, answers, config, public_dir, answers_path, output_dir
        )
    if kind == "decoy_topk":
        return grade_decoy_topk_task(submission, answers, config, public_dir, answers_path)
    if kind == "nested_grouped_pearson":
        return grade_nested_grouped_pearson_task(
            submission, answers, config, public_dir, answers_path
        )
    raise InvalidSubmissionError(f"Unsupported grading kind for {task_key}: {kind}")
