#!/usr/bin/env python3
"""Evaluate BioXArena-Pro submissions produced by LLM training runs.

Examples:
    python evaluation/evaluate_llm_agents.py --model openai/gpt-5.4 --task chemical-biology/solubility
    python evaluation/evaluate_llm_agents.py --model openai/gpt-5.4 --domain sequence-genomics --max-workers 4
    python evaluation/evaluate_llm_agents.py --model openai/gpt-5.4 --all-tasks --max-workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()  # /<work_root>/2_BioXArena-Pro-Code/evaluation/evaluate_llm_agents.py
EVAL_DIR = SCRIPT_PATH.parent  # /<work_root>/2_BioXArena-Pro-Code/evaluation
EVAL_ROOT = EVAL_DIR.parent  # /<work_root>/2_BioXArena-Pro-Code
WORKSPACE_ROOT = EVAL_ROOT.parent # /<work_root>
DEFAULT_PREFIX_ROOT = WORKSPACE_ROOT
DEFAULT_TASKS_ROOT = EVAL_ROOT / "tasks"  # /<work_root>/2_BioXArena-Pro-Code/tasks
DEFAULT_PUBLIC_ROOT_NAME = "2_BioXArena-Pro-Data-Public"
DEFAULT_PRIVATE_ROOT_NAME = "2_BioXArena-Pro-Data-Private"
DEFAULT_OUTPUT_ROOT_NAME = "2_BioXArena-Pro-Output"
DEFAULT_ROUND_NAME = "round1"

if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from benchmark_layout import canonicalize_task_key, require_canonical_layout  # noqa: E402

SUBMISSION_FILE_MANIFEST_PATH_COLUMNS: dict[str, tuple[str, ...]] = {
    "chemical-biology/kano-molecular-property-prediction": ("prediction_file",),
    "chemical-biology/uspto-reaction-product": ("prediction_file",),
    "imaging/calcium-neuron-segmentation": ("prediction_file",),
    "imaging/microscopy-instance-segmentation": ("prediction_file",),
    "network-biology/ppi-cross-species": ("prediction_file",),
    "network-biology/swissprot-protein-function": ("prediction_file",),
    "perturbation/pdtc-nb-clinical-drug-response": ("prediction_file",),
    "perturbation/perturbench-response": ("response_file",),
    "perturbation/sciplex-distribution": ("prediction_file",),
    "perturbation/xatlas-orion-perturb": ("response_file",),
    "phenotype-disease/clinvar-disease-variant-effect": ("prediction_file",),
    "phenotype-disease/covid-hospitalization-forecast": ("prediction_file",),
    "phenotype-disease/cptac-pan-protein-imputation": ("prediction_file",),
    "sequence-genomics/de-novo-peptide-sequencing-from-tandem-mass-spectrometry": ("prediction_file",),
    "sequence-genomics/deepstarr-genomic-sequence": ("prediction_file",),
    "sequence-genomics/yeast-promoter-expression": ("prediction_file",),
    "single-cell-spatial/openproblems-rna-atac": ("prediction_file",),
    "single-cell-spatial/openproblems-rna-protein": ("prediction_file",),
    "single-cell-spatial/hest-coad-spatial-expression": ("prediction_file",),
    "structure/cath-inverse-folding": ("prediction_file",),
    "structure/denoise-cryo": ("type_label_file", "residue_label_file"),
}

NB_FILE_MANIFEST_TASKS = {
    "chemical-biology/kano-molecular-property-prediction",
    "chemical-biology/uspto-reaction-product",
    "imaging/calcium-neuron-segmentation",
    "imaging/microscopy-instance-segmentation",
    "network-biology/ppi-cross-species",
    "network-biology/swissprot-protein-function",
    "perturbation/pdtc-nb-clinical-drug-response",
    "phenotype-disease/clinvar-disease-variant-effect",
    "phenotype-disease/covid-hospitalization-forecast",
    "sequence-genomics/de-novo-peptide-sequencing-from-tandem-mass-spectrometry",
    "sequence-genomics/deepstarr-genomic-sequence",
    "sequence-genomics/yeast-promoter-expression",
    "structure/cath-inverse-folding",
}

ANSWER_MANIFEST_PATH_COLUMNS = dict(SUBMISSION_FILE_MANIFEST_PATH_COLUMNS)
for _task_key in NB_FILE_MANIFEST_TASKS:
    ANSWER_MANIFEST_PATH_COLUMNS[_task_key] = ("answer_path",)

@dataclass(frozen=True)
class TaskSpec:
    domain: str
    task: str
    grade_path: Path
    answers_path: Path
    public_dir: Path
    output_dir: Path

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.task}"


@dataclass(frozen=True)
class MetricInfo:
    name: str
    direction: str
    normalization: str


@dataclass
class TaskEvaluationResult:
    task_key: str
    status: str
    score: float | None
    normalized_score: float
    metric: dict[str, str]
    remark: str
    results_path: str
    submission_path: str
    answers_path: str
    failed_marker_present: bool


def sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "__", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "unknown"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_grade_module(grade_path: Path, task_key: str) -> Any:
    module_name = "grade_" + sanitize_path_component(task_key)
    spec = importlib.util.spec_from_file_location(module_name, grade_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load grade module from {grade_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grade_fn = getattr(module, "grade", None)
    if not callable(grade_fn):
        raise RuntimeError(f"`grade` function not found in {grade_path}")
    return module


def discover_tasks(
    tasks_root: Path,
    private_root: Path,
    output_root: Path,
    public_root: Path,
) -> list[TaskSpec]:
    task_specs: list[TaskSpec] = []
    for grade_path in sorted(tasks_root.glob("*/*/grade.py")):
        rel = grade_path.relative_to(tasks_root)
        domain, task = rel.parts[0], rel.parts[1]
        answers_path = private_root / domain / task / "private" / "answers.csv"
        public_dir = public_root / domain / task / "public"
        output_dir = output_root / domain / task
        task_specs.append(
            TaskSpec(
                domain=domain,
                task=task,
                grade_path=grade_path,
                answers_path=answers_path,
                public_dir=public_dir,
                output_dir=output_dir,
            )
        )
    require_canonical_layout((spec.key for spec in task_specs), "code task layout")
    return task_specs


def resolve_task_selector(selector: str, tasks_by_key: dict[str, TaskSpec]) -> TaskSpec:
    selector = canonicalize_task_key(selector.strip())
    if selector in tasks_by_key:
        return tasks_by_key[selector]

    matches = [spec for spec in tasks_by_key.values() if spec.task == selector]
    if not matches:
        raise ValueError(f"Unknown task selector: {selector}")
    if len(matches) > 1:
        keys = ", ".join(spec.key for spec in matches)
        raise ValueError(f"Ambiguous task selector '{selector}'. Use one of: {keys}")
    return matches[0]


def select_tasks(
    all_tasks: list[TaskSpec],
    selected_domains: list[str],
    selected_tasks: list[str],
    run_all_tasks: bool,
) -> list[TaskSpec]:
    tasks_by_key = {task.key: task for task in all_tasks}
    selected: dict[str, TaskSpec] = {}

    for domain in selected_domains:
        domain_matches = [task for task in all_tasks if task.domain == domain]
        if not domain_matches:
            raise ValueError(f"Unknown domain: {domain}")
        for task in domain_matches:
            selected[task.key] = task

    for selector in selected_tasks:
        task = resolve_task_selector(selector, tasks_by_key)
        selected[task.key] = task

    if run_all_tasks or (not selected and not selected_domains and not selected_tasks):
        for task in all_tasks:
            selected[task.key] = task

    ordered_keys = sorted(selected)
    return [selected[key] for key in ordered_keys]


def infer_metric_info(grade_module: Any) -> MetricInfo:
    module_metric = getattr(grade_module, "METRIC_INFO", None)
    if isinstance(module_metric, dict):
        return MetricInfo(
            name=str(module_metric.get("name", "score")),
            direction=str(module_metric.get("direction", "higher_is_better")),
            normalization=str(module_metric.get("normalization", "auto_by_score_range")),
        )

    try:
        grade_source = inspect.getsource(grade_module.grade)
    except (OSError, TypeError):
        grade_source = ""
    source = grade_source.lower()

    if "grade_macro_roc_auc" in source:
        return MetricInfo(
            name="macro_roc_auc",
            direction="higher_is_better",
            normalization="already_in_0_1",
        )
    if "grade_roc_auc" in source:
        return MetricInfo(
            name="roc_auc",
            direction="higher_is_better",
            normalization="already_in_0_1",
        )
    if "grade_macro_f1" in source:
        return MetricInfo(
            name="macro_f1",
            direction="higher_is_better",
            normalization="already_in_0_1",
        )
    if "grade_accuracy" in source or "preds == labels" in source:
        return MetricInfo(
            name="accuracy",
            direction="higher_is_better",
            normalization="already_in_0_1",
        )
    if "grade_auprc" in source or "average_precision_score" in source:
        return MetricInfo(
            name="auprc",
            direction="higher_is_better",
            normalization="already_in_0_1",
        )
    if "grade_c_index" in source:
        return MetricInfo(
            name="c_index",
            direction="higher_is_better",
            normalization="already_in_0_1",
        )
    if (
        "grade_pearson" in source
        or "pearsonr(" in source
        or "np.corrcoef(" in source
        or "corrcoef(" in source
        or "pearson correlation" in source
    ):
        return MetricInfo(
            name="pearson",
            direction="higher_is_better",
            normalization="map_minus1_1_to_0_1",
        )
    if (
        "grade_spearman" in source
        or "spearmanr(" in source
        or "spearman rank correlation" in source
    ):
        return MetricInfo(
            name="spearman",
            direction="higher_is_better",
            normalization="map_minus1_1_to_0_1",
        )
    return MetricInfo(
        name="score",
        direction="higher_is_better",
        normalization="auto_by_score_range",
    )


def default_metric_info() -> MetricInfo:
    return MetricInfo(
        name="score",
        direction="higher_is_better",
        normalization="auto_by_score_range",
    )


def infer_task_metric_info(task_spec: TaskSpec) -> MetricInfo:
    try:
        grade_module = load_grade_module(task_spec.grade_path, task_spec.key)
    except Exception:
        return default_metric_info()
    return infer_metric_info(grade_module)


def normalize_score(score: float, metric_info: MetricInfo) -> float:
    # v4 (2026-10-16) chance-zero scale: correlation/MCC-type scores are reported as
    # clip(r, 0, 1) instead of (r + 1) / 2, so chance and anti-correlation score 0.
    # The name "map_minus1_1_to_0_1" is kept because callers/results depend on it.
    if metric_info.normalization == "map_minus1_1_to_0_1":
        return max(0.0, min(1.0, score))
    if metric_info.normalization == "already_in_0_1":
        return max(0.0, min(1.0, score))

    # v4: negative scores are clipped at 0 (previously mapped by (s + 1) / 2).
    return max(0.0, min(1.0, score))


def build_results_payload(
    *,
    status: str,
    metric_info: MetricInfo,
    score: float | None,
    normalized_score: float,
    remark: str,
) -> dict[str, Any]:
    return {
        "Status": status,
        "Metric": {
            "name": metric_info.name,
            "direction": metric_info.direction,
            "normalization": metric_info.normalization,
        },
        "Score": score,
        "Normalized_Score": normalized_score,
        "Remark": remark,
    }


def resolve_manifest_path(value: Any, root: Path, label: str) -> Any:
    if pd.isna(value):
        return value

    candidate = Path(str(value))
    if label == "Submission" and candidate.is_absolute():
        raise ValueError(
            f"Submission file paths must be relative to the task output directory: {value!r}"
        )
    if not candidate.is_absolute() and any(part == ".." for part in candidate.parts):
        raise ValueError(f"{label} file path may not contain '..': {value!r}")

    resolved_root = root.resolve()
    resolved_candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"{label} file path must stay inside its task directory: {value!r}"
        ) from exc
    return str(resolved_candidate)


def resolve_file_manifest_columns(
    task_spec: TaskSpec,
    frame: pd.DataFrame,
    *,
    root: Path,
    label: str,
) -> pd.DataFrame:
    # Expand file-manifest paths before handing dataframes to task graders.
    path_columns_by_task = (
        SUBMISSION_FILE_MANIFEST_PATH_COLUMNS
        if label == "Submission"
        else ANSWER_MANIFEST_PATH_COLUMNS
    )
    path_columns = path_columns_by_task.get(task_spec.key)
    if not path_columns:
        return frame

    resolved_df = frame.copy()
    for column in path_columns:
        if column not in resolved_df.columns:
            raise ValueError(
                f"{label} for {task_spec.key} is missing required path column: {column}"
            )
        resolved_df[column] = resolved_df[column].map(
            lambda value: resolve_manifest_path(value, root, label)
        )
    return resolved_df


def call_grade(
    grade_module: Any,
    task_spec: TaskSpec,
    submission_df: pd.DataFrame,
    answers_df: pd.DataFrame,
    submission_path: Path,
) -> Any:
    grade_fn = grade_module.grade
    grade_context = {
        "task_key": task_spec.key,
        "grade_path": task_spec.grade_path,
        "answers_path": task_spec.answers_path,
        "public_dir": task_spec.public_dir,
        "output_dir": task_spec.output_dir,
        "submission_path": submission_path,
    }
    signature = inspect.signature(grade_fn)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return grade_fn(submission_df, answers_df, **grade_context)

    supported_kwargs = {
        name: value for name, value in grade_context.items() if name in signature.parameters
    }
    return grade_fn(submission_df, answers_df, **supported_kwargs)


def evaluate_single_task(task_spec: TaskSpec, print_lock: threading.Lock) -> TaskEvaluationResult:
    task_spec.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = task_spec.output_dir / "submission.csv"
    failed_marker_path = task_spec.output_dir / "FAILED.json"
    results_path = task_spec.output_dir / "results.json"

    metric_info = infer_task_metric_info(task_spec)

    if failed_marker_path.exists():
        remark = "Training marked this task as failed because FAILED.json is present."
        payload = build_results_payload(
            status="Fail",
            metric_info=metric_info,
            score=None,
            normalized_score=0.0,
            remark=remark,
        )
        write_json(results_path, payload)
        with print_lock:
            print(f"[FAIL] {task_spec.key}: {remark}")
        return TaskEvaluationResult(
            task_key=task_spec.key,
            status="Fail",
            score=None,
            normalized_score=0.0,
            metric=payload["Metric"],
            remark=remark,
            results_path=str(results_path),
            submission_path=str(submission_path),
            answers_path=str(task_spec.answers_path),
            failed_marker_present=True,
        )

    if not submission_path.exists():
        remark = "Missing submission.csv in the task output directory."
        payload = build_results_payload(
            status="Fail",
            metric_info=metric_info,
            score=None,
            normalized_score=0.0,
            remark=remark,
        )
        write_json(results_path, payload)
        with print_lock:
            print(f"[FAIL] {task_spec.key}: {remark}")
        return TaskEvaluationResult(
            task_key=task_spec.key,
            status="Fail",
            score=None,
            normalized_score=0.0,
            metric=payload["Metric"],
            remark=remark,
            results_path=str(results_path),
            submission_path=str(submission_path),
            answers_path=str(task_spec.answers_path),
            failed_marker_present=False,
        )

    if not task_spec.answers_path.exists():
        remark = f"Missing answers.csv: {task_spec.answers_path}"
        payload = build_results_payload(
            status="Fail",
            metric_info=metric_info,
            score=None,
            normalized_score=0.0,
            remark=remark,
        )
        write_json(results_path, payload)
        with print_lock:
            print(f"[FAIL] {task_spec.key}: {remark}")
        return TaskEvaluationResult(
            task_key=task_spec.key,
            status="Fail",
            score=None,
            normalized_score=0.0,
            metric=payload["Metric"],
            remark=remark,
            results_path=str(results_path),
            submission_path=str(submission_path),
            answers_path=str(task_spec.answers_path),
            failed_marker_present=False,
        )

    try:
        grade_module = load_grade_module(task_spec.grade_path, task_spec.key)
        metric_info = infer_metric_info(grade_module)
        submission_df = pd.read_csv(submission_path)
        submission_df = resolve_file_manifest_columns(
            task_spec,
            submission_df,
            root=task_spec.output_dir,
            label="Submission",
        )
        answers_df = pd.read_csv(task_spec.answers_path)
        answers_df = resolve_file_manifest_columns(
            task_spec,
            answers_df,
            root=task_spec.answers_path.parent,
            label="Answer",
        )
        raw_score = call_grade(
            grade_module,
            task_spec,
            submission_df,
            answers_df,
            submission_path,
        )
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"Grade returned a non-finite value: {raw_score!r}")
    except Exception as exc:
        remark = f"Evaluation failed: {exc}"
        payload = build_results_payload(
            status="Fail",
            metric_info=metric_info,
            score=None,
            normalized_score=0.0,
            remark=remark,
        )
        write_json(results_path, payload)
        with print_lock:
            print(f"[FAIL] {task_spec.key}: {remark}")
        return TaskEvaluationResult(
            task_key=task_spec.key,
            status="Fail",
            score=None,
            normalized_score=0.0,
            metric=payload["Metric"],
            remark=remark,
            results_path=str(results_path),
            submission_path=str(submission_path),
            answers_path=str(task_spec.answers_path),
            failed_marker_present=False,
        )

    normalized_score = normalize_score(score, metric_info)
    remark = "Evaluation succeeded."
    payload = build_results_payload(
        status="Succeed",
        metric_info=metric_info,
        score=score,
        normalized_score=normalized_score,
        remark=remark,
    )
    write_json(results_path, payload)
    with print_lock:
        print(
            f"[SUCCEED] {task_spec.key}: "
            f"{metric_info.name}={score:.6f}, normalized={normalized_score:.6f}"
        )
    return TaskEvaluationResult(
        task_key=task_spec.key,
        status="Succeed",
        score=score,
        normalized_score=normalized_score,
        metric=payload["Metric"],
        remark=remark,
        results_path=str(results_path),
        submission_path=str(submission_path),
        answers_path=str(task_spec.answers_path),
        failed_marker_present=False,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate BioXArena-Pro submission.csv outputs.")
    parser.add_argument("--task", action="append", default=[], help="Task selector, e.g. chemical-biology/cyp-inhibition-multi-label or cyp-inhibition-multi-label.")
    parser.add_argument("--domain", action="append", default=[], help="Run all tasks in a domain, e.g. chemical-biology.")
    parser.add_argument("--all-tasks", action="store_true", help="Evaluate all discovered tasks.")
    parser.add_argument("--list-tasks", action="store_true", help="List discoverable tasks and exit.")
    parser.add_argument("--model", required=True, help="Model name used by the training runner.")
    parser.add_argument("--round-name", default=DEFAULT_ROUND_NAME, help="Round/output subdirectory name appended after the model-specific output directory.")
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_ROOT, help="Shared prefix containing 2_BioXArena-Pro-Output and BioXArena-Data-*-Pro.")
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT, help="Path to 2_BioXArena-Pro-Code/tasks.")
    parser.add_argument("--private-root", type=Path, default=None, help="Base root for private answers. Overrides --prefix-dir/2_BioXArena-Pro-Data-Private.")
    parser.add_argument("--public-root", type=Path, default=None, help="Base root for public Pro task data. Overrides --prefix-dir/2_BioXArena-Pro-Data-Public.")
    parser.add_argument("--output-root", type=Path, default=None, help="Base output root. Overrides --prefix-dir/2_BioXArena-Pro-Output.")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of task evaluations to run in parallel.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    resolved_prefix_dir = args.prefix_dir.resolve()
    resolved_private_root = (
        args.private_root.resolve()
        if args.private_root
        else resolved_prefix_dir / DEFAULT_PRIVATE_ROOT_NAME
    )
    resolved_public_root = (
        args.public_root.resolve()
        if args.public_root
        else resolved_prefix_dir / DEFAULT_PUBLIC_ROOT_NAME
    )
    base_output_root = (
        args.output_root.resolve()
        if args.output_root
        else resolved_prefix_dir / DEFAULT_OUTPUT_ROOT_NAME
    )
    resolved_model_dir = sanitize_path_component(args.model)
    resolved_round_name = sanitize_path_component(args.round_name)
    resolved_output_root = base_output_root / resolved_model_dir / resolved_round_name

    all_tasks = discover_tasks(
        args.tasks_root,
        resolved_private_root,
        resolved_output_root,
        resolved_public_root,
    )
    try:
        selected_tasks = select_tasks(all_tasks, args.domain, args.task, args.all_tasks)
    except ValueError as exc:
        parser.error(str(exc))

    if args.list_tasks:
        for task in selected_tasks or all_tasks:
            print(task.key)
        return 0

    if not selected_tasks:
        parser.error("No tasks selected.")

    print(f"Selected {len(selected_tasks)} task(s) for evaluation.")
    print(f"Model: {args.model}")
    print(f"Model dir: {resolved_model_dir}")
    print(f"Round name: {resolved_round_name}")
    print(f"Public root: {resolved_public_root}")
    print(f"Private root: {resolved_private_root}")
    print(f"Output root: {resolved_output_root}")

    max_workers = max(1, min(args.max_workers, len(selected_tasks)))
    print(f"Parallel workers: {max_workers}")

    resolved_output_root.mkdir(parents=True, exist_ok=True)
    print_lock = threading.Lock()
    results: list[TaskEvaluationResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(evaluate_single_task, task_spec=task, print_lock=print_lock): task
            for task in selected_tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                task.output_dir.mkdir(parents=True, exist_ok=True)
                results_path = task.output_dir / "results.json"
                fallback_metric = infer_task_metric_info(task)
                remark = f"Evaluation runner crashed: {exc}"
                payload = build_results_payload(
                    status="Fail",
                    metric_info=fallback_metric,
                    score=None,
                    normalized_score=0.0,
                    remark=remark,
                )
                write_json(results_path, payload)
                result = TaskEvaluationResult(
                    task_key=task.key,
                    status="Fail",
                    score=None,
                    normalized_score=0.0,
                    metric=payload["Metric"],
                    remark=remark,
                    results_path=str(results_path),
                    submission_path=str(task.output_dir / "submission.csv"),
                    answers_path=str(task.answers_path),
                    failed_marker_present=(task.output_dir / "FAILED.json").exists(),
                )
                with print_lock:
                    print(f"[FAIL] {task.key}: {remark}")
            results.append(result)

    ordered_results = sorted(results, key=lambda item: item.task_key)
    summary_path = resolved_output_root / (
        f"evaluation_summary_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json"
    )
    overall_normalized_score = (
        sum(result.normalized_score for result in ordered_results) / len(ordered_results)
        if ordered_results else 0.0
    )
    summary_payload = {
        "overall_normalized_score": overall_normalized_score,
        "aggregation": "equal_weight_mean_across_selected_tasks",
        "model": args.model,
        "model_dir": resolved_model_dir,
        "round_name": resolved_round_name,
        "public_root": str(resolved_public_root),
        "private_root": str(resolved_private_root),
        "resolved_output_root": str(resolved_output_root),
        "selected_tasks": [task.key for task in selected_tasks],
        "max_workers": max_workers,
        "results": [asdict(result) for result in ordered_results],
    }
    write_json(summary_path, summary_payload)

    success_count = sum(result.status == "Succeed" for result in ordered_results)
    fail_count = len(ordered_results) - success_count
    print(f"Completed {len(ordered_results)} task(s): {success_count} succeeded, {fail_count} failed.")
    print(f"Overall normalized score (task-equal): {overall_normalized_score:.6f}")
    print(f"Summary written to: {summary_path}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
