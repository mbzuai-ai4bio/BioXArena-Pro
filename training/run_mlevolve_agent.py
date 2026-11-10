# BioXArena-Pro: harness/runners/*.py import only the task-level functions of this file (run_single_task and
# helpers). Its standalone command-line entry point (main, .env / default-endpoint handling) is legacy and not used.
#!/usr/bin/env python3
"""Run MLEvolve (MCGS-based) agent on BioXArena-Pro tasks.

MLEvolve uses Monte Carlo Graph Search with multi-agent collaboration:
  - Draft Agent: initial solution generation
  - Improve Agent: solution improvement
  - Debug Agent: bug fixing
  - Evolution Agent: trajectory-aware evolution
  - Fusion Agent: cross-branch fusion

Examples:
    python run_mlevolve_agent.py --task chemical-biology/tdc-dti-binding-affinity
    python run_mlevolve_agent.py --all-tasks --max-workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from submission_preflight import SubmissionPreflightError, validate_public_submission

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
TRAINING_DIR = SCRIPT_PATH.parent
EVAL_ROOT = TRAINING_DIR.parent
WORKSPACE_ROOT = EVAL_ROOT.parent
DEFAULT_TASKS_ROOT = EVAL_ROOT / "tasks"
DEFAULT_ENV_PATH = EVAL_ROOT / ".env"
DEFAULT_ROUND_NAME = "round1"
MLEVOLVE_DIR = EVAL_ROOT / "agents" / "MLEvolve"
MLEVOLVE_EXEC_TIMEOUT = int(os.environ.get("BX_MLE_EXEC_TIMEOUT", 5400))  # BioXArena-Pro runner sets 2400

if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from benchmark_layout import canonicalize_task_key, require_canonical_layout  # noqa: E402

REQUIRED_OUTPUT_LABELS = ["submission.csv", "metrics.json", "solution.py"]
_PATH_COLUMN_SUFFIXES = ("_file", "_path", "_dir")
_MANIFEST_ARTIFACT_SUFFIXES = ("_file", "_path")
_REPAIRABLE_PREFLIGHT_MESSAGES = (
    "artifact paths in submission.csv must be relative",
    "artifact path escapes the task output directory",
    "referenced artifact is missing or is not a regular file",
    "prediction ZIP has an extra 'output/' wrapper",
    "prediction ZIP has an extra ",
)


@dataclass(frozen=True)
class TaskSpec:
    domain: str
    task: str
    task_dir: Path
    description_path: Path
    output_dir: Path

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.task}"


@dataclass
class TaskRunResult:
    task_key: str
    status: str
    output_dir: str
    error: str | None
    duration_sec: float


@dataclass(frozen=True)
class SubmissionRepairResult:
    """Changes made by the conservative post-export artifact repair."""

    relocated_artifacts: tuple[str, ...]
    normalized_archives: tuple[str, ...]


class SubmissionArtifactRepairError(ValueError):
    """Raised when a known export defect cannot be repaired safely."""


def discover_tasks(tasks_root: Path, output_root: Path) -> list[TaskSpec]:
    specs: list[TaskSpec] = []
    if not tasks_root.exists():
        return specs
    for domain_dir in sorted(tasks_root.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name.startswith("."):
            continue
        for task_dir in sorted(domain_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            public_dir = task_dir / "public"
            desc_path = public_dir / "description.md"
            if not desc_path.exists():
                continue
            specs.append(
                TaskSpec(
                    domain=domain_dir.name,
                    task=task_dir.name,
                    task_dir=public_dir,
                    description_path=desc_path,
                    output_dir=output_root / domain_dir.name / task_dir.name,
                )
            )
    return specs


def filter_tasks(
    all_tasks: list[TaskSpec],
    selected_domains: list[str],
    selected_tasks: list[str],
    run_all: bool,
) -> list[TaskSpec]:
    if run_all:
        return sorted(all_tasks, key=lambda t: t.key)
    selected_task_keys = {canonicalize_task_key(key) for key in selected_tasks}
    selected: dict[str, TaskSpec] = {}
    for t in all_tasks:
        if t.domain in selected_domains:
            selected[t.key] = t
        if t.key in selected_task_keys:
            selected[t.key] = t
    return [selected[k] for k in sorted(selected)]


def find_latest_artifact(search_root: Path, relative_pattern: str) -> Path | None:
    matches = [p for p in search_root.rglob(relative_pattern) if p.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def find_latest_bundle_dir(search_root: Path, relative_pattern: str) -> Path | None:
    bundle_file = find_latest_artifact(search_root, relative_pattern)
    if bundle_file is None:
        return None
    return bundle_file.parent


def find_matching_submission_bundle_dirs(
    search_root: Path,
    selected_bundle_dir: Path | None,
) -> tuple[Path, ...]:
    """Find known export locations carrying the selected manifest.

    MLEvolve can write the same best manifest to ``best_submission`` without
    copying its sidecars, while ``top_solution/top1`` still contains the full
    selected bundle.  Deliberately do not search top2/top3/etc.: substituting
    another ranked candidate would change the agent's selected answer.  The
    byte-identical manifest check also prevents recovery from an unrelated run.
    """
    if selected_bundle_dir is None:
        return ()

    selected_manifest = selected_bundle_dir / "submission.csv"
    try:
        selected_bytes = selected_manifest.read_bytes()
    except OSError:
        return ()

    resolved_search_root = search_root.resolve()
    resolved_selected = selected_bundle_dir.resolve()
    if selected_bundle_dir.name == "best_submission":
        run_root = selected_bundle_dir.parent
    elif (
        selected_bundle_dir.name == "top1"
        and selected_bundle_dir.parent.name == "top_solution"
    ):
        run_root = selected_bundle_dir.parent.parent
    else:
        return ()
    resolved_run_root = run_root.resolve()
    if not _is_within(resolved_run_root, resolved_search_root):
        return ()

    matches: list[Path] = []
    seen: set[Path] = {resolved_selected}
    for manifest in (
        run_root / "best_submission/submission.csv",
        run_root / "top_solution/top1/submission.csv",
    ):
        if not manifest.is_file():
            continue
        candidate_dir = manifest.parent
        resolved_candidate = candidate_dir.resolve()
        if resolved_candidate in seen:
            continue
        seen.add(resolved_candidate)
        try:
            if manifest.read_bytes() == selected_bytes:
                matches.append(candidate_dir)
        except OSError:
            continue
    return tuple(matches)


def copy_missing_submission_artifacts_from_matching_top_solution(
    output_dir: Path,
    workspace_root: Path,
    selected_bundle_dir: Path | None,
) -> tuple[str, ...]:
    """Copy only missing, manifest-referenced files from selected top1.

    This repairs MLEvolve's incomplete ``best_submission`` export without
    changing the manifest or the generated artifact bytes.  In particular,
    malformed predictions are copied as-is and remain subject to the normal
    submission preflight, so format correctness is still part of agent ability.
    Existing output paths are never overwritten.
    """
    if selected_bundle_dir is None:
        return ()

    submission_path = output_dir / "submission.csv"
    try:
        with submission_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return ()
            path_columns = [
                name
                for name in reader.fieldnames
                if name.lower().endswith(_MANIFEST_ARTIFACT_SUFFIXES)
            ]
            raw_paths = {
                (row.get(column) or "").strip()
                for row in reader
                for column in path_columns
                if (row.get(column) or "").strip()
            }
    except (OSError, csv.Error):
        return ()

    fallback_dirs = find_matching_submission_bundle_dirs(
        workspace_root,
        selected_bundle_dir,
    )
    if not fallback_dirs:
        return ()

    resolved_output = output_dir.resolve()
    resolved_workspace = workspace_root.resolve()
    copied: list[str] = []
    for raw_path in sorted(raw_paths):
        relative = PurePosixPath(raw_path.replace("\\", "/"))
        if relative.is_absolute() or not relative.name or ".." in relative.parts:
            # Absolute/workspace paths are handled by the existing conservative
            # relocation pass.  This helper is intentionally only for the
            # unambiguous missing-relative-sidecar case.
            continue

        destination = output_dir.joinpath(*relative.parts)
        if os.path.lexists(destination):
            continue
        resolved_destination = destination.resolve()
        if not _is_within(resolved_destination, resolved_output):
            continue

        for fallback_dir in fallback_dirs:
            resolved_fallback = fallback_dir.resolve()
            if not _is_within(resolved_fallback, resolved_workspace):
                continue
            source = _safe_regular_file(
                fallback_dir.joinpath(*relative.parts),
                resolved_workspace,
            )
            if source is None:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if os.path.lexists(destination):
                break
            shutil.copy2(source, destination)
            copied.append(relative.as_posix())
            break

    return tuple(copied)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_numeric_metric(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"`{key}` must be a number or numeric string in metrics.json.")


def parse_nonnegative_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return 0
        return max(0, int(float(stripped)))
    return 0


def parse_optional_numeric_metric(payload: dict[str, Any], key: str) -> float | None:
    try:
        return parse_numeric_metric(payload, key)
    except Exception:
        return None


def parse_execution_time_metric(metric_txt_path: Path | None) -> float | None:
    if metric_txt_path is None or not metric_txt_path.exists():
        return None

    for line in metric_txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("Execution Time(s):"):
            continue
        raw_value = line.split(":", 1)[1].strip()
        if not raw_value or raw_value.upper() == "N/A":
            return None
        return float(raw_value)

    return None


def resolve_solution_generated_time(task_start_time: float, solution_path: Path | None) -> float:
    if solution_path is None or not solution_path.exists():
        return time.time()
    try:
        return min(time.time(), max(task_start_time, solution_path.stat().st_mtime))
    except OSError:
        return time.time()


def collect_llm_usage(search_root: Path) -> tuple[int, int, bool]:
    if not search_root.exists():
        return 0, 0, False

    total_input_tokens = 0
    total_output_tokens = 0
    found_usage = False

    for usage_path in sorted(search_root.rglob("llm_usage.jsonl")):
        for raw_line in usage_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            total_input_tokens += parse_nonnegative_int(payload.get("input_tokens"))
            total_output_tokens += parse_nonnegative_int(payload.get("output_tokens"))
            found_usage = True

    return total_input_tokens, total_output_tokens, found_usage


def infer_model_used(solution_path: Path | None) -> str:
    if solution_path is None or not solution_path.exists():
        return "Unknown"

    try:
        source = solution_path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return "Unknown"

    keyword_map = (
        ("xgboost", "XGBoost"),
        ("xgbclassifier", "XGBoost"),
        ("xgbregressor", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("lgbmclassifier", "LightGBM"),
        ("lgbmregressor", "LightGBM"),
        ("catboost", "CatBoost"),
        ("randomforest", "RandomForest"),
        ("extratrees", "ExtraTrees"),
        ("logisticregression", "LogisticRegression"),
        ("linearregression", "LinearRegression"),
        ("svc(", "SVM"),
        ("svr(", "SVM"),
        ("transformers", "Transformers"),
        ("automodel", "Transformers"),
        ("bert", "Transformers"),
        ("resnet", "CNN"),
        ("conv2d", "CNN"),
        ("torch.nn", "PyTorch"),
        ("tensorflow", "TensorFlow"),
        ("sklearn", "Scikit-learn"),
    )
    for keyword, model_label in keyword_map:
        if keyword in source:
            return model_label

    return "Unknown"


def combine_notes(*note_groups: Any) -> str:
    combined: list[str] = []
    seen: set[str] = set()
    for note_group in note_groups:
        if note_group is None:
            continue
        if isinstance(note_group, str):
            candidates = [note_group]
        elif isinstance(note_group, (list, tuple, set)):
            candidates = [str(item) for item in note_group if str(item).strip()]
        else:
            candidates = [str(note_group)]
        for note in candidates:
            normalized = note.strip()
            if not normalized or normalized in seen:
                continue
            combined.append(normalized)
            seen.add(normalized)
    return "\n".join(combined)


def write_final_metrics_json(
    output_path: Path,
    source_metrics_path: Path | None,
    metric_txt_path: Path | None,
    solution_path: Path | None,
    task_start_time: float,
    solution_generated_time: float,
    input_tokens: int,
    output_tokens: int,
    usage_found: bool,
    model: str,
    temperature: float,
    steps: int,
) -> None:
    source_payload: dict[str, Any] = {}
    if source_metrics_path is not None and source_metrics_path.exists():
        payload = read_json(source_metrics_path)
        if not isinstance(payload, dict):
            raise ValueError("source metrics.json must contain a JSON object.")
        source_payload = dict(payload)

    aggregate_exec_time = parse_execution_time_metric(metric_txt_path)
    train_time_sec = parse_optional_numeric_metric(source_payload, "train_time_sec")
    test_time_sec = parse_optional_numeric_metric(source_payload, "test_time_sec")

    fallback_notes: list[str] = []
    if train_time_sec is None or test_time_sec is None:
        if aggregate_exec_time is not None:
            if train_time_sec is None:
                train_time_sec = aggregate_exec_time
            if test_time_sec is None:
                test_time_sec = 0.0
            fallback_notes.append(
                "MLEvolve did not emit phase-separated train/test timings; runner used aggregate candidate execution time from metric.txt and set any missing phase timing to 0."
            )
        else:
            if train_time_sec is None:
                train_time_sec = 0.0
            if test_time_sec is None:
                test_time_sec = 0.0
            fallback_notes.append(
                "MLEvolve did not emit train/test timings; runner defaulted missing timing fields to 0."
            )

    model_used = source_payload.get("model_used") or infer_model_used(solution_path)
    if model_used == "Unknown":
        model_used = "Unknown (inspect solution.py)"

    phase_execution_time_sec = float(train_time_sec) + float(test_time_sec)
    solution_generation_time_sec = max(
        0.0,
        solution_generated_time - task_start_time - phase_execution_time_sec,
    )

    notes = combine_notes(
        source_payload.get("notes"),
        None if source_metrics_path is not None else "metrics.json synthesized by run_mlevolve_agent.py from MLEvolve artifacts.",
        None if usage_found else "MLEvolve did not persist token usage metadata; input_tokens/output_tokens defaulted to 0.",
        fallback_notes,
        f"MLEvolve search metadata: steps={steps}, code_model={model}, temperature={temperature}.",
    )

    final_payload = dict(source_payload)
    final_payload["solution_generation_time_sec"] = round(solution_generation_time_sec, 6)
    final_payload["train_time_sec"] = round(float(train_time_sec), 6)
    final_payload["test_time_sec"] = round(float(test_time_sec), 6)
    final_payload["code_total_time_sec"] = round(
        solution_generation_time_sec + float(train_time_sec) + float(test_time_sec), 6
    )
    final_payload["input_tokens"] = input_tokens if usage_found else 0
    final_payload["output_tokens"] = output_tokens if usage_found else 0
    final_payload["model_used"] = model_used
    final_payload["model_param_count"] = source_payload.get("model_param_count")
    final_payload["notes"] = notes

    write_json(output_path, final_payload)


def validate_metrics_json(metrics_path: Path) -> None:
    payload = read_json(metrics_path)
    if not isinstance(payload, dict):
        raise ValueError("metrics.json must contain a JSON object.")

    required_fields = (
        "solution_generation_time_sec",
        "train_time_sec",
        "test_time_sec",
        "code_total_time_sec",
        "input_tokens",
        "output_tokens",
        "model_used",
        "model_param_count",
        "notes",
    )
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        raise ValueError(
            "metrics.json is missing required field(s): " + ", ".join(missing_fields)
        )

    parse_numeric_metric(payload, "solution_generation_time_sec")
    parse_numeric_metric(payload, "train_time_sec")
    parse_numeric_metric(payload, "test_time_sec")
    parse_numeric_metric(payload, "code_total_time_sec")
    parse_nonnegative_int(payload.get("input_tokens"))
    parse_nonnegative_int(payload.get("output_tokens"))


def verify_required_outputs(output_dir: Path) -> list[str]:
    missing: list[str] = []
    for filename in REQUIRED_OUTPUT_LABELS:
        if not (output_dir / filename).exists():
            missing.append(filename)
    return missing


def looks_like_artifact_value(value: str, column_name: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    normalized = stripped.replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute():
        return True
    if "/" in stripped or "\\" in stripped:
        return True
    return column_name.lower().endswith(_PATH_COLUMN_SUFFIXES)


def collect_submission_artifact_targets(submission_csv: Path, output_dir: Path) -> set[Path]:
    if not submission_csv.exists():
        return set()

    targets: set[Path] = set()
    with submission_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return targets

        for row in reader:
            for column_name in reader.fieldnames[1:]:
                raw_value = (row.get(column_name) or "").strip()
                if not looks_like_artifact_value(raw_value, column_name):
                    continue
                artifact_path = Path(raw_value.replace("\\", "/"))
                if artifact_path.is_absolute():
                    try:
                        artifact_path = artifact_path.relative_to(output_dir)
                    except ValueError:
                        continue
                if not artifact_path.parts:
                    continue
                top_level_target = output_dir / artifact_path.parts[0]
                targets.add(top_level_target)

    return targets


def clear_exported_submission_bundle(output_dir: Path) -> None:
    submission_csv = output_dir / "submission.csv"
    for target in sorted(
        collect_submission_artifact_targets(submission_csv, output_dir),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()

    if submission_csv.exists():
        submission_csv.unlink()


def copy_bundle_dir_contents(source_dir: Path, target_dir: Path) -> None:
    for source_path in source_dir.iterdir():
        target_path = target_dir / source_path.name
        if source_path.is_dir():
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_regular_file(candidate: Path, allowed_root: Path) -> Path | None:
    """Return a resolved regular file only when it stays inside allowed_root."""
    resolved_root = allowed_root.resolve()
    resolved_candidate = candidate.resolve()
    if not _is_within(resolved_candidate, resolved_root):
        return None
    if not resolved_candidate.is_file() or candidate.is_symlink():
        return None
    return resolved_candidate


def _resolve_repair_source(
    raw_path: str,
    output_dir: Path,
    workspace_root: Path,
    source_submission_dir: Path | None,
    fallback_submission_dirs: tuple[Path, ...],
) -> Path:
    """Resolve an exported artifact without allowing arbitrary filesystem reads."""
    normalized = raw_path.replace("\\", "/")
    manifest_path = Path(normalized)
    candidates: list[tuple[Path, Path]] = []

    if manifest_path.is_absolute():
        candidates.extend(
            (manifest_path, root) for root in (output_dir, workspace_root)
        )
    else:
        # Prefer the already-exported artifact.  Fall back to the exact source
        # bundle only when it remains inside this task's MLEvolve workspace.
        candidates.append((output_dir / manifest_path, output_dir))
        if source_submission_dir is not None:
            candidates.append((source_submission_dir / manifest_path, workspace_root))
        candidates.extend(
            (source_dir / manifest_path, workspace_root)
            for source_dir in fallback_submission_dirs
        )

    for candidate, allowed_root in candidates:
        resolved = _safe_regular_file(candidate, allowed_root)
        if resolved is not None:
            return resolved

    raise SubmissionArtifactRepairError(
        "cannot safely locate referenced artifact in the task output or its "
        f"MLEvolve workspace: {raw_path!r}"
    )


def _safe_instance_name(instance: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in instance)
    return safe.strip("._") or "instance"


def _preferred_repair_destination(
    raw_path: str,
    source: Path,
    output_dir: Path,
) -> PurePosixPath:
    normalized = raw_path.replace("\\", "/")
    manifest_path = Path(normalized)
    if not manifest_path.is_absolute():
        resolved_candidate = (output_dir / manifest_path).resolve()
        if _is_within(resolved_candidate, output_dir.resolve()):
            relative = PurePosixPath(normalized)
            if relative.parts and ".." not in relative.parts and relative.name:
                return relative
    return PurePosixPath("predictions", source.name)


def _zip_members_for_repair(archive: Path) -> list[zipfile.ZipInfo]:
    try:
        handle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SubmissionArtifactRepairError(f"invalid ZIP artifact: {archive}") from exc

    with handle:
        members = handle.infolist()
        if not members:
            raise SubmissionArtifactRepairError(f"prediction ZIP is empty: {archive}")
        seen: set[str] = set()
        files: list[zipfile.ZipInfo] = []
        for member in members:
            if "\\" in member.filename:
                raise SubmissionArtifactRepairError(
                    f"ZIP member must use POSIX separators: {member.filename!r}"
                )
            normalized = member.filename.rstrip("/")
            if not normalized or normalized in seen:
                raise SubmissionArtifactRepairError(
                    f"empty or duplicate ZIP member: {member.filename!r}"
                )
            seen.add(normalized)
            if member.flag_bits & 0x1:
                raise SubmissionArtifactRepairError(
                    f"encrypted ZIP member cannot be repaired: {member.filename!r}"
                )
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise SubmissionArtifactRepairError(
                    f"unsafe path in prediction ZIP: {member.filename!r}"
                )
            if stat.S_ISLNK(member.external_attr >> 16):
                raise SubmissionArtifactRepairError(
                    f"symbolic link in prediction ZIP: {member.filename!r}"
                )
            if not member.is_dir():
                files.append(member)
        if not files:
            raise SubmissionArtifactRepairError(
                f"prediction ZIP contains no files: {archive}"
            )
        return files


def _wrapper_depth(files: list[zipfile.ZipInfo], instance: str) -> int:
    paths = [PurePosixPath(member.filename) for member in files]
    depth = 0
    if all(len(path.parts) > depth and path.parts[depth] == "output" for path in paths):
        depth += 1
    if instance and all(
        len(path.parts) > depth and path.parts[depth] == instance for path in paths
    ):
        depth += 1
    return depth


def _clone_zip_info(member: zipfile.ZipInfo, filename: str) -> zipfile.ZipInfo:
    cloned = zipfile.ZipInfo(filename=filename, date_time=member.date_time)
    for attribute in (
        "comment",
        "extra",
        "create_system",
        "create_version",
        "extract_version",
        "reserved",
        "flag_bits",
        "volume",
        "internal_attr",
        "external_attr",
        "compress_type",
    ):
        setattr(cloned, attribute, getattr(member, attribute))
    if hasattr(member, "_compresslevel"):
        cloned._compresslevel = member._compresslevel
    return cloned


def _copy_or_normalize_artifact(source: Path, destination: Path, instance: str) -> bool:
    """Copy an artifact, stripping only preflight-forbidden ZIP wrappers."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() != ".zip":
        shutil.copy2(source, destination)
        return False

    files = _zip_members_for_repair(source)
    wrapper_depth = _wrapper_depth(files, instance)
    if wrapper_depth == 0:
        # Preserve valid archives byte-for-byte.
        shutil.copy2(source, destination)
        return False

    remapped: list[tuple[zipfile.ZipInfo, str]] = []
    seen_destinations: set[str] = set()
    for member in files:
        parts = PurePosixPath(member.filename).parts[wrapper_depth:]
        if not parts:
            raise SubmissionArtifactRepairError(
                f"ZIP wrapper removal produced an empty member: {member.filename!r}"
            )
        remapped_name = PurePosixPath(*parts).as_posix()
        if remapped_name in seen_destinations:
            raise SubmissionArtifactRepairError(
                "ZIP wrapper removal would create duplicate member "
                f"{remapped_name!r}"
            )
        seen_destinations.add(remapped_name)
        remapped.append((member, remapped_name))

    with zipfile.ZipFile(source) as source_zip, zipfile.ZipFile(
        destination, mode="w", allowZip64=True
    ) as destination_zip:
        destination_zip.comment = source_zip.comment
        for member, remapped_name in remapped:
            cloned = _clone_zip_info(member, remapped_name)
            with source_zip.open(member, mode="r") as source_handle, \
                    destination_zip.open(cloned, mode="w", force_zip64=True) as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
    return True


def _is_known_repairable_preflight_error(error: SubmissionPreflightError) -> bool:
    message = str(error)
    return any(fragment in message for fragment in _REPAIRABLE_PREFLIGHT_MESSAGES)


def _repair_exported_submission_bundle(
    task_dir: Path,
    output_dir: Path,
    task_key: str,
    workspace_root: Path,
    source_submission_dir: Path | None,
) -> SubmissionRepairResult:
    submission_path = output_dir / "submission.csv"
    with submission_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SubmissionArtifactRepairError("submission.csv has no header")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    path_columns = [
        name for name in fieldnames if name.lower().endswith(_MANIFEST_ARTIFACT_SUFFIXES)
    ]
    if not path_columns:
        raise SubmissionArtifactRepairError(
            "submission.csv has no artifact path columns"
        )

    source_submission_dir = (
        source_submission_dir.resolve() if source_submission_dir is not None else None
    )
    resolved_workspace_root = workspace_root.resolve()
    if source_submission_dir is not None and not _is_within(
        source_submission_dir, resolved_workspace_root
    ):
        raise SubmissionArtifactRepairError(
            "source submission directory is outside the task workspace"
        )
    resolved_fallback_dirs: list[Path] = []
    for fallback_dir in find_matching_submission_bundle_dirs(
        workspace_root,
        source_submission_dir,
    ):
        resolved_fallback = fallback_dir.resolve()
        if not _is_within(resolved_fallback, resolved_workspace_root):
            raise SubmissionArtifactRepairError(
                "fallback submission directory is outside the task workspace"
            )
        if resolved_fallback != source_submission_dir:
            resolved_fallback_dirs.append(resolved_fallback)

    assignments: dict[PurePosixPath, tuple[Path, str]] = {}
    relocated: set[str] = set()
    for row in rows:
        instance = (row.get(fieldnames[0]) or "").strip()
        for column in path_columns:
            raw_path = (row.get(column) or "").strip()
            if not raw_path:
                raise SubmissionArtifactRepairError(
                    f"blank artifact path for instance {instance!r}, column {column!r}"
                )
            source = _resolve_repair_source(
                raw_path=raw_path,
                output_dir=output_dir,
                workspace_root=workspace_root,
                source_submission_dir=source_submission_dir,
                fallback_submission_dirs=tuple(resolved_fallback_dirs),
            )
            destination = _preferred_repair_destination(raw_path, source, output_dir)
            prior = assignments.get(destination)
            if prior is not None and prior[0] != source:
                destination = PurePosixPath(
                    "predictions", f"{_safe_instance_name(instance)}__{source.name}"
                )
                prior = assignments.get(destination)
                if prior is not None and prior[0] != source:
                    raise SubmissionArtifactRepairError(
                        f"artifact destination collision: {destination.as_posix()!r}"
                    )
            if destination == PurePosixPath("submission.csv"):
                raise SubmissionArtifactRepairError(
                    "artifact destination collides with submission.csv"
                )
            assignments[destination] = (source, instance)
            if destination.as_posix() != raw_path.replace("\\", "/") or not _is_within(
                source, output_dir.resolve()
            ):
                relocated.add(destination.as_posix())
            row[column] = destination.as_posix()

    normalized_archives: set[str] = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".submission-repair-", dir=output_dir) as tmp:
        staging_dir = Path(tmp)
        for destination, (source, instance) in assignments.items():
            staged_artifact = staging_dir.joinpath(*destination.parts)
            normalized = _copy_or_normalize_artifact(
                source=source,
                destination=staged_artifact,
                instance=instance,
            )
            if normalized:
                normalized_archives.add(destination.as_posix())

        with (staging_dir / "submission.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        # Nothing is committed unless the repaired bundle passes the same strict
        # validator used by evaluation.
        validate_public_submission(
            task_dir=task_dir,
            output_dir=staging_dir,
            task_key=task_key,
        )

        staged_files = sorted(
            (
                path for path in staging_dir.rglob("*")
                if path.is_file() and path.name != "submission.csv"
            ),
            key=lambda path: path.relative_to(staging_dir).as_posix(),
        )
        for staged_path in staged_files:
            relative = staged_path.relative_to(staging_dir)
            target_path = output_dir / relative
            if target_path.exists() and target_path.is_dir():
                raise SubmissionArtifactRepairError(
                    f"artifact target is an existing directory: {relative.as_posix()!r}"
                )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_path, target_path)
        os.replace(staging_dir / "submission.csv", submission_path)

    # Verify the installed files as well, so callers remove FAILED.json only
    # after the final on-disk bundle is known to be valid.
    validate_public_submission(
        task_dir=task_dir,
        output_dir=output_dir,
        task_key=task_key,
    )
    return SubmissionRepairResult(
        relocated_artifacts=tuple(sorted(relocated)),
        normalized_archives=tuple(sorted(normalized_archives)),
    )


def validate_or_repair_public_submission(
    task_dir: Path,
    output_dir: Path,
    task_key: str,
    workspace_root: Path,
    source_submission_dir: Path | None,
) -> SubmissionRepairResult | None:
    """Validate first; repair only the two known MLEvolve export defects.

    A valid bundle returns before creating or modifying any file.  Unknown
    validation failures are re-raised unchanged.
    """
    try:
        validate_public_submission(
            task_dir=task_dir,
            output_dir=output_dir,
            task_key=task_key,
        )
        return None
    except SubmissionPreflightError as original_error:
        if not _is_known_repairable_preflight_error(original_error):
            raise
        try:
            return _repair_exported_submission_bundle(
                task_dir=task_dir,
                output_dir=output_dir,
                task_key=task_key,
                workspace_root=workspace_root,
                source_submission_dir=source_submission_dir,
            )
        except Exception as repair_error:
            raise SubmissionPreflightError(
                f"{original_error}; conservative artifact repair failed: {repair_error}"
            ) from repair_error


def clear_benchmark_outputs(output_dir: Path) -> None:
    clear_exported_submission_bundle(output_dir)
    for filename in REQUIRED_OUTPUT_LABELS + ["FAILED.json"]:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def clear_mlevolve_run_dirs(output_dir: Path) -> None:
    for dirname in ("logs", "workspaces"):
        path = output_dir / dirname
        if path.exists():
            shutil.rmtree(path)


def write_failed_marker(
    task_spec: TaskSpec,
    error: str | None,
    duration_sec: float,
    missing_outputs: list[str],
    exit_code: int | None = None,
) -> None:
    payload = {
        "task_key": task_spec.key,
        "status": "failed",
        "output_dir": str(task_spec.output_dir),
        "final_error": error,
        "duration_sec": duration_sec,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "missing_required_outputs": missing_outputs,
        "exit_code": exit_code,
    }
    write_json(task_spec.output_dir / "FAILED.json", payload)


def run_single_task(
    task_spec: TaskSpec,
    mlevolve_dir: Path,
    dataset_root: Path,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    steps: int,
    time_limit: int,
    print_lock: threading.Lock,
) -> TaskRunResult:
    """Run MLEvolve MCGS agent on a single task via subprocess."""
    task_start = time.time()
    task_spec.output_dir.mkdir(parents=True, exist_ok=True)
    clear_benchmark_outputs(task_spec.output_dir)
    clear_mlevolve_run_dirs(task_spec.output_dir)
    try:
        from _core_pkg_guard import restore_core_packages
        restore_core_packages()
    except Exception:
        pass

    with print_lock:
        print(f"[START] {task_spec.key}")

    try:
        log_root = task_spec.output_dir / "logs"
        workspace_root = task_spec.output_dir / "workspaces"
        exp_id = task_spec.key
        exp_name = task_spec.task

        cmd = [
            "timeout",
            "--foreground",
            "--signal=TERM",
            "--kill-after=10s",
            f"{time_limit}s",
            sys.executable,
            str(mlevolve_dir / "run.py"),
            f"exp_id={exp_id}",
            f"data_dir={task_spec.task_dir}",
            f"dataset_dir={dataset_root}",
            f"desc_file={task_spec.description_path}",
            f"exp_name={exp_name}",
            f"log_dir={log_root}",
            f"workspace_dir={workspace_root}",
            f"agent.code.model={model}",
            f"agent.code.temp={temperature}",
            f"agent.code.base_url={base_url}",
            f"agent.code.api_key={api_key}",
            f"agent.feedback.model={model}",
            f"agent.feedback.temp={temperature}",
            f"agent.feedback.base_url={base_url}",
            f"agent.feedback.api_key={api_key}",
            f"agent.steps={steps}",
            f"agent.time_limit={time_limit}",
            f"exec.timeout={MLEVOLVE_EXEC_TIMEOUT}",
            "preprocess_data=False",
            "copy_data=False",
            "start_cpu_id=0",
            "cpu_number=16",
        ]

        (task_spec.output_dir / "mlevolve_cmd.txt").write_text(
            " \\\n  ".join(cmd), encoding="utf-8"
        )

        result = subprocess.run(
            cmd,
            cwd=str(mlevolve_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        (task_spec.output_dir / "mlevolve_stdout.log").write_text(
            result.stdout or "", encoding="utf-8"
        )
        (task_spec.output_dir / "mlevolve_stderr.log").write_text(
            result.stderr or "", encoding="utf-8"
        )

        best_submission_dir = (
            find_latest_bundle_dir(workspace_root, "best_submission/submission.csv")
            or find_latest_bundle_dir(workspace_root, "top_solution/top1/submission.csv")
        )
        best_submission = (
            best_submission_dir / "submission.csv"
            if best_submission_dir is not None
            else find_latest_artifact(workspace_root, "submission.csv")
        )
        best_solution = (
            find_latest_artifact(workspace_root, "best_solution/solution.py")
            or find_latest_artifact(log_root, "best_solution.py")
            or find_latest_artifact(log_root, "solution.py")
        )
        best_metric_txt = (
            find_latest_artifact(workspace_root, "best_solution/metric.txt")
            or find_latest_artifact(workspace_root, "top_solution/top1/metric.txt")
        )
        best_metrics = (
            find_latest_artifact(workspace_root, "metrics.json")
            or find_latest_artifact(log_root, "metrics.json")
        )

        # Copy outputs to expected locations
        if best_submission_dir is not None:
            copy_bundle_dir_contents(best_submission_dir, task_spec.output_dir)
            recovered_sidecars = copy_missing_submission_artifacts_from_matching_top_solution(
                output_dir=task_spec.output_dir,
                workspace_root=workspace_root,
                selected_bundle_dir=best_submission_dir,
            )
            if recovered_sidecars:
                with print_lock:
                    print(
                        f"[EXPORTED SIDECARS] {task_spec.key}: "
                        f"{list(recovered_sidecars)}"
                    )
        elif best_submission is not None:
            shutil.copy2(best_submission, task_spec.output_dir / "submission.csv")
        if best_solution is not None:
            shutil.copy2(best_solution, task_spec.output_dir / "solution.py")
        duration = time.time() - task_start
        input_tokens, output_tokens, usage_found = collect_llm_usage(log_root)
        solution_generated_time = resolve_solution_generated_time(task_start, best_solution)

        missing = verify_required_outputs(task_spec.output_dir)
        metrics_error: str | None = None
        if best_solution is not None:
            try:
                write_final_metrics_json(
                    output_path=task_spec.output_dir / "metrics.json",
                    source_metrics_path=best_metrics,
                    metric_txt_path=best_metric_txt,
                    solution_path=best_solution,
                    task_start_time=task_start,
                    solution_generated_time=solution_generated_time,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    usage_found=usage_found,
                    model=model,
                    temperature=temperature,
                    steps=steps,
                )
            except Exception as exc:
                metrics_error = f"metrics.json synthesis failed: {exc}"

        metrics_path = task_spec.output_dir / "metrics.json"
        missing = verify_required_outputs(task_spec.output_dir)
        if metrics_error is None and metrics_path.exists():
            try:
                validate_metrics_json(metrics_path)
            except Exception as exc:
                metrics_error = f"metrics.json validation failed: {exc}"

        submission_error: str | None = None
        if not missing:
            try:
                repair_result = validate_or_repair_public_submission(
                    task_dir=task_spec.task_dir,
                    output_dir=task_spec.output_dir,
                    task_key=task_spec.key,
                    workspace_root=workspace_root,
                    source_submission_dir=best_submission_dir,
                )
                if repair_result is not None:
                    with print_lock:
                        print(
                            f"[REPAIRED] {task_spec.key}: "
                            f"relocated={list(repair_result.relocated_artifacts)}, "
                            f"normalized_zips={list(repair_result.normalized_archives)}"
                        )
            except Exception as exc:
                submission_error = f"submission preflight failed: {exc}"

        outputs_ready = (
            metrics_error is None
            and metrics_path.exists()
            and not missing
            and submission_error is None
        )
        if outputs_ready:
            status = "success"
            error = None
            failed_marker = task_spec.output_dir / "FAILED.json"
            if failed_marker.exists():
                failed_marker.unlink()
        else:
            status = "failed"
            if metrics_error is not None:
                error = metrics_error
                if "metrics.json" not in missing:
                    missing.append("valid metrics.json")
            elif submission_error is not None:
                error = submission_error
                missing.append("valid submission.csv and referenced artifacts")
            elif result.returncode == 124:
                error = f"Timeout after {time_limit}s"
            elif result.returncode != 0:
                error = f"MLEvolve exited with code {result.returncode}"
            elif missing:
                error = "Missing required outputs: " + ", ".join(missing)
            else:
                error = "metrics.json was not generated"

        if status == "failed":
            write_failed_marker(
                task_spec=task_spec,
                error=error,
                duration_sec=duration,
                missing_outputs=missing,
                exit_code=result.returncode,
            )

        with print_lock:
            print(f"[{status.upper()}] {task_spec.key} ({duration:.0f}s)")

        return TaskRunResult(
            task_key=task_spec.key, status=status,
            output_dir=str(task_spec.output_dir), error=error, duration_sec=duration,
        )

    except Exception as exc:
        duration = time.time() - task_start
        with print_lock:
            print(f"[ERROR] {task_spec.key} ({duration:.0f}s): {exc}")
        write_failed_marker(
            task_spec=task_spec,
            error=str(exc),
            duration_sec=duration,
            missing_outputs=verify_required_outputs(task_spec.output_dir),
        )
        return TaskRunResult(
            task_key=task_spec.key, status="failed",
            output_dir=str(task_spec.output_dir), error=str(exc), duration_sec=duration,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLEvolve agent on BioXArena-Pro tasks")
    parser.add_argument("--prefix-dir", type=str, default=str(WORKSPACE_ROOT))
    parser.add_argument("--model", type=str, default="google/gemma-4-31b-it")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--time-limit", type=int, default=7200)
    parser.add_argument("--round-name", type=str, default=DEFAULT_ROUND_NAME)
    parser.add_argument("--task", action="append", default=[], dest="tasks")
    parser.add_argument("--domain", action="append", default=[], dest="domains")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prefix_dir = Path(args.prefix_dir).resolve()
    tasks_root = (prefix_dir / "2_BioXArena-Pro-Data-Public").resolve()
    model_dir_name = args.model.replace("/", "__").replace(":", "_")
    output_root = prefix_dir / "2_BioXArena-Pro-Output" / f"mlevolve__{model_dir_name}" / args.round_name
    mlevolve_dir = MLEVOLVE_DIR.resolve()
    env_path = prefix_dir / "2_BioXArena-Pro-Code" / ".env"

    if args.api_key is None or args.base_url is None:
        if env_path.exists():
            env_vars = {}
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip()
            if args.api_key is None:
                args.api_key = env_vars.get("api_key", "")
            if args.base_url is None:
                args.base_url = env_vars.get("api_url", "https://openrouter.ai/api/v1")

    if args.api_key is None:
        args.api_key = ""
    if args.base_url is None:
        args.base_url = "https://openrouter.ai/api/v1"

    all_tasks = discover_tasks(tasks_root, output_root)
    require_canonical_layout((task.key for task in all_tasks), "public task layout")
    selected = filter_tasks(all_tasks, args.domains, args.tasks, args.all_tasks)

    if not selected:
        print("No tasks selected. Use --task, --domain, or --all-tasks.")
        sys.exit(1)

    print(f"Agent: MLEvolve (MCGS)")
    print(f"Backend LLM: {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"Temperature: {args.temperature}")
    print(f"MCGS steps: {args.steps}")
    print(f"Time limit: {args.time_limit}s")
    print(f"Tasks: {len(selected)}")
    print(f"Output root: {output_root}")
    print(f"Max workers: {args.max_workers}")
    print()

    if args.dry_run:
        for t in selected:
            print(f"  [DRY] {t.key} -> {t.output_dir}")
        return

    print_lock = threading.Lock()
    results: list[TaskRunResult] = []
    if args.max_workers <= 1:
        for task_spec in selected:
            results.append(run_single_task(
                task_spec=task_spec,
                mlevolve_dir=mlevolve_dir,
                dataset_root=tasks_root,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                temperature=args.temperature,
                steps=args.steps,
                time_limit=args.time_limit,
                print_lock=print_lock,
            ))
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(
                    run_single_task,
                    task_spec=ts,
                    mlevolve_dir=mlevolve_dir,
                    dataset_root=tasks_root,
                    model=args.model,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    temperature=args.temperature,
                    steps=args.steps,
                    time_limit=args.time_limit,
                    print_lock=print_lock,
                ): ts for ts in selected
            }
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda r: r.task_key)
    success = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {success} success, {failed} failed, {len(results)} total")
    print(f"{'='*60}")
    for r in results:
        marker = "✓" if r.status == "success" else "✗"
        print(f"  {marker} {r.task_key} ({r.duration_sec:.0f}s)" + (f" - {r.error}" if r.error else ""))

    summary_path = output_root / f"mlevolve_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "agent": "mlevolve", "model": args.model, "base_url": args.base_url,
        "temperature": args.temperature, "steps": args.steps,
        "time_limit": args.time_limit, "round_name": args.round_name,
        "output_root": str(output_root),
        "selected_tasks": [t.key for t in selected],
        "results": [asdict(r) for r in results],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
