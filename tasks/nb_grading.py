"""Adapter for NatureBench tasks using BioXArena-Pro's CSV/file-manifest API."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class InvalidNBSubmissionError(ValueError):
    """Raised when an NB submission does not satisfy its published contract."""


def _load_contract(task_dir: Path) -> Any:
    contract_path = task_dir / "contract.py"
    module_name = "nb_contract_" + task_dir.name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, contract_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load NB metric contract: {contract_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_info_from_contract(grade_file: str | Path) -> dict[str, str]:
    contract = _load_contract(Path(grade_file).resolve().parent)
    return dict(contract.METRIC)


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    if not archive.is_file():
        raise FileNotFoundError(f"Prediction artifact not found: {archive}")
    if archive.suffix.lower() != ".zip":
        raise InvalidNBSubmissionError(
            f"NB prediction artifacts must be ZIP files, got: {archive.name}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        if not members:
            raise InvalidNBSubmissionError(f"Prediction ZIP is empty: {archive}")
        seen_members: set[str] = set()
        for member in members:
            normalized_name = member.filename.rstrip("/")
            if not normalized_name or normalized_name in seen_members:
                raise InvalidNBSubmissionError(
                    f"Empty or duplicate member in prediction ZIP {archive.name}: "
                    f"{member.filename!r}"
                )
            seen_members.add(normalized_name)
            if member.flag_bits & 0x1:
                raise InvalidNBSubmissionError(
                    f"Encrypted ZIP members are not supported: {member.filename!r}"
                )
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise InvalidNBSubmissionError(
                    f"Unsafe path in prediction ZIP {archive.name}: {member.filename!r}"
                )
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise InvalidNBSubmissionError(
                    f"Symbolic links are not allowed in prediction ZIPs: {member.filename!r}"
                )
            resolved = (destination / member_path).resolve()
            try:
                resolved.relative_to(destination_root)
            except ValueError as exc:
                raise InvalidNBSubmissionError(
                    f"Unsafe path in prediction ZIP {archive.name}: {member.filename!r}"
                ) from exc
        handle.extractall(destination)


def _to_unit(value: Any, transform: str) -> float:
    if value is None:
        raise RuntimeError("NB evaluator omitted the selected primary metric")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise RuntimeError(
            f"NB evaluator returned a non-finite primary metric: {value!r}"
        )
    if transform == "map_minus1_1":
        # v4 chance-zero scale: clip(r, 0, 1) instead of (r + 1) / 2
        return float(np.clip(numeric, 0.0, 1.0))
    if transform == "percent":
        return float(np.clip(numeric / 100.0, 0.0, 1.0))
    return float(np.clip(numeric, 0.0, 1.0))


def _reduce_primary(results: dict[str, Any], instance_plan: list[dict[str, Any]]) -> float:
    if not isinstance(results, dict):
        raise RuntimeError("NB evaluator score.json must contain a JSON object")

    scores: list[float] = []
    for item in instance_plan:
        instance = str(item["instance"])
        entry = results.get(instance)
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"NB evaluator omitted or malformed result for instance {instance!r}"
            )
        error = entry.get("error")
        if error:
            error_text = str(error)
            if error_text.startswith("Validation:"):
                # A malformed or incomplete model output is not a scored
                # submission.  Propagate a dedicated exception so the outer
                # evaluator records Status=Fail and Score=null instead of
                # silently turning the invalid artifact into a successful 0.
                raise InvalidNBSubmissionError(
                    f"Invalid NB submission for instance {instance!r}: "
                    f"{error_text.removeprefix('Validation:').strip()}"
                )
            # Ground-truth loading, dependency, and evaluator implementation errors
            # must never be reported as successful model evaluations with score zero.
            raise RuntimeError(
                f"NB evaluator failed internally for instance {instance!r}: {error_text}"
            )
        metric_name = str(item["metric"])
        value = entry.get(metric_name)
        if value is None:
            normalized_keys = {
                str(key).lower().replace(" ", ""): candidate
                for key, candidate in entry.items()
            }
            value = normalized_keys.get(metric_name.lower().replace(" ", ""))
        if value is None:
            raise RuntimeError(
                f"NB evaluator omitted primary metric {metric_name!r} "
                f"for instance {instance!r}"
            )
        scores.append(_to_unit(value, str(item["transform"])))
    if not scores:
        raise RuntimeError("NB metric contract contains no instances")
    return float(np.mean(scores))


def _validate_ids(submission: pd.DataFrame, answers: pd.DataFrame) -> list[str]:
    if "id" not in submission.columns or "id" not in answers.columns:
        raise InvalidNBSubmissionError("submission.csv and answers.csv must contain an 'id' column")
    submission_ids = submission["id"].astype(str).tolist()
    answer_ids = answers["id"].astype(str).tolist()
    if submission_ids != answer_ids:
        raise InvalidNBSubmissionError(
            "submission ids must match answers ids exactly, including order"
        )
    if len(set(submission_ids)) != len(submission_ids):
        raise InvalidNBSubmissionError("submission ids must be unique")
    return submission_ids


def _validate_answer_paths(
    answers: pd.DataFrame,
    ids: list[str],
    private_dir: Path,
) -> None:
    """Verify that every answer row targets its canonical hidden instance directory."""
    if "answer_path" not in answers.columns:
        raise InvalidNBSubmissionError(
            "File-Manifest answers.csv must contain an 'answer_path' column"
        )

    private_root = private_dir.resolve()
    answers_root = (private_dir / "answers").resolve()
    for instance, value in zip(ids, answers["answer_path"].astype(str)):
        answer_path = Path(value).resolve()
        expected_path = (answers_root / instance).resolve()
        try:
            answer_path.relative_to(private_root)
        except ValueError as exc:
            raise InvalidNBSubmissionError(
                f"Answer path escapes the private task directory: {value!r}"
            ) from exc
        if answer_path != expected_path:
            raise InvalidNBSubmissionError(
                f"Answer path for {instance} must resolve to {expected_path}"
            )
        if not answer_path.is_dir() or answer_path.is_symlink():
            raise InvalidNBSubmissionError(
                f"Answer path is missing or is not a directory: {answer_path}"
            )
        payload_files = []
        for path in answer_path.rglob("*"):
            if path.is_symlink():
                raise InvalidNBSubmissionError(
                    f"Symbolic links are not allowed in hidden answers: {path}"
                )
            if path.is_file():
                payload_files.append(path)
        if not payload_files:
            raise InvalidNBSubmissionError(
                f"Answer directory for {instance} must contain at least one hidden file"
            )


def grade_from_contract(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    *,
    grade_path: str | Path,
    answers_path: str | Path,
    public_dir: str | Path,
    **_: Any,
) -> float:
    """Run the original NB evaluator after adapting the unified submission protocol."""
    task_dir = Path(grade_path).resolve().parent
    contract = _load_contract(task_dir)
    ids = _validate_ids(submission, answers)
    private_dir = Path(answers_path).resolve().parent
    if not contract.ROWWISE:
        _validate_answer_paths(answers, ids, private_dir)

    with tempfile.TemporaryDirectory(prefix="bioxarena_pro_nb_") as temp_dir:
        working_dir = Path(temp_dir)
        native_output = working_dir / "output"

        if contract.ROWWISE:
            rowwise = contract.ROWWISE
            expected_columns = ["id", *rowwise["pred_cols"]]
            if submission.columns.tolist() != expected_columns:
                raise InvalidNBSubmissionError(
                    "submission.csv columns must exactly match: "
                    + ", ".join(expected_columns)
                )
            instance = contract.INSTANCE_PLAN[0]["instance"]
            instance_dir = native_output / instance
            instance_dir.mkdir(parents=True, exist_ok=True)
            native_frame = submission[expected_columns].copy()
            native_frame.columns = [rowwise["id_col"], *rowwise["pred_cols"]]
            native_frame.to_csv(instance_dir / rowwise["output_file"], index=False)

            expected_answer_columns = ["id", *rowwise["pred_cols"]]
            if answers.columns.tolist() != expected_answer_columns:
                raise InvalidNBSubmissionError(
                    "answers.csv columns must exactly match: "
                    + ", ".join(expected_answer_columns)
                )
            native_ground_truth = working_dir / "ground_truth" / instance
            native_ground_truth.mkdir(parents=True, exist_ok=True)
            answer_frame = answers[expected_answer_columns].copy()
            answer_frame.columns = [
                rowwise["id_col"],
                *(f"labels{index}" for index in range(1, len(rowwise["pred_cols"]) + 1)),
            ]
            answer_frame.to_csv(native_ground_truth / "y_ref.csv", index=False)
            ground_truth_root = working_dir / "ground_truth"
        else:
            expected_columns = ["id", "prediction_file"]
            if submission.columns.tolist() != expected_columns:
                raise InvalidNBSubmissionError(
                    "File-Manifest submission.csv columns must exactly match: "
                    + ", ".join(expected_columns)
                )
            for instance, artifact in zip(ids, submission["prediction_file"].astype(str)):
                _safe_extract_zip(Path(artifact), native_output / instance)
            ground_truth_root = private_dir / "answers"

        evaluator_path = task_dir / "evaluator.py"
        if not evaluator_path.is_file():
            raise FileNotFoundError(f"Missing original NB evaluator: {evaluator_path}")
        env = dict(
            os.environ,
            OUTPUT_DIR=str(native_output),
            GROUND_TRUTH_DIR=str(ground_truth_root),
            DATA_DIR=str(Path(public_dir).resolve() / "data"),
        )
        process = subprocess.run(
            [sys.executable, str(evaluator_path)],
            cwd=working_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
            check=False,
        )
        score_path = working_dir / "score.json"
        if process.returncode != 0 or not score_path.is_file():
            raise RuntimeError(
                f"Original NB evaluator failed with exit code {process.returncode}. "
                f"stdout tail: {process.stdout[-1000:]} stderr tail: {process.stderr[-1000:]}"
            )
        results = json.loads(score_path.read_text(encoding="utf-8"))
        return _reduce_primary(results, list(contract.INSTANCE_PLAN))
