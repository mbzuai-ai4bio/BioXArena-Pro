#!/usr/bin/env python3
"""Validate BioXArena-Pro public/private/code contracts without model outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
DEFAULT_PUBLIC_ROOT = WORKSPACE_ROOT / "2_BioXArena-Pro-Data-Public"
DEFAULT_PRIVATE_ROOT = WORKSPACE_ROOT / "2_BioXArena-Pro-Data-Private"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark_layout import (  # noqa: E402
    PARTITION_DOMAINS,
    TASK_KEY_SET,
    layout_issues,
    partition_task_keys,
)


def _rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _task_set(root: Path, leaf: str) -> set[str]:
    return {
        f"{path.parents[1].name}/{path.parent.name}"
        for path in root.glob(f"*/*/{leaf}")
    }


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _check_assets(
    public_dir: Path,
    rows: list[dict[str, str]],
    issues: list[str],
) -> None:
    for row in rows:
        for column, raw in row.items():
            if not column.endswith("_assets"):
                continue
            try:
                values = json.loads(raw)
            except json.JSONDecodeError as exc:
                issues.append(f"{public_dir}: invalid JSON in {column}: {exc}")
                continue
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                issues.append(f"{public_dir}: {column} must be a JSON string array")
                continue
            for value in values:
                candidate = (public_dir / value).resolve()
                if not _inside(public_dir, candidate):
                    issues.append(f"{public_dir}: asset escapes public directory: {value}")
                elif not candidate.exists():
                    issues.append(f"{public_dir}: missing public asset: {value}")


def _check_deepbacs(public_dir: Path, private_dir: Path, issues: list[str]) -> None:
    _, test_rows = _rows(public_dir / "test.csv")
    row = next((item for item in test_rows if item["id"] == "deepbacs"), None)
    if row is None:
        issues.append("microscopy: test.csv is missing deepbacs")
        return
    public_stems = {
        Path(value).stem for value in json.loads(row["test_assets"])
    }
    answer_dir = private_dir / "answers" / "deepbacs"
    hidden_stems = {
        path.stem for path in answer_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff", ".npy"}
    }
    if public_stems != hidden_stems:
        issues.append(
            "microscopy/deepbacs public test stems do not match hidden answer stems: "
            f"public_only={sorted(public_stems-hidden_stems)}, "
            f"private_only={sorted(hidden_stems-public_stems)}"
        )


def validate(public_root: Path, private_root: Path, code_root: Path) -> list[str]:
    issues: list[str] = []
    public_tasks = _task_set(public_root, "public")
    private_tasks = _task_set(private_root, "private")
    code_tasks = {
        f"{path.parents[1].name}/{path.parent.name}"
        for path in (code_root / "tasks").glob("*/*/grade.py")
    }
    issues.extend(layout_issues(public_tasks, "public task layout"))
    issues.extend(layout_issues(private_tasks, "private task layout"))
    issues.extend(layout_issues(code_tasks, "code task layout"))
    if public_tasks != private_tasks:
        issues.append(
            f"public/private task-set mismatch: public_only={sorted(public_tasks-private_tasks)}, "
            f"private_only={sorted(private_tasks-public_tasks)}"
        )
    if public_tasks != code_tasks:
        issues.append(
            f"public/code task-set mismatch: public_only={sorted(public_tasks-code_tasks)}, "
            f"code_only={sorted(code_tasks-public_tasks)}"
        )

    partition_union: set[str] = set()
    for part in sorted(PARTITION_DOMAINS):
        keys = partition_task_keys(part)
        if len(keys) != 16 or len(set(keys)) != 16:
            issues.append(f"training partition {part} is not 16 unique tasks")
        overlap = partition_union.intersection(keys)
        if overlap:
            issues.append(f"training partition {part} overlaps earlier parts: {sorted(overlap)}")
        partition_union.update(keys)
    if partition_union != TASK_KEY_SET:
        issues.append(
            "training partitions do not cover the canonical task set: "
            f"missing={sorted(TASK_KEY_SET-partition_union)}, "
            f"extra={sorted(partition_union-TASK_KEY_SET)}"
        )

    training_dir = code_root / "training"
    slurm_scripts = sorted(training_dir.glob("Slurm_run_*_part[1-4].sh"))
    if not slurm_scripts:
        issues.append("no partitioned training Slurm scripts were found")
    for script in slurm_scripts:
        match = re.search(r"_part([1-4])\.sh$", script.name)
        if match is None:
            issues.append(f"cannot determine partition from training script: {script.name}")
            continue
        part = int(match.group(1))
        marker = f'benchmark_layout.py" --part {part}'
        text = script.read_text(encoding="utf-8")
        if marker not in text:
            issues.append(
                f"{script.name}: does not load canonical training partition {part}"
            )

    invalid_policy_text = "fails the task with `Status=Fail` and `Score=null`"
    for task_key in sorted(public_tasks & private_tasks & code_tasks):
        domain, task = task_key.split("/", 1)
        public_dir = public_root / domain / task / "public"
        private_dir = private_root / domain / task / "private"
        required = [
            public_dir / "description.md",
            public_dir / "train.csv",
            public_dir / "test.csv",
            public_dir / "sample_submission.csv",
            private_dir / "answers.csv",
            code_root / "tasks" / domain / task / "grade.py",
        ]
        for path in required:
            if not path.is_file():
                issues.append(f"{task_key}: missing required file: {path}")
        if any(not path.is_file() for path in required):
            continue

        sample_header, sample_rows = _rows(public_dir / "sample_submission.csv")
        _, test_rows = _rows(public_dir / "test.csv")
        _, train_rows = _rows(public_dir / "train.csv")
        _, answer_rows = _rows(private_dir / "answers.csv")
        sample_ids = [row[sample_header[0]] for row in sample_rows]
        test_ids = [row["id"] for row in test_rows]
        answer_ids = [row["id"] for row in answer_rows]
        if sample_ids != test_ids:
            issues.append(f"{task_key}: sample_submission IDs/order differ from test.csv")
        if answer_ids != test_ids:
            issues.append(f"{task_key}: answers IDs/order differ from test.csv")
        if len(set(test_ids)) != len(test_ids):
            issues.append(f"{task_key}: duplicate test IDs")
        _check_assets(public_dir, train_rows + test_rows, issues)

        for row in answer_rows:
            for column, raw in row.items():
                if not column.endswith(("_path", "_file")):
                    continue
                candidate = Path(raw)
                if not candidate.is_absolute():
                    candidate = private_dir / candidate
                if not _inside(private_dir, candidate):
                    issues.append(f"{task_key}: answer path escapes private directory: {raw}")
                elif not candidate.exists():
                    issues.append(f"{task_key}: missing private answer artifact: {raw}")

        path_columns = [name for name in sample_header if name.endswith(("_file", "_path"))]
        if path_columns:
            description = (public_dir / "description.md").read_text(encoding="utf-8")
            if invalid_policy_text not in description:
                issues.append(f"{task_key}: description lacks the Fail/null invalid policy")

    microscopy_public = public_root / "imaging" / "microscopy-instance-segmentation" / "public"
    microscopy_private = private_root / "imaging" / "microscopy-instance-segmentation" / "private"
    if microscopy_public.is_dir() and microscopy_private.is_dir():
        _check_deepbacs(microscopy_public, microscopy_private, issues)
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--code-root", type=Path, default=CODE_ROOT)
    args = parser.parse_args()
    issues = validate(args.public_root, args.private_root, args.code_root)
    if issues:
        print(f"FAIL: {len(issues)} benchmark integrity issue(s)")
        for issue in issues:
            print(f"- {issue}")
        return 1
    task_count = len(_task_set(args.public_root, "public"))
    print(
        f"PASS: canonical 8-domain layout; {task_count} public/private/code tasks "
        "and all training partitions are structurally consistent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
