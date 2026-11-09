"""Public-only validation for BioXArena-Pro training outputs.

The checks in this module intentionally use only ``sample_submission.csv`` and
other public task assets.  They catch malformed manifests and artifacts before
an agent run is marked successful, without importing private labels or grader
data.
"""

from __future__ import annotations

import csv
import io
import itertools
import math
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable


class SubmissionPreflightError(ValueError):
    """Raised when a generated submission violates its public contract."""


_PATH_COLUMN_SUFFIXES = ("_file", "_path")
_CLASSIFICATION_KANO_INSTANCES = {
    "bace", "bbbp", "clintox", "hiv", "muv", "sider", "tox21", "toxcast"
}
_PEPTIDE_RESIDUES = set("ARNDCEQGHILKMFPSTWYV")
_PEPTIDE_MOD_ALIASES = {
    "ox", "oxidation", "cam", "carbamidomethyl",
    "carbamidomethylation", "deamidation", "deamidated",
}


def _is_path_column(name: str) -> bool:
    return name.lower().endswith(_PATH_COLUMN_SUFFIXES)


def _finite_number(value: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SubmissionPreflightError(f"prediction value is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise SubmissionPreflightError(f"prediction value is not finite: {value!r}")
    return number


def _read_header(
    reader: csv.reader,
    path: Path,
    *,
    allow_unnamed_index: bool = False,
) -> list[str]:
    try:
        header = next(reader)
    except StopIteration as exc:
        raise SubmissionPreflightError(f"CSV is empty: {path}") from exc
    checked_fields = header[1:] if allow_unnamed_index else header
    if not header or any(not name for name in checked_fields):
        raise SubmissionPreflightError(f"CSV has an empty header field: {path}")
    return header


def _validate_submission_csv(
    sample_path: Path,
    submission_path: Path,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Stream-compare public schema/IDs and collect referenced artifacts."""
    if not sample_path.is_file():
        raise RuntimeError(f"benchmark is missing sample_submission.csv: {sample_path}")
    if not submission_path.is_file():
        raise SubmissionPreflightError(f"submission.csv is missing: {submission_path}")

    artifacts: list[tuple[str, str, str]] = []
    with sample_path.open(newline="", encoding="utf-8-sig") as sample_handle, \
            submission_path.open(newline="", encoding="utf-8-sig") as submission_handle:
        sample_reader = csv.reader(sample_handle)
        submission_reader = csv.reader(submission_handle)
        expected_header = _read_header(sample_reader, sample_path)
        actual_header = _read_header(submission_reader, submission_path)
        if actual_header != expected_header:
            raise SubmissionPreflightError(
                "submission.csv columns must exactly match sample_submission.csv; "
                f"expected {expected_header}, got {actual_header}"
            )

        path_indices = [
            index for index, name in enumerate(expected_header) if _is_path_column(name)
        ]
        seen_manifest_ids: set[str] = set()
        for row_number, pair in enumerate(
            itertools.zip_longest(sample_reader, submission_reader), start=2
        ):
            expected_row, actual_row = pair
            if expected_row is None or actual_row is None:
                raise SubmissionPreflightError(
                    "submission.csv row count must exactly match sample_submission.csv"
                )
            if len(expected_row) != len(expected_header):
                raise RuntimeError(
                    f"benchmark sample row {row_number} has {len(expected_row)} fields; "
                    f"expected {len(expected_header)}"
                )
            if len(actual_row) != len(actual_header):
                raise SubmissionPreflightError(
                    f"submission.csv row {row_number} has {len(actual_row)} fields; "
                    f"expected {len(actual_header)}"
                )
            if actual_row[0] != expected_row[0]:
                raise SubmissionPreflightError(
                    f"submission.csv id/order mismatch at row {row_number}: "
                    f"expected {expected_row[0]!r}, got {actual_row[0]!r}"
                )

            instance = actual_row[0]
            if path_indices:
                if instance in seen_manifest_ids:
                    raise SubmissionPreflightError(
                        f"submission.csv contains duplicate id {instance!r}"
                    )
                seen_manifest_ids.add(instance)

            for index, (expected, actual) in enumerate(zip(expected_row[1:], actual_row[1:]), start=1):
                column = expected_header[index]
                if index in path_indices:
                    if not actual.strip():
                        raise SubmissionPreflightError(
                            f"artifact path is blank for id {instance!r}, column {column!r}"
                        )
                    artifacts.append((instance, column, actual.strip()))
                    continue
                if not actual.strip():
                    raise SubmissionPreflightError(
                        f"prediction is blank at row {row_number}, column {column!r}"
                    )
                # A numeric sample placeholder identifies a numeric prediction
                # column.  Text-label tasks remain valid and are schema-checked
                # by their task grader.
                try:
                    float(expected)
                except (TypeError, ValueError):
                    continue
                _finite_number(actual)

    return expected_header, artifacts


def _resolve_artifact(output_dir: Path, raw_path: str) -> Path:
    relative = Path(raw_path.replace("\\", "/"))
    if relative.is_absolute():
        raise SubmissionPreflightError(
            f"artifact paths in submission.csv must be relative: {raw_path!r}"
        )
    output_root = output_dir.resolve()
    candidate = (output_dir / relative).resolve()
    try:
        candidate.relative_to(output_root)
    except ValueError as exc:
        raise SubmissionPreflightError(
            f"artifact path escapes the task output directory: {raw_path!r}"
        ) from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise SubmissionPreflightError(
            f"referenced artifact is missing or is not a regular file: {raw_path!r}"
        )
    return candidate


def _zip_file_members(archive: Path, instance: str) -> list[str]:
    try:
        handle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SubmissionPreflightError(f"invalid ZIP artifact: {archive}") from exc

    with handle:
        members = handle.infolist()
        if not members:
            raise SubmissionPreflightError(f"prediction ZIP is empty: {archive.name}")
        seen: set[str] = set()
        files: list[str] = []
        for member in members:
            if "\\" in member.filename:
                raise SubmissionPreflightError(
                    f"ZIP member must use POSIX separators: {member.filename!r}"
                )
            normalized = member.filename.rstrip("/")
            if not normalized or normalized in seen:
                raise SubmissionPreflightError(
                    f"empty or duplicate ZIP member: {member.filename!r}"
                )
            seen.add(normalized)
            if member.flag_bits & 0x1:
                raise SubmissionPreflightError(
                    f"encrypted ZIP members are not supported: {member.filename!r}"
                )
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise SubmissionPreflightError(
                    f"unsafe path in prediction ZIP: {member.filename!r}"
                )
            if stat.S_ISLNK(member.external_attr >> 16):
                raise SubmissionPreflightError(
                    f"symbolic links are not allowed in prediction ZIPs: {member.filename!r}"
                )
            if not member.is_dir():
                files.append(normalized)

        if not files:
            raise SubmissionPreflightError(f"prediction ZIP contains no files: {archive.name}")
        first_parts = [PurePosixPath(name).parts[0] for name in files]
        if all(part == "output" for part in first_parts):
            raise SubmissionPreflightError(
                "prediction ZIP has an extra 'output/' wrapper; native files must be at ZIP root"
            )
        if instance and all(part == instance for part in first_parts):
            raise SubmissionPreflightError(
                f"prediction ZIP has an extra {instance!r} wrapper; native files must be at ZIP root"
            )
        return files


def _require_root_members(archive: Path, files: Iterable[str], names: Iterable[str]) -> None:
    present = set(files)
    missing = [name for name in names if name not in present]
    if missing:
        raise SubmissionPreflightError(
            f"{archive.name} is missing required ZIP-root file(s): {', '.join(missing)}"
        )


def _csv_rows(
    path: Path,
    *,
    allow_unnamed_index: bool = False,
) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        return _read_header(
            reader,
            path,
            allow_unnamed_index=allow_unnamed_index,
        ), list(reader)


def _archive_csv_rows(archive: Path, member: str) -> tuple[list[str], list[list[str]]]:
    try:
        with zipfile.ZipFile(archive) as handle, handle.open(member) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
            reader = csv.reader(text)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise SubmissionPreflightError(
                    f"{archive.name}:{member} is empty"
                ) from exc
            return header, list(reader)
    except KeyError as exc:
        raise SubmissionPreflightError(
            f"{archive.name} is missing ZIP-root file {member!r}"
        ) from exc


def _validate_kano_artifact(task_dir: Path, archive: Path, instance: str) -> None:
    train_path = task_dir / "data" / instance / "train.csv"
    test_path = task_dir / "data" / instance / "x_test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise RuntimeError(f"KANO public assets are incomplete for {instance!r}")
    train_header, _ = _csv_rows(train_path)
    test_header, test_rows = _csv_rows(test_path)
    if not test_header:
        raise RuntimeError(f"KANO x_test.csv has no header for {instance!r}")
    targets = [name for name in train_header if name.lower() != "smiles"]
    header, rows = _archive_csv_rows(archive, "predictions.csv")
    if header != targets:
        raise SubmissionPreflightError(
            f"KANO {instance} predictions.csv columns must be {targets}, got {header}"
        )
    if len(rows) != len(test_rows):
        raise SubmissionPreflightError(
            f"KANO {instance} predictions.csv must have {len(test_rows)} rows, got {len(rows)}"
        )
    is_classification = instance in _CLASSIFICATION_KANO_INSTANCES
    for row_number, row in enumerate(rows, start=2):
        if len(row) != len(targets):
            raise SubmissionPreflightError(
                f"KANO {instance} predictions.csv row {row_number} has the wrong width"
            )
        for value in row:
            number = _finite_number(value)
            if is_classification and not 0.0 <= number <= 1.0:
                raise SubmissionPreflightError(
                    f"KANO {instance} classification probabilities must be in [0,1]"
                )


def _public_index(path: Path) -> tuple[str, list[str]]:
    # Pandas writes an unnamed first field when an index has no name.  Several
    # published NatureBench assets use that representation, and the adjacent
    # evaluator intentionally reads them with ``index_col=0``.  Accept only the
    # first field being unnamed; feature/data column names remain strict.
    header, rows = _csv_rows(path, allow_unnamed_index=True)
    if not header:
        raise RuntimeError(f"public test CSV has no columns: {path}")
    return header[0], [row[0] for row in rows]


def _validate_indexed_probabilities(
    archive: Path,
    member: str,
    expected_index: list[str],
    expected_columns: list[str],
) -> None:
    header, rows = _archive_csv_rows(archive, member)
    if len(header) < 2 or header[1:] != expected_columns:
        raise SubmissionPreflightError(
            f"{archive.name}:{member} prediction columns must be {expected_columns}, "
            f"got {header[1:]}"
        )
    if len(rows) != len(expected_index):
        raise SubmissionPreflightError(
            f"{archive.name}:{member} must have {len(expected_index)} rows, got {len(rows)}"
        )
    for row_number, (row, expected_id) in enumerate(zip(rows, expected_index), start=2):
        if len(row) != len(header) or row[0] != expected_id:
            raise SubmissionPreflightError(
                f"{archive.name}:{member} index/order mismatch at row {row_number}"
            )
        for value in row[1:]:
            number = _finite_number(value)
            if not 0.0 <= number <= 1.0:
                raise SubmissionPreflightError(
                    f"{archive.name}:{member} probabilities must be in [0,1]"
                )


def _validate_pdtc_artifact(task_dir: Path, archive: Path, instance: str) -> None:
    data_dir = task_dir / "data" / instance
    if instance == "pdtc":
        _, expected_index = _public_index(data_dir / "x_test.csv")
        mapping_header, mapping_rows = _csv_rows(
            data_dir / "pdtc_gdsc_drug_mapping.csv",
            allow_unnamed_index=True,
        )
        try:
            drug_index = mapping_header.index("gdsc_name")
        except ValueError as exc:
            raise RuntimeError("PDTC mapping is missing the public gdsc_name column") from exc
        drugs = [row[drug_index] for row in mapping_rows]
        _validate_indexed_probabilities(
            archive, "predictions.csv", expected_index, drugs
        )
        return

    drug_codes = {
        "clinical_diagnosis": ("gem", "fu"),
        "clinical_relapse": ("tem", "gem", "cis", "sor", "fu"),
    }[instance]
    for code in drug_codes:
        _, expected_index = _public_index(data_dir / f"x_test_{code}.csv")
        _validate_indexed_probabilities(
            archive, f"predictions_{code}.csv", expected_index, ["score"]
        )


def _valid_modification(text: str) -> bool:
    normalized = text.strip().lower().replace("da", "")
    return normalized in _PEPTIDE_MOD_ALIASES or bool(
        re.search(r"[-+]?\d+(?:\.\d+)?", normalized)
    )


def _validate_peptide_sequence(sequence: str) -> None:
    text = sequence.strip().strip(".")
    index = 0
    while index < len(text):
        char = text[index]
        if char in ".-_ ":
            index += 1
            continue
        if char.upper() not in _PEPTIDE_RESIDUES:
            raise SubmissionPreflightError(f"unsupported peptide residue: {char!r}")
        index += 1
        while index < len(text) and text[index] in "([":
            closing = ")" if text[index] == "(" else "]"
            end = text.find(closing, index + 1)
            if end < 0 or not _valid_modification(text[index + 1:end]):
                raise SubmissionPreflightError(
                    f"invalid peptide modification in sequence {sequence!r}"
                )
            index = end + 1


def _grader_peptide_masses():
    # the official grader's notation parser, vendored (the agent sandbox cannot read tasks/)
    import sys as _sys
    here = str(Path(__file__).resolve().parent)
    if here not in _sys.path:
        _sys.path.insert(0, here)
    from peptide_v2_format import _peptide_v2_masses
    return _peptide_v2_masses


def _validate_peptide_artifact(archive: Path) -> None:
    try:
        with zipfile.ZipFile(archive) as handle, handle.open("predictions.csv") as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
            reader = csv.reader(text)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise SubmissionPreflightError(
                    f"{archive.name}:predictions.csv is empty"
                ) from exc
            # v5.1 format (description.md): columns spectrum_id,sequence; one row per test spectrum, any order
            if header != ["spectrum_id", "sequence"]:
                raise SubmissionPreflightError(
                    f"{archive.name}:predictions.csv columns must be ['spectrum_id', 'sequence']"
                )
            seen = set()
            peptide_masses = _grader_peptide_masses()
            for row in reader:
                if len(row) != 2 or not row[0] or row[0] in seen:
                    raise SubmissionPreflightError(
                        f"{archive.name}:predictions.csv needs one row per spectrum_id (missing, empty or duplicated id)"
                    )
                seen.add(row[0])
                try:  # same parser as the official grader (vendored in peptide_v2_format.py)
                    peptide_masses(row[1])
                except Exception as exc:
                    raise SubmissionPreflightError(f"{archive.name}: invalid sequence {row[1]!r}: {exc}") from exc
    except KeyError as exc:
        raise SubmissionPreflightError(
            f"{archive.name} is missing ZIP-root file 'predictions.csv'"
        ) from exc


def _required_root_members(task_key: str, instance: str) -> tuple[str, ...]:
    if task_key in {
        "chemical-biology/kano-molecular-property-prediction",
        "sequence-genomics/de-novo-peptide-sequencing-from-tandem-mass-spectrometry",
        "sequence-genomics/deepstarr-genomic-sequence",
    }:
        return ("predictions.csv",)
    if task_key == "imaging/microscopy-instance-segmentation":
        return ("predictions.npz",)
    if task_key == "perturbation/pdtc-nb-clinical-drug-response":
        if instance == "pdtc":
            return ("predictions.csv",)
        if instance == "clinical_diagnosis":
            return ("predictions_gem.csv", "predictions_fu.csv")
        if instance == "clinical_relapse":
            return tuple(
                f"predictions_{code}.csv" for code in ("tem", "gem", "cis", "sor", "fu")
            )
    return ()


def _validate_artifact(
    task_dir: Path,
    output_dir: Path,
    task_key: str,
    instance: str,
    raw_path: str,
) -> None:
    archive = _resolve_artifact(output_dir, raw_path)
    if archive.suffix.lower() != ".zip":
        return
    files = _zip_file_members(archive, instance)
    required = _required_root_members(task_key, instance)
    if required:
        _require_root_members(archive, files, required)

    if task_key == "chemical-biology/kano-molecular-property-prediction":
        _validate_kano_artifact(task_dir, archive, instance)
    elif task_key == "perturbation/pdtc-nb-clinical-drug-response":
        _validate_pdtc_artifact(task_dir, archive, instance)
    elif task_key == "sequence-genomics/de-novo-peptide-sequencing-from-tandem-mass-spectrometry":
        _validate_peptide_artifact(archive)


def validate_public_submission(task_dir: Path, output_dir: Path, task_key: str) -> None:
    """Validate generated outputs using public information only.

    Raises ``SubmissionPreflightError`` for model-output problems.  A caller
    should mark that run failed and, when retries are available, include the
    message in the repair prompt.
    """
    task_dir = Path(task_dir)
    output_dir = Path(output_dir)
    _, artifacts = _validate_submission_csv(
        task_dir / "sample_submission.csv",
        output_dir / "submission.csv",
    )
    for instance, _column, raw_path in artifacts:
        _validate_artifact(task_dir, output_dir, task_key, instance, raw_path)
