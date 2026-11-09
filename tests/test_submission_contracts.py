from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "training"))

from tasks.native_manifest_metrics import _peptide_masses, _peptide_match
from tasks.nb_grading import InvalidNBSubmissionError, _reduce_primary
from submission_preflight import SubmissionPreflightError, validate_public_submission


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InvalidSubmissionPolicyTests(unittest.TestCase):
    def test_central_nb_validation_error_is_not_zero_score(self) -> None:
        results = {"case": {"metric": None, "error": "Validation: missing output"}}
        plan = [{"instance": "case", "metric": "metric", "transform": "identity"}]
        with self.assertRaisesRegex(InvalidNBSubmissionError, "missing output"):
            _reduce_primary(results, plan)

    def test_every_generated_contract_rejects_validation_errors(self) -> None:
        contract_paths = sorted((PROJECT_ROOT / "tasks").glob("*/*/contract.py"))
        self.assertGreater(len(contract_paths), 0)
        for index, path in enumerate(contract_paths):
            module = _load_module(path, f"contract_test_{index}")
            first = str(module.INSTANCE_PLAN[0]["instance"])
            with self.subTest(contract=str(path)):
                with self.assertRaisesRegex(ValueError, "Invalid submission"):
                    module._reduce({first: {"error": "Validation: bad archive"}})


class PeptideParserTests(unittest.TestCase):
    def test_multiple_consecutive_modifications_are_summed(self) -> None:
        sequence = ".C(+57.02)[-17.0265]AHFE."
        masses = _peptide_masses(sequence)
        self.assertEqual(len(masses), 5)
        self.assertAlmostEqual(masses[0], 103.00919 + 57.02 - 17.0265, places=6)
        self.assertTrue(_peptide_match(sequence, sequence))


class PublicPreflightTests(unittest.TestCase):
    def _manifest_fixture(self, root: Path, artifact_name: str = "predictions/case.zip"):
        task_dir = root / "task"
        output_dir = root / "output"
        task_dir.mkdir()
        (output_dir / "predictions").mkdir(parents=True)
        (task_dir / "sample_submission.csv").write_text(
            "id,prediction_file\ncase,predictions/case.zip\n", encoding="utf-8"
        )
        (output_dir / "submission.csv").write_text(
            f"id,prediction_file\ncase,{artifact_name}\n", encoding="utf-8"
        )
        return task_dir, output_dir, output_dir / artifact_name

    def test_rejects_extra_output_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir, output_dir, archive = self._manifest_fixture(Path(tmp))
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("output/case/predictions.csv", "value\n0\n")
            with self.assertRaisesRegex(SubmissionPreflightError, "extra 'output/' wrapper"):
                validate_public_submission(task_dir, output_dir, "example/task")

    def test_accepts_root_member_for_generic_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir, output_dir, archive = self._manifest_fixture(Path(tmp))
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("predictions.csv", "value\n0\n")
            validate_public_submission(task_dir, output_dir, "example/task")

    def test_kano_rejects_nan_predictions_using_public_assets_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir, output_dir, archive = self._manifest_fixture(root)
            data_dir = task_dir / "data" / "case"
            data_dir.mkdir(parents=True)
            (data_dir / "train.csv").write_text("smiles,target\nC,1\n", encoding="utf-8")
            (data_dir / "x_test.csv").write_text("smiles\nC\n", encoding="utf-8")
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("predictions.csv", "target\nNaN\n")
            with self.assertRaisesRegex(SubmissionPreflightError, "not finite"):
                validate_public_submission(
                    task_dir,
                    output_dir,
                    "chemical-biology/kano-molecular-property-prediction",
                )

    def test_peptide_preflight_accepts_consecutive_modifications(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir, output_dir, archive = self._manifest_fixture(Path(tmp))
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(
                    "predictions.csv",
                    "index,sequence\n0,C(+57.02)[-17.0265]AH\n",
                )
            validate_public_submission(
                task_dir,
                output_dir,
                "sequence-genomics/de-novo-peptide-sequencing-from-tandem-mass-spectrometry",
            )

    def test_pdtc_requires_every_published_clinical_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            output_dir = root / "output"
            task_dir.mkdir()
            output_dir.mkdir()
            (task_dir / "sample_submission.csv").write_text(
                "id,prediction_file\n"
                "clinical_diagnosis,diagnosis.zip\n",
                encoding="utf-8",
            )
            (output_dir / "submission.csv").write_text(
                "id,prediction_file\n"
                "clinical_diagnosis,diagnosis.zip\n",
                encoding="utf-8",
            )
            with zipfile.ZipFile(output_dir / "diagnosis.zip", "w") as handle:
                handle.writestr("predictions_gem.csv", "id,score\na,0.5\n")
            with self.assertRaisesRegex(SubmissionPreflightError, "predictions_fu.csv"):
                validate_public_submission(
                    task_dir,
                    output_dir,
                    "perturbation/pdtc-nb-clinical-drug-response",
                )

    def _pdtc_fixture(self, root: Path, prediction_column: str) -> tuple[Path, Path]:
        task_dir = root / "task"
        output_dir = root / "output"
        data_dir = task_dir / "data" / "pdtc"
        data_dir.mkdir(parents=True)
        output_dir.mkdir()
        (task_dir / "sample_submission.csv").write_text(
            "id,prediction_file\npdtc,pdtc.zip\n",
            encoding="utf-8",
        )
        (output_dir / "submission.csv").write_text(
            "id,prediction_file\npdtc,pdtc.zip\n",
            encoding="utf-8",
        )
        (data_dir / "x_test.csv").write_text(
            ",gene_a\nsample_a,1.0\nsample_b,2.0\n",
            encoding="utf-8",
        )
        (data_dir / "pdtc_gdsc_drug_mapping.csv").write_text(
            ",gdsc_name,drug_name,pdtc_count\n"
            "wee,Wee1 Inhibitor,wee,2\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(output_dir / "pdtc.zip", "w") as handle:
            handle.writestr(
                "predictions.csv",
                f"Sample,{prediction_column}\nsample_a,0.25\nsample_b,0.75\n",
            )
        return task_dir, output_dir

    def test_pdtc_accepts_unnamed_public_index_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir, output_dir = self._pdtc_fixture(
                Path(tmp),
                prediction_column="Wee1 Inhibitor",
            )
            validate_public_submission(
                task_dir,
                output_dir,
                "perturbation/pdtc-nb-clinical-drug-response",
            )

    def test_pdtc_still_rejects_short_drug_code_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir, output_dir = self._pdtc_fixture(
                Path(tmp),
                prediction_column="wee",
            )
            with self.assertRaisesRegex(SubmissionPreflightError, "prediction columns"):
                validate_public_submission(
                    task_dir,
                    output_dir,
                    "perturbation/pdtc-nb-clinical-drug-response",
                )


if __name__ == "__main__":
    unittest.main()
