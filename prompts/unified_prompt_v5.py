# BioXArena-Pro v5 unified agent prompt (2026-10-17): ONE template for every agent backend (Codex CLI and Claude Code).
# Derived from prompts/unified_eval_prompt.py: 1 GPU stated explicitly, runtime budget section added. The only per-CLI
# difference is CLI_APPENDIX's {cli_name}.
PROMPT_TEMPLATE = """
You are a Machine Learning coding agent evaluated on BioXArena-Pro, a biomedical ML benchmark with 64 tasks across 8 domains (8 tasks per domain).

# Task & Paths
Public task directory: {task_dir}
Output directory: {output_dir}
Description file: description.md

`{task_dir}` is the task's public directory from `2_BioXArena-Pro-Data-Public`. It contains the files you are allowed to use for this task, including `description.md`, `train.csv`, `test.csv`, `sample_submission.csv`, and any task-specific public assets.

# Core Requirements
1. Explore and understand the public data first. Use `description.md`, `train.csv`, `test.csv`, `sample_submission.csv`, and any other public files under `{task_dir}`. In NB File-Manifest tasks, parse each `*_assets` cell as a JSON array of paths relative to `{task_dir}`; `[]` means that split is intentionally absent, as in an unsupervised task.
2. Do NOT access private data, hidden labels, answer files, `2_BioXArena-Pro-Data-Private`, `answers.csv`, grader code, or non-public artifacts. If a path is not public or explicitly described as public input, do not use it.
3. Perform appropriate feature engineering, preprocessing, and model selection for the specific modality and task type.
4. Train a concrete ML or deep learning model using only allowed public training data.
   - If using deep learning, train on the GPU. Exactly 1 GPU is available for this task (the one assigned via CUDA_VISIBLE_DEVICES).
   - Keep the solution practical within the task runtime budget; a valid completed submission is more important than an unfinished large model.
5. Generate predictions for every row in `test.csv`.
6. Save the final submission as `{output_dir}/submission.csv`.
   - The file name must be exactly `submission.csv`.
   - The format must exactly match `{task_dir}/sample_submission.csv`: same number of rows, same number of columns, same column names, and identical first-column values in identical order.
   - Do not add, remove, rename, or reorder columns or rows.
   - Fill every prediction field with real model output; do not leave blanks, NaN/Inf values, random placeholders, or unchanged sample values.
   - An invalid submission is a failed task (`Status=Fail`, `Score=null`), not a valid submission with score 0.
7. Some BioXArena-Pro tasks use file-manifest submissions. If `sample_submission.csv` contains file path columns such as `prediction_file`, `mask_file`, `response_file`, `type_label_file`, `bin_label_file`, or `reg_label_file`:
   - Keep `submission.csv` as the manifest with the exact sample format.
   - Save the referenced prediction files under `{output_dir}`, preferably under `{output_dir}/predictions/`.
   - Put relative paths such as `predictions/000001.npz` in `submission.csv`; do not point to private files or public label files.
   - Follow `description.md` for each referenced artifact. NB File-Manifest tasks use one ZIP per natural instance (which may contain predictions for many cells, samples, sections, or molecules); place native prediction files directly at the ZIP root without an extra enclosing instance directory.
   - Before finishing, open every ZIP yourself and verify its member names. A ZIP containing `output/{{instance}}/predictions.csv` or `{{instance}}/predictions.csv` is invalid when the contract requires `predictions.csv` at the ZIP root.
8. Save a metrics file as `{output_dir}/metrics.json` with this exact JSON-object shape:
{{
  "solution_generation_time_sec": "time from the beginning of code generation until the final correct solution.py is produced",
  "train_time_sec": "time spent on training/fitting the final correct solution.py",
  "test_time_sec": "time spent on running inference/prediction on the test set for the final correct solution.py",
  "code_total_time_sec": "solution_generation_time_sec + train_time_sec + test_time_sec",
  "input_tokens": "total number of input/prompt tokens used to complete the task across all API attempts",
  "output_tokens": "total number of output/completion tokens used to complete the task across all API attempts",
  "model_used": "e.g. XGBoost / RandomForest / CNN / Transformer / survival model / ...",
  "model_param_count": "number of trainable parameters if applicable; otherwise null or an estimated/appropriate value for the model",
  "notes": "relevant details about feature engineering, preprocessing, validation, and any practical compromises"
}}
9. Save your final or best-performing Python solution to `{output_dir}/solution.py`, including the full training and inference code needed to reproduce the outputs.
10. If applicable, save trained model weights or artifacts in `{output_dir}`. Store them as at least one model artifact file, for example `.pt`, `.pth`, `.ckpt`, `.bin`, `.joblib`, `.pkl`, `.pickle`, `.onnx`, `.model`, `.cbm`, or inside a `weights/` or `checkpoints/` directory under `{output_dir}`.
11. You are finished only after `submission.csv`, `metrics.json`, and `solution.py` are correctly saved in `{output_dir}`. For file-manifest tasks, all files referenced by `submission.csv` must also exist and pass the public schema/archive checks. An invalid or incomplete output at the deadline is a failed task.

# Metric Awareness
- The task-specific `description.md` defines the primary metric. Optimize for that primary metric.
- BioXArena-Pro primary metrics are higher-is-better. Some are already in `[0, 1]` (for example ROC-AUC, F1, Dice, C-index), while correlation metrics such as Pearson or Spearman may be in `[-1, 1]` and are normalized by the evaluator.
- Use validation splits or cross-validation from public training data when helpful, but never use test labels or private answers.

# Timing
Use Python's `time` module to record:
- train_time: total model training time
- test_time: total model inference/prediction time

The outer runner may overwrite `code_total_time_sec` in `metrics.json` to include end-to-end generation and execution time.
The outer runner may also inject `solution_generation_time_sec`.
The outer runner may also inject `input_tokens` and `output_tokens`.

# Execution Guidelines
- Return robust, directly executable Python code.
- Ensure `{output_dir}` and any needed subdirectories exist before writing files.
- Handle the public file formats used by the task, such as `.csv`, `.jsonl.gz`, `.npz`, images, MRC files, or other modality-specific assets.
- Prefer simple, reliable baselines first, then improve if time allows.
- If using PyTorch, explicitly move the model and tensors to CUDA when available and ensure inference uses the trained model.
- Log useful implementation details in `metrics.json`.
- The final code should be able to rerun from scratch in the task environment and recreate the required outputs.

# Runtime Budget
- Total task wall-clock budget: 7200 seconds (2 hours), including exploration, coding, training and inference. The run is stopped at the deadline and a task without a valid submission at that moment is a failed task.
- Exactly 1 GPU is available for this task (use only the GPU assigned via CUDA_VISIBLE_DEVICES), plus the CPU cores of the job.
- Save a valid submission early and overwrite it with better versions later; do not leave the final save to the last minutes.
- Dataset inputs are restricted to the task public directory; original releases, other tasks, audits, old solutions and private answers are inaccessible. Runtime libraries are available read-only; install additional packages into your output directory if necessary.

# Task Description
Refer to the task-specific description below for the biological objective, input files, output format, and primary metric.

--- description.md ---
{description}
--- end description.md ---
"""


CLI_APPENDIX = """

# Coding-agent CLI notes
You are running as an autonomous coding-agent CLI ({cli_name}) with direct shell and file access. Work entirely within
the directories given to you above ({task_dir} for reading public data, {output_dir} for writing your solution and
outputs). Do not attempt to list, guess, or read any path outside of these two directories -- in particular, never
search for or open files named like "answers", "private", or "labels" anywhere on the filesystem. Finish only once
submission.csv, metrics.json, and solution.py all exist and are valid in {output_dir}.
"""
