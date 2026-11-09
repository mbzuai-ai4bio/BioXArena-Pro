#!/usr/bin/env python3
"""Codex CLI (`codex exec`, we used 0.163.0) for the unified harness (called by harness/run_agent.py).

Codex CLI is a general-purpose coding-agent CLI: it owns its own tool-use loop (shell exec, file read/write, patches) and
its own retry/repair behaviour. This runner only (1) stages the task (public task dir + output dir), (2) invokes
`codex exec` non-interactively against an OpenAI-compatible endpoint (the harness's model_tap, which forwards to
bx_config.ROUTES["openai"], e.g. OpenRouter or any Responses/Chat-Completions-compatible server) with a hard wall-clock kill,
and (3) collects submission.csv / metrics.json / solution.py, the JSONL transcript and the CLI's own token usage.
Codex is pointed at the endpoint through an isolated CODEX_HOME (<output>/.runtime/cli) holding a small config.toml with a
custom model_provider; the key reaches Codex only through the environment variable BX_LLM_API_KEY. The operator's personal
~/.codex is never touched. Wire protocol: bx_config.CODEX_WIRE_API ("responses" by default, as in all BioXArena-Pro runs).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths (mirror the Biomni/STELLA runner layout)
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
TRAINING_DIR = SCRIPT_PATH.parent
EVAL_ROOT = TRAINING_DIR.parents[1]  # 2_BioXArena-Pro-Code/
WORKSPACE_ROOT = EVAL_ROOT.parent
DEFAULT_TASKS_ROOT = EVAL_ROOT / "tasks"
DEFAULT_PROMPT_PATH = EVAL_ROOT / "prompts" / "unified_prompt_v5.py"
DEFAULT_ROUND_NAME = "round1"

KEY_ENV = "BX_LLM_API_KEY"  # the only way the key reaches the CLI (set per subprocess by run_single_task)
DEFAULT_MODEL = "gpt-6-astra"

REQUIRED_OUTPUT_LABELS = ["submission.csv", "metrics.json", "solution.py"]

CODEX_APPENDIX = """

# Coding-agent CLI notes
You are running as an autonomous coding-agent CLI (Codex) with direct shell and file
access. Work entirely within the directories given to you above ({task_dir} for
reading public data, {output_dir} for writing your solution and outputs). Do not
attempt to list, guess, or read any path outside of these two directories -- in
particular, never search for or open files named like "answers", "private", or
"labels" anywhere on the filesystem. Finish only once submission.csv, metrics.json,
and solution.py all exist and are valid in {output_dir}.
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
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
    input_tokens: int = 0
    output_tokens: int = 0
    exit_code: int | None = None


# ---------------------------------------------------------------------------
# Task discovery (identical contract to run_biomni_agent.py / run_stella_agent.py)
# ---------------------------------------------------------------------------
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
    selected: dict[str, TaskSpec] = {}
    for t in all_tasks:
        if t.domain in selected_domains:
            selected[t.key] = t
        if t.key in selected_tasks:
            selected[t.key] = t
    return [selected[k] for k in sorted(selected)]


# ---------------------------------------------------------------------------
# Prompt builder (reuses the same unified_prompt_v5.py template as every other
# agent in this project, plus a short CLI-specific isolation reminder appendix)
# ---------------------------------------------------------------------------
def load_prompt_template(prompt_path: Path) -> str:
    ns: dict[str, Any] = {}
    exec(prompt_path.read_text(encoding="utf-8"), ns)
    return ns["PROMPT_TEMPLATE"]


def build_prompt(prompt_template: str, task_spec: TaskSpec) -> str:
    description = task_spec.description_path.read_text(encoding="utf-8")
    base = prompt_template.format(
        task_dir=str(task_spec.task_dir),
        output_dir=str(task_spec.output_dir),
        description=description,
    )
    appendix = CODEX_APPENDIX.format(task_dir=str(task_spec.task_dir), output_dir=str(task_spec.output_dir))
    return base + appendix


# ---------------------------------------------------------------------------
# Credentials / isolated Codex config
# ---------------------------------------------------------------------------
def load_api_key(explicit_key: str | None) -> str:
    """The harness passes the key explicitly (bx_config.api_key); OPENROUTER_API_KEY / OPENAI_API_KEY are the fallbacks."""
    key = explicit_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("No API key: set OPENROUTER_API_KEY or OPENAI_API_KEY (see config.env.example)")
    return key


def ensure_codex_home(codex_home: Path, base_url: str, model: str) -> None:
    """Write an isolated CODEX_HOME/config.toml pointing at the harness endpoint (model_tap).

    Idempotent: safe to call before every task/worker. Never touches the operator's
    real ~/.codex directory because CODEX_HOME is overridden per-subprocess.
    """
    codex_home.mkdir(parents=True, exist_ok=True)
    config_path = codex_home / "config.toml"
    wire_api = os.environ.get("BX_CODEX_WIRE_API", "responses")
    config_text = f"""model_provider = "bioxarena"
model = "{model}"
approval_policy = "never"
sandbox_mode = "danger-full-access"

[model_providers.bioxarena]
name = "bioxarena"
base_url = "{base_url.rstrip('/')}/v1"
env_key = "{KEY_ENV}"
wire_api = "{wire_api}"
"""
    # Only rewrite if different. Multiple SLURM jobs/processes call this concurrently
    # at startup, so the temp file must be unique per-process (PID) -- a shared tmp
    # filename caused a real FileNotFoundError race (two processes both renaming the
    # same "config.toml.tmp" -> config.toml, the second finding it already consumed).
    try:
        needs_write = not config_path.exists() or config_path.read_text(encoding="utf-8") != config_text
    except OSError:
        needs_write = True
    if needs_write:
        tmp_path = codex_home / f".config.toml.tmp.{os.getpid()}"
        try:
            tmp_path.write_text(config_text, encoding="utf-8")
            os.replace(tmp_path, config_path)
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Output validation / idempotency
# ---------------------------------------------------------------------------
def _missing_required_outputs(output_dir: Path) -> list[str]:
    missing = [f for f in REQUIRED_OUTPUT_LABELS if not (output_dir / f).exists()]
    if "submission.csv" not in missing:
        sub_path = output_dir / "submission.csv"
        try:
            if sub_path.stat().st_size < 5:
                missing.append("submission.csv")
        except OSError:
            missing.append("submission.csv")
    return missing


def task_already_done(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    if (output_dir / "FAILED.json").exists():
        return False
    return not _missing_required_outputs(output_dir)


# ---------------------------------------------------------------------------
# Transcript parsing (Codex --json JSONL events)
# ---------------------------------------------------------------------------
def parse_codex_usage(transcript_path: Path) -> dict[str, int]:
    input_tokens = 0
    output_tokens = 0
    if not transcript_path.exists():
        return {"input_tokens": 0, "output_tokens": 0}
    try:
        for line in transcript_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("type") == "turn.completed":
                usage = evt.get("usage") or {}
                input_tokens += int(usage.get("input_tokens", 0) or 0)
                output_tokens += int(usage.get("output_tokens", 0) or 0)
    except Exception:
        pass
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def backfill_metrics(output_dir: Path, duration_sec: float, token_usage: dict[str, int]) -> None:
    metrics_path = output_dir / "metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}

    def _f(v: Any) -> float:
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    train_time = max(0.0, _f(metrics.get("train_time_sec")))
    test_time = max(0.0, _f(metrics.get("test_time_sec")))
    code_total_time = round(max(0.0, duration_sec), 2)
    solution_generation_time = round(max(0.0, code_total_time - train_time - test_time), 2)

    metrics["solution_generation_time_sec"] = metrics.get("solution_generation_time_sec", solution_generation_time)
    metrics["train_time_sec"] = round(train_time, 2)
    metrics["test_time_sec"] = round(test_time, 2)
    metrics["code_total_time_sec"] = code_total_time
    metrics["input_tokens"] = token_usage.get("input_tokens", 0)
    metrics["output_tokens"] = token_usage.get("output_tokens", 0)
    metrics.setdefault("model_used", metrics.get("model_used", "N/A"))
    metrics.setdefault("model_param_count", metrics.get("model_param_count", "N/A"))
    metrics.setdefault("notes", metrics.get("notes", ""))

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Single-task runner (subprocess with hard wall-clock kill)
# ---------------------------------------------------------------------------
def run_single_task(
    task_spec: TaskSpec,
    prompt_template: str,
    model: str,
    base_url: str,
    api_key: str,
    codex_home: Path,
    wall_clock_sec: int,
    gpu_id: int | None,
    print_lock: threading.Lock,
) -> TaskRunResult:
    task_spec.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    with print_lock:
        print(f"[START] {task_spec.key} (gpu={gpu_id})")

    prompt = build_prompt(prompt_template, task_spec)
    (task_spec.output_dir / "agent_prompt.txt").write_text(prompt, encoding="utf-8")

    transcript_path = task_spec.output_dir / "codexcli_transcript.jsonl"
    stderr_path = task_spec.output_dir / "codexcli_stderr.log"

    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    env[KEY_ENV] = api_key
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--json",
        "-C",
        str(task_spec.output_dir),
        "-m",
        model,
        prompt,
    ]

    proc = None
    exit_code: int | None = None
    error: str | None = None
    timed_out = False
    try:
        with open(transcript_path, "w", encoding="utf-8") as out_f, open(stderr_path, "w", encoding="utf-8") as err_f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(task_spec.output_dir),
                env=env,
                stdout=out_f,
                stderr=err_f,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                exit_code = proc.wait(timeout=wall_clock_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    time.sleep(5)
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=15)
                exit_code = proc.returncode
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    duration = time.time() - start
    token_usage = parse_codex_usage(transcript_path)
    backfill_metrics(task_spec.output_dir, duration, token_usage)

    if timed_out:
        (task_spec.output_dir / "FAILED.json").write_text(
            json.dumps(
                {
                    "task": task_spec.key,
                    "error": f"Task wall-clock exceeded {wall_clock_sec}s",
                    "duration_sec": duration,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        status = "failed"
        error = f"Task wall-clock exceeded {wall_clock_sec}s"
    else:
        missing = _missing_required_outputs(task_spec.output_dir)
        if missing:
            status = "failed"
            error = error or f"Missing/invalid outputs: {missing} (exit_code={exit_code})"
            (task_spec.output_dir / "FAILED.json").write_text(
                json.dumps(
                    {"task": task_spec.key, "error": error, "duration_sec": duration, "exit_code": exit_code},
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            status = "success"

    with print_lock:
        print(
            f"[{status.upper()}] {task_spec.key} ({duration:.0f}s) exit={exit_code} "
            f"in_tok={token_usage['input_tokens']} out_tok={token_usage['output_tokens']}"
        )

    return TaskRunResult(
        task_key=task_spec.key,
        status=status,
        output_dir=str(task_spec.output_dir),
        error=error,
        duration_sec=duration,
        input_tokens=token_usage["input_tokens"],
        output_tokens=token_usage["output_tokens"],
        exit_code=exit_code,
    )
