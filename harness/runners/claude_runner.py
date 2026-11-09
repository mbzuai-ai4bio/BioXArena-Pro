#!/usr/bin/env python3
"""Claude Code CLI (`claude --bare -p`, we used 2.1.302) for the unified harness (called by harness/run_agent.py).

Claude Code is a first-party, general-purpose coding-agent CLI: it owns its own tool-use loop (Bash, file read/write/edit,
grep/glob) and its own retry/repair behaviour. This runner only (1) stages the task (public task dir + output dir),
(2) invokes `claude --bare -p ...` non-interactively against an Anthropic Messages API endpoint (the harness's model_tap,
which forwards to bx_config.ROUTES["anthropic"]) with a hard wall-clock kill, and (3) collects submission.csv /
metrics.json / solution.py, the stream-json transcript and the CLI's own usage / cost from its final `result` event.
Isolation: CLAUDE_CONFIG_DIR = <output>/.runtime/cli, ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY set per subprocess, `--bare`
(API-key auth only, never OAuth / keychain) and `--no-session-persistence`; the operator's personal ~/.claude is never read.
Main round: the CLI's own refusal fallback to another model is disabled (CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK=1); a make-up
run with BX_ALLOW_REFUSAL_FALLBACK=1 allows it, and run_agent.py records every switch.
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
# Paths (mirror the Codex CLI / OpenCode runner layout)
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
TRAINING_DIR = SCRIPT_PATH.parent
EVAL_ROOT = TRAINING_DIR.parents[1]  # 2_BioXArena-Pro-Code/
WORKSPACE_ROOT = EVAL_ROOT.parent
DEFAULT_TASKS_ROOT = EVAL_ROOT / "tasks"
DEFAULT_PROMPT_PATH = EVAL_ROOT / "prompts" / "unified_prompt_v5.py"
DEFAULT_ROUND_NAME = "round1"

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_MAX_BUDGET_USD = None  # no cost cap in BioXArena-Pro (same as Codex CLI, which has none)

REQUIRED_OUTPUT_LABELS = ["submission.csv", "metrics.json", "solution.py"]

# Env vars from a parent Claude Code session that must never leak into the child
# `claude` subprocess we spawn -- otherwise the child may believe it is a nested
# session and change behavior. Harmless if unset already.
PARENT_SESSION_ENV_PREFIXES = ("CLAUDECODE", "CLAUDE_CODE_", "CLAUDE_AGENT_SDK", "CLAUDE_PID", "CLAUDE_EFFORT", "AI_AGENT")

CLAUDECODE_APPENDIX = """

# Coding-agent CLI notes
You are running as an autonomous coding-agent CLI (Claude Code) with direct Bash shell and
file access. Work entirely within the directories given to you above ({task_dir} for
reading public data, {output_dir} for writing your solution and outputs). Do not attempt to
list, guess, or read any path outside of these two directories -- in particular, never
search for or open files named like "answers", "private", or "labels" anywhere on the
filesystem. Finish only once submission.csv, metrics.json, and solution.py all exist and
are valid in {output_dir}.
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
    total_cost_usd: float = 0.0
    num_turns: int = 0
    result_subtype: str = ""
    exit_code: int | None = None


# ---------------------------------------------------------------------------
# Task discovery (identical contract to run_codexcli_agent.py / run_opencode_agent.py)
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
    appendix = CLAUDECODE_APPENDIX.format(task_dir=str(task_spec.task_dir), output_dir=str(task_spec.output_dir))
    return base + appendix


# ---------------------------------------------------------------------------
# Credentials / isolated Claude Code config
# ---------------------------------------------------------------------------
def load_api_key(explicit_key: str | None) -> str:
    """The harness passes the key explicitly (bx_config.api_key); ANTHROPIC_API_KEY is the fallback."""
    key = explicit_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No API key: set ANTHROPIC_API_KEY (see config.env.example)")
    return key


def ensure_claude_config_dir(config_dir: Path) -> None:
    """Create an isolated CLAUDE_CONFIG_DIR. Never touches the operator's real ~/.claude.

    With --bare + --no-session-persistence, Claude Code writes almost nothing here at
    runtime (no CLAUDE.md discovery, no auto-memory, no session files), so a single
    shared directory across concurrent workers is safe.
    """
    config_dir.mkdir(parents=True, exist_ok=True)


def sanitize_env_for_child(env: dict[str, str]) -> dict[str, str]:
    """Strip parent Claude Code session env vars so the child `claude` process never
    thinks it's a nested session (relevant when this runner itself is launched from
    inside an interactive Claude Code session; harmless no-op under plain SLURM)."""
    return {
        k: v
        for k, v in env.items()
        if not any(k == p or k.startswith(p) for p in PARENT_SESSION_ENV_PREFIXES)
    }


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
# Transcript parsing (Claude Code --output-format stream-json NDJSON events)
# ---------------------------------------------------------------------------
def parse_claudecode_result(transcript_path: Path) -> dict[str, Any]:
    """Find the final `type: result` event (last line) and extract usage/cost."""
    info: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost_usd": 0.0,
        "num_turns": 0,
        "result_subtype": "",
        "is_error": None,
    }
    if not transcript_path.exists():
        return info
    try:
        lines = [
            ln.strip()
            for ln in transcript_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip()
        ]
        for line in reversed(lines):
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("type") == "result":
                usage = evt.get("usage") or {}
                info["input_tokens"] = int(usage.get("input_tokens", 0) or 0)
                info["output_tokens"] = int(usage.get("output_tokens", 0) or 0)
                info["total_cost_usd"] = float(evt.get("total_cost_usd", 0.0) or 0.0)
                info["num_turns"] = int(evt.get("num_turns", 0) or 0)
                info["result_subtype"] = str(evt.get("subtype", ""))
                info["is_error"] = evt.get("is_error")
                break
    except Exception:
        pass
    return info


def backfill_metrics(output_dir: Path, duration_sec: float, run_info: dict[str, Any]) -> None:
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
    metrics["input_tokens"] = run_info.get("input_tokens", 0)
    metrics["output_tokens"] = run_info.get("output_tokens", 0)
    metrics.setdefault("model_used", metrics.get("model_used", "N/A"))
    metrics.setdefault("model_param_count", metrics.get("model_param_count", "N/A"))
    metrics.setdefault("notes", metrics.get("notes", ""))
    metrics["claudecode_cli"] = {
        "total_cost_usd": run_info.get("total_cost_usd", 0.0),
        "num_turns": run_info.get("num_turns", 0),
        "result_subtype": run_info.get("result_subtype", ""),
        "is_error": run_info.get("is_error"),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Transient-backend-error retry: an API gateway can return a fast 503 / 502 or
# close the connection under load. Claude Code CLI's own internal retries
# (~10x over ~3 min) are sometimes not enough, so this adds one more, coarser
# retry layer at the task level: a handful of attempts with a cooldown between
# them, bounded by the task's overall wall-clock budget. Never retries a real
# wall-clock timeout or a non-transient failure (e.g. missing outputs from a
# model logic error) -- only when the gateway/network signature looks transient.
# ---------------------------------------------------------------------------
TRANSIENT_BACKEND_MARKERS = (
    "no available accounts",
    '"error_status":503',
    '"error_status":502',
    "bad gateway",
    "connection closed mid-response",
    "overloaded_error",
    "rate_limit_error",
    "socket hang up",
)
MAX_TASK_ATTEMPTS = 3
TRANSIENT_RETRY_COOLDOWN_SEC = 90


def _is_transient_backend_error(transcript_path: Path) -> bool:
    if not transcript_path.exists():
        return False
    try:
        tail = transcript_path.read_text(encoding="utf-8", errors="replace")[-4000:].lower()
    except Exception:
        return False
    return any(marker in tail for marker in TRANSIENT_BACKEND_MARKERS)


# ---------------------------------------------------------------------------
# Single-task runner (subprocess with hard wall-clock kill)
# ---------------------------------------------------------------------------
def run_single_task(
    task_spec: TaskSpec,
    prompt_template: str,
    model: str,
    base_url: str,
    api_key: str,
    claude_config_dir: Path,
    wall_clock_sec: int,
    max_budget_usd: float,
    gpu_id: int | None,
    print_lock: threading.Lock,
) -> TaskRunResult:
    task_spec.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    with print_lock:
        print(f"[START] {task_spec.key} (gpu={gpu_id})")

    prompt = build_prompt(prompt_template, task_spec)
    (task_spec.output_dir / "agent_prompt.txt").write_text(prompt, encoding="utf-8")

    transcript_path = task_spec.output_dir / "claudecode_transcript.jsonl"
    stderr_path = task_spec.output_dir / "claudecode_stderr.log"

    env = sanitize_env_for_child(os.environ.copy())
    env["CLAUDE_CONFIG_DIR"] = str(claude_config_dir)
    # v5: no silent model switch. Claude Code would otherwise re-run a refused turn (e.g. a "bio" safeguard refusal of
    # Opus 5.5) and the REST OF THE SESSION on another model (Opus 5). A refusal now stays a refusal of the requested model.
    # Round 2 (safety make-up runs, PI decision 2026-10-17) sets BX_ALLOW_REFUSAL_FALLBACK=1: the CLI's own refusal fallback is
    # then allowed, and every switch is recorded (served models per response, by the api tap).
    if os.environ.get("BX_ALLOW_REFUSAL_FALLBACK") != "1":
        env["CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK"] = "1"
    env["ANTHROPIC_BASE_URL"] = base_url
    env["ANTHROPIC_API_KEY"] = api_key
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "claude",
        "--bare",
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--add-dir",
        str(task_spec.task_dir),
        "--disallowedTools",
        "Task",
    ]
    if max_budget_usd is not None:  # v5: None = no cost cap (same as Codex, which has none)
        cmd[-2:-2] = ["--max-budget-usd", str(max_budget_usd)]

    proc = None
    exit_code: int | None = None
    error: str | None = None
    timed_out = False
    attempt = 0
    while True:
        attempt += 1
        remaining_budget = wall_clock_sec - (time.time() - start)
        if remaining_budget <= 60:
            # Out of wall-clock budget; whatever the last attempt produced stands.
            break
        proc = None
        exit_code = None
        error = None
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
                    exit_code = proc.wait(timeout=remaining_budget)
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

        if timed_out:
            break  # real wall-clock timeout: never retried, handled below as-is

        run_info_attempt = parse_claudecode_result(transcript_path)
        missing_attempt = _missing_required_outputs(task_spec.output_dir)
        if not missing_attempt:
            break  # success on this attempt

        transient = _is_transient_backend_error(transcript_path)
        remaining_after_cooldown = wall_clock_sec - (time.time() - start) - TRANSIENT_RETRY_COOLDOWN_SEC
        if transient and attempt < MAX_TASK_ATTEMPTS and remaining_after_cooldown > 60:
            with print_lock:
                print(
                    f"[RETRY] {task_spec.key} attempt {attempt} hit a transient backend error "
                    f"(no available accounts / 502 / connection reset) -- cooling down "
                    f"{TRANSIENT_RETRY_COOLDOWN_SEC}s then retrying (attempt {attempt + 1}/{MAX_TASK_ATTEMPTS})"
                )
            # Preserve this attempt's transcript for audit before the next attempt overwrites it.
            attempt_backup = task_spec.output_dir / f"claudecode_transcript_attempt{attempt}.jsonl"
            try:
                transcript_path.replace(attempt_backup)
            except OSError:
                pass
            time.sleep(TRANSIENT_RETRY_COOLDOWN_SEC)
            continue
        break  # non-transient failure, or out of attempts/budget: stop here

    duration = time.time() - start
    run_info = parse_claudecode_result(transcript_path)
    backfill_metrics(task_spec.output_dir, duration, run_info)

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
            error = error or f"Missing/invalid outputs: {missing} (exit_code={exit_code}, result_subtype={run_info.get('result_subtype')})"
            (task_spec.output_dir / "FAILED.json").write_text(
                json.dumps(
                    {"task": task_spec.key, "error": error, "duration_sec": duration, "exit_code": exit_code, "run_info": run_info},
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            status = "success"

    with print_lock:
        print(
            f"[{status.upper()}] {task_spec.key} ({duration:.0f}s) exit={exit_code} "
            f"in_tok={run_info['input_tokens']} out_tok={run_info['output_tokens']} "
            f"cost=${run_info['total_cost_usd']:.3f} subtype={run_info['result_subtype']}"
        )

    return TaskRunResult(
        task_key=task_spec.key,
        status=status,
        output_dir=str(task_spec.output_dir),
        error=error,
        duration_sec=duration,
        input_tokens=run_info["input_tokens"],
        output_tokens=run_info["output_tokens"],
        total_cost_usd=run_info["total_cost_usd"],
        num_turns=run_info["num_turns"],
        result_subtype=run_info["result_subtype"],
        exit_code=exit_code,
    )
