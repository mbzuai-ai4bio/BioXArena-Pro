"""Gemini CLI (@google/gemini-cli, pinned 0.64.0) for the unified harness -- the Google counterpart of the Codex CLI and
Claude Code runners used for the GPT / Claude configurations. UNTESTED against the live API (no Google key on the
maintainers' side when the kit was built): smoke-test one short task first (README).

Invocation (headless): gemini -p <prompt> -m <model> --approval-mode yolo --output-format stream-json
                       --skip-trust --include-directories <task public dir>
run inside the Landlock sandbox (bin/gemini shim), cwd = output dir, isolated home GEMINI_CLI_HOME=<output>/.runtime/cli
holding .gemini/settings.json (written below; the operator's ~/.gemini is never read). The API goes through model_tap:
GOOGLE_GEMINI_BASE_URL=<tap> (honoured by the CLI's genai client for API-key auth; checked in the 0.64.0 bundle), key from
GEMINI_API_KEY. Settings, chosen to match the Claude Code runs (all built-in tools on, no sub-agents):
  auth gemini-api-key; auto-update, usage statistics, telemetry, checkpointing off; folder trust off;
  experimental.enableAgents=false (no sub-agents, as Claude Code ran with --disallowedTools Task);
  model.skipNextSpeakerCheck=true and model.disableLoopDetection=true (both are extra LLM calls on helper models and
  loop detection can end a headless run early).
The CLI may still make small utility calls on helper models (e.g. edit correction, web-fetch fallback, compression);
run_agent.py therefore checks the requested model's share of OUTPUT TOKENS (>= 95 %), not of responses.
Wall clock: process group SIGTERM, then SIGKILL at the deadline. A failed attempt caused by a transient backend error
(429 RESOURCE_EXHAUSTED / 5xx / connection reset) is retried after a cooldown while budget remains (max 3 attempts, like the
Claude Code runner); the previous transcript is kept as gemini_transcript_attempt<N>.jsonl."""
import json, os, re, signal, subprocess, time
from dataclasses import dataclass
from pathlib import Path

REQUIRED = ["submission.csv", "metrics.json", "solution.py"]
MAX_TASK_ATTEMPTS = 3
TRANSIENT_RETRY_COOLDOWN_SEC = 120
TRANSIENT = re.compile(r"RESOURCE_EXHAUSTED|UNAVAILABLE|\b(429|500|502|503|504)\b|ECONNRESET|ETIMEDOUT|socket hang up|overloaded", re.I)
SETTINGS = {
    "security": {"auth": {"selectedType": "gemini-api-key"}, "folderTrust": {"enabled": False}},
    "general": {"enableAutoUpdate": False, "enableAutoUpdateNotification": False, "checkpointing": {"enabled": False}},
    "privacy": {"usageStatisticsEnabled": False},
    "telemetry": {"enabled": False},
    "experimental": {"enableAgents": False},
    "model": {"skipNextSpeakerCheck": True, "disableLoopDetection": True},
}


@dataclass
class TaskRunResult:
    task_key: str
    status: str
    output_dir: str
    error: str | None
    duration_sec: float
    input_tokens: int = 0
    output_tokens: int = 0
    num_attempts: int = 0
    exit_code: int | None = None


def missing_outputs(out: Path) -> list[str]:
    miss = [f for f in REQUIRED if not (out / f).exists()]
    if "submission.csv" not in miss and (out / "submission.csv").stat().st_size < 5:
        miss.append("submission.csv")
    return miss


def parse_result(tp: Path) -> dict:
    """CLI-reported stats of the final `result` event (informational; run_summary.json uses the tap's exact numbers)."""
    info = {"input_tokens": 0, "output_tokens": 0, "result_status": None, "models": {}}
    if tp.exists():
        for line in tp.read_text(errors="replace").splitlines():
            try: e = json.loads(line)
            except Exception: continue
            if e.get("type") == "result":
                st = e.get("stats") or {}
                info.update(input_tokens=int(st.get("input_tokens") or 0), output_tokens=int(st.get("output_tokens") or 0),
                            result_status=e.get("status"), models=st.get("models") or {})
    return info


def error_text(tp: Path) -> str:
    """Text of the CLI's error events and of an error result (agent output is ignored: numbers like 500 appear there)."""
    parts = []
    if tp.exists():
        for line in tp.read_text(errors="replace").splitlines():
            try: e = json.loads(line)
            except Exception: continue
            if e.get("type") == "error" or (e.get("type") == "result" and e.get("status") == "error"):
                parts.append(json.dumps(e))
    return "\n".join(parts)


def backfill_metrics(out: Path, duration: float) -> None:
    p = out / "metrics.json"
    if not p.exists(): return
    try: m = json.loads(p.read_text())
    except Exception: return
    if not isinstance(m, dict): return
    f = lambda v: float(v) if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).isdigit()) else 0.0
    m["code_total_time_sec"] = round(duration, 2)
    m.setdefault("solution_generation_time_sec", round(max(0.0, duration - f(m.get("train_time_sec")) - f(m.get("test_time_sec"))), 2))
    p.write_text(json.dumps(m, indent=2))


def run_single_task(*, domain, task, public, out, prompt, model, base_url, api_key, cli_home, wall_clock_sec, print_lock):
    out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    (out / "agent_prompt.txt").write_text(prompt)
    sp = cli_home / ".gemini" / "settings.json"; sp.parent.mkdir(parents=True, exist_ok=True); sp.write_text(json.dumps(SETTINGS, indent=2))
    env = os.environ.copy()
    for k in ("GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_GCA", "GOOGLE_CLOUD_PROJECT", "GOOGLE_VERTEX_BASE_URL"):
        env.pop(k, None)  # API-key auth on the Gemini API only (no Vertex / OAuth)
    env.update(GEMINI_CLI_HOME=str(cli_home), GEMINI_API_KEY=api_key, GOOGLE_GEMINI_BASE_URL=base_url.rstrip("/"),
               GEMINI_TELEMETRY_ENABLED="false", GEMINI_SANDBOX="false", NO_BROWSER="1", CI="1")
    cmd = ["gemini", "-p", prompt, "-m", model, "--approval-mode", "yolo", "--output-format", "stream-json",
           "--skip-trust", "--include-directories", str(public)]
    tp, ep = out / "gemini_transcript.jsonl", out / "gemini_stderr.log"
    exit_code = err = None; timed_out = False; attempt = 0
    while True:
        attempt += 1
        remaining = wall_clock_sec - (time.time() - t0)
        if remaining <= 60: break
        exit_code = None; timed_out = False
        with open(tp, "w") as fo, open(ep, "w") as fe:
            proc = subprocess.Popen(cmd, cwd=str(out), env=env, stdout=fo, stderr=fe, stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                exit_code = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try: os.killpg(proc.pid, sig)
                    except ProcessLookupError: pass
                    time.sleep(5)
                proc.wait(timeout=30); exit_code = proc.returncode
        if timed_out or not missing_outputs(out):
            break
        log = error_text(tp) + ep.read_text(errors="replace")[-20000:]
        left = wall_clock_sec - (time.time() - t0) - TRANSIENT_RETRY_COOLDOWN_SEC
        if TRANSIENT.search(log) and attempt < MAX_TASK_ATTEMPTS and left > 60:
            with print_lock:
                print(f"[RETRY] {domain}/{task} attempt {attempt}: transient backend error, retry in {TRANSIENT_RETRY_COOLDOWN_SEC}s", flush=True)
            for f in (tp, ep):
                try: f.replace(f.with_name(f"{f.stem}_attempt{attempt}{f.suffix}"))
                except OSError: pass
            time.sleep(TRANSIENT_RETRY_COOLDOWN_SEC)
            continue
        break
    dur = time.time() - t0; info = parse_result(tp); backfill_metrics(out, dur)
    if timed_out:
        err = f"Task wall-clock exceeded {wall_clock_sec}s"
    else:
        miss = missing_outputs(out)
        if miss: err = f"Missing/invalid outputs: {miss} (exit_code={exit_code}, result_status={info['result_status']})"
    status = "failed" if err else "success"
    if err:
        (out / "FAILED.json").write_text(json.dumps({"task": f"{domain}/{task}", "error": err, "duration_sec": dur,
                                                     "exit_code": exit_code, "attempts": attempt}, indent=2))
    with print_lock:
        print(f"[{status.upper()}] {domain}/{task} ({dur:.0f}s) exit={exit_code} attempts={attempt} "
              f"cli_in={info['input_tokens']} cli_out={info['output_tokens']}", flush=True)
    return TaskRunResult(f"{domain}/{task}", status, str(out), err, dur, info["input_tokens"], info["output_tokens"], attempt, exit_code)
