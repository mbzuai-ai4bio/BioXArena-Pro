"""OpenCode CLI (`opencode run`, we used 1.18.5) for the unified harness.
OpenCode runs inside the Landlock sandbox (bin/opencode shim) with an isolated opencode.json, XDG dirs and HOME under
<output>/.runtime, and is killed (process group SIGTERM, then SIGKILL) at the wall clock. Providers, all pointed at model_tap:
  route openai    : @ai-sdk/openai (Responses API, as Codex CLI uses) for bare ids, @ai-sdk/openai-compatible (Chat
                    Completions) for vendor-prefixed ids (OpenRouter style); baseURL <tap>/v1
  route anthropic : @ai-sdk/anthropic (Messages API), baseURL <tap>/v1, adaptive thinking as in Claude Code
  route gemini    : @ai-sdk/google (Gemini native API, keeps thought signatures across tool calls), baseURL <tap>/v1beta
OpenCode downloads the provider package into its XDG cache on first use (network needed).
webfetch is denied as in all BioXArena OpenCode runs; bash/edit are allowed (the sandbox bounds the filesystem)."""
import json, os, signal, subprocess, time
from dataclasses import dataclass
from pathlib import Path

H = Path(__file__).resolve().parents[1]
REQUIRED = ["submission.csv", "metrics.json", "solution.py"]
PROVIDER_ID = "bioxarena"
KEY_ENV = "BX_LLM_API_KEY"


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


def write_config(path: Path, base_url: str, model: str, route: str) -> None:
    base = base_url.rstrip("/")
    if route == "anthropic":
        npm, url = "@ai-sdk/anthropic", base + "/v1"
        mcfg = {"name": model, "limit": {"context": 200000, "output": 64000}, "options": {"thinking": {"type": "adaptive"}}}
    elif route == "gemini":
        npm, url = "@ai-sdk/google", base + "/v1beta"
        mcfg = {"name": model, "limit": {"context": 1048576, "output": 65536}}
    else:
        npm, url = ("@ai-sdk/openai-compatible" if "/" in model else "@ai-sdk/openai"), base + "/v1"
        mcfg = {"name": model, "limit": {"context": 400000, "output": 64000}}
    payload = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {PROVIDER_ID: {"npm": npm, "name": f"BioXArena model_tap ({route})",
                                   "options": {"baseURL": url, "apiKey": "{env:" + KEY_ENV + "}"}, "models": {model: mcfg}}},
        "model": f"{PROVIDER_ID}/{model}",
        "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny", "external_directory": "allow"},  # the sandbox bounds paths
        "autoupdate": False, "share": "disabled",
    }
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2))


def missing_outputs(out: Path) -> list[str]:
    miss = [f for f in REQUIRED if not (out / f).exists()]
    if "submission.csv" not in miss and (out / "submission.csv").stat().st_size < 5:
        miss.append("submission.csv")
    return miss


def parse_usage(tp: Path) -> dict:
    i = o = 0
    if tp.exists():
        for line in tp.read_text(errors="replace").splitlines():
            try: e = json.loads(line)
            except Exception: continue
            part = e.get("part") or {}
            if e.get("type") == "step_finish" or part.get("type") == "step-finish":
                t = part.get("tokens") or {}; i += int(t.get("input", 0) or 0); o += int(t.get("output", 0) or 0)
    return {"input_tokens": i, "output_tokens": o}


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


def run_single_task(*, domain, task, public, out, prompt, model, route, base_url, api_key, cli_home, wall_clock_sec, print_lock):
    out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    (out / "agent_prompt.txt").write_text(prompt)
    cfg = cli_home / "opencode.json"; write_config(cfg, base_url, model, route)
    env = os.environ.copy()
    env.update({KEY_ENV: api_key, "OPENCODE_CONFIG": str(cfg), "XDG_DATA_HOME": str(cli_home / "data"), "XDG_CONFIG_HOME": str(cli_home / "config"),
                "XDG_STATE_HOME": str(cli_home / "state"), "OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
                "OPENCODE_DISABLE_SHARE": "1"})
    cmd = ["opencode", "run", "--dir", str(out), "-m", f"{PROVIDER_ID}/{model}", "--format", "json", prompt]
    tp, ep = out / "opencode_transcript.jsonl", out / "opencode_stderr.log"
    exit_code = err = None; timed_out = False
    with open(tp, "w") as fo, open(ep, "w") as fe:
        proc = subprocess.Popen(cmd, cwd=str(out), env=env, stdout=fo, stderr=fe, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            exit_code = proc.wait(timeout=wall_clock_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try: os.killpg(proc.pid, sig)
                except ProcessLookupError: pass
                time.sleep(5)
            proc.wait(timeout=30); exit_code = proc.returncode
    dur = time.time() - t0; usage = parse_usage(tp); backfill_metrics(out, dur)
    if timed_out:
        err = f"Task wall-clock exceeded {wall_clock_sec}s"
    else:
        miss = missing_outputs(out)
        if miss: err = f"Missing/invalid outputs: {miss} (exit_code={exit_code})"
    status = "failed" if err else "success"
    if err:
        (out / "FAILED.json").write_text(json.dumps({"task": f"{domain}/{task}", "error": err, "duration_sec": dur, "exit_code": exit_code}, indent=2))
    with print_lock:
        print(f"[{status.upper()}] {domain}/{task} ({dur:.0f}s) exit={exit_code} in={usage['input_tokens']} out={usage['output_tokens']}", flush=True)
    return TaskRunResult(f"{domain}/{task}", status, str(out), err, dur, usage["input_tokens"], usage["output_tokens"], exit_code)
