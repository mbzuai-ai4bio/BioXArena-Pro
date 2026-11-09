"""Biomni (A1 agent, agents/Biomni = Biomni 0.0.8 with the BioXArena llm.py / a1.py patches) for the unified harness.
The whole Biomni process (its ReAct loop, in-process Python execution, bash/R subprocesses) runs inside the Landlock sandbox
(runners/biomni_child.py); the Biomni data lake (BIOMNI_DATA_ROOT/biomni_data, ~15 GB) is read-only inside it and linked as
<output>/biomni_data so A1 skips its S3 download. Single attempt, hard wall clock (process-group kill), per-execution
timeout = Biomni's default 600 s. Backends (all through model_tap): route openai = langchain ChatOpenAI (Chat Completions,
<tap>/v1); route anthropic = ChatAnthropic (Messages API, <tap>); route gemini = ChatOpenAI on Gemini's OpenAI-compatible
route <tap>/v1beta/openai (the route upstream Biomni uses for Gemini)."""
import json, os, signal, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path

H = Path(__file__).resolve().parents[1]; CODE = H.parent
sys.path.insert(0, str(H))
import bx_config as C  # noqa: E402
DATA_LAKE = str(Path(C.BIOMNI_DATA_ROOT) / "biomni_data") if C.BIOMNI_DATA_ROOT else ""
EXEC_TIMEOUT_SEC = 600


@dataclass
class TaskRunResult:
    task_key: str
    status: str
    output_dir: str
    error: str | None
    duration_sec: float
    exit_code: int | None = None


def run_single_task(*, domain, task, public, out, prompt_template, model, route, base_url, api_key, cli_home, wall_clock_sec, print_lock):
    out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    if not DATA_LAKE or not Path(DATA_LAKE, "data_lake").is_dir():
        raise SystemExit("BIOMNI_DATA_ROOT must point to a dir containing biomni_data/data_lake (see README)")
    base = base_url.rstrip("/")
    if route == "anthropic": source, url = "Anthropic", base
    elif route == "gemini": source, url = "OpenAI", base + C.OPENAI_COMPAT_PATH
    else: source, url = "OpenAI", base + "/v1"
    rp = cli_home / "biomni_result.json"
    cfg = dict(code_root=str(CODE), domain=domain, task=task, public=str(public), output=str(out), model=model, source=source, base_url=url,
               exec_timeout_sec=EXEC_TIMEOUT_SEC, data_lake=DATA_LAKE, prompt_template=prompt_template, result_path=str(rp))
    cp = cli_home / "biomni_config.json"; cp.write_text(json.dumps(cfg, indent=1))
    env = os.environ.copy()
    env["BX_LLM_API_KEY"] = api_key
    env["BX_EXTRA_RO"] = ":".join([p for p in env.get("BX_EXTRA_RO", "").split(":") if p] + [DATA_LAKE, str(CODE / "agents"), str(CODE / "training"),
                                    str(CODE / "benchmark_layout.py"), str(CODE / "tasks"), str(H / "runners")])
    cmd = [sys.executable, "-I", str(H / "sandbox.py"), sys.executable, str(H / "runners" / "biomni_child.py"), str(cp)]
    exit_code = err = None; timed_out = False
    with open(out / "biomni_stdout.log", "w") as fo, open(out / "biomni_stderr.log", "w") as fe:
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
    dur = time.time() - t0
    r = json.loads(rp.read_text()) if rp.exists() else {}
    if timed_out:
        err = f"Task wall-clock exceeded {wall_clock_sec}s"
        (out / "FAILED.json").write_text(json.dumps({"task": f"{domain}/{task}", "error": err, "duration_sec": dur}, indent=2))
    elif r.get("status") != "success":
        err = r.get("error") or f"biomni child exited {exit_code} without a result"
        if not (out / "FAILED.json").exists():
            (out / "FAILED.json").write_text(json.dumps({"task": f"{domain}/{task}", "error": err, "duration_sec": dur, "exit_code": exit_code}, indent=2))
    status = "failed" if err else "success"
    with print_lock:
        print(f"[{status.upper()}] {domain}/{task} ({dur:.0f}s) exit={exit_code}", flush=True)
    return TaskRunResult(f"{domain}/{task}", status, str(out), err, dur, exit_code)
