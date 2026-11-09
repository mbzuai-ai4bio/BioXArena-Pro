"""BioXArena-Pro: run ONE agent configuration on ONE task with the unified harness. Usage: run_agent.py <runs.json> <index>

Scaffolds (item "cli"): codex (Codex CLI), claude (Claude Code), geminicli (Gemini CLI) -- the vendor CLIs -- and the
model-agnostic minimal, opencode, biomni, mlevolve (runners/*_runner.py). Every scaffold gets the SAME prompt
(prompts/unified_prompt_v5.py: 7200 s budget, exactly 1 GPU; the tool-using scaffolds add CLI_APPENDIX with their own name),
the SAME Landlock sandbox (sandbox.py: task public dir read-only, own output dir writable, everything else inaccessible), the
SAME 7200 s hard wall clock, 1 GPU (CUDA_VISIBLE_DEVICES) and the same cache layout under <output>/.runtime. After the run
every process still alive in the output dir is killed (no post-deadline writes).

API route (item "route", default bx_config.default_route): "openai" (OpenAI-compatible endpoint, e.g. OpenRouter),
"anthropic" (Anthropic Messages API) or "gemini" (Google Gemini API). All traffic goes through model_tap.py, a local
pass-through that records the SERVER-reported model and token usage of every response, so both are exact for every run,
including runs stopped at the deadline. run_summary.json gets exact tokens (input incl. cache, output) and runtime_sec.
Model check: the requested model must serve >= 95 % of responses (Gemini CLI: >= 95 % of output tokens, because the CLI makes
small utility calls on helper models; Claude Code additionally: >= 95 % of transcript output tokens), else MODEL_MISMATCH.json
and exit 3 (infrastructure: the worker archives the attempt and retries).
Safety: a refusal / policy error / safety block of the requested model in a run that then fails -> SAFETY_REFUSAL.json and
exit 4 (archived by the worker, not retried automatically). Claude Code's own refusal fallback to another model is disabled in
the main round; a make-up run with item allow_refusal_fallback=true allows it and records every switch in SAFETY_FALLBACK.json.
Exit codes: 0 success, 1 task failed (genuine: timeout, invalid / missing outputs), 3 model mismatch, 4 safety refusal."""
import csv, dataclasses, importlib.util, json, os, re, signal, subprocess, sys, threading, time, types
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.append(str(H))
import bx_config as C  # noqa: E402
from model_tap import Tap  # noqa: E402

WALL = C.WALL_SEC
NS = {}; exec((C.ROOT / "prompts/unified_prompt_v5.py").read_text(), NS)
PROMPT, APPENDIX = NS["PROMPT_TEMPLATE"], NS["CLI_APPENDIX"]
CLI_NAMES = {"codex": "Codex CLI", "claude": "Claude Code", "geminicli": "Gemini CLI", "opencode": "OpenCode", "biomni": "Biomni"}
TOOLS = {"codex": ("BX_CODEX_BIN", C.CODEX_BIN), "claude": ("BX_CLAUDE_BIN", C.CLAUDE_BIN), "geminicli": ("BX_GEMINI_CLI_JS", C.GEMINI_CLI_JS),
         "opencode": ("BX_OPENCODE_BIN", C.OPENCODE_BIN),
         "biomni": ("BIOMNI_DATA_ROOT", C.BIOMNI_DATA_ROOT and str(Path(C.BIOMNI_DATA_ROOT) / "biomni_data" / "data_lake"))}


def load(name):
    spec = importlib.util.spec_from_file_location(name, H / "runners" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def kill_leftovers(out):
    """SIGKILL every process of this user whose cwd or command line is inside the output dir."""
    me, killed, outs = os.getuid(), [], str(out)
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            if p.stat().st_uid != me: continue
            cwd = os.readlink(p / "cwd"); cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            if cwd.startswith(outs) or outs in cmd:
                os.kill(int(p.name), signal.SIGKILL); killed.append(int(p.name))
        except (OSError, ProcessLookupError):
            continue
    return killed


def resource_sampler(out, stop):
    """Optional (item resource_log=true): every 30 s the memory / utilisation of the assigned GPU and the job cgroup memory."""
    gpu = (os.getenv("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]; cg = None
    try:
        rel = open("/proc/self/cgroup").read().strip().split(":")[-1]
        cg = next((c for c in [f"/sys/fs/cgroup{rel}/memory.current", f"/sys/fs/cgroup/memory{rel}/memory.usage_in_bytes"] if os.path.exists(c)), None)
    except Exception:
        pass
    with open(out / "resource_log.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["t_sec", "gpu_mem_mib", "gpu_util_pct", "job_mem_gib"]); t0 = time.time()
        while not stop.is_set():
            gm = gu = jm = ""
            try:
                q = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=20).stdout.strip().split(",")
                gm, gu = q[0].strip(), q[1].strip()
            except Exception:
                pass
            try:
                jm = round(int(open(cg).read()) / 2**30, 2) if cg else ""
            except Exception:
                pass
            w.writerow([round(time.time() - t0), gm, gu, jm]); fh.flush(); stop.wait(30)


def claude_transcript_check(transcript, model):
    """Share of transcript output tokens produced by the requested model (Claude Code's modelUsage in the result event)."""
    usage = {}
    txt = transcript.read_text(errors="ignore") if transcript.exists() else ""
    for line in txt.splitlines():
        if '"type":"result"' in line:
            try: usage = json.loads(line).get("modelUsage") or usage
            except Exception: pass
    if not usage:  # timed-out run without a result line: count per-message models
        for m in re.findall(r'"model":"([^"]+)"', txt):
            usage.setdefault(m, {"outputTokens": 0})["outputTokens"] += 1
    tot = sum((v.get("outputTokens") or 0) for v in usage.values())
    mine = sum((v.get("outputTokens") or 0) for k, v in usage.items() if C.same_model(k, model))
    return (tot == 0 or mine / tot >= 0.95), {k: v.get("outputTokens") for k, v in usage.items()}


def main():
    item = json.loads(Path(sys.argv[1]).read_text())[int(sys.argv[2])]
    cli, model = item["cli"], item["model"]
    route = item.get("route") or C.default_route(cli, model)
    public, out = Path(item["public"]), Path(item["output"])
    if (out / "run_summary.json").exists():
        raise SystemExit("existing attempt: refusing to overwrite (archive it first)")
    key = C.api_key(route)  # exits with a clear message when the key is missing
    if cli in TOOLS and not (TOOLS[cli][1] and Path(TOOLS[cli][1]).exists()):
        raise SystemExit(f"{cli}: {TOOLS[cli][0]} is not set or does not exist (see README); nothing was run")
    rt = out / ".runtime"
    cache = {k: str(rt / "cache" / v) for k, v in {"HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub",
             "HUGGINGFACE_HUB_CACHE": "huggingface/hub", "TRANSFORMERS_CACHE": "huggingface/transformers", "TORCH_HOME": "torch",
             "HF_XET_CACHE": "huggingface/xet", "HF_DATASETS_CACHE": "huggingface/datasets", "CUDA_CACHE_PATH": "cuda",
             "NUMBA_CACHE_DIR": "numba", "MPLCONFIGDIR": "matplotlib"}.items()}
    for d in [rt / "home", rt / "tmp", rt / "cli", *map(Path, cache.values())]:
        d.mkdir(parents=True, exist_ok=True)
    os.environ.update(cache, BX_PUBLIC=str(public), BX_OUTPUT=str(out), HOME=str(rt / "home"), TMPDIR=str(rt / "tmp"),
                      XDG_CACHE_HOME=str(rt / "cache"), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", BX_PYTHON=sys.executable,
                      PATH=f"{H / 'bin'}:{os.environ['PATH']}")
    os.environ.pop("PYTHONPATH", None)
    for k in [k for k in os.environ if k in ROUTE_KEYS]:  # the agent gets its key through the runner only
        os.environ.pop(k)
    os.environ["BX_ALLOW_REFUSAL_FALLBACK"] = "1" if item.get("allow_refusal_fallback") else "0"
    os.environ["BX_EXTRA_RO"] = ":".join(item.get("extra_readonly", []))
    prompt = PROMPT + item.get("prompt_addendum", "").replace("{", "{{").replace("}", "}}")
    domain, task = item["task"].split("/")
    tap = Tap(C.upstream(route))  # tap.url = http://127.0.0.1:<port>; each runner adds the API path it needs
    cfg = {**item, "route": route, "api_upstream": C.upstream(route), "wall_clock_sec": WALL, "gpu_count": 1,
           "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
           "host": os.uname().nodename, "prompt": "prompts/unified_prompt_v5.py",
           "sandbox": "sandbox.py (Landlock ABI>=3; public read-only, output writable)", "cache_environment": cache,
           "extra_readonly": item.get("extra_readonly", []), "prompt_addendum": item.get("prompt_addendum", "")}
    (out / "execution_config.json").write_text(json.dumps(cfg, indent=2))
    lock = threading.Lock(); stop = threading.Event()
    if item.get("resource_log"):
        threading.Thread(target=resource_sampler, args=(out, stop), daemon=True).start()
    kw = dict(domain=domain, task=task, public=public, out=out, model=model, route=route, base_url=tap.url, api_key=key,
              wall_clock_sec=WALL, print_lock=lock)
    t_start = time.time()
    if cli == "codex":
        r = load("codex_runner"); r.CODEX_APPENDIX = APPENDIX.replace("{cli_name}", "Codex CLI")
        ts = r.TaskSpec(domain, task, public, public / "description.md", out)
        r.ensure_codex_home(rt / "cli", tap.url, model)
        res = r.run_single_task(task_spec=ts, prompt_template=prompt, model=model, base_url=tap.url, api_key=key,
                                codex_home=rt / "cli", wall_clock_sec=WALL, gpu_id=None, print_lock=lock)
    elif cli == "claude":
        r = load("claude_runner"); r.CLAUDECODE_APPENDIX = APPENDIX.replace("{cli_name}", "Claude Code")
        ts = r.TaskSpec(domain, task, public, public / "description.md", out)
        res = r.run_single_task(task_spec=ts, prompt_template=prompt, model=model, base_url=tap.url, api_key=key,
                                claude_config_dir=rt / "cli", wall_clock_sec=WALL, max_budget_usd=None, gpu_id=None, print_lock=lock)
    elif cli == "minimal":  # no tools: the CLI appendix does not apply; the runner adds its one-code-block response format
        r = load("minimal_runner")
        signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError(f"wall clock {WALL}s")))
        signal.alarm(WALL)
        try:
            res = r.run_single_task(prompt_template=prompt, **kw)
        except TimeoutError:  # wall clock hit: record a failed run (tokens/runtime still come from the tap below)
            err = f"Task wall-clock exceeded {WALL}s"
            (out / "FAILED.json").write_text(json.dumps({"task": item["task"], "error": err, "duration_sec": WALL}, indent=2))
            res = types.SimpleNamespace(task_key=item["task"], status="failed", output_dir=str(out), error=err, duration_sec=float(WALL))
        finally:
            signal.alarm(0)
    elif cli in ("geminicli", "opencode"):
        r = load("gemini_runner" if cli == "geminicli" else "opencode_runner")
        full = (prompt + APPENDIX.replace("{cli_name}", CLI_NAMES[cli])).format(task_dir=str(public), output_dir=str(out),
                                                                                description=(public / "description.md").read_text())
        res = r.run_single_task(prompt=full, cli_home=rt / "cli", **kw)
    elif cli == "biomni":
        res = load("biomni_runner").run_single_task(prompt_template=prompt + APPENDIX.replace("{cli_name}", "Biomni"), cli_home=rt / "cli", **kw)
    elif cli == "mlevolve":  # unified prompt as MLEvolve's task description (paths ./input, ./submission)
        res = load("mlevolve_runner").run_single_task(prompt_template=prompt, cli_home=rt / "cli", **kw)
    else:
        raise SystemExit(f"unknown cli {cli}")
    t_end = time.time(); stop.set()
    time.sleep(2); killed = kill_leftovers(out); time.sleep(1)
    rep = tap.report(); tap.close()
    summ = dataclasses.asdict(res) if dataclasses.is_dataclass(res) else dict(vars(res)); summ["leftover_processes_killed"] = killed
    u = rep["usage_total"]
    if route == "anthropic":  # Anthropic: input_tokens excludes cache writes / reads
        tok = {"input_tokens_total": u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"],
               "input_tokens_uncached": u["input_tokens"], "input_tokens_cache_write": u["cache_creation_input_tokens"],
               "input_tokens_cache_read": u["cache_read_input_tokens"], "output_tokens_total": u["output_tokens"]}
    else:  # OpenAI-compatible and Gemini: input_tokens already includes cached tokens, output includes reasoning
        tok = {"input_tokens_total": u["input_tokens"], "input_tokens_cached": u["cached_input_tokens"],
               "output_tokens_total": u["output_tokens"], "reasoning_output_tokens": u["reasoning_output_tokens"]}
    mine_usage = {m: c for m, c in rep["usage_by_served_model"].items() if C.same_model(m, model)}
    tok["requested_model_output_tokens"] = sum(c.get("output_tokens", 0) for c in mine_usage.values())
    summ.update(runtime_sec=round(t_end - t_start, 1), start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
                end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end)), tokens=tok, api_tap=rep,
                token_source="api_tap (server-reported usage of every response)")
    rc = 0 if res.status == "success" else 1
    served = rep["served_models"]; tot = sum(served.values())
    n_mine = sum(n for m, n in served.items() if C.same_model(m, model))
    other = {m: n for m, n in served.items() if not C.same_model(m, model)}
    allow_fb = bool(item.get("allow_refusal_fallback"))
    refusal = None
    if cli == "claude":
        txt = (out / "claudecode_transcript.jsonl").read_text(errors="ignore") if (out / "claudecode_transcript.jsonl").exists() else ""
        cats = sorted(set(re.findall(r'"api_refusal_category":"([a-z_]+)"', txt)))
        if '"stop_reason":"refusal"' in txt or "model_refusal_" in txt or rep["refusal_responses"]:
            refusal = {"categories": cats, "fallback_events": txt.count('"subtype":"model_refusal_fallback"'),
                       "no_fallback_events": txt.count('"subtype":"model_refusal_no_fallback"'), "refusal_responses": rep["refusal_responses"]}
    elif rep["refusal_responses"] or rep["policy_errors"] or (cli == "codex" and other):
        refusal = {"refusal_responses": rep["refusal_responses"], "policy_errors": rep["policy_errors"], "other_served_models": other}
    summ["safety_refusal"] = refusal
    summ["round"] = "safety_makeup_fallback_allowed" if allow_fb else "main_no_fallback"
    if cli == "geminicli":
        out_tot = u["output_tokens"]; mismatch = tot == 0 or (tok["requested_model_output_tokens"] / out_tot if out_tot else 0.0) < 0.95
    else:
        mismatch = tot == 0 or n_mine / tot < 0.95
    if allow_fb:  # make-up round: a model switch after a safeguard refusal is allowed and recorded, never retried
        summ["fallback_record"] = {"requested": model, "main_round_model": item.get("main_round_model", model),
                                   "responses_by_served_model": served, "usage_by_served_model": rep["usage_by_served_model"], "refusal": refusal}
        (out / "SAFETY_FALLBACK.json").write_text(json.dumps(summ["fallback_record"], indent=1))
    elif refusal and (res.status != "success" or (cli == "codex" and other)):
        (out / "SAFETY_REFUSAL.json").write_text(json.dumps({"requested": model, "refusal": refusal, "responses_by_served_model": served}, indent=1)); rc = 4
    elif mismatch:
        (out / "MODEL_MISMATCH.json").write_text(json.dumps({"requested": model, "responses_by_served_model": served,
                                                             "usage_by_served_model": rep["usage_by_served_model"]}, indent=1)); rc = 3
    elif cli == "claude":
        ok, usage = claude_transcript_check(out / "claudecode_transcript.jsonl", model); summ["transcript_model_output_tokens"] = usage
        if not ok:
            (out / "MODEL_MISMATCH.json").write_text(json.dumps({"requested": model, "transcript_output_tokens": usage}, indent=1)); rc = 3
    (out / "run_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({"task": item["task"], "model": model, "route": route, "status": res.status, "rc": rc,
                      "runtime_sec": summ["runtime_sec"], "served": served, "tokens": tok, "killed": len(killed)}), flush=True)
    sys.exit(rc)


ROUTE_KEYS = {k for r in C.ROUTES.values() for k in r["keys"]}

if __name__ == "__main__":
    main()
