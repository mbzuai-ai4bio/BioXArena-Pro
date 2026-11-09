"""MLEvolve (agents/MLEvolve, MCGS search agent) for the unified harness: wraps training/run_mlevolve_agent.py::run_single_task.
Harness changes vs upstream MLEvolve: run.py runs inside the Landlock sandbox; the API key is never on the command line or
in mlevolve_cmd.txt (OmegaConf interpolation ${oc.env:BX_LLM_API_KEY}); the core-package pip guard is disabled (env is
read-only); code and feedback models are the requested model, reached through model_tap. Backends: route gemini = MLEvolve's
own Gemini route (llm/gemini.py, chosen for every model id starting with "gemini"; base_url = tap root); routes openai /
anthropic = MLEvolve's OpenAI Chat Completions route (<tap>/v1; for Claude use an OpenAI-compatible endpoint, e.g. OpenRouter,
since MLEvolve has no native Anthropic backend). Temperature 1.0 (MLEvolve's default). Prompt: MLEvolve's desc_file is the
unified prompt (prompts/unified_prompt_v5.py) with description.md embedded and MLEvolve's own paths (task_dir = ./input,
output_dir = ./submission), so it states the same rules, 1-GPU budget and submission contract as for the other scaffolds.
Budget: the search stops at agent.time_limit = wall - 300 s so it can export best_submission before the hard `timeout` at wall - 60 s.
2026-10-21 adapter fixes (all BioXArena-Pro MLEvolve results were produced with them): (1) ./input is a single symlink to the
public dir (agents/MLEvolve/config/__init__.py), so agents' own public-path checks pass; (2) TMPDIR is a short dir under /dev/shm
(<output>/.runtime/tmp is usually over the 108-character AF_UNIX limit, so PyTorch DataLoader workers fail and hang);
(3) LLM output cap BX_MLE_MAX_TOKENS = 65536 instead of 16384 (merged solutions were cut off and the truncated text was
executed as code; reasoning tokens count toward the cap); (4) one code execution may take at most 2400 s instead of 5400 s,
so a single hung run cannot use the whole 2 h budget; (5) MLEvolve's CPU-affinity lines are inserted after a leading docstring
and `from __future__` imports instead of before them (agents/MLEvolve/engine/executor.py)."""
import os, shutil, subprocess, sys, types
from pathlib import Path

H = Path(__file__).resolve().parents[1]; CODE = H.parent
sys.path.insert(0, str(CODE / "training"))
sys.modules["_core_pkg_guard"] = types.SimpleNamespace(restore_core_packages=lambda *a, **k: None)
import run_mlevolve_agent as M  # noqa: E402

M.MLEVOLVE_EXEC_TIMEOUT = 2400  # fix (4)
MAX_TOKENS = os.environ.get("BX_MLE_MAX_TOKENS", "65536")  # fix (3), read by agents/MLEvolve/llm/{openai,gemini}.py

KEY_REF = "${oc.env:BX_LLM_API_KEY}"
HARD_LIMIT = {"sec": None}


def _sandboxed_run(cmd, **kw):
    i = cmd.index(sys.executable)
    if HARD_LIMIT["sec"]:  # `timeout <N>s` (the hard stop) gets the later deadline; agent.time_limit stays the search budget
        cmd = [f"{HARD_LIMIT['sec']}s" if (j < i and c.endswith("s") and c[:-1].isdigit()) else c for j, c in enumerate(cmd)]
    cmd = cmd[:i] + [sys.executable, "-I", str(H / "sandbox.py"), sys.executable] + cmd[i + 1:]
    return subprocess.run(cmd, **kw)


M.subprocess = types.SimpleNamespace(run=_sandboxed_run, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)


def run_single_task(*, domain, task, public, out, prompt_template, cli_home, model, route, base_url, api_key, wall_clock_sec, print_lock):
    if route == "gemini":
        assert model.lower().startswith("gemini"), "MLEvolve's Gemini route is chosen by model ids starting with 'gemini'"
        url = base_url.rstrip("/")
    else:
        assert not model.lower().startswith("gemini"), "a bare gemini-* id selects MLEvolve's Gemini backend; use route gemini"
        url = base_url.rstrip("/") + "/v1"
    os.environ["BX_LLM_API_KEY"] = api_key
    os.environ["BX_MLE_MAX_TOKENS"] = MAX_TOKENS
    tmp = Path(f"/dev/shm/bxml_{os.environ.get('SLURM_JOB_ID', 'x')}_{os.getpid()}")  # fix (2): short AF_UNIX-safe path
    tmp.mkdir(parents=True, exist_ok=True); os.environ["TMPDIR"] = str(tmp)
    os.environ["BX_EXTRA_RO"] = ":".join([p for p in os.environ.get("BX_EXTRA_RO", "").split(":") if p] + [str(CODE / "agents" / "MLEvolve")])
    HARD_LIMIT["sec"] = max(360, wall_clock_sec - 60)
    desc = cli_home / "task_prompt.md"; desc.parent.mkdir(parents=True, exist_ok=True)
    desc.write_text(prompt_template.format(task_dir="./input", output_dir="./submission", description=(public / "description.md").read_text()))
    ts = M.TaskSpec(domain=domain, task=task, task_dir=public, description_path=desc, output_dir=out)
    try:
        res = M.run_single_task(task_spec=ts, mlevolve_dir=CODE / "agents" / "MLEvolve", dataset_root=public, model=model,
                                api_key=KEY_REF, base_url=url, temperature=1.0, steps=999,
                                time_limit=max(300, wall_clock_sec - 300), print_lock=print_lock)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return res
