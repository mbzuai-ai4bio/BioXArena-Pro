"""Create queues/runs_<CFG>.json + queues/queue_<CFG>.json: the list of (task, scaffold, model, route, output dir) items that
worker.sh / run_agent.py execute.
Usage: make_runs.py <cfg>[,<cfg>...|paper] [--cli C --model M [--route R]] [--prefix P] [--out-root DIR] [--tasks t1,t2,...]
  <cfg>      a named configuration from CONFIGS below (the 25 configurations of the BioXArena-Pro leaderboard), or
  --cli/--model  a free-form configuration (cfg name = <cli>_<model>); --route overrides bx_config.default_route.
Output dirs: <out-root (BX_OUTPUT_ROOT)>/<scaffold>__<model>/<domain>/<task>. Queue order: harness/task_order.txt (longest
runs first). Refuses to overwrite an existing queue."""
import argparse, json, sys
from pathlib import Path

H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import bx_config as C  # noqa: E402

# name: (scaffold, model id as sent to the API, route). Bare ids go to the vendor API; vendor-prefixed ids are OpenRouter ids.
CONFIGS = {
    # vendor CLIs
    "codex_g6a": ("codex", "gpt-6-astra", "openai"), "codex_g61": ("codex", "gpt-6.1-sol", "openai"), "codex_g56": ("codex", "gpt-5.6-sol", "openai"),
    "claude_o55": ("claude", "claude-opus-5-5", "anthropic"), "claude_o5": ("claude", "claude-opus-5", "anthropic"),
    "gc_g38f": ("geminicli", "gemini-3.8-flash", "gemini"), "gc_g31p": ("geminicli", "gemini-3.1-pro-preview", "gemini"),
    # model-agnostic scaffolds x the same backbones
    "mn_g6a": ("minimal", "gpt-6-astra", "openai"), "mn_o55": ("minimal", "claude-opus-5-5", "anthropic"),
    "mn_g38f": ("minimal", "gemini-3.8-flash", "gemini"), "mn_g31p": ("minimal", "gemini-3.1-pro-preview", "gemini"),
    "mn_dsf": ("minimal", "deepseek/deepseek-v4.1-flash-1017", "openai"), "mn_dsp": ("minimal", "deepseek/deepseek-v4-pro-0813", "openai"),
    "oc_g6a": ("opencode", "gpt-6-astra", "openai"), "oc_o55": ("opencode", "claude-opus-5-5", "anthropic"),
    "oc_g38f": ("opencode", "gemini-3.8-flash", "gemini"), "oc_dsf": ("opencode", "deepseek/deepseek-v4.1-flash-1017", "openai"),
    "bm_g6a": ("biomni", "gpt-6-astra", "openai"), "bm_o55": ("biomni", "claude-opus-5-5", "anthropic"),
    "bm_g38f": ("biomni", "google/gemini-3.8-flash", "openai"), "bm_dsf": ("biomni", "deepseek/deepseek-v4.1-flash-1017", "openai"),
    "ml_g6a": ("mlevolve", "gpt-6-astra", "openai"), "ml_o55": ("mlevolve", "claude-opus-5-5", "openai"), "ml_o5": ("mlevolve", "claude-opus-5", "openai"),
    "ml_g38f": ("mlevolve", "gemini-3.8-flash", "gemini"), "ml_dsf": ("mlevolve", "deepseek/deepseek-v4.1-flash-1017", "openai"),
}
ap = argparse.ArgumentParser(); ap.add_argument("cfgs", nargs="?", default="")
ap.add_argument("--cli", choices=sorted(C.SCAFFOLDS)); ap.add_argument("--model"); ap.add_argument("--route", choices=sorted(C.ROUTES))
ap.add_argument("--prefix", default=""); ap.add_argument("--out-root", default=str(C.OUTPUT_ROOT)); ap.add_argument("--tasks", default="")
a = ap.parse_args()
pub = C.DATA_PUBLIC
tasks = sorted(f"{d.name}/{t.name}" for d in pub.iterdir() if d.is_dir() for t in d.iterdir() if (t / "public/description.md").exists())
assert len(tasks) == 64, f"{pub}: found {len(tasks)} tasks, expected 64 (set BX_DATA_PUBLIC)"
if a.tasks:
    sel = a.tasks.split(","); assert all(t in tasks for t in sel), sel; tasks = sel
order = [l.strip() for l in (H / "task_order.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
tasks = sorted(tasks, key=lambda t: order.index(t) if t in order else len(order))
if a.cli:
    assert a.model, "--model is required with --cli"
    todo = {f"{a.cli}_{a.model.split('/')[-1]}": (a.cli, a.model, a.route or C.default_route(a.cli, a.model))}
else:
    names = list(CONFIGS) if a.cfgs == "paper" else a.cfgs.split(",")
    todo = {n: CONFIGS[n] for n in names}
for cfg, (cli, model, route) in todo.items():
    mdir = C.model_dir(cli, model); name = a.prefix + cfg
    rp, qp = H / "queues" / f"runs_{name}.json", H / "queues" / f"queue_{name}.json"
    rp.parent.mkdir(exist_ok=True)
    if qp.exists(): raise SystemExit(f"{qp} exists; refusing to overwrite")
    runs = [dict(index=i, task=t, cli=cli, model=model, route=route, model_dir=mdir, public=str(pub / t / "public"),
                 output=str(Path(a.out_root) / mdir / t)) for i, t in enumerate(tasks)]
    rp.write_text(json.dumps(runs, indent=1)); qp.write_text(json.dumps({"todo": [str(i) for i in range(len(runs))], "claimed": {}}, indent=1))
    print(name, len(runs), route, "->", runs[0]["output"])
