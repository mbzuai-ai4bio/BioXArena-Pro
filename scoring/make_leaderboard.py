"""Build the BioXArena-Pro leaderboard from scoring/results/<model_dir>.csv (written by score_final.py).

Writes into results/ (BX_LEADERBOARD_DIR):
  LEADERBOARD.md            ranked table of finished configurations, per-domain scores, configurations still running
  leaderboard.csv           one row per finished configuration (mean64, per-domain means, status counts, runtime, tokens)
  leaderboard_per_task.csv  task x configuration score matrix (x100), plus the per-task oracle of the 22 paper configs
  LEADERBOARD.html          the same tables as a self-contained page (make_leaderboard_html.py)
A configuration counts as finished when all 64 run dirs have run_summary.json and its results csv is newer than every
run_summary.json; otherwise it is listed under "In progress". Re-run after scoring a newly finished configuration:
  python score_final.py <model_dir> && python make_leaderboard.py

SUBST below lets one configuration borrow single task scores from another when the requested model could not be served
for that task at all (2026-10-23: the proxy returned empty completions for MLEvolve + Opus 5.5 on both
*-knockdown-efficacy tasks, so they were rerun on Opus 5). A borrowed cell is marked in every output and the
unsubstituted mean is printed next to the substituted one, so the substitution can never be mistaken for an own result.
"""
import csv, json, os, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlsplit

CODE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(CODE / "harness"))
import bx_config as C  # noqa: E402
OUT = C.OUTPUT_ROOT            # run dirs (optional: when absent, the leaderboard is built from scoring/results/*.csv alone)
RES = CODE / "scoring" / "results"
LB = Path(os.environ.get("BX_LEADERBOARD_DIR") or CODE / "results"); LB.mkdir(parents=True, exist_ok=True)
DOMAINS = ["chemical-biology", "imaging", "network-biology", "perturbation", "phenotype-disease",
           "sequence-genomics", "single-cell-spatial", "structure"]
SCAFFOLD = {"codexcli": "Codex CLI", "claudecode": "Claude Code", "minimal": "Minimal", "opencode": "OpenCode",
            "biomni": "Biomni", "mlevolve": "MLEvolve", "geminicli": "Gemini CLI"}
MODEL = {"gpt-6-astra": "GPT-6 Astra", "gpt-6.1-sol": "GPT-6.1 Sol", "gpt-5.6-sol": "GPT-5.6 Sol",
         "claude-opus-5-5": "Claude Opus 5.5", "claude-opus-5": "Claude Opus 5",
         "deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash (1017)", "deepseek-v4-pro-0813": "DeepSeek V4 Pro (0813)",
         "gemini-4-argon": "Gemini 4 Argon", "gemini-3.8-flash": "Gemini 3.8 Flash",
         "gemini-3.1-pro-preview": "Gemini 3.1 Pro (preview)"}
UPSTREAM = {"maas.devops.rednote.life": "MaaS gateway (rednote)", "openrouter.ai": "OpenRouter (paid)"}
# API that served each published configuration (used when the run directories are not available, i.e. when the leaderboard
# is rebuilt from scoring/results/*.csv alone). "institutional proxy" = the OpenAI- / Anthropic-compatible API gateway of
# the authors' institution, which served the GPT and Claude runs of the paper.
API_LABEL = {m: "institutional proxy" for m in [
    "codexcli__gpt-6-astra", "codexcli__gpt-6.1-sol", "codexcli__gpt-5.6-sol", "claudecode__claude-opus-5-5", "claudecode__claude-opus-5",
    "opencode__gpt-6-astra", "opencode__claude-opus-5-5", "mlevolve__claude-opus-5-5", "mlevolve__claude-opus-5", "mlevolve__gpt-6-astra",
    "biomni__claude-opus-5-5", "biomni__gpt-6-astra", "minimal__claude-opus-5-5", "minimal__gpt-6-astra"]}
API_LABEL.update({m: "OpenRouter (paid)" for m in ["opencode__deepseek-v4.1-flash-1017", "biomni__gemini-3.8-flash", "biomni__deepseek-v4.1-flash-1017",
                                                  "mlevolve__deepseek-v4.1-flash-1017", "minimal__deepseek-v4-pro-0813", "minimal__deepseek-v4.1-flash-1017"]})
API_LABEL.update({m: "MaaS gateway (rednote)" for m in ["opencode__gemini-3.8-flash", "geminicli__gemini-3.8-flash", "mlevolve__gemini-3.8-flash",
                                                       "geminicli__gemini-3.1-pro-preview", "minimal__gemini-3.8-flash", "minimal__gemini-3.1-pro-preview"]})


def API(sums, m=None):
    """Which API actually served the run, read from the run itself rather than guessed from the model name.
    First choice is api_upstream in execution_config.json, the endpoint the runner actually dialled; runs made on the
    Google kit carry it and are served by a third-party MaaS gateway that no model name reveals. Our own runs predate
    that field, so they fall back to the api_tap's served model ids: this harness routes a vendor-prefixed id
    (deepseek/..., google/...) to OpenRouter and a bare id to the school proxy, so the prefix is the route.
    Both signals beat guessing by name, which labelled Gemini "Google" when it had run on the paid OpenRouter key."""
    if not sums:
        return API_LABEL.get(m, "unknown")
    for s in sums[:3]:
        cfg = s.parent / "execution_config.json"
        if cfg.exists():
            up = json.load(open(cfg)).get("api_upstream")
            if up:
                host = urlsplit(up if "//" in up else "//" + up).hostname or up
                return UPSTREAM.get(host, host)
    served = set()
    for s in sums[:3]:
        served |= set((json.load(open(s)).get("api_tap") or {}).get("served_models", {}))
    if not served: return "unknown"
    return "OpenRouter (paid)" if all("/" in x for x in served) else "institutional proxy"
AD = timezone(timedelta(hours=4))  # Abu Dhabi
# target config -> {task: config the score is taken from}; see the module docstring.
SUBST = {"mlevolve__claude-opus-5-5": {
    "sequence-genomics/sirna-knockdown-efficacy": "mlevolve__claude-opus-5",
    "sequence-genomics/aso-knockdown-efficacy": "mlevolve__claude-opus-5"}}
# configs that exist only to supply such cells: they are not ranked as configurations of their own.
DONOR_ONLY = {"mlevolve__claude-opus-5"}
SUB_MARK = "†"
# Rows run on the Google kit (2026-10-23..27) by a collaborator, through a third-party MaaS gateway: different sandbox
# backend and shared GPUs, so scores compare but agent time does not. Marked in every table and footnoted.
KIT_MARK = "‡"
KIT_API = "MaaS gateway (rednote)"
# Per-task oracle: for every task the best score among the 22 configurations the NMI paper reports (all ranked rows
# except these three). It is a ceiling for "pick the right agent per task", not a configuration anyone ran, and it is
# biased upward because each cell is one noisy run and the maximum picks the luckiest one.
ORACLE_EXCLUDE = {"codexcli__gpt-6.1-sol", "minimal__deepseek-v4-pro-0813", "minimal__gemini-3.1-pro-preview"}
ORACLE_PAIR = ["codexcli__gpt-6-astra", "claudecode__claude-opus-5-5"]


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return 0.0


def summaries(d):
    return list(d.glob("*/*/run_summary.json"))


def upstream(s):
    c = s.parent / "execution_config.json"
    return json.load(open(c)).get("api_upstream") if c.exists() else None


def local_reruns(sums):
    """In a row run elsewhere (its runs record an api_upstream), the tasks rerun on this cluster, which do not."""
    if not any(upstream(s) for s in sums): return []
    return sorted(f"{s.parent.parent.name}/{s.parent.name}" for s in sums if not upstream(s))


rows, running, per_task, donors, per_status = [], [], {}, [], {}
names = sorted({p.name for p in OUT.iterdir() if p.is_dir() and "__" in p.name} if OUT.exists() else set()) 
names = sorted(set(names) | {p.stem for p in RES.glob("*.csv")})
for m in names:
    d = OUT / m; scaf, model = m.split("__", 1); res = RES / f"{m}.csv"; sums = summaries(d) if d.exists() else []
    last = max((s.stat().st_mtime for s in sums), default=0)
    subs = SUBST.get(m, {})
    if m in DONOR_ONLY:
        donors.append((m, len(sums) or 64)); continue
    if not res.exists() or (sums and (len(sums) + len(subs) < 64 or res.stat().st_mtime < last)):
        n_fail = sum(json.load(open(s)).get("status") != "success" for s in sums)
        running.append((m, len(sums), n_fail)); continue
    rs = list(csv.DictReader(open(res))); assert len(rs) == 64, (m, len(rs))
    mean_own = 100 * sum(fnum(r["score"]) for r in rs) / 64  # before substitution, for the footnote
    for task, src in subs.items():
        src_rs = {q["task"]: q for q in csv.DictReader(open(RES / f"{src}.csv"))}
        for i, r in enumerate(rs):
            if r["task"] == task:
                rs[i] = {**src_rs[task], "note": f"score taken from {src} ({src_rs[task].get('note') or 'substituted run'})"}
    st = {}
    for r in rs: st[r["status"]] = st.get(r["status"], 0) + 1
    dom = {k: 100 * sum(fnum(r["score"]) for r in rs if r["domain"] == k) / 8 for k in DOMAINS}
    rows.append(dict(model_dir=m, scaffold=SCAFFOLD.get(scaf, scaf), backend=MODEL.get(model, model), api=API(sums, m),
                     mean64=100 * sum(fnum(r["score"]) for r in rs) / 64, ok=st.get("ok", 0) + st.get("ok_safety_fallback", 0),
                     fail=64 - st.get("ok", 0) - st.get("ok_safety_fallback", 0), statuses=st,
                     runtime_h=sum(fnum(r["runtime_sec"]) for r in rs) / 3600,
                     in_m=sum(fnum(r["input_tokens_total"]) for r in rs) / 1e6, out_m=sum(fnum(r["output_tokens_total"]) for r in rs) / 1e6,
                     subs=dict(subs), mean_own=mean_own,
                     local_reruns=local_reruns(sums),
                     scored=datetime.fromtimestamp(res.stat().st_mtime, AD).strftime("%Y-%m-%d %H:%M"), **{f"d_{k}": v for k, v in dom.items()}))
    for r in rs:
        per_task.setdefault(r["task"], {})[m] = 100 * fnum(r["score"])
        per_status.setdefault(r["task"], {})[m] = dict(status=r["status"], runtime=fnum(r["runtime_sec"]),
                                                       note=r.get("note") or "", sub=r["task"] in subs)

rows.sort(key=lambda r: -r["mean64"])
ocfg = [r["model_dir"] for r in rows if r["model_dir"] not in ORACLE_EXCLUDE]
def oracle(cs):
    best = {t: max(per_task[t].get(c, 0) for c in cs) for t in per_task}
    arg = {t: [c for c in cs if per_task[t].get(c, 0) == best[t]] for t in per_task}
    dom = {k: sum(v for t, v in best.items() if t.startswith(k + "/")) / 8 for k in DOMAINS}
    return best, arg, sum(best.values()) / 64, dom
o_best, o_arg, o_mean, o_dom = oracle(ocfg)
p_best, p_arg, p_mean, p_dom = oracle(ORACLE_PAIR)
now = datetime.now(AD).strftime("%Y-%m-%d %H:%M")

with open(LB / "leaderboard.csv", "w", newline="") as f:
    keys = ["model_dir", "scaffold", "backend", "api", "mean64", "mean64_own_runs_only", "substituted_tasks",
            "ok", "fail", "runtime_h", "in_m", "out_m", "scored"] + [f"d_{k}" for k in DOMAINS]
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
    for r in rows:
        v = {**r, "mean64_own_runs_only": r["mean_own"],
             "substituted_tasks": ";".join(f"{t}<-{s}" for t, s in sorted(r["subs"].items()))}
        w.writerow({k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items()})
with open(LB / "leaderboard_per_task.csv", "w", newline="") as f:
    cols = [r["model_dir"] for r in rows]; w = csv.writer(f)
    w.writerow(["task"] + cols + [f"oracle{len(ocfg)}_best", f"oracle{len(ocfg)}_argmax"])
    for t in sorted(per_task):
        w.writerow([t] + [round(per_task[t].get(c, 0), 2) for c in cols] + [round(o_best[t], 2), ";".join(o_arg[t])])

L = [f"# BioXArena-Pro leaderboard", "",
     f"Updated {now} (Abu Dhabi time). 64 tasks (8 domains x 8), release v5.1.1 (2026-10-21: pring v5.3, tahoe v5.2,",
     "gastrulation v5.2.1, covid v5.1). Scores x100; every task score is in [0, 100] and a failed task scores 0.",
     "Generated by `scoring/make_leaderboard.py` from `scoring/results/<config>.csv`; do not edit by hand.", "",
     "## Finished configurations", "",
     "| # | Agent scaffold | Backend model | API | Mean64 | OK / fail | Agent time (h) | Input / output tokens (M) |",
     "|---:|---|---|---|---:|---:|---:|---:|"]
for i, r in enumerate(rows, 1):
    mark = (SUB_MARK if r["subs"] else "") + (KIT_MARK if r["api"] == KIT_API else "")
    L.append(f"| {i} | {r['scaffold']} | {r['backend']}{mark} | {r['api']} | **{r['mean64']:.1f}** | {r['ok']} / {r['fail']} | "
             f"{r['runtime_h']:.1f} | {r['in_m']:.2f} / {r['out_m']:.2f} |")
L.append(f"| – | *Per-task best of {len(ocfg)} (oracle)* | *see below* | – | ***{o_mean:.1f}*** | – | – | – |")
L += ["", "Input tokens include cached input. Agent time is the sum of the 64 task wall times.", "",
      f"**Per-task oracle.** For each task, the best score among the {len(ocfg)} configurations used in the NMI paper (all",
      "ranked rows except " + ", ".join(f"`{c}`" for c in sorted(ORACLE_EXCLUDE)) + "), averaged over 64 tasks.",
      "It is the ceiling of choosing the right existing agent for every task, not a run, and it is biased upward because",
      "each cell is a single run and the maximum picks the luckiest one. Per-task values and the winning configuration",
      f"are in the last two columns of `leaderboard_per_task.csv`.", "",
      "| Pool | Mean64 | Gain over best single row |", "|---|---:|---:|",
      f"| Best single configuration ({rows[0]['scaffold']} + {rows[0]['backend']}) | {rows[0]['mean64']:.1f} | – |",
      f"| Per-task best of Codex CLI + GPT-6 Astra and Claude Code + Opus 5.5 | {p_mean:.1f} | +{p_mean - rows[0]['mean64']:.1f} |",
      f"| Per-task best of all {len(ocfg)} | {o_mean:.1f} | +{o_mean - rows[0]['mean64']:.1f} |", "",
      "Tasks won (ties count for every tied configuration): " + ", ".join(
          f"{c} {n}" for c, n in sorted(((c, sum(c in a for a in o_arg.values())) for c in ocfg), key=lambda x: -x[1]) if n) + ".", ""]
sub_rows = [r for r in rows if r["subs"]]
if sub_rows:
    L += [f"{SUB_MARK} **Substituted task scores.** The proxy served empty completions for these tasks on the requested",
          "model, so no own result exists and the cell is taken from the configuration named below. Everything else in the",
          "row is that configuration's own work; the mean without the substitution is given for comparison.", "",
          "| Configuration | Task | Score taken from | Mean64 as shown | Mean64 without substitution |",
          "|---|---|---|---:|---:|"]
    for r in sub_rows:
        for t, s in sorted(r["subs"].items()):
            L.append(f"| {r['scaffold']} + {r['backend']} | `{t}` | `{s}` | {r['mean64']:.1f} | {r['mean_own']:.1f} |")
    L += ["", "Why (authors' archive, not released): `v5_failed_attempts/mlevolve__claude-opus-5-5__empty_api_responses_20261022/`",
          "- 74 % and 64 % of calls came back with `output_tokens: 0`, so every MLEvolve node was discarded and no code was",
          "produced. The gateway answered HTTP 200 with no refusal and other lanes on the same model and key were",
          "unaffected, so this is a serving failure rather than a property of the model or the task.", ""]
kit_rows = [r for r in rows if r["api"] == KIT_API]
if kit_rows:
    L += [f"{KIT_MARK} **Run on the Google kit by a collaborator** (2026-10-23 to 10-27), not on this cluster. Same prompt,",
          "task set, 1 GPU and 7200 s limit, but three things differ, so **scores compare and agent time does not**:",
          "",
          "- API: `https://maas.devops.rednote.life/gateway-v2/genai`, a third-party MaaS gateway speaking the native Gemini",
          "  protocol. Gemini CLI speaks it natively; Minimal, OpenCode and MLEvolve reach it through an OpenAI-to-native",
          "  bridge that changes no prompt, model or parameter (every recorded call hit `:streamGenerateContent` or",
          "  `:generateContent`; the bridge changes no prompt, model or parameter).",
          "- Sandbox: `sandbox.py` uid backend (unprivileged user, file permissions) instead of Landlock; public data is",
          "  read-only and answers are absent in both.",
          "- GPUs were shared with other workloads for part of the round, so wall times include contention.",
          "",
          "Infrastructure failures in this round were rerun as everywhere else: 8 runs that died on gateway HTML/404 or on an",
          "external SIGKILL (cgroup evidence: no OOM kill) were rerun by the kit operator, all successfully",
          "(rerun records in the authors' archive)."]
    loc = [(r, t) for r in kit_rows for t in r["local_reruns"]]
    if loc:
        L += ["The OpenCode gateway failures were rerun on this cluster over the paid OpenRouter key instead (OpenCode",
              "takes an OpenAI-compatible endpoint), so these cells used a different transport from the rest of their row:", "",
              "| Configuration | Task |", "|---|---|"] + [f"| {r['scaffold']} + {r['backend']} | `{t}` |" for r, t in loc]
    L += [""]
L += ["## Per-domain mean (x100, 8 tasks each)", "",
      "| Agent scaffold | Backend model | " + " | ".join(DOMAINS) + " |", "|---|---|" + "---:|" * len(DOMAINS)]
for r in rows:
    L.append(f"| {r['scaffold']} | {r['backend']}{SUB_MARK if r['subs'] else ''}{KIT_MARK if r['api'] == KIT_API else ''} | "
             + " | ".join(f"{r['d_' + k]:.1f}" for k in DOMAINS) + " |")
L.append(f"| *Per-task best of {len(ocfg)} (oracle)* | | " + " | ".join(f"*{o_dom[k]:.1f}*" for k in DOMAINS) + " |")
if donors:
    L += ["", "## Substitution sources (not ranked)", "",
          "| Config | Runs |", "|---|---:|"] + [f"| `{m}` | {n} |" for m, n in donors]
    L += ["", f"These exist only to fill the {SUB_MARK} cells above and are not configurations of their own."]
L += ["", "## In progress", ""]
if running:
    L += ["| Config | Runs finished | Of which failed |", "|---|---:|---:|"]
    L += [f"| `{m}` | {n} / 64 | {k} |" for m, n, k in running]
    L += ["", "Not scored yet; some failures above may still be rerun as infrastructure failures."]
else:
    L.append("None.")
L += ["", "## Rules", "",
      "- One prompt for every scaffold (`prompts/unified_prompt_v5.py`), 1 GPU, 7200 s wall clock per task, Landlock sandbox",
      f"  (public data read-only, no access to answers); rows marked {KIT_MARK} used the uid sandbox instead, see above.",
      "- A run over 2 h, a missing or invalid submission, or an out-of-memory kill counts as a failure (score 0).",
      "- Infrastructure failures (API gateway errors, network drops, runs killed by another job) are rerun; the replaced",
      "  outputs are kept in the authors' archive (`$BX_ARCHIVE_ROOT` in this harness).",
      "- The served model is checked on every API response; a run is rejected if under 95 % of responses come from the",
      "  requested model.",
      "- Status counts per config: `ok_safety_fallback` (Claude runs that hit a safety refusal and were rerun with the",
      "  fallback prompt) counts as OK.", "",
      "Files: `leaderboard.csv` (this table), `leaderboard_per_task.csv` (task x config scores, x100), per-run details in",
      "`<config>/<domain>/<task>/run_summary.json`.", ""]
(LB / "LEADERBOARD.md").write_text("\n".join(L))
from make_leaderboard_html import build_html
(LB / "LEADERBOARD.html").write_text(build_html(
    rows=rows, per_task=per_task, per_status=per_status, domains=DOMAINS, now=now, ocfg=ocfg, o_best=o_best,
    o_arg=o_arg, o_mean=o_mean, o_dom=o_dom, p_mean=p_mean, pair=ORACLE_PAIR, sub_mark=SUB_MARK, kit_mark=KIT_MARK,
    kit_api=KIT_API, excluded=sorted(ORACLE_EXCLUDE)))
print(f"{len(rows)} finished, {len(running)} in progress -> {LB / 'LEADERBOARD.md'}")
