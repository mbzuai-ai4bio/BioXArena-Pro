"""Build the interactive leaderboard page docs/index.html (GitHub Pages), in the same format as the BioXArena leaderboard
(https://mbzuai-ai4bio.github.io/BioXArena-ProjectPage/leaderboard.html), from:
  results/leaderboard.csv, results/leaderboard_per_task.csv   (written by scoring/make_leaderboard.py)
  scoring/results/<config>.csv                                 (per-task status, raw metric, runtime, tokens)
  $BX_DATA_PUBLIC/<domain>/<task>/public/                      (task descriptions and file lists; optional)
  $BX_OUTPUT_ROOT/<config>/<domain>/<task>/metrics.json         (the agent's own run report; optional)
The page template is scoring/leaderboard_template.html. Usage: python scoring/make_leaderboard_page.py"""
import csv, json, os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, SRES, OUT = ROOT / "results", ROOT / "scoring" / "results", ROOT / "docs" / "index.html"
PUB = Path(os.environ.get("BX_DATA_PUBLIC", "/nonexistent"))
RUNS = Path(os.environ.get("BX_OUTPUT_ROOT", "/nonexistent"))
RENAME = {}  # release name -> run-directory name (run directories keep the development task names)
if os.environ.get("BX_TASK_RENAME"):
    for r in csv.DictReader(open(os.environ["BX_TASK_RENAME"])):
        RENAME[f"{r['domain']}/{r['new_name']}"] = f"{r['domain']}/{r['old_name']}"
DOM = ["chemical-biology", "imaging", "network-biology", "perturbation", "phenotype-disease", "sequence-genomics",
       "single-cell-spatial", "structure"]
GROUP = {"codexcli": "Coding agent", "claudecode": "Coding agent", "geminicli": "Coding agent", "opencode": "Coding agent",
         "biomni": "Biomedical agent", "mlevolve": "ML-engineering agent", "minimal": "Minimal agent"}
COLORS = ["#58a6ff", "#d2a8ff", "#fb923c", "#3fb950", "#f778ba", "#2dd4bf", "#e3b341", "#a5d6ff", "#ff7b72", "#7ee787",
          "#bc8cff", "#ffa657", "#79c0ff", "#56d364", "#db61a2", "#d29922", "#8b949e", "#f0883e", "#a371f7", "#39c5cf",
          "#e6edf3", "#ffdf5d"]
RED = [(re.compile(r"/(?:lus" r"tre|home|users?|scratch|tmp)/[^\s\"'`:,;)]*"), "<path>"),   # absolute cluster paths
       (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "<key>"), (re.compile(r"https?://[^\s\"'`]*gateway[^\s\"'`]*"), "<api>")]
def redact(t):
    t = str(t)
    for p, r in RED: t = p.sub(r, t)
    return t

lb = list(csv.DictReader(open(RES / "leaderboard.csv")))
pt = {r["task"]: r for r in csv.DictReader(open(RES / "leaderboard_per_task.csv"))}
task_keys = list(pt)
agents, summaries, cells = [], [], {}
rows = {k: {"task": k.split("/")[1], "domain": k.split("/")[0]} for k in task_keys}
for i, r in enumerate(lb):
    c = r["model_dir"]; scaf = c.split("__")[0]
    name = f"{r['scaffold']} ({r['backend']})"
    if r["substituted_tasks"]: name += " †"
    if r["api"] == "Gemini-protocol gateway": name += " ‡"
    key = re.sub(r"[^a-z0-9]", "_", c)
    SS = {"Codex CLI": "Codex", "Claude Code": "Claude Code", "Gemini CLI": "Gemini CLI", "OpenCode": "OpenCode",
          "Biomni": "Biomni", "MLEvolve": "MLEvolve", "Minimal": "Minimal"}
    SB = {"GPT-6 Astra": "GPT-6 Astra", "GPT-5.6 Sol": "GPT-5.6 Sol", "Claude Opus 5.5": "Opus 5.5", "Claude Opus 5": "Opus 5",
          "Gemini 3.8 Flash": "Gem 3.8 Flash", "Gemini 3.1 Pro (preview)": "Gem 3.1 Pro", "DeepSeek V4.1 Flash (1017)": "DS V4.1 Flash"}
    mk = (" \u2020" if r["substituted_tasks"] else "") + (" \u2021" if r["api"] == "Gemini-protocol gateway" else "")
    short = f"{SS[r['scaffold']]}<br>{SB.get(r['backend'], r['backend'])}{mk}"
    agents.append({"key": key, "name": name, "short": short, "color": COLORS[i % len(COLORS)]})
    res = {q["task"]: q for q in csv.DictReader(open(SRES / f"{c}.csv"))}
    cells[key] = {}
    for k in task_keys:
        q = res.get(k, {}); s = pt[k].get(c); sc = float(s) / 100 if s not in (None, "") else 0.0
        st = q.get("status") or ""
        subst = k.split("/")[1] in (r["substituted_tasks"] or "") or k in (r["substituted_tasks"] or "")
        ok = st.startswith("ok") or (subst and sc > 0)
        if subst: q = dict(q, note="score taken from MLEvolve + Claude Opus 5 (the API returned empty completions for Opus 5.5)")
        elif st == "ok_safety_fallback": q = dict(q, note="completed by the CLI's built-in refusal fallback (Opus 5) after a biosafety refusal")
        raw = q.get("raw"); raw = float(raw) if raw not in (None, "", "nan") else None
        rows[k][key] = sc; rows[k][key + "_raw"] = raw
        if not ok:
            rows[k][key + "_status"] = "Fail"; rows[k][key + "_remark"] = redact(q.get("note") or q.get("status") or "failed")
        d = {"score": sc, "raw": raw, "metric_name": q.get("metric"), "ok": ok, "eval_status": "Succeed" if ok else "Fail",
             "r_status": q.get("status"), "r_duration_sec": float(q["runtime_sec"]) if q.get("runtime_sec") else None,
             "r_input_tokens": int(float(q["input_tokens_total"])) if q.get("input_tokens_total") else None,
             "r_output_tokens": int(float(q["output_tokens_total"])) if q.get("output_tokens_total") else None}
        if not ok or q.get("note"): d["eval_remark"] = redact(q.get("note") or "")
        mp = RUNS / c / RENAME.get(k, k) / "metrics.json"
        if mp.exists():
            try:
                m = json.load(open(mp))
                for f in ["model_used", "solution_generation_time_sec", "train_time_sec", "test_time_sec", "code_total_time_sec"]:
                    if m.get(f) is not None: d["m_" + f] = redact(m[f]) if isinstance(m[f], str) else m[f]
                if m.get("notes"): d["m_notes"] = redact(m["notes"] if isinstance(m["notes"], str) else json.dumps(m["notes"]))[:1500]
            except Exception: pass
        cells[key][k] = d
    sc_all = [rows[k][key] for k in task_keys]
    ok_tasks = [rows[k][key] for k in task_keys if rows[k].get(key + "_status") != "Fail"]
    dm = {d_: sum(rows[k][key] for k in task_keys if k.startswith(d_ + "/")) / 8 for d_ in DOM}
    summaries.append({"key": key, "scaffold": r["scaffold"], "backbone": r["backend"], "display": name, "color": agents[-1]["color"], "group": GROUP[scaf], "status": "final",
                      "total": 64, "succeeded": int(r["ok"]), "sub_count": 64, "eval_ok": int(r["ok"]),
                      "mean_succ": sum(ok_tasks) / max(1, len(ok_tasks)), "mean_all": sum(sc_all) / 64,
                      "domain_avg": sum(dm.values()) / 8, "domains": dm})
summaries.sort(key=lambda s: -s["mean_all"])
order = [s["key"] for s in summaries]; agents.sort(key=lambda a: order.index(a["key"]))

# task facts (from the public release, when available)
cat = {}
catp = ROOT / "results" / "task_facts.json"
tdet = {}
CAT, MOD = {}, {}
if os.environ.get("BX_TASK_CATALOG"):
    CAT = {f"{r['domain']}/{r['task']}": r for r in csv.DictReader(open(os.environ["BX_TASK_CATALOG"]))}
if os.environ.get("BX_SOURCE_MODALITY"):
    MOD = {f"{r['domain']}/{r['task']}": r for r in csv.DictReader(open(os.environ["BX_SOURCE_MODALITY"])) if r["version"] != "v1"}
for k in task_keys:
    dom, t = k.split("/"); pub = PUB / dom / t / "public"
    meta = {"domain": dom, "task": t, "was_corrected": "no"}
    if (pub / "description.md").exists():
        meta["description_md"] = redact((pub / "description.md").read_text(errors="replace"))
        meta["public_files"] = sorted(p.name for p in pub.iterdir())[:60]
    dev = RENAME.get(k, k)
    c_ = CAT.get(dev); m_ = MOD.get(dev)
    if c_:
        meta.update({"train_units_updated": str(int(float(c_["train_rows"]))) if c_["train_rows"] not in ("", "nan") else None,
                     "test_units_updated": str(int(float(c_["test_rows"]))) if c_["test_rows"] not in ("", "nan") else None,
                     "count_basis": "prediction units (rows of train.csv / test.csv, or instances counted from the data files)",
                     "metric": c_["metric"], "split": c_["split_strategy"]})
    if m_:
        meta.update({"n_input_sources": m_["n_input_sources"], "input_sources": m_["input_sources"].replace(" ; ", " | "),
                     "modalities_considered": m_["modality_families"],
                     "is_multimodal_paper_standard": {"yes": "yes", "borderline": "borderline", "no": "no"}.get(m_["multimodal_status"], m_["multimodal_status"])})
    tdet[k] = meta
DETAILS = {"cells": cells, "tasks": tdet}

tpl = (Path(__file__).with_name("leaderboard_template.html")).read_text()
R = [("<title>BioXArena Leaderboard</title>", "<title>BioXArena-Pro Leaderboard</title>"),
     ("<h1>Bio<span>X</span>Arena Leaderboard</h1>", "<h1>Bio<span>X</span>Arena-Pro Leaderboard</h1>"),
     ("Main results of the NeurIPS 2026 paper (Figure 3): 18 agent / backbone configurations on the 76 BioXArena tasks across 9 domains.",
      "Main results of the paper (Fig. 2): 22 agent / backbone configurations on the 64 BioXArena-Pro tasks across 8 domains."),
     ('id="stat-tasks">76<', 'id="stat-tasks">64<'), ('id="stat-domains">9<', 'id="stat-domains">8<'),
     ("tasks (out of 76) whose", "tasks (out of 64) whose"), ("over all 76 tasks", "over all 64 tasks"),
     ("Per-task scores (76 tasks &times; 18 configs)", "Per-task scores (64 tasks &times; 22 configs)"),
     ("BioXArena — Multi-modal Biomedical Benchmark for LLM Agents", "BioXArena-Pro — a research-grade benchmark for autonomous biomedical machine learning agents"),
     ('<a href="https://mbzuai-ai4bio.github.io/BioXArena-ProjectPage/" style="color:var(--accent)">Project page</a> &middot; <a href="https://github.com/mbzuai-ai4bio/BioXArena" style="color:var(--accent)">Code</a>',
      '<a href="https://github.com/mbzuai-ai4bio/BioXArena-Pro" style="color:var(--accent)">Code</a> &middot; <a href="https://huggingface.co/datasets/mbzuai-ai4bio/BioXArena-Pro-Public" style="color:var(--accent)">Data</a> &middot; <a href="https://www.codabench.org/competitions/18583/" style="color:var(--accent)">Codabench</a>'),
     ("document.getElementById('update-date').textContent = '2026-10-13';", "document.getElementById('update-date').textContent = '2026-11-10';"),
     ('const doms = ["chemical-biology", "imaging", "network-biology", "perturbation-dynamics", "phenotype-disease", "sequence", "single-cell", "structure", "text-integrated"];',
      "const doms = " + json.dumps(DOM) + ";"),
     ("const short = {'chemical-biology':'Chem','imaging':'Img','network-biology':'Net','perturbation-dynamics':'Pert','phenotype-disease':'Phen','sequence':'Seq','single-cell':'SC','structure':'Str','text-integrated':'Text'};",
      "const short = {'chemical-biology':'Chem','imaging':'Img','network-biology':'Net','perturbation':'Pert','phenotype-disease':'Phen','sequence-genomics':'Seq','single-cell-spatial':'SC','structure':'Str'};"),
     ("<th>OK/76</th>", "<th>OK/64</th>"),
     ("""${meta.was_corrected === 'yes' ? '<span class="pill yes" style="margin-left:8px">corrected in Oct 2026</span>' : '<span class="pill no" style="margin-left:8px">unchanged task</span>'}""", ""),
     ("  .domain-badge.text-integrated {", "  .domain-badge.perturbation { background:rgba(210,153,34,0.15); color:var(--yellow); }\n  .domain-badge.sequence-genomics { background:rgba(45,212,191,0.15); color:var(--teal); }\n  .domain-badge.single-cell-spatial { background:rgba(248,81,73,0.15); color:#f97583; }\n  .domain-badge.text-integrated {")]
for a, b in R:
    assert tpl.count(a) == 1, a[:70]
    tpl = tpl.replace(a, b)
note = ('<p class="note">&dagger; Two task scores taken from MLEvolve with Opus 5, because the API returned empty completions for '
        'Opus 5.5 on those tasks. &Dagger; Run through a gateway for the native Gemini protocol on a separate machine of the same '
        'hardware class; scores are comparable, agent times are not.</p>\n<div id="summary-cards"')
tpl = tpl.replace('<div id="summary-cards"', note, 1)
dump = lambda o: json.dumps(o, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
html = (tpl.replace("__TASKS__", dump([rows[k] for k in task_keys])).replace("__AGENTS__", dump(agents))
        .replace("__SUMMARIES__", dump(summaries)).replace("__DETAILS__", dump(DETAILS)))
OUT.parent.mkdir(exist_ok=True); OUT.write_text(html)
print("wrote", OUT, f"{len(html)/1e6:.2f} MB", "configs", len(agents), "tasks", len(rows),
      "descriptions", sum("description_md" in v for v in tdet.values()),
      "run reports", sum(1 for a in cells.values() for v in a.values() if "m_model_used" in v))
