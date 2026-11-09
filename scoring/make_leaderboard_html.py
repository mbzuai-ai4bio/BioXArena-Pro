"""LEADERBOARD.html for BioXArena-Pro: the tables of LEADERBOARD.md as one self-contained page (no external files).
Called by make_leaderboard.py with the data it has already computed, so the page and the markdown can never disagree."""
import html, json

OK = {"ok", "ok_safety_fallback"}


def build_html(*, rows, per_task, per_status, domains, now, ocfg, o_best, o_arg, o_mean, o_dom, p_mean, pair,
               sub_mark, kit_mark, kit_api, excluded):
    cols = [r["model_dir"] for r in rows]
    label = {r["model_dir"]: f"{r['scaffold']} + {r['backend']}" for r in rows}
    marks = {r["model_dir"]: (sub_mark if r["subs"] else "") + (kit_mark if r["api"] == kit_api else "") for r in rows}
    tasks = sorted(per_task, key=lambda t: (domains.index(t.split("/")[0]), t))
    data = dict(
        cols=cols, label=label, marks=marks, domains=domains, excluded=excluded, ocfg=ocfg,
        rows=[dict(m=r["model_dir"], scaffold=r["scaffold"], backend=r["backend"], api=r["api"], mean=round(r["mean64"], 2),
                   ok=r["ok"], fail=r["fail"], time=round(r["runtime_h"], 1), tin=round(r["in_m"], 2), tout=round(r["out_m"], 2),
                   dom=[round(r["d_" + d], 1) for d in domains], paper=r["model_dir"] not in excluded) for r in rows],
        tasks=[dict(t=t, s=[round(per_task[t].get(c, 0), 1) for c in cols],
                    f=[int(per_status[t][c]["status"] not in OK) for c in cols],
                    rt=[round(per_status[t][c]["runtime"] / 60) for c in cols],
                    sub=[int(per_status[t][c]["sub"]) for c in cols],
                    best=round(o_best[t], 1), arg=o_arg[t]) for t in tasks],
        oracle=dict(mean=round(o_mean, 2), dom=[round(o_dom[d], 1) for d in domains], pair=round(p_mean, 2),
                    pair_names=[label[c] for c in pair], n=len(ocfg)),
        wins=sorted(((label[c] + marks[c], sum(c in a for a in o_arg.values())) for c in ocfg), key=lambda x: -x[1]))
    best = rows[0]
    return PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__NOW__", html.escape(now)) \
        .replace("__N__", str(len(rows))).replace("__NO__", str(len(ocfg))).replace("__OMEAN__", f"{o_mean:.1f}") \
        .replace("__BEST__", f"{best['mean64']:.1f}").replace("__BESTNAME__", html.escape(label[best["model_dir"]])) \
        .replace("__PAIR__", f"{p_mean:.1f}").replace("__KIT__", kit_mark).replace("__SUB__", sub_mark) \
        .replace("__EXCL__", ", ".join(f"<code>{html.escape(e)}</code>" for e in excluded))


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BioXArena-Pro Leaderboard</title>
<style>
:root { --bg:#ffffff; --panel:#f6f7f9; --text:#1b1f24; --text2:#5b6470; --border:#dde1e6; --accent:#2563eb;
        --fail:#c62828; --failbg:#fdecea; --best:#111; --row:#fafbfc; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0f1216; --panel:#171b21; --text:#e6e9ee;
        --text2:#9aa3ad; --border:#2a3038; --accent:#6ea8ff; --fail:#ff6b6b; --failbg:#3a1c1c; --best:#fff; --row:#13171c; } }
:root[data-theme="dark"] { --bg:#0f1216; --panel:#171b21; --text:#e6e9ee; --text2:#9aa3ad; --border:#2a3038;
        --accent:#6ea8ff; --fail:#ff6b6b; --failbg:#3a1c1c; --best:#fff; --row:#13171c; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
.wrap { max-width:1280px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:1.6em; margin:0 0 4px; } h2 { font-size:1.2em; margin:32px 0 10px; }
.sub { color:var(--text2); margin:0 0 18px; }
.stats { display:flex; flex-wrap:wrap; gap:10px; margin:14px 0 6px; }
.stat { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:10px 14px; min-width:130px; }
.stat .v { font-size:1.45em; font-weight:650; font-variant-numeric:tabular-nums; } .stat .l { color:var(--text2); font-size:.82em; }
.tw { overflow-x:auto; border:1px solid var(--border); border-radius:8px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { padding:6px 9px; border-bottom:1px solid var(--border); white-space:nowrap; text-align:right; }
th { background:var(--panel); font-weight:600; position:sticky; top:0; z-index:1; }
th.s { cursor:pointer; user-select:none; } th.s:hover { color:var(--accent); }
.l, th.l { text-align:left; }
tr.ex td { color:var(--text2); }
tr.or td { background:var(--panel); font-style:italic; font-weight:600; }
.bar { display:inline-block; height:8px; border-radius:4px; background:var(--accent); vertical-align:middle; margin-right:6px; opacity:.75; }
.pill { font-size:.75em; padding:1px 7px; border-radius:9px; border:1px solid var(--border); color:var(--text2); margin-left:6px; font-style:normal; }
.note { color:var(--text2); font-size:.88em; max-width:980px; } .note b { color:var(--text); }
code { background:var(--panel); padding:1px 4px; border-radius:4px; font-size:.9em; }
.mx td, .mx th { padding:3px 5px; font-size:12px; }
.mx th.c { writing-mode:vertical-rl; transform:rotate(180deg); text-align:left; height:190px; vertical-align:bottom; font-weight:500; }
.mx td.t { position:sticky; left:0; background:var(--bg); z-index:1; text-align:left; max-width:260px; overflow:hidden; text-overflow:ellipsis; }
.mx th.t { position:sticky; left:0; z-index:2; }
.mx td.b { outline:2px solid var(--best); outline-offset:-2px; font-weight:700; }
.mx td.f { color:var(--fail); }
.mx tr.dh td { background:var(--panel); font-weight:600; text-align:left; position:static; }
.mx td.o { border-left:2px solid var(--border); font-weight:650; }
.ctl { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:8px 0; }
input[type=search] { padding:6px 10px; border:1px solid var(--border); border-radius:6px; background:var(--bg); color:var(--text); min-width:220px; }
label { color:var(--text2); font-size:.9em; }
.wins { columns:2 280px; font-size:.9em; color:var(--text2); padding-left:18px; }
footer { margin-top:36px; color:var(--text2); font-size:.82em; }
</style>
</head>
<body>
<div class="wrap">
<h1>BioXArena-Pro leaderboard</h1>
<p class="sub">64 tasks (8 domains &times; 8), release v5.1.1. Scores &times;100; a failed task scores 0. Updated __NOW__ (Abu Dhabi time).</p>
<div class="stats">
  <div class="stat"><div class="v">__N__</div><div class="l">configurations</div></div>
  <div class="stat"><div class="v">__BEST__</div><div class="l">best single: __BESTNAME__</div></div>
  <div class="stat"><div class="v">__PAIR__</div><div class="l">per-task best of top 2</div></div>
  <div class="stat"><div class="v">__OMEAN__</div><div class="l">per-task best of __NO__ (oracle)</div></div>
</div>

<h2>Finished configurations</h2>
<div class="ctl"><label><input type="checkbox" id="onlyPaper"> only the __NO__ configurations in the NMI paper</label></div>
<div class="tw"><table id="main"></table></div>
<p class="note">Click a column header to sort. Input tokens include cached input. Agent time is the sum of the 64 task wall
times. Rows marked <span class="pill">not in paper</span> are on the leaderboard but outside the __NO__ configurations
the NMI paper reports. __SUB__ = two task cells taken from MLEvolve + Opus 5 because the proxy served empty completions
for Opus 5.5 on those tasks. __KIT__ = run by a collaborator on the Google kit (MaaS gateway, uid sandbox, shared GPUs):
scores compare, agent time does not. Full footnotes in <code>LEADERBOARD.md</code>.</p>

<h2>Per-task oracle</h2>
<p class="note"><b>For each task, the best score among the __NO__ paper configurations</b> (all rows except __EXCL__),
averaged over 64 tasks: <b>__OMEAN__</b>, against __BEST__ for the best single configuration. Taking the better of
Codex CLI + GPT-6 Astra and Claude Code + Opus 5.5 per task gives __PAIR__. The oracle is the ceiling of picking the right
existing agent for every task, not a run, and it is biased upward: each cell is one run and the maximum picks the
luckiest one. Tasks won (ties count for every tied configuration):</p>
<ul class="wins" id="wins"></ul>

<h2>Per-domain mean (8 tasks each)</h2>
<div class="tw"><table id="dom"></table></div>

<h2>Per-task scores</h2>
<div class="ctl">
  <input type="search" id="q" placeholder="Filter tasks…">
  <label><input type="checkbox" id="mxPaper" checked> only paper configurations</label>
</div>
<p class="note">Boxed cell = best of the __NO__ paper configurations for that task (the oracle column). Red = failed run
(score 0). Hover a cell for its run time. Columns are in leaderboard order.</p>
<div class="tw" style="max-height:80vh"><table class="mx" id="mx"></table></div>

<footer>Generated by <code>2_BioXArena-Pro-Code/scoring/make_leaderboard.py</code> from <code>scoring/results/&lt;config&gt;.csv</code>;
same data as <code>LEADERBOARD.md</code>, <code>leaderboard.csv</code> and <code>leaderboard_per_task.csv</code>. Do not edit by hand.</footer>
</div>
<script>
const D = __DATA__;
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const heat = v => { const a = Math.max(0, Math.min(1, v / 90)); return `rgba(37,99,235,${(0.06 + 0.5 * a).toFixed(3)})`; };
let sortKey = "mean", sortDir = -1;
const COLS = [["#", null, "r"], ["Agent scaffold", "scaffold", "l"], ["Backend model", "backend", "l"], ["API", "api", "l"],
  ["Mean64", "mean", "r"], ["OK / fail", "fail", "r"], ["Agent time (h)", "time", "r"], ["Input / output tokens (M)", "tin", "r"]];
const rank = {}; D.rows.forEach((r, i) => rank[r.m] = i + 1);
function renderMain() {
  const only = document.getElementById("onlyPaper").checked;
  let rs = D.rows.filter(r => !only || r.paper).slice();
  rs.sort((a, b) => { const x = a[sortKey], y = b[sortKey]; return (typeof x === "string" ? x.localeCompare(y) : x - y) * sortDir; });
  const mx = Math.max(...D.rows.map(r => r.mean));
  let h = "<thead><tr>" + COLS.map(([n, k, c]) => `<th class="${c === "l" ? "l " : ""}${k ? "s" : ""}" data-k="${k || ""}">${n}${k === sortKey ? (sortDir < 0 ? " ▾" : " ▴") : ""}</th>`).join("") + "</tr></thead><tbody>";
  for (const r of rs) h += `<tr class="${r.paper ? "" : "ex"}"><td>${rank[r.m]}</td><td class="l">${esc(r.scaffold)}</td>` +
    `<td class="l">${esc(r.backend)}${esc(D.marks[r.m])}${r.paper ? "" : '<span class="pill">not in paper</span>'}</td><td class="l">${esc(r.api)}</td>` +
    `<td><span class="bar" style="width:${(60 * r.mean / mx).toFixed(0)}px"></span><b>${r.mean.toFixed(1)}</b></td>` +
    `<td>${r.ok} / ${r.fail}</td><td>${r.time.toFixed(1)}</td><td>${r.tin.toFixed(2)} / ${r.tout.toFixed(2)}</td></tr>`;
  h += `<tr class="or"><td>–</td><td class="l" colspan="3">Per-task best of ${D.oracle.n} (oracle)</td><td><b>${D.oracle.mean.toFixed(1)}</b></td><td>–</td><td>–</td><td>–</td></tr>`;
  h += `<tr class="or"><td>–</td><td class="l" colspan="3">Per-task best of ${esc(D.oracle.pair_names.join(" / "))}</td><td><b>${D.oracle.pair.toFixed(1)}</b></td><td>–</td><td>–</td><td>–</td></tr>`;
  document.getElementById("main").innerHTML = h + "</tbody>";
  document.querySelectorAll("#main th.s").forEach(th => th.onclick = () => {
    const k = th.dataset.k; if (k === sortKey) sortDir = -sortDir; else { sortKey = k; sortDir = (k === "scaffold" || k === "backend" || k === "api") ? 1 : -1; }
    renderMain(); });
}
function renderDom() {
  let h = "<thead><tr><th class='l'>Configuration</th>" + D.domains.map(d => `<th>${esc(d)}</th>`).join("") + "</tr></thead><tbody>";
  for (const r of D.rows) h += `<tr class="${r.paper ? "" : "ex"}"><td class="l">${esc(D.label[r.m] + D.marks[r.m])}</td>` +
    r.dom.map(v => `<td style="background:${heat(v)}">${v.toFixed(1)}</td>`).join("") + "</tr>";
  h += `<tr class="or"><td class="l">Per-task best of ${D.oracle.n} (oracle)</td>` + D.oracle.dom.map(v => `<td>${v.toFixed(1)}</td>`).join("") + "</tr>";
  document.getElementById("dom").innerHTML = h + "</tbody>";
}
function renderMx() {
  const q = document.getElementById("q").value.toLowerCase(), only = document.getElementById("mxPaper").checked;
  const idx = D.cols.map((c, i) => i).filter(i => !only || !D.excluded.includes(D.cols[i]));
  let h = "<thead><tr><th class='l t'>Task</th>" + idx.map(i => `<th class="c" title="${esc(D.cols[i])}">${esc(D.label[D.cols[i]] + D.marks[D.cols[i]])}</th>`).join("") +
    `<th class="c">Oracle (best of ${D.oracle.n})</th></tr></thead><tbody>`;
  let dom = "";
  for (const t of D.tasks) {
    if (q && !t.t.toLowerCase().includes(q)) continue;
    const d = t.t.split("/")[0];
    if (d !== dom) { dom = d; h += `<tr class="dh"><td colspan="${idx.length + 2}">${esc(d)}</td></tr>`; }
    h += `<tr><td class="l t" title="${esc(t.t)}">${esc(t.t.split("/")[1])}</td>` + idx.map(i => {
      const c = D.cols[i], v = t.s[i], cls = (t.arg.includes(c) ? "b " : "") + (t.f[i] ? "f" : "");
      const tip = `${D.label[c]} · ${t.f[i] ? "FAILED · " : ""}${t.rt[i]} min${t.sub[i] ? " · substituted cell" : ""}`;
      return `<td class="${cls}" style="background:${t.f[i] ? "var(--failbg)" : heat(v)}" title="${esc(tip)}">${t.f[i] ? "F" : v.toFixed(1)}</td>`;
    }).join("") + `<td class="o" title="${esc(t.arg.map(c => D.label[c]).join(" / "))}">${t.best.toFixed(1)}</td></tr>`;
  }
  document.getElementById("mx").innerHTML = h + "</tbody>";
}
document.getElementById("wins").innerHTML = D.wins.filter(w => w[1]).map(w => `<li>${esc(w[0])}: ${w[1]}</li>`).join("");
document.getElementById("onlyPaper").onchange = renderMain;
document.getElementById("q").oninput = renderMx; document.getElementById("mxPaper").onchange = renderMx;
renderMain(); renderDom(); renderMx();
</script>
</body>
</html>
"""
