"""Build the interactive leaderboard page docs/index.html (served by GitHub Pages) from results/leaderboard.csv and
results/leaderboard_per_task.csv, which scoring/make_leaderboard.py writes. Self-contained HTML, no external assets.
Usage: python scoring/make_leaderboard_page.py"""
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "docs" / "index.html"
DOM = ["chemical-biology", "imaging", "network-biology", "perturbation", "phenotype-disease", "sequence-genomics",
       "single-cell-spatial", "structure"]
DLAB = {"chemical-biology": "Chemical biology", "imaging": "Imaging", "network-biology": "Network biology",
        "perturbation": "Perturbation", "phenotype-disease": "Phenotype-disease", "sequence-genomics": "Sequence & genomics",
        "single-cell-spatial": "Single-cell & spatial", "structure": "Structure"}

rows = list(csv.DictReader(open(RES / "leaderboard.csv")))
pt = list(csv.DictReader(open(RES / "leaderboard_per_task.csv")))
cfgs = [r["model_dir"] for r in rows]
# exact means from the per-task scores (leaderboard.csv rounds to 2 decimals, which would mis-round e.g. 52.947 -> 53.0)
def exact(c, dom=None):
    v = [float(r[c] or 0) for r in pt if dom is None or r["task"].startswith(dom + "/")]
    return sum(v) / len(v)
conf = [{"id": r["model_dir"], "scaffold": r["scaffold"], "backbone": r["backend"], "mean": exact(r["model_dir"]),
         "ok": int(r["ok"]), "fail": int(r["fail"]), "time": float(r["runtime_h"]), "tin": float(r["in_m"]),
         "tout": float(r["out_m"]), "sub": bool(r["substituted_tasks"]), "gw": r["api"] == "Gemini-protocol gateway",
         "dom": [exact(r["model_dir"], d) for d in DOM]} for r in rows]
tasks = [{"t": r["task"].split("/")[1], "d": r["task"].split("/")[0],
          "s": [None if r[c] in ("", None) else float(r[c]) for c in cfgs]} for r in pt]
oracle = sum(float(r["oracle22_best"]) for r in pt) / len(pt)
data = {"conf": conf, "tasks": tasks, "dom": DOM, "dlab": DLAB, "oracle": round(oracle, 1)}

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BioXArena-Pro Leaderboard</title>
<style>
:root{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;--text2:#8b949e;--accent:#58a6ff;--green:#3fb950;--red:#f85149}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);line-height:1.5}
.container{max-width:1400px;margin:0 auto;padding:24px 16px}
.header{text-align:center;padding:40px 0 28px;border-bottom:1px solid var(--border);margin-bottom:28px}
.header h1{font-size:2.3em;font-weight:700}.header h1 span{color:var(--accent)}
.header p{color:var(--text2);font-size:1.05em;max-width:820px;margin:8px auto 0}
.links{margin-top:14px}.links a{color:var(--accent);text-decoration:none;margin:0 10px;font-size:.95em}
.stats{display:flex;justify-content:center;gap:40px;margin-top:22px;flex-wrap:wrap}
.stat .value{font-size:1.9em;font-weight:700;color:var(--accent);text-align:center}.stat .label{font-size:.82em;color:var(--text2);text-align:center}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-bottom:28px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:18px}
.card h3{font-size:.8em;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.card .name{font-size:1.15em;font-weight:600}.card .score{font-size:2em;font-weight:700;color:var(--green)}
.card .meta{color:var(--text2);font-size:.85em;margin-top:6px}
.tabs{display:flex;border-bottom:1px solid var(--border);overflow-x:auto}
.tab{padding:11px 18px;cursor:pointer;color:var(--text2);border:none;border-bottom:2px solid transparent;font-size:.95em;background:none;font-family:inherit;white-space:nowrap}
.tab:hover{color:var(--text)}.tab.active{color:var(--accent);border-bottom-color:var(--accent);font-weight:600}
.panel{display:none;padding-top:16px}.panel.active{display:block}
.controls{display:flex;gap:10px;margin:0 0 12px;flex-wrap:wrap}
.controls input,.controls select{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:7px 11px;color:var(--text);font-size:.9em}
.tw{background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.88em;font-variant-numeric:tabular-nums}
th{padding:10px 12px;text-align:left;background:rgba(110,118,129,.1);color:var(--text2);font-weight:600;font-size:.82em;text-transform:uppercase;letter-spacing:.3px;cursor:pointer;white-space:nowrap;position:sticky;top:0}
th:hover{color:var(--text)}td{padding:8px 12px;border-top:1px solid var(--border);white-space:nowrap}
tr:hover td{background:rgba(110,118,129,.06)}td.n{text-align:right}th.n{text-align:right}
.bar{display:inline-block;height:8px;border-radius:4px;background:var(--accent);vertical-align:middle;margin-left:8px;opacity:.8}
.best{outline:2px solid var(--green);outline-offset:-2px;font-weight:700}
.fail{color:var(--red)}.mark{color:var(--text2);font-size:.85em}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:.78em;background:rgba(88,166,255,.15);color:var(--accent)}
.note{color:var(--text2);font-size:.85em;margin-top:12px;max-width:1100px}
.mx td{padding:5px 8px;text-align:center}.mx th{font-size:.72em}.mx td.t{text-align:left;position:sticky;left:0;background:var(--surface)}
.grid td{text-align:center;font-weight:600;padding:14px}
footer{color:var(--text2);font-size:.82em;text-align:center;margin:36px 0 10px}
</style></head><body><div class="container">
<div class="header">
<h1>BioXArena-<span>Pro</span> Leaderboard</h1>
<p>64 research-level biomedical machine-learning tasks in 8 domains, each rebuilt from a published study and audited before release. An agent gets the task files, one GPU and two hours, and must return predictions for a held-out test set. Scores are on a common 0&ndash;100 scale (higher is better); a failed run scores 0.</p>
<div class="links"><a href="https://github.com/mbzuai-ai4bio/BioXArena-Pro">Code</a><a href="https://huggingface.co/datasets/mbzuai-ai4bio/BioXArena-Pro-Public">Data</a><a href="https://www.codabench.org/competitions/18583/">Submit to Codabench</a></div>
<div class="stats">
<div class="stat"><div class="value">64</div><div class="label">tasks</div></div>
<div class="stat"><div class="value">8</div><div class="label">domains</div></div>
<div class="stat"><div class="value">22</div><div class="label">configurations</div></div>
<div class="stat"><div class="value">7</div><div class="label">agent scaffolds</div></div>
<div class="stat"><div class="value" id="sBest"></div><div class="label">best configuration</div></div>
<div class="stat"><div class="value" id="sOr"></div><div class="label">per-task best of 22</div></div>
</div></div>
<div class="cards" id="cards"></div>
<div class="tabs">
<button class="tab active" data-p="pOverall">Overall</button>
<button class="tab" data-p="pDomain">By domain</button>
<button class="tab" data-p="pGrid">Scaffold &times; backbone</button>
<button class="tab" data-p="pTask">Per task</button>
</div>
<div class="panel active" id="pOverall"><div class="tw"><table id="tOverall"></table></div>
<p class="note">Click a column header to sort. Mean is the mean score over the 64 tasks (&times;100). OK / fail counts valid and failed runs. Agent time is the wall-clock time summed over the 64 tasks; tokens are in millions (input includes cached prompt tokens). &dagger; Two task scores taken from MLEvolve with Opus 5, because the API returned empty completions for Opus 5.5 on those tasks. &Dagger; Run through a gateway for the native Gemini protocol on a separate machine of the same hardware class; scores are comparable, agent times are not.</p></div>
<div class="panel" id="pDomain"><div class="tw"><table id="tDomain"></table></div><p class="note">Mean score over the 8 tasks of each domain (&times;100). Darker cells are higher.</p></div>
<div class="panel" id="pGrid"><div class="tw"><table class="grid" id="tGrid"></table></div><p class="note">The four model-agnostic scaffolds crossed with four backbones (mean over 64 tasks, &times;100), with row and column means.</p></div>
<div class="panel" id="pTask"><div class="controls"><input id="q" placeholder="Search task..."><select id="fd"></select></div>
<div class="tw"><table class="mx" id="tTask"></table></div><p class="note">Per-task scores (&times;100) of all 22 configurations; the best configuration on each task is outlined; &ldquo;&ndash;&rdquo; marks a failed run.</p></div>
<footer>BioXArena-Pro &middot; code MIT, data CC BY-NC 4.0 &middot; generated by <code>scoring/make_leaderboard_page.py</code></footer>
</div>
<script>
(function () {
const D = __DATA__;
const C = D.conf, f1 = x => (Math.round(x * 10 + 1e-9) / 10).toFixed(1);
const lab = c => c.backbone + (c.sub ? ' <span class="mark">&dagger;</span>' : '') + (c.gw ? ' <span class="mark">&Dagger;</span>' : '');
document.getElementById('sBest').textContent = f1(Math.max(...C.map(c => c.mean)));
document.getElementById('sOr').textContent = D.oracle.toFixed(1);
const leaders = [...C].sort((a, b) => b.mean - a.mean).slice(0, 3);
document.getElementById('cards').innerHTML = leaders.map((c, i) => `<div class="card"><h3>#${i + 1}</h3><div class="name">${c.scaffold} + ${c.backbone}</div><div class="score">${f1(c.mean)}</div><div class="meta">${c.ok} / ${c.fail} valid / failed &middot; ${c.time.toFixed(1)} agent-hours</div></div>`).join('');
function sortable(id, cols, data, render, key0) {
  let key = key0, dir = -1; const el = document.getElementById(id);
  function draw() {
    const d = [...data].sort((a, b) => { const x = cols[key].v(a), y = cols[key].v(b); return (x > y ? 1 : x < y ? -1 : 0) * dir; });
    el.innerHTML = '<tr>' + cols.map((c, i) => `<th class="${c.n ? 'n' : ''}" data-i="${i}">${c.h}${i === key ? (dir < 0 ? ' &#9662;' : ' &#9652;') : ''}</th>`).join('') + '</tr>' + d.map((r, i) => render(r, i)).join('');
    el.querySelectorAll('th').forEach(th => th.onclick = () => { const i = +th.dataset.i; dir = (i === key) ? -dir : -1; key = i; draw(); });
  }
  draw();
}
const max = Math.max(...C.map(c => c.mean));
const rank = {}; [...C].sort((a, b) => b.mean - a.mean).forEach((c, i) => rank[c.id] = i + 1);
sortable('tOverall', [
  {h: 'Rank', v: c => c.mean, n: 1}, {h: 'Scaffold', v: c => c.scaffold}, {h: 'Backbone', v: c => c.backbone},
  {h: 'Mean', v: c => c.mean, n: 1}, {h: 'OK / fail', v: c => c.ok, n: 1}, {h: 'Agent time (h)', v: c => c.time, n: 1},
  {h: 'Tokens in (M)', v: c => c.tin, n: 1}, {h: 'Tokens out (M)', v: c => c.tout, n: 1}], C,
  (c, i) => `<tr><td class="n">${rank[c.id]}</td><td>${c.scaffold}</td><td>${lab(c)}</td><td class="n"><b>${f1(c.mean)}</b><span class="bar" style="width:${(c.mean / max * 120).toFixed(0)}px"></span></td><td class="n">${c.ok} / <span class="${c.fail ? 'fail' : ''}">${c.fail}</span></td><td class="n">${c.time.toFixed(1)}</td><td class="n">${c.tin.toFixed(1)}</td><td class="n">${c.tout.toFixed(2)}</td></tr>`, 3);
const heat = v => `background:rgba(88,166,255,${Math.max(0.05, Math.min(0.85, (v - 20) / 60)).toFixed(2)})`;
sortable('tDomain', [{h: 'Configuration', v: c => c.scaffold + c.backbone}].concat(D.dom.map((d, j) => ({h: D.dlab[d], v: c => c.dom[j], n: 1}))).concat([{h: 'Mean', v: c => c.mean, n: 1}]), C,
  c => `<tr><td>${c.scaffold} + ${lab(c)}</td>` + c.dom.map(v => `<td class="n" style="${heat(v)}">${f1(v)}</td>`).join('') + `<td class="n"><b>${f1(c.mean)}</b></td></tr>`, 9);
(function () {
  const S = ['OpenCode', 'Biomni', 'MLEvolve', 'Minimal'], B = ['Claude Opus 5.5', 'GPT-6 Astra', 'Gemini 3.8 Flash', 'DeepSeek V4.1 Flash (1017)'];
  const get = (s, b) => (C.find(c => c.scaffold === s && c.backbone === b) || {}).mean;
  let h = '<tr><th></th>' + B.map(b => `<th>${b.replace(' (1017)', '')}</th>`).join('') + '<th>Mean</th></tr>';
  S.forEach(s => { const v = B.map(b => get(s, b)); h += `<tr><td style="text-align:left">${s}</td>` + v.map(x => `<td style="${heat(x)}">${f1(x)}</td>`).join('') + `<td>${f1(v.reduce((a, b) => a + b) / 4)}</td></tr>`; });
  h += '<tr><td style="text-align:left">Mean</td>' + B.map(b => f1(S.map(s => get(s, b)).reduce((a, c) => a + c) / 4)).map(x => `<td>${x}</td>`).join('') + '<td></td></tr>';
  document.getElementById('tGrid').innerHTML = h;
})();
const fd = document.getElementById('fd');
fd.innerHTML = '<option value="">All domains</option>' + D.dom.map(d => `<option value="${d}">${D.dlab[d]}</option>`).join('');
const order = C.map((c, i) => [c.mean, i]).sort((a, b) => b[0] - a[0]).map(x => x[1]);
function drawTasks() {
  const q = document.getElementById('q').value.toLowerCase(), d = fd.value;
  let h = '<tr><th>Task</th><th>Domain</th>' + order.map(i => `<th title="${C[i].scaffold} + ${C[i].backbone}">${C[i].scaffold}<br>${C[i].backbone.replace(' (1017)', '').replace('Claude ', '')}</th>`).join('') + '</tr>';
  D.tasks.filter(t => (!d || t.d === d) && t.t.includes(q)).forEach(t => {
    const m = Math.max(...t.s.filter(x => x !== null));
    h += `<tr><td class="t">${t.t}</td><td><span class="badge">${D.dlab[t.d]}</span></td>` + order.map(i => { const v = t.s[i]; return v === null || v === 0 ? '<td class="fail">&ndash;</td>' : `<td class="${v === m ? 'best' : ''}">${f1(v)}</td>`; }).join('') + '</tr>';
  });
  document.getElementById('tTask').innerHTML = h;
}
document.getElementById('q').oninput = drawTasks; fd.onchange = drawTasks; drawTasks();
document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active')); document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
  t.classList.add('active'); document.getElementById(t.dataset.p).classList.add('active');
});
})();
</script></body></html>"""
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print("wrote", OUT, "configs", len(conf), "tasks", len(tasks), "oracle", round(oracle, 1))
