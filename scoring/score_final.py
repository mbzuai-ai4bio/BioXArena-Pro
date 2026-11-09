"""Score every task of the 64-task BioXArena-Pro release (v5.1.1 data, 2026-10-21) for one or more agent configurations.
Run dir: <BX_OUTPUT_ROOT>/<model_dir>/<domain>/<task>. Failure (FAILED.json, no submission.csv, wall time > 7200 s, invalid
submission) = score 0; a run still holding SAFETY_REFUSAL.json or MODEL_MISMATCH.json is reported separately (should
have been archived and retried). Graders: tasks/<domain>/<task>/grade.py; scores normalised by
evaluation/evaluate_llm_agents.py. Needs the PRIVATE answers (BX_DATA_PRIVATE). Read-only on run dirs.
Usage: score_final.py <model_dir> [...]   e.g. score_final.py codexcli__gpt-6-astra claudecode__claude-opus-5-5
Output: scoring/results/<model_dir>.csv with task, domain, status, score, raw, metric, runtime_sec, input_tokens_total
(incl. cached), output_tokens_total, served_models (all exact, from the api tap in run_summary.json), note."""
import csv, importlib.util, json, math, os, sys
from pathlib import Path
import pandas as pd
CODE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "harness"))
import bx_config as C  # noqa: E402
spec = importlib.util.spec_from_file_location("ev", CODE / "evaluation/evaluate_llm_agents.py")
ev = importlib.util.module_from_spec(spec); sys.modules["ev"] = ev; spec.loader.exec_module(ev)
PUB, PRIV = C.DATA_PUBLIC, C.DATA_PRIVATE
OUT = Path(os.environ.get("BX_SCORE_OUT") or C.OUTPUT_ROOT)
TASKS = sorted(f"{d.name}/{t.name}" for d in PUB.iterdir() if d.is_dir() for t in d.iterdir() if (t / "public").is_dir())
assert len(TASKS) == 64, len(TASKS)


def grade(t, d):
    dom, name = t.split("/"); pub, priv = PUB / t / "public", PRIV / t / "private"
    ans = priv / "answers.csv"
    if not ans.exists(): ans = next(priv.glob("answers*"))
    ts = ev.TaskSpec(domain=dom, task=name, grade_path=CODE / "tasks" / t / "grade.py", answers_path=ans, public_dir=pub, output_dir=d)
    g = ev.load_grade_module(ts.grade_path, t); info = ev.infer_metric_info(g)
    sub = ev.resolve_file_manifest_columns(ts, pd.read_csv(d / "submission.csv"), root=d, label="Submission")
    a = ev.resolve_file_manifest_columns(ts, pd.read_csv(ts.answers_path), root=ts.answers_path.parent, label="Answer")
    raw = float(ev.call_grade(g, ts, sub, a, d / "submission.csv")); assert math.isfinite(raw)
    return raw, info.name, float(ev.normalize_score(raw, info))


def summary(d):
    """Exact accounting from run_summary.json (api tap): tokens, runtime, served models."""
    s = d / "run_summary.json"
    if not s.exists(): return {}
    j = json.loads(s.read_text()); t = j.get("tokens") or {}
    return {"runtime_sec": j.get("runtime_sec"), "input_tokens_total": t.get("input_tokens_total"),
            "output_tokens_total": t.get("output_tokens_total"),
            "served_models": json.dumps((j.get("api_tap") or {}).get("served_models", {}))}


for mdir in sys.argv[1:]:
    rows = []
    for t in TASKS:
        d = OUT / mdir / t; r = dict(task=t, domain=t.split("/")[0], run_dir=str(d), status="not_run", score=0.0, raw=None,
                                     metric=None, runtime_sec=None, input_tokens_total=None, output_tokens_total=None,
                                     served_models=None, note="")
        if d.exists():
            s = d / "run_summary.json"; r.update(summary(d))
            sf = d / "SAFETY_FALLBACK.json"
            if sf.exists():
                fb = json.loads(sf.read_text()); r["note"] = ("safety make-up run (refusal fallback allowed): served "
                    + json.dumps(fb.get("responses_by_served_model")) + "; refusal " + json.dumps((fb.get("refusal") or {}).get("categories")) + "; ")
            if not s.exists():
                r["status"] = "running"; r["note"] = "run not finished (no run_summary.json)"
            elif (d / "SAFETY_REFUSAL.json").exists():
                r["status"] = "safety_refusal_pending_makeup"
                r["note"] = "main round: requested model refused on safety grounds " + json.dumps(json.loads((d / "SAFETY_REFUSAL.json").read_text()).get("refusal"))
            elif (d / "FAILED.json").exists():
                r["status"] = "fail"; r["note"] = json.loads((d / "FAILED.json").read_text()).get("error", "")[:200]
                tr = d / "claudecode_transcript.jsonl"
                if tr.exists() and '"stop_reason":"refusal"' in tr.read_text(errors="ignore"):
                    r["note"] = "model safeguard refusal (no fallback model allowed); " + r["note"]
            elif not (d / "submission.csv").exists():
                r["status"] = "fail"; r["note"] = "no submission.csv"
            elif (d / "MODEL_MISMATCH.json").exists():
                r["status"] = "invalid_model"; r["note"] = "served model != requested (should have been retried)"
            elif r["runtime_sec"] and r["runtime_sec"] > 7200 + 120:
                r["status"] = "fail"; r["note"] = f"wall time {r['runtime_sec']:.0f}s > 7200s"
            else:
                try:
                    r["raw"], r["metric"], r["score"] = grade(t, d); r["status"] = "ok_safety_fallback" if sf.exists() else "ok"
                except Exception as e:
                    r["status"] = "fail"; r["note"] = f"invalid submission: {type(e).__name__}: {str(e)[:160]}"
        rows.append(r); print(mdir, t, r["status"], round(r["score"], 4), flush=True)
    out = Path(os.environ.get("BX_SCORE_RESULTS") or Path(__file__).resolve().parent / "results"); out.mkdir(parents=True, exist_ok=True)
    with open(out / (mdir.replace("/", "__") + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r["status"] in ("ok", "ok_safety_fallback", "fail")]
    print(f"{mdir}: mean64 {sum(r['score'] for r in rows)/64:.4f} done {len(ok)}/64 fails {sum(r['status']=='fail' for r in rows)}")
