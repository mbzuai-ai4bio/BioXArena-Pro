"""Progress of every configuration that has a queue (harness/queues/): done ok / failed / running / todo, archived infra attempts."""
import json, sys
from pathlib import Path
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import bx_config as C  # noqa: E402
for rp in sorted((H / "queues").glob("runs_*.json")):
    cfg = rp.stem[5:]; runs = json.loads(rp.read_text()); q = json.loads((H / "queues" / f"queue_{cfg}.json").read_text())
    ok = fail = run = 0; fails = []
    for r in runs:
        d = Path(r["output"])
        if (d / "run_summary.json").exists():
            if (d / "submission.csv").exists() and not (d / "FAILED.json").exists(): ok += 1
            else:
                fail += 1; e = json.loads((d / "FAILED.json").read_text()).get("error", "")[:70] if (d / "FAILED.json").exists() else "no submission"
                fails.append(f"{r['task']}: {e}")
        elif d.exists(): run += 1
    arch = len(list((C.ARCHIVE_ROOT / cfg).glob("*/*/*"))) if (C.ARCHIVE_ROOT / cfg).exists() else 0
    print(f"{cfg}: ok {ok}  fail {fail}  running {run}  todo {len(q['todo'])}  archived attempts {arch}")
    for f in fails: print("   fail", f)
