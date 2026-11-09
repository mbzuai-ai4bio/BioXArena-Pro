"""Shared task queue (queues/queue_<CFG>.json: {"todo": [...], "claimed": {index: lane}}), flock-protected; path from env BXQ_QUEUE.
claim <lane> -> print next index (moves it from todo to claimed) or nothing; unclaim <idx>... -> back to the front of todo;
add <idx>... -> append to todo if neither queued nor claimed (to rerun a claimed index, use unclaim)."""
import fcntl, json, sys
from pathlib import Path
import os
Q = Path(os.environ.get("BXQ_QUEUE") or (Path(__file__).resolve().parent / "queue.json"))
with open(str(Q) + ".lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    q = json.loads(Q.read_text()) if Q.exists() else {"todo": [], "claimed": {}}
    if sys.argv[1] == "claim":
        if q["todo"]:
            t = q["todo"].pop(0); q["claimed"][t] = sys.argv[2]; print(t)
    elif sys.argv[1] == "unclaim":  # put claimed indices back at the front of todo
        for t in sys.argv[2:]:
            q["claimed"].pop(t, None)
            if t not in q["todo"]: q["todo"].insert(0, t)
    elif sys.argv[1] == "add":
        for t in sys.argv[2:]:
            if t not in q["todo"] and t not in q["claimed"]: q["todo"].append(t)
    Q.write_text(json.dumps(q, indent=1))
