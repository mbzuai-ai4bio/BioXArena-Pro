"""Exit 0 if the agent transcript in <out> ended with a turn failure caused by the API gateway (503 / 429 / 401 / retry
limit), i.e. the agent was cut off by infrastructure even though an early submission may exist -> worker retries."""
import glob, json, re, sys
ts = glob.glob(sys.argv[1] + "/*transcript.jsonl")
last, msg = None, ""
for l in (open(ts[0], errors="ignore") if ts else []):
    try: j = json.loads(l)
    except Exception: continue
    if "type" in j: last = j["type"]
    if j.get("type") == "turn.failed": msg = (j.get("error") or {}).get("message", "")
sys.exit(0 if last == "turn.failed" and re.search(r"503|429|401|retry limit|Reconnect|unavailable", msg) else 1)
