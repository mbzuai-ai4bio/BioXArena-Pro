#!/bin/bash
# Smoke test: the Minimal scaffold on the smallest task (chemical-biology/solubility, 0.6 MB) with a 15-minute wall clock,
# then scoring if the private answers are available. Needs: the Python env, `source config.env` (BX_DATA_PUBLIC and the key
# of the route you choose), optionally a GPU. Usage: bash tests/smoke_minimal.sh [model] [route]
#   bash tests/smoke_minimal.sh                                   # deepseek/deepseek-v4.1-flash-1017 over OpenRouter
#   bash tests/smoke_minimal.sh gemini-3.8-flash gemini           # Gemini API
#   bash tests/smoke_minimal.sh claude-opus-5-5 anthropic         # Anthropic API
set -euo pipefail
MODEL=${1:-deepseek/deepseek-v4.1-flash-1017}; ROUTE=${2:-}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT/harness"
export BX_WALL_SEC=${BX_WALL_SEC:-900}
SMOKE=${BX_SMOKE_ROOT:-$ROOT/smoke}; CFG=smoke_minimal_$(date +%Y%m%d_%H%M%S)
python make_runs.py --cli minimal --model "$MODEL" ${ROUTE:+--route $ROUTE} --prefix "${CFG}_" --out-root "$SMOKE" --tasks chemical-biology/solubility
RUNS=$(ls queues/runs_${CFG}_*.json)
python run_agent.py "$RUNS" 0 || echo "run_agent exit code $? (1 = the agent failed the task; the harness itself worked if run_summary.json exists)"
OUT=$(python -c "import json;print(json.load(open('$RUNS'))[0]['output'])")
echo "--- $OUT"; ls "$OUT"
python -c "import json;s=json.load(open('$OUT/run_summary.json'));print('status',s['status'],'served',s['api_tap']['served_models'],'tokens',s['tokens'])"
if [ -d "${BX_DATA_PRIVATE:-/nonexistent}/chemical-biology/solubility/private" ]; then
  BX_SCORE_OUT=$SMOKE BX_SCORE_RESULTS=$SMOKE/results python ../scoring/score_final.py "$(basename "$(dirname "$(dirname "$OUT")")")" | grep -E "solubility|mean64"
else
  echo "BX_DATA_PRIVATE not set or no private dir: skipping scoring"
fi
