#!/bin/bash
# ---- Slurm header: EDIT for your cluster (account / partition / qos). Keep 1 GPU, 16 CPUs, 96 GB: the BioXArena-Pro compute contract.
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# One lane = one job = 1 GPU. Pulls run indices from queues/queue_<CFG>.json until empty and runs each with run_agent.py.
#   Slurm:      cd harness && sbatch -J bx_<CFG>_W<i> --export=ALL,CFG=<cfg>,LANE=<i> worker.sh
#   No Slurm:   cd harness && CFG=<cfg> LANE=<i> CUDA_VISIBLE_DEVICES=<gpu> bash worker.sh > logs/<cfg>_W<i>.out 2>&1 &
# Before: source ../config.env (paths, API keys, tool paths) and activate the Python env (see README).
# Infra failures (model mismatch, API 429/5xx with no submission) are archived to $BX_ARCHIVE_ROOT/<CFG>/ and retried (max 4
# attempts, 5 min apart). Safety refusals (exit 4) are archived to $BX_ARCHIVE_ROOT/<CFG>/safety_refusal_* and NOT retried here
# (README: retry once by hand, then decide on a make-up run). Genuine failures (timeouts, invalid outputs) are final.
set -uo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); [ -f "$H/run_agent.py" ] || H=${SLURM_SUBMIT_DIR:-$PWD}
[ -n "${BX_CONFIG_ENV:-}" ] && source "$BX_CONFIG_ENV"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
: "${CFG:?set CFG}" "${LANE:?set LANE}"
mkdir -p "$H/logs"
ROOT=$(cd "$H/.." && pwd); ARCH=${BX_ARCHIVE_ROOT:-$ROOT/archive}
RUNS=$H/queues/runs_$CFG.json
read -r CLI MODEL ROUTE < <(python -c "import json;r=json.load(open('$RUNS'))[0];print(r['cli'],r['model'],r['route'])")
API=$(python "$H/bx_config.py" upstream "$ROUTE"); KEY=$(python "$H/bx_config.py" key "$ROUTE") || exit 2
echo "cfg $CFG cli $CLI model $MODEL route $ROUTE lane $LANE host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}"
# API probe (no tokens are spent): a model-list / model-metadata GET on the configured upstream
probe() {
  case "$ROUTE" in
    gemini)    [ "$(curl -s -m 60 -o /dev/null -w '%{http_code}' "$API/v1beta/models/$MODEL" -H "x-goog-api-key: $KEY")" = 200 ] ;;
    anthropic) [ "$(curl -s -m 60 -o /dev/null -w '%{http_code}' "$API/v1/models" -H "x-api-key: $KEY" -H "anthropic-version: 2023-06-01")" = 200 ] ;;
    *)         [ "$(curl -s -m 60 -o /dev/null -w '%{http_code}' "$API/v1/models" -H "Authorization: Bearer $KEY")" = 200 ] ;;
  esac
}
wait_for_api() { local ok=0; while [ $ok -lt 2 ]; do if probe; then ok=$((ok+1)); sleep 15; else echo "[$(date '+%F %T')] API probe failed, waiting 5 min"; ok=0; sleep 300; fi; done; }
while idx=$(BXQ_QUEUE=$H/queues/queue_$CFG.json python "$H/task_queue.py" claim ${CFG}_W$LANE) && [ -n "$idx" ]; do
  read -r task out < <(python -c "import json;r=json.load(open('$RUNS'))[$idx];print(r['task'],r['output'])")
  echo "[$(date '+%F %T')] lane $LANE claimed $idx $task"
  if [ -s "$out/run_summary.json" ]; then echo "skip done $task"; continue; fi
  for attempt in 1 2 3 4; do
    echo "===== [$(date '+%F %T')] $task attempt $attempt"
    wait_for_api
    python "$H/run_agent.py" "$RUNS" "$idx"; rc=$?
    if [ ! -d "$out" ]; then  # configuration error before anything ran: put the task back and stop this lane
      BXQ_QUEUE=$H/queues/queue_$CFG.json python "$H/task_queue.py" unclaim "$idx"; echo "[$(date '+%F %T')] configuration error (rc=$rc), lane stops"; exit 2
    fi
    reason=""
    [ $rc -eq 3 ] && reason=model_mismatch
    if [ -z "$reason" ] && [ ! -s "$out/submission.csv" ] && grep -qsE "RESOURCE_EXHAUSTED|\"code\": ?(429|500|502|503|504)|UNAVAILABLE|503 Service Unavailable|No available accounts|\"error_status\":50[234]" \
         "$out"/*transcript*.jsonl "$out"/*stderr*.log "$out"/api_response_attempt_*.json "$out"/*.log 2>/dev/null; then reason=api_unavailable; fi
    if [ -z "$reason" ] && python "$H/api_cut.py" "$out" 2>/dev/null; then reason=api_cut; fi
    if [ $rc -eq 4 ]; then
      dest=$ARCH/$CFG/safety_refusal_$(date +%Y%m%d_%H%M%S)_W$LANE/$task; mkdir -p "$(dirname "$dest")"; mv "$out" "$dest"
      echo "[$(date '+%F %T')] SAFETY REFUSAL -> archived $dest (not retried; see README)"; break
    elif [ -n "$reason" ]; then
      dest=$ARCH/$CFG/${reason}_$(date +%Y%m%d_%H%M%S)_W$LANE/$task; mkdir -p "$(dirname "$dest")"; mv "$out" "$dest"
      echo "[$(date '+%F %T')] $reason -> archived $dest; retry in 5 min"; sleep 300
    else
      echo "[$(date '+%F %T')] $task final rc=$rc: $(ls "$out" | grep -E '^(submission.csv|FAILED.json)$' | tr '\n' ' ')"; break
    fi
  done
done
echo "[$(date '+%F %T')] lane $LANE: queue empty, exit"
