#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../../.." && pwd)"
PRISM="${PRISM_BIN:-$ROOT/lab/tools/installed/prism-4.11-linux64-x86/bin/prism}"
MODEL="$HERE/prism/amr_shared_priority_mdp.prism"
PROPS="$HERE/prism/amr_shared_priority_mdp.props"
SCENARIOS="$HERE/prism/shared_scenarios.csv"
LOG_DIR="$HERE/logs/prism_shared_scenarios"
mkdir -p "$LOG_DIR"

tail -n +2 "$SCENARIOS" | while IFS=, read -r scenario p_loss p_fail p_recover p_batt_low max_wait reason; do
  log="$LOG_DIR/${scenario}.log"
  {
    echo "Scenario: $scenario"
    echo "Reason: $reason"
    echo "PRISM: $("$PRISM" -version 2>&1 | head -2 | tr '\n' ' ')"
    echo "Constants: p_loss=$p_loss,p_fail=$p_fail,p_recover=$p_recover,p_batt_low=$p_batt_low,MAX_WAIT=$max_wait"
    echo "Started: $(date -Is)"
    "$PRISM" "$MODEL" "$PROPS" \
      -const "p_loss=$p_loss,p_fail=$p_fail,p_recover=$p_recover,p_batt_low=$p_batt_low,MAX_WAIT=$max_wait" \
      -explicit -javamaxmem 2g
    echo "Finished: $(date -Is)"
  } > "$log" 2>&1
done

echo "Completed PRISM scenario matrix; logs: $LOG_DIR"
