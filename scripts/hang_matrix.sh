#!/bin/bash
# Controlled runs around the concurrent-assistant GPU stall (docs/results-phase3.md section 3).
# Each run ends at the first stalled worker (watchdog, exit code 3) or after --duration.
# Usage on the Jetson, inside tmux:  bash scripts/hang_matrix.sh [run letters, default all]
cd ~/fieldbench
mkdir -p logs
BASE="--jpeg cpu --prep gpu --workers 2 --duration 600 --series 30 --stall-s 60"
declare -A RUNS=(
  [A]="assistant --order concurrent $BASE --power-modes 15W"
  [B]="ocr --sizes 1280 $BASE"
  [C]="assistant --order concurrent $BASE --serialize rec"
  [D]="assistant --order concurrent $BASE --serialize all"
  [E]="assistant --order sequential $BASE"
)
SUMMARY=logs/hang-matrix.log
for k in ${@:-A B C D E}; do
  echo "=== run $k: ${RUNS[$k]}"
  start=$(date +%s)
  .venv/bin/python -u -m fieldbench pipeline ${RUNS[$k]} --label hang-$k
  code=$?
  [ "$(nvpmodel -q | head -1)" != "NV Power Mode: MAXN_SUPER" ] && .venv/bin/python -m fieldbench power --set MAXN_SUPER
  echo "$(date -Is) run $k exit $code ($([ $code = 3 ] && echo STALLED || echo completed)) after $(( $(date +%s) - start )) s: ${RUNS[$k]}" | tee -a $SUMMARY
  sleep 30  # cool down between runs
done
echo "matrix done"
