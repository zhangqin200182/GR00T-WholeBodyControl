#!/bin/bash
# Fine gain sweep for release zero-shot (experiment 2 continuation).
# Fixed: isaac-action-space + DRIVE_ANALYTICAL=1. Sweep MULT / KD_MULT.
# Runs sequentially; appends one summary line per config to the results file.
set -u
cd /root/GR00T-WholeBodyControl
RESULT_FILE=/tmp/gain_sweep_results.txt
EVAL=/tmp/physx_cross_eval.py
CKPT=/root/sonic_release/last.pt

: > "$RESULT_FILE"
echo "ts=$(date '+%H:%M:%S') gain sweep start" >> "$RESULT_FILE"

run() {
  local mult=$1 kd=$2 tag=$3
  local log=/tmp/gain_sweep_${tag}.log
  SONIC_PHYSX_DRIVE_ANALYTICAL=1 SONIC_PHYSX_DRIVE_MULT="$mult" \
    SONIC_PHYSX_DRIVE_KD_MULT="$kd" \
    python3 "$EVAL" --ckpt "$CKPT" --ori 0.35 --ank 0.35 \
    --episodes 10 --trust 1.0 --isaac-space > "$log" 2>&1
  local line
  line=$(grep "RESULT:" "$log" | tail -1)
  echo "mult=${mult} kd=${kd} :: ${line}" >> "$RESULT_FILE"
  echo "DONE ${tag} ${line}"
}

run 10 8   m10k8
run 10 16  m10k16
run 10 32  m10k32
run 10 64  m10k64
run 5  16  m5k16
run 20 16  m20k16
run 30 16  m30k16

echo "=== ALL RESULTS ==="
cat "$RESULT_FILE"
