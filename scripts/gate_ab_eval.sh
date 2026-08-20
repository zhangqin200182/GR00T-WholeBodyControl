#!/bin/bash
# gate_ab_eval.sh — joint-order gate A/B zero-shot on the FULL Isaac stack.
#
# v1 (container /tmp/gate_ab.sh) set only ISAAC_FEET + GROUND_FRICTION and
# was documented as "full Isaac stack" — corrected: that was the 2x2-matrix
# stack.  This version sets the complete env-var set:
#   SONIC_PHYSX_ISAAC_JOINT_ORDER  1 (A/B) | 0 (C/D)   obs/action order gate
#   SONIC_PHYSX_ISAAC_FEET         1                   7-capsule feet
#   SONIC_PHYSX_GROUND_FRICTION    1.0,1.0             ground friction
#   SONIC_PHYSX_BOUNCE_THRESHOLD   0.5                 Isaac A3
#   SONIC_PHYSX_FRICTION_OFFSET    0.04                Isaac A3
#   SONIC_PHYSX_FRICTION_CORR_DIST 0.025               Isaac A3
#   SONIC_PHYSX_ROOT_Z_OFFSET      0.02                reset root z offset
# cross_eval flags: FORCE + vel_iters 4 + dt 0.005x4 + isaac-space (Isaac A3)
#
# NOTE: A/B contrast (gate ON vs OFF) is invariant to the contact env vars
# (both sides identical); the vars only move the ABSOLUTE values.
cd /root/GR00T-WholeBodyControl
CKPT=/root/sonic_release/last.pt
OUT=/tmp/gate_ab
mkdir -p "$OUT"
COMMON="--ckpt $CKPT --pkl /sample_data/robot_filtered --ori 0.35 --ank 0.35 --episodes 24 --motion_seed 0 --isaac-space --drive-type FORCE --vel-iters 4 --native-dt 0.005 --decimation 4"
STACK="SONIC_PHYSX_ISAAC_FEET=1 SONIC_PHYSX_GROUND_FRICTION=1.0,1.0 SONIC_PHYSX_BOUNCE_THRESHOLD=0.5 SONIC_PHYSX_FRICTION_OFFSET=0.04 SONIC_PHYSX_FRICTION_CORR_DIST=0.025 SONIC_PHYSX_ROOT_Z_OFFSET=0.02"

run () {
  local name=$1; shift
  echo "===== $name ====="
  env $STACK "$@" python3 scripts/physx_cross_eval.py $COMMON > "$OUT/$name.log" 2>&1
  grep -E "RESULT|DEATH" "$OUT/$name.log"
}

run A_full_gateON_release
run C_full_gateOFF_release SONIC_PHYSX_ISAAC_JOINT_ORDER=0
echo "===== GATE FULL AB DONE ====="
