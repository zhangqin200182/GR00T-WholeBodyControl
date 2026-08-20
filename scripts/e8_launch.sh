#!/bin/bash
# E8 launch — FULL pinned stack.  Every lever that differs from the legacy
# default is set EXPLICITLY here (audit 2026-08-20: the manager never passed
# drive_type; production training had silently run ACCELERATION defaults).
#
# Stack (Isaac-faithful, verdicts in docs/physx-training-log.md):
#   drive FORCE (real Isaac semantics; corr 0.52 vs ACCEL 0.25)
#   7-capsule feet + foot material (SONIC_PHYSX_ISAAC_FEET)
#   ground friction 1.0/1.0 (SONIC_PHYSX_GROUND_FRICTION)
#   bounce 0.5 / frictionOffset 0.04 / corrDist 0.025 (Isaac A3)
#   dt 0.005 x 4 (exact 50Hz) + vel_iters 4 (Isaac A3)
#   isaaclab joint-order interface (SONIC_PHYSX_ISAAC_JOINT_ORDER=1)
#   no action clip (removed in env code, E7)
set -euo pipefail
cd /root/GR00T-WholeBodyControl

export SONIC_PHYSX_DRIVE_TYPE=FORCE
export SONIC_PHYSX_ISAAC_JOINT_ORDER=1
export SONIC_PHYSX_ISAAC_FEET=1
export SONIC_PHYSX_GROUND_FRICTION=1.0,1.0
export SONIC_PHYSX_BOUNCE_THRESHOLD=0.5
export SONIC_PHYSX_FRICTION_OFFSET=0.04
export SONIC_PHYSX_FRICTION_CORR_DIST=0.025
export SONIC_PHYSX_VEL_ITERS=4
export SONIC_PHYSX_NATIVE_DT=0.005
export SONIC_PHYSX_DECIMATION=4
export SONIC_PHYSX_ROOT_Z_OFFSET=0.02
# rz note: 0.02 = Isaac-faithful reset height (capsule fan bottom at ground);
# the gate-ON -13% (12.17->10.62) vs the 2x2 stack was attributed to rz but
# never single-variable verified — BC smoke runs an rz 0.04 control group.
# armature / velocity limits / depenetration: default ON (Isaac values)

echo "E8 stack:"
echo "  drive=$(echo $SONIC_PHYSX_DRIVE_TYPE) joint_order=$SONIC_PHYSX_ISAAC_JOINT_ORDER"
echo "  feet=$SONIC_PHYSX_ISAAC_FEET ground_fric=$SONIC_PHYSX_GROUND_FRICTION"
echo "  bounce=$SONIC_PHYSX_BOUNCE_THRESHOLD fric_off=$SONIC_PHYSX_FRICTION_OFFSET corr=$SONIC_PHYSX_FRICTION_CORR_DIST"
echo "  vel_iters=$SONIC_PHYSX_VEL_ITERS dt=$SONIC_PHYSX_NATIVE_DT x $SONIC_PHYSX_DECIMATION"

# BC warmup (E8 phase 1) — replace with the actual trainer entry once the
# E8 pre-flight checklist (docs/physx-review-request-gate-ab.md §2) is closed.
echo "TODO: insert BC warmup command here (train_physx_ppo.py --phase bc ...)"
