#!/bin/bash
# Batch runner v2: splits 12 clips into 4 batches of 3.
# Fix: force motion loader to load all 12 clips, then CLIP_LIST selects batch subset.
set -u
POLICY=$1; OUT=$2; LOGPFX=$3
ENVS=3
MOTDIR=sample_data/robot_filtered_fixed12/fixed12
STEPS=500
PAT='baseline_[c]ollect'

CLIPS=(
  walk_ff_loop_180_R_003__A050
  walk_the_dog_ff_180_loop_R_001__A476
  injured_R_leg_walk_ff_start_315_R_002__A232
  walk_sideway_045_loop_003__A033
  crutches_walk_arc_cw_start_R_001__A516
  walk_ff_stop_360_R_001__A418
  crutch_walk_turn_270_R_001__A518
  walk_ff_stop_270_002__A051_M
  walk_into_door_R_001__A514
  inj_right_leg_walk_180_R_max_003__A078
  big_heavy_one_hand_walk_ff_start_360_R_001__A509
  injured_torso_walk_ff_start_225_R_003__A338
)

clean_gpu() {
  for P in $(pgrep -f "$PAT"); do kill -9 "$P" 2>/dev/null; done
  sleep 2
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read p; do
    [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done
  sleep 1; rm -f /tmp/isaaclab_app_launcher.lock
}

mkdir -p "$OUT"
BATCH_NUM=0
for ((i=0; i<12; i+=ENVS)); do
  BATCH_NUM=$((BATCH_NUM+1))
  BATCH_CLIPS=("${CLIPS[@]:i:ENVS}")
  CLIPS_JSON=$(printf '["%s"' "${BATCH_CLIPS[0]}")
  for ((j=1; j<${#BATCH_CLIPS[@]}; j++)); do
    CLIPS_JSON="$CLIPS_JSON,\"${BATCH_CLIPS[$j]}\""
  done
  CLIPS_JSON="$CLIPS_JSON]"
  LOG="${LOGPFX}_batch${BATCH_NUM}.log"

  echo "=== BATCH $BATCH_NUM: ${BATCH_CLIPS[*]} ===" | tee -a "${LOGPFX}_summary.log"

  clean_gpu
  cd /root/GR00T-WholeBodyControl
  COLLECT_POLICY=$POLICY COLLECT_OUT=$OUT COLLECT_STEPS=$STEPS COLLECT_CLIPS="$CLIPS_JSON" \
    timeout --signal=KILL 25m /opt/Anaconda3/envs/isaac/bin/python gear_sonic/isaac_baseline_collect.py \
    checkpoint=sonic_release/last.pt \
    manager_env/recorders=empty \
    +num_envs=$ENVS +headless=true \
    +run_once=false +use_wandb=false \
    +manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTDIR \
    +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered \
    +manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load=12 \
    > "$LOG" 2>&1
  RC=$?
  NNPZ=$(ls "$OUT"/${POLICY}_*.npz 2>/dev/null | wc -l)
  echo "  rc=$RC npz=$NNPZ gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" | tee -a "${LOGPFX}_summary.log"
  if [ "$RC" -ne 0 ]; then
    echo "  FAILED — stopping. Check $LOG" | tee -a "${LOGPFX}_summary.log"
    clean_gpu
    exit 1
  fi
done
clean_gpu
echo "ALL_BATCHES_DONE policy=$POLICY total_npz=$(ls "$OUT"/${POLICY}_*.npz 2>/dev/null | wc -l)"
