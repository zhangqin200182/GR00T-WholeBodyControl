#!/bin/bash
# Isaac baseline collection wrapper: clean -> run (bounded) -> clean.
# usage: run_collect.sh <POLICY> <OUTDIR> <CLIPS_JSON> <ENVS> <MOTION_DIR> <LOG> [STEPS]
set -u
POLICY=$1; OUT=$2; CLIPS=$3; ENVS=$4; MOTDIR=$5; LOG=$6; STEPS=${7:-500}
PAT='baseline_[c]ollect'
clean_gpu() {
  for P in $(pgrep -f "$PAT"); do kill -9 "$P" 2>/dev/null; done
  sleep 2
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read p; do
    [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done
  sleep 1
  rm -f /tmp/isaaclab_app_launcher.lock
}
clean_gpu
mkdir -p "$OUT"
cd /root/GR00T-WholeBodyControl
COLLECT_POLICY=$POLICY COLLECT_OUT=$OUT COLLECT_STEPS=$STEPS COLLECT_CLIPS="$CLIPS" \
  timeout --signal=KILL 45m /opt/Anaconda3/envs/isaac/bin/python gear_sonic/isaac_baseline_collect.py \
  checkpoint=sonic_release/last.pt \
  +manager_env.config.render_results=True \
  manager_env/recorders=empty \
  +num_envs=$ENVS +headless=true \
  +run_once=false +use_wandb=false \
  +manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTDIR \
  +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered > "$LOG" 2>&1
RC=$?
clean_gpu
echo "RUN_DONE rc=$RC gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
