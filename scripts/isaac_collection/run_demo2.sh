#!/bin/bash
# demo 片段基线: 2 clips x {release,pd}, 单批 2 env
set -u
POLICY=$1
PAT='baseline_[c]ollect'
clean_gpu() {
  for P in $(pgrep -f "$PAT"); do kill -9 "$P" 2>/dev/null; done
  sleep 2
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read p; do
    [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done
  sleep 1; rm -f /tmp/isaaclab_app_launcher.lock
}
clean_gpu
mkdir -p /root/isaac_baseline_out_demo/$POLICY
cd /root/GR00T-WholeBodyControl
CLIPS='["walk_forward_amateur_001__A001","walk_forward_amateur_001__A001_M"]'
COLLECT_POLICY=$POLICY COLLECT_OUT=/root/isaac_baseline_out_demo/$POLICY COLLECT_STEPS=500 COLLECT_CLIPS="$CLIPS" \
  timeout --signal=KILL 25m /opt/Anaconda3/envs/isaac/bin/python gear_sonic/isaac_baseline_collect.py \
  checkpoint=sonic_release/last.pt \
  manager_env/recorders=empty \
  +num_envs=2 +headless=true +run_once=false +use_wandb=false \
  +manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/demo2 \
  +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered \
  +manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load=2 \
  > /root/demo2_${POLICY}.log 2>&1
RC=$?
clean_gpu
echo "DEMO2_${POLICY}_DONE rc=$RC npz=$(ls /root/isaac_baseline_out_demo/$POLICY/*.npz 2>/dev/null | wc -l)"
