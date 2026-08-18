#!/bin/bash
# P1 single-joint drive response runner: clean -> run (bounded 20m) -> clean
set -u
PAT="isaac_p1_[d]rive"
clean_gpu() {
  for P in $(pgrep -f "$PAT"); do kill -9 "$P" 2>/dev/null; done
  sleep 2
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read p; do
    [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done
  sleep 1; rm -f /tmp/isaaclab_app_launcher.lock
}
clean_gpu
mkdir -p /root/isaac_baseline_out/p1
cd /root/GR00T-WholeBodyControl
P1_OUT=/root/isaac_baseline_out/p1 \
  timeout --signal=KILL 20m /opt/Anaconda3/envs/isaac/bin/python gear_sonic/isaac_p1_drive.py \
  checkpoint=sonic_release/last.pt \
  manager_env/recorders=empty \
  +num_envs=1 +headless=true +run_once=false +use_wandb=false \
  +manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered_fixed12/fixed12 \
  +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered \
  > /root/p1_run.log 2>&1
RC=$?
clean_gpu
echo "P1_DONE rc=$RC npz=$(ls /root/isaac_baseline_out/p1/*.npz 2>/dev/null | wc -l) gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
