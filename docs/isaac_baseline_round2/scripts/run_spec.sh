#!/bin/bash
set -u
PAT="isaac_spec_[d]ump"
clean_gpu() {
  for P in $(pgrep -f "$PAT"); do kill -9 "$P" 2>/dev/null; done
  sleep 2
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read p; do
    [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done
  sleep 1; rm -f /tmp/isaaclab_app_launcher.lock
}
clean_gpu
cd /root/GR00T-WholeBodyControl
timeout --signal=KILL 20m /opt/Anaconda3/envs/isaac/bin/python gear_sonic/isaac_spec_dump.py \
  checkpoint=sonic_release/last.pt \
  manager_env/recorders=empty \
  +num_envs=1 +headless=true +run_once=false +use_wandb=false \
  +manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered_fixed12/fixed12 \
  +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered \
  > /root/spec_run.log 2>&1
RC=$?
clean_gpu
echo "SPEC_DONE rc=$RC"
