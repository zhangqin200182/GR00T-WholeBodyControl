#!/usr/bin/env python3
"""Our-side obs_step0 capture at reset — direct obs-vs-obs counterpart of the
round-2 B3 per-episode files (their obs_step0 is a RESET-moment obs: ah all
zeros, jph static — captured before any policy step).

For each of the 12 forced clips: phase 0 + _forced_idx, env.reset(), save
actor_obs (930,) + reset state (qpos/qvel/root_quat) for state-noise
attribution.  No policy needed — obs is computed in reset().

Usage (NPU): python3 scripts/obs_step0_ours.py --out /tmp/obs_step0_ours
"""
import argparse
import os
import sys
import numpy as np

os.environ.setdefault("PHYSX_CONTACT_DEBUG", "1")
_build = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "gear_sonic", "envs", "physx", "build"),
    "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build",
]
for _d in _build:
    if os.path.isdir(_d):
        sys.path.insert(0, _d)
        break
import physx_core  # noqa: E402

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"

CLIPS = ["walk_ff_loop_180_R_003__A050", "walk_the_dog_ff_180_loop_R_001__A476",
         "injured_R_leg_walk_ff_start_315_R_002__A232",
         "walk_sideway_045_loop_003__A033",
         "crutches_walk_arc_cw_start_R_001__A516",
         "walk_ff_stop_360_R_001__A418", "crutch_walk_turn_270_R_001__A518",
         "walk_ff_stop_270_002__A051_M", "walk_into_door_R_001__A514",
         "inj_right_leg_walk_180_R_max_003__A078",
         "big_heavy_one_hand_walk_ff_start_360_R_001__A509",
         "injured_torso_walk_ff_start_225_R_003__A338"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/obs_step0_ours")
    ap.add_argument("--pkl", default="/sample_data/robot_filtered_fixed12")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from omegaconf import OmegaConf
    physx_core.init_foundation()
    from gear_sonic.envs.physx_env import PhysXEnv

    for i, clip in enumerate(CLIPS):
        env = PhysXEnv(physx_core, XML, args.pkl,
                       config=OmegaConf.create({
                           "alive_bonus": 0.0, "ori_thresh": 0.35,
                           "ank_pos_thresh": 0.35, "ank_h_mult": 1.5,
                           "action_trust": 1.0, "isaac_action_space": True,
                           "ignore_terminations": False,
                           "skip_termination": False,
                           "max_episode_length": 500,
                       }),
                       native_dt=0.005, decimation=4, pos_iters=8, vel_iters=4,
                       static_pose=False, root_z_offset=0.04, standing_prob=0.0,
                       drive_type="FORCE")
        env._forced_idx = i
        env._forced_ref_time = 0.0
        obs = env.reset()
        qpos = env.art.get_joint_positions()
        qvel = env.art.get_joint_velocities()
        rpos, rquat = env.art.get_root_world_pose()
        np.savez(os.path.join(args.out, f"ours_{clip}.npz"),
                 actor_obs=obs["actor_obs"], qpos=qpos, qvel=qvel,
                 root_pos=rpos, root_quat=rquat)
        print(f"{clip}: obs saved, |qpos|max={np.abs(qpos).max():.3f}, "
              f"|qvel|max={np.abs(qvel).max():.4f}, root_z={rpos[2]:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
