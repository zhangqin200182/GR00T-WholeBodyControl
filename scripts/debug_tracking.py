"""Diagnose: per-joint position error during ref PD."""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
import physx_core as px
import physx_env

px.init_foundation()
env = physx_env.PhysXEnv(px, "/gear_sonic_deploy/g1/g1_29dof_v17.xml", "/sample_data/robot_filtered")

# Gather joint tracking error over multiple episodes
all_errors = []
for ep in range(3):
    env.reset()
    ep_errors = []
    for step in range(500):
        ref_qpos = env.get_current_ref_qpos()
        actual_qpos = env.art.get_joint_positions()
        joint_err = np.abs(actual_qpos - ref_qpos)
        ep_errors.append(joint_err)
        action = (ref_qpos - env.jm) / env.jh
        obs, reward, done, info = env.step(action.astype(np.float64))
        if done:
            break
    ep_errors = np.array(ep_errors)
    all_errors.append(ep_errors.mean(axis=0))
    print(f"ep {ep}: survival={step+1}, "
          f"q_err_mean={ep_errors.mean():.4f}, "
          f"q_err_max={ep_errors.max(axis=0).max():.4f}, "
          f"q_err_ankle_L={ep_errors[:,4:6].mean():.4f}")

all_errors = np.array(all_errors)
print(f"\nPer-joint mean error across 3 eps:")
# Joint indices: 0-5 left leg, 6-11 right leg, 12-14 waist, 15-21 left arm, 22-28 right arm
labels = ["L_hip_yaw","L_hip_roll","L_hip_pitch","L_knee","L_ankle_pitch","L_ankle_roll",
          "R_hip_yaw","R_hip_roll","R_hip_pitch","R_knee","R_ankle_pitch","R_ankle_roll",
          "waist_yaw","waist_roll","waist_pitch",
          "L_shoulder_pitch","L_shoulder_roll","L_shoulder_yaw","L_elbow","L_wrist_roll","L_wrist_pitch","L_wrist_yaw",
          "R_shoulder_pitch","R_shoulder_roll","R_shoulder_yaw","R_elbow","R_wrist_roll","R_wrist_pitch","R_wrist_yaw"]
for i in range(29):
    print(f"  {labels[i]:20s}: {all_errors[:,i].mean():.4f}")

# Also check root position trajectory
env.reset()
print("\nRoot pos trajectory (first 5 steps):")
for step in range(5):
    ref_qpos = env.get_current_ref_qpos()
    action = (ref_qpos - env.jm) / env.jh
    obs, reward, done, info = env.step(action.astype(np.float64))
    rp = env.art.get_root_world_pose()[0]
    rr = env._ref_root_pos()
    print(f"  step {step}: actual_z={rp[2]:.4f} ref_z={rr[2]:.4f} err_z={abs(rp[2]-rr[2]):.4f}")

px.release_foundation()
