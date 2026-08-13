"""Quick debug: check why α and reward are NaN."""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
import physx_core as px
import physx_env

px.init_foundation()
env = physx_env.PhysXEnv(px, "/gear_sonic_deploy/g1/g1_29dof_v17.xml", "/sample_data/robot_filtered")
env.reset()

for i in range(5):
    ref_qpos = env.get_current_ref_qpos()
    action = (ref_qpos - env.jm) / env.jh
    obs, reward, done, info = env.step(action.astype(np.float64))
    rp = env.art.get_root_world_pose()[0]
    rr = env._ref_root_pos()
    print(f"step {i}: root_pos={rp} ref_root={rr} drift={np.linalg.norm(rp - rr):.6f} reward={reward:.6f}")
    jp = env.art.get_joint_positions()
    print(f"  joint_pos range=[{jp.min():.4f}, {jp.max():.4f}]")

px.release_foundation()
