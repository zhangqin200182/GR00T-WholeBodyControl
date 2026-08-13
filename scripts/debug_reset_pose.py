"""Check link poses RIGHT after reset, before any physics step."""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
import physx_core as px
import physx_env, physx_fk

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

px.init_foundation()
env = physx_env.PhysXEnv(px, XML, PKL)

# Manually set a known reference state
import joblib, glob, os
pk = joblib.load(glob.glob(os.path.join(PKL, "**/*.pkl"), recursive=True)[0])
if isinstance(pk, dict) and "dof" not in pk:
    pk = list(pk.values())[0]

idx = 500  # middle of motion
ref_qpos = pk["dof"][idx].astype(np.float64)
ref_root_pos = pk["root_trans_offset"][idx]
pk_quat = pk["root_rot"][idx]
ref_root_quat = np.array([pk_quat[3], pk_quat[0], pk_quat[1], pk_quat[2]], dtype=np.float64)

# Set robot state
env.art.set_root_world_pose(ref_root_pos.astype(np.float32), ref_root_quat.astype(np.float32))
env.art.set_joint_positions(ref_qpos.astype(np.float32))
env.art.set_joint_velocities(np.zeros(29, dtype=np.float32))
env.art.set_joint_drive_targets(ref_qpos.astype(np.float32))

# Compute Python FK
fk = physx_fk.G1ForwardKinematics(XML)
all_poses = fk.compute(ref_root_pos, ref_root_quat, ref_qpos)

print("Link poses RIGHT AFTER reset (no physics step):")
for i, name in enumerate(physx_env.BODY_NAMES):
    pidx = env._body_idx[name]
    px_pos = env.art.get_link_world_pose(pidx)[0]
    fk_all = all_poses[fk.get_link_index(name)]
    fk_pos = fk_all[0]
    diff = np.linalg.norm(px_pos - fk_pos)
    marker = "***" if diff > 0.01 else ""
    print(f"  {name:30s}: PhysX={px_pos}  FK={fk_pos}  diff={diff:.4f}m {marker}")

# Check root
rp = env.art.get_root_world_pose()
print(f"\nRoot: PhysX={rp[0]} FK={ref_root_pos} diff={np.linalg.norm(rp[0]-ref_root_pos):.6f}")
print(f"Joint pos roundtrip max err: {np.max(np.abs(env.art.get_joint_positions() - ref_qpos)):.6f}")

px.release_foundation()
