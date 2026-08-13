"""Check: does get_link_world_pose return CoM or body origin?"""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
import physx_core as px
import physx_env
import physx_loader

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

px.init_foundation()
env = physx_env.PhysXEnv(px, XML, PKL)
env.reset()

# Compare FK body positions vs PhysX get_link_world_pose
fk_pos, fk_quat = env._get_body_state_fk()
print("Body position comparison (FK vs PhysX getGlobalPose):")
for i, name in enumerate(physx_env.BODY_NAMES):
    pidx = env._body_idx[name]
    px_pos = env.art.get_link_world_pose(pidx)[0]
    fk_p = fk_pos[i]
    diff = np.linalg.norm(px_pos - fk_p)
    if diff > 0.01:
        print(f"  {name:30s}: FK={fk_p}  PhysX={px_pos}  diff={diff:.4f}m ***")
    else:
        print(f"  {name:30s}: diff={diff:.6f}m")

# Also check: does set_joint_positions + get_joint_positions roundtrip?
ref_qpos = env.get_current_ref_qpos()
env.art.set_joint_positions(ref_qpos.astype(np.float32))
got = env.art.get_joint_positions()
print(f"\nJoint pos roundtrip error: max={np.max(np.abs(got - ref_qpos)):.6f}")

px.release_foundation()
