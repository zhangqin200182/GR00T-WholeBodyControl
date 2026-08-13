"""2-link: test createLink FK, then set_joint_positions FK."""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
import physx_core as px

px.init_foundation()

art = px.Articulation()
art.add_link(-1, "parent")
art.add_link(0, "child")
art.add_joint(0, 1, 1, kp=10.0, kd=1.0, force_limit=10.0)

parent_pos = np.array([0, 0, 1.0], dtype=np.float32)
child_local = np.array([0, 0, -0.5], dtype=np.float32)
child_world = parent_pos + child_local  # [0, 0, 0.5]

positions = np.array([0,0,1.0, 0,0,-0.5], dtype=np.float32)
quats     = np.array([1,0,0,0, 1,0,0,0], dtype=np.float32)
parents   = np.array([-1, 0], dtype=np.int32)
masses    = np.array([1.0, 0.5], dtype=np.float32)
inertias  = np.array([0.01]*6, dtype=np.float32)
com_pos   = np.zeros(6, dtype=np.float32)
com_quat  = np.array([1,0,0,0]*2, dtype=np.float32)
axis      = np.array([1], dtype=np.int32)
lower     = np.array([-1.57], dtype=np.float32)
upper     = np.array([1.57], dtype=np.float32)
fric      = np.array([0.0], dtype=np.float32)

art.finalize(link_masses=masses, link_diag_inertia=inertias,
             link_local_pos=positions, link_local_quat=quats,
             link_parents=parents, link_com_pos=com_pos, link_com_quat=com_quat,
             joint_axis=axis, joint_lower=lower, joint_upper=upper,
             joint_friction=fric, position_iters=8, velocity_iters=1)

# After createLink - no set_joint_positions yet
print(f"After createLink:")
print(f"  parent: {art.get_link_world_pose(0)[0]}")
print(f"  child:  {art.get_link_world_pose(1)[0]}")
print(f"  joint_pos: {art.get_joint_positions()}")
print()

# NOW call set_root_world_pose + set_joint_positions (like reset does)
new_root = np.array([0, 0, 2.0], dtype=np.float32)
rot_quat = np.array([0.7071, 0.7071, 0, 0], dtype=np.float32)  # 90° X
art.set_root_world_pose(new_root, rot_quat)
art.set_joint_positions(np.array([0.0], dtype=np.float32))

print(f"After set_root + set_joint_pos:")
print(f"  parent: {art.get_link_world_pose(0)[0]}")
print(f"  child:  {art.get_link_world_pose(1)[0]}")
# After 90° X rotation of local (0,0,-0.5): (0, 0.5, 0) in world → child should be at (0, 0.5, 2.0)
expected = np.array([0, 0.5, 2.0], dtype=np.float32)
print(f"  expected child: {expected}")
print(f"  error: {np.linalg.norm(art.get_link_world_pose(1)[0] - expected):.4f}")
print(f"  joint_pos: {art.get_joint_positions()}")

# Now set non-zero joint angle
art.set_joint_positions(np.array([0.5], dtype=np.float32))
print(f"\nAfter set joint to 0.5:")
print(f"  parent: {art.get_link_world_pose(0)[0]}")
print(f"  child:  {art.get_link_world_pose(1)[0]}")
cx = 2.0 + np.sin(0.5)*0.5  # after X rotation, Y axis → Z axis, Z axis → -Y
print(f"  joint_pos: {art.get_joint_positions()}")

px.release_foundation()
print("Done")
