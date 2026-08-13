"""2-link: test if adding to scene fixes kinematic propagation."""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
import physx_core as px

px.init_foundation()

art = px.Articulation()
art.add_link(-1, "parent")
art.add_link(0, "child")
art.add_joint(0, 1, 1, kp=10.0, kd=1.0, force_limit=10.0)

positions = np.array([0,0,1.0, 0,0,-0.5], dtype=np.float32)
quats     = np.array([1,0,0,0]*2, dtype=np.float32)
parents   = np.array([-1, 0], dtype=np.int32)
masses    = np.array([1.0, 0.5], dtype=np.float32)
inertias  = np.array([0.01]*6, dtype=np.float32)
com_pos   = np.zeros(6, dtype=np.float32)
com_quat  = np.array([1,0,0,0]*2, dtype=np.float32)
axis      = np.array([1], dtype=np.int32)
lower     = np.array([-3.14], dtype=np.float32)
upper     = np.array([3.14], dtype=np.float32)
fric      = np.array([0.0], dtype=np.float32)

art.finalize(link_masses=masses, link_diag_inertia=inertias,
             link_local_pos=positions, link_local_quat=quats,
             link_parents=parents, link_com_pos=com_pos, link_com_quat=com_quat,
             joint_axis=axis, joint_lower=lower, joint_upper=upper,
             joint_friction=fric, position_iters=8, velocity_iters=1)

# Add to scene (zero gravity)
scene = px.create_scene(gravity=np.array([0,0,0], dtype=np.float32))
mat = scene.create_material(0.6, 0.5, 0.0)
scene.add_ground_plane(mat, np.array([0,0,1], dtype=np.float32))
scene.add_articulation(art)

print("A) After add to scene (before physics):")
print(f"  parent: {art.get_link_world_pose(0)[0]}")
print(f"  child:  {art.get_link_world_pose(1)[0]}")
print(f"  joint_pos: {art.get_joint_positions()}")

# Set joint to 0.5 (no root change)
art.set_joint_positions(np.array([0.5], dtype=np.float32))
print(f"\nB) After set_joint(0.5) — NO physics step:")
print(f"  parent: {art.get_link_world_pose(0)[0]}")
print(f"  child:  {art.get_link_world_pose(1)[0]}")
sin05 = np.sin(0.5); cos05 = np.cos(0.5)
expected = np.array([0,0,1.0]) + np.array([-0.5*sin05, 0, -0.5*cos05])
print(f"  expected child: {expected}")
print(f"  error: {np.linalg.norm(art.get_link_world_pose(1)[0] - expected):.4f}")

# Run one physics step then check
scene.simulate(0.001)
scene.fetch_results()
print(f"\nC) After 1 physics step (0.001s, zero-G):")
print(f"  parent: {art.get_link_world_pose(0)[0]}")
print(f"  child:  {art.get_link_world_pose(1)[0]}")
print(f"  joint_pos: {art.get_joint_positions()}")

# Set root to rotated position, then simulate
art.set_root_world_pose(np.array([0,0,2.0], dtype=np.float32),
                        np.array([0.7071, 0.7071, 0, 0], dtype=np.float32))
art.set_joint_positions(np.array([0.0], dtype=np.float32))
print(f"\nD) After set_root(rotated) — before physics:")
print(f"  parent: {art.get_link_world_pose(0)[0]}")
print(f"  child:  {art.get_link_world_pose(1)[0]}")

scene.simulate(0.001)
scene.fetch_results()
print(f"\nE) After physics step:")
print(f"  parent: {art.get_link_world_pose(0)[0]}")
print(f"  child:  {art.get_link_world_pose(1)[0]}")

# Fresh test: simple translation (no rotation)
art2 = px.Articulation()
art2.add_link(-1, "p"); art2.add_link(0, "c")
art2.add_joint(0, 1, 1, kp=10.0, kd=1.0, force_limit=10.0)
art2.finalize(link_masses=masses, link_diag_inertia=inertias,
              link_local_pos=positions, link_local_quat=quats,
              link_parents=parents, link_com_pos=com_pos, link_com_quat=com_quat,
              joint_axis=axis, joint_lower=lower, joint_upper=upper,
              joint_friction=fric, position_iters=8, velocity_iters=1)
scene2 = px.create_scene(gravity=np.array([0,0,0], dtype=np.float32))
mat2 = scene2.create_material(0.6, 0.5, 0.0)
scene2.add_ground_plane(mat2, np.array([0,0,1], dtype=np.float32))
scene2.add_articulation(art2)

print("\nF) Fresh art2: after createLink + add to scene:")
print(f"  parent: {art2.get_link_world_pose(0)[0]}")
print(f"  child:  {art2.get_link_world_pose(1)[0]}")

art2.set_root_world_pose(np.array([0,0,3.0], dtype=np.float32),
                         np.array([1,0,0,0], dtype=np.float32))
art2.set_joint_positions(np.array([0.0], dtype=np.float32))
print(f"\nG) After set_root(z=3) — before physics:")
print(f"  parent: {art2.get_link_world_pose(0)[0]}")
print(f"  child:  {art2.get_link_world_pose(1)[0]}")
# Expected: child at (0,0,3) + local(0,0,-0.5) = (0,0,2.5)

scene2.simulate(0.001)
scene2.fetch_results()
print(f"\nH) After physics:")
print(f"  parent: {art2.get_link_world_pose(0)[0]}")
print(f"  child:  {art2.get_link_world_pose(1)[0]}")

px.release_foundation()
print("Done")
