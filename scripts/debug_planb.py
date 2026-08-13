"""Plan B: local poses + real CoM + set_joint([0]) + updateKinematic."""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
import physx_core as px

px.init_foundation()

# ── 2-link test ──
print("=== 2-Link: Plan B (local + real CoM + set_root + set_joint[0] + updateKinematic) ===")

art = px.Articulation()
art.add_link(-1, "parent")
art.add_link(0, "child")
art.add_joint(0, 1, 1, kp=10.0, kd=1.0, force_limit=10.0)  # Y axis

# LOCAL positions (parent at z=1, child at local [0,0,-0.5] = z=0.5)
positions = np.array([0,0,1.0, 0,0,-0.5], dtype=np.float32)
quats     = np.array([1,0,0,0]*2, dtype=np.float32)
parents   = np.array([-1, 0], dtype=np.int32)
masses    = np.array([1.0, 0.5], dtype=np.float32)
inertias  = np.array([0.01]*6, dtype=np.float32)
# REAL CoM offsets (non-zero)
com_pos   = np.array([0,0,0.1, 0,0,0.05], dtype=np.float32)
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

print(f"After createLink (BEFORE set_root):")
print(f"  parent: {art.get_link_world_pose(0)[0]}")
print(f"  child:  {art.get_link_world_pose(1)[0]}")

# Plan B step 3-5: set_root + set_joint([0]) + updateKinematic
art.set_root_world_pose(np.array([0,0,3.0], dtype=np.float32),
                        np.array([1,0,0,0], dtype=np.float32))
art.set_joint_positions(np.array([0.0], dtype=np.float32))

print(f"\nAfter set_root(z=3) + set_joint(0) + updateKinematic:")
print(f"  parent: {art.get_link_world_pose(0)[0]} (expected: [0,0,3])")
print(f"  child:  {art.get_link_world_pose(1)[0]} (expected: [0,0,2.5])")
ce = np.linalg.norm(np.array(art.get_link_world_pose(1)[0]) - np.array([0,0,2.5]))

# Now add to scene and test physics
scene = px.create_scene(gravity=np.array([0,0,-9.81], dtype=np.float32))
mat = scene.create_material(0.6, 0.5, 0.0)
scene.add_articulation(art)

print(f"\nAfter add_articulation (no physics step):")
print(f"  parent: {art.get_link_world_pose(0)[0]}")
print(f"  child:  {art.get_link_world_pose(1)[0]}")

# Re-do set_root + set_joint after adding to scene (scene might change state)
art.set_root_world_pose(np.array([0,0,3.0], dtype=np.float32),
                        np.array([1,0,0,0], dtype=np.float32))
art.set_joint_positions(np.array([0.0], dtype=np.float32))
print(f"\nAfter re-set root + joint (post-add):")
print(f"  parent: {art.get_link_world_pose(0)[0]}")
print(f"  child:  {art.get_link_world_pose(1)[0]}")

# Physics stability test
print(f"\nPhysics steps:")
for step in range(20):
    scene.simulate(0.002)
    scene.fetch_results()
    rp = art.get_root_world_pose()[0]
    jp = art.get_joint_positions()
    cp = art.get_link_world_pose(1)[0]
    if np.any(np.isnan(jp)) or np.any(np.isnan(rp)):
        print(f"  step {step}: NaN!")
        break
    if step < 3 or step % 5 == 0:
        print(f"  step {step}: parent_z={rp[2]:.4f} child_z={cp[2]:.4f} joint={jp[0]:.4f}")

px.release_foundation()

# ── Test with non-zero joint angle ──
print(f"\n=== 2-Link: Plan B with θ=0.5 (Y-axis revolute) ===")
px.init_foundation()

art2 = px.Articulation()
art2.add_link(-1, "p")
art2.add_link(0, "c")
art2.add_joint(0, 1, 1, kp=10.0, kd=1.0, force_limit=10.0)

art2.finalize(link_masses=masses, link_diag_inertia=inertias,
              link_local_pos=positions, link_local_quat=quats,
              link_parents=parents, link_com_pos=com_pos, link_com_quat=com_quat,
              joint_axis=axis, joint_lower=lower, joint_upper=upper,
              joint_friction=fric, position_iters=8, velocity_iters=1)

art2.set_root_world_pose(np.array([0,0,3.0], dtype=np.float32),
                         np.array([1,0,0,0], dtype=np.float32))
art2.set_joint_positions(np.array([0.5], dtype=np.float32))

# Expected: child at parent + R_y(0.5) * local
# local = [0,0,-0.5], R_y(0.5) * [0,0,-0.5] = [-0.5*sin(0.5), 0, -0.5*cos(0.5)]
sin05, cos05 = np.sin(0.5), np.cos(0.5)
expected = np.array([-0.5*sin05, 0, 3.0 - 0.5*cos05])
cp = art2.get_link_world_pose(1)[0]
print(f"  parent: {art2.get_link_world_pose(0)[0]} (expected: [0,0,3])")
print(f"  child:  {cp} (expected: {expected})")
print(f"  FK error: {np.linalg.norm(np.array(cp) - expected):.6f}m")

scene2 = px.create_scene(gravity=np.array([0,0,0], dtype=np.float32))
mat2 = scene2.create_material(0.6, 0.5, 0.0)
scene2.add_articulation(art2)

# Re-apply after add_articulation
art2.set_root_world_pose(np.array([0,0,3.0], dtype=np.float32),
                         np.array([1,0,0,0], dtype=np.float32))
art2.set_joint_positions(np.array([0.5], dtype=np.float32))
print(f"  After re-set: child={art2.get_link_world_pose(1)[0]}")

art2.set_joint_drive_targets(np.array([0.5], dtype=np.float32))
for step in range(10):
    scene2.simulate(0.002)
    scene2.fetch_results()
    rp = art2.get_root_world_pose()[0]
    jp = art2.get_joint_positions()
    if np.any(np.isnan(jp)):
        print(f"  step {step}: NaN!")
        break
    print(f"  step {step}: joint={jp[0]:.4f} (target=0.5)")

px.release_foundation()
print("\nDone.")
