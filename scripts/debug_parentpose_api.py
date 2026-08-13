"""Test: world-pose createLink → fix parentPose via API → verify propagation."""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
import physx_core as px

px.init_foundation()

# ═══ Test A: world-pose createLink → setParentPose → set_root → check ═══
print("=== A: World createLink + setParentPose(correct) ===")
art = px.Articulation()
art.add_link(-1, "parent")
art.add_link(0, "child")
art.add_joint(0, 1, 1, kp=10.0, kd=1.0, force_limit=10.0)

# World-accumulated poses (parent at z=1, child at z=0.5)
positions = np.array([0,0,1.0, 0,0,1.5], dtype=np.float32)  # child world = parent + [0,0,0.5] = [0,0,1.5]
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

print(f"After createLink:  parent={art.get_link_world_pose(0)[0]}  child={art.get_link_world_pose(1)[0]}")
# Expected: child at [0,0,1.5] (world coordinate)

# Fix parentPose to correct local value: child_local = [0,0,-0.5] in parent frame
art.set_joint_parent_pose(0, np.array([0,0,-0.5], dtype=np.float32), np.array([1,0,0,0], dtype=np.float32))

# Now set root and check propagation
art.set_root_world_pose(np.array([0,0,3.0], dtype=np.float32), np.array([1,0,0,0], dtype=np.float32))
art.set_joint_positions(np.array([0.0], dtype=np.float32))

print(f"After fix+set_root: parent={art.get_link_world_pose(0)[0]}  child={art.get_link_world_pose(1)[0]}")
# Expected: parent at [0,0,3], child at [0,0,2.5]
err = np.linalg.norm(np.array(art.get_link_world_pose(1)[0]) - np.array([0,0,2.5]))
print(f"  Child error: {err:.6f}m {'✅' if err < 0.01 else '❌'}")

# Add to scene, check again
scene = px.create_scene(gravity=np.array([0,0,0], dtype=np.float32))
mat = scene.create_material(0.6, 0.5, 0.0)
scene.add_articulation(art)
print(f"After add_artic:  parent={art.get_link_world_pose(0)[0]}  child={art.get_link_world_pose(1)[0]}")

# Re-apply after add
art.set_root_world_pose(np.array([0,0,3.0], dtype=np.float32), np.array([1,0,0,0], dtype=np.float32))
art.set_joint_positions(np.array([0.0], dtype=np.float32))
print(f"After re-set:     parent={art.get_link_world_pose(0)[0]}  child={art.get_link_world_pose(1)[0]}")

# Physics test
for step in range(10):
    scene.simulate(0.002); scene.fetch_results()
    cp = art.get_link_world_pose(1)[0]
    jp = art.get_joint_positions()
    if np.any(np.isnan(jp)):
        print(f"  step {step}: NaN!"); break
    if step < 3 or step % 3 == 0:
        print(f"  step {step}: child_z={cp[2]:.4f} joint={jp[0]:.4f}")

px.release_foundation()

# ═══ Test B: setParentPose AFTER add_articulation ═══
print("\n=== B: setParentPose AFTER add_articulation ===")
px.init_foundation()
art = px.Articulation()
art.add_link(-1, "parent")
art.add_link(0, "child")
art.add_joint(0, 1, 1, kp=10.0, kd=1.0, force_limit=10.0)
art.finalize(link_masses=masses, link_diag_inertia=inertias,
             link_local_pos=positions, link_local_quat=quats,
             link_parents=parents, link_com_pos=com_pos, link_com_quat=com_quat,
             joint_axis=axis, joint_lower=lower, joint_upper=upper,
             joint_friction=fric, position_iters=8, velocity_iters=1)

scene2 = px.create_scene(gravity=np.array([0,0,0], dtype=np.float32))
scene2.add_articulation(art)

art.set_joint_parent_pose(0, np.array([0,0,-0.5], dtype=np.float32), np.array([1,0,0,0], dtype=np.float32))
art.set_root_world_pose(np.array([0,0,3.0], dtype=np.float32), np.array([1,0,0,0], dtype=np.float32))
art.set_joint_positions(np.array([0.0], dtype=np.float32))

print(f"After add+fix+set: parent={art.get_link_world_pose(0)[0]} child={art.get_link_world_pose(1)[0]}")
err = np.linalg.norm(np.array(art.get_link_world_pose(1)[0]) - np.array([0,0,2.5]))
print(f"  Child error: {err:.6f}m {'✅' if err < 0.01 else '❌'}")

for step in range(10):
    scene2.simulate(0.002); scene2.fetch_results()
    cp = art.get_link_world_pose(1)[0]; jp = art.get_joint_positions()
    if np.any(np.isnan(jp)):
        print(f"  step {step}: NaN!"); break
    if step < 3:
        print(f"  step {step}: child_z={cp[2]:.4f} joint={jp[0]:.4f}")

px.release_foundation()

# ═══ Test C: setParentPose + setChildPose (both) ═══
print("\n=== C: setParentPose + setChildPose ===")
px.init_foundation()
art = px.Articulation()
art.add_link(-1, "parent")
art.add_link(0, "child")
art.add_joint(0, 1, 1, kp=10.0, kd=1.0, force_limit=10.0)
art.finalize(link_masses=masses, link_diag_inertia=inertias,
             link_local_pos=positions, link_local_quat=quats,
             link_parents=parents, link_com_pos=com_pos, link_com_quat=com_quat,
             joint_axis=axis, joint_lower=lower, joint_upper=upper,
             joint_friction=fric, position_iters=8, velocity_iters=1)

scene3 = px.create_scene(gravity=np.array([0,0,0], dtype=np.float32))
scene3.add_articulation(art)

art.set_joint_parent_pose(0, np.array([0,0,-0.5], dtype=np.float32), np.array([1,0,0,0], dtype=np.float32))
art.set_joint_child_pose(0, np.array([0,0,0], dtype=np.float32), np.array([1,0,0,0], dtype=np.float32))
art.set_root_world_pose(np.array([0,0,3.0], dtype=np.float32), np.array([1,0,0,0], dtype=np.float32))
art.set_joint_positions(np.array([0.0], dtype=np.float32))

print(f"After both+set: parent={art.get_link_world_pose(0)[0]} child={art.get_link_world_pose(1)[0]}")
err = np.linalg.norm(np.array(art.get_link_world_pose(1)[0]) - np.array([0,0,2.5]))
print(f"  Child error: {err:.6f}m {'✅' if err < 0.01 else '❌'}")

for step in range(10):
    scene3.simulate(0.002); scene3.fetch_results()
    cp = art.get_link_world_pose(1)[0]; jp = art.get_joint_positions()
    if np.any(np.isnan(jp)):
        print(f"  step {step}: NaN!"); break
    if step < 3:
        print(f"  step {step}: child_z={cp[2]:.4f} joint={jp[0]:.4f}")

px.release_foundation()
print("\nDone")
