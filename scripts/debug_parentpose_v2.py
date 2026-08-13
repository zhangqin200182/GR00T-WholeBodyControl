"""Verify: world-pose + setParentPose with non-zero angle + gravity."""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
import physx_core as px

px.init_foundation()

# ═══ Non-zero joint, gravity ═══
print("=== Test X: θ=0.5, gravity, PD hold ===")
art = px.Articulation()
art.add_link(-1, "parent")
art.add_link(0, "child")
art.add_joint(0, 1, 1, kp=10.0, kd=1.0, force_limit=10.0)

positions = np.array([0,0,1.0, 0,0,1.5], dtype=np.float32)
quats     = np.array([1,0,0,0]*2, dtype=np.float32)
parents   = np.array([-1, 0], dtype=np.int32)
masses    = np.array([1.0, 0.5], dtype=np.float32)
inertias  = np.array([0.01]*6, dtype=np.float32)
com_pos   = np.zeros(6, dtype=np.float32); com_quat  = np.array([1,0,0,0]*2, dtype=np.float32)
axis      = np.array([1], dtype=np.int32)
lower     = np.array([-3.14], dtype=np.float32); upper = np.array([3.14], dtype=np.float32)
fric      = np.array([0.0], dtype=np.float32)
art.finalize(link_masses=masses, link_diag_inertia=inertias,
             link_local_pos=positions, link_local_quat=quats,
             link_parents=parents, link_com_pos=com_pos, link_com_quat=com_quat,
             joint_axis=axis, joint_lower=lower, joint_upper=upper,
             joint_friction=fric, position_iters=8, velocity_iters=1)

scene = px.create_scene(gravity=np.array([0,0,-9.81], dtype=np.float32))
mat = scene.create_material(0.6, 0.5, 0.0)
scene.add_ground_plane(mat, np.array([0,0,1], dtype=np.float32))
scene.add_articulation(art)

art.set_joint_parent_pose(0, np.array([0,0,-0.5], dtype=np.float32), np.array([1,0,0,0], dtype=np.float32))
art.set_root_world_pose(np.array([0,0,2.0], dtype=np.float32), np.array([1,0,0,0], dtype=np.float32))
art.set_joint_positions(np.array([0.5], dtype=np.float32))
art.set_joint_drive_targets(np.array([0.5], dtype=np.float32))

sin05, cos05 = np.sin(0.5), np.cos(0.5)
expected = np.array([-0.5*sin05, 0, 2.0 - 0.5*cos05])
cp = art.get_link_world_pose(1)[0]
err = np.linalg.norm(np.array(cp) - expected)
print(f"  child={cp} expected={expected} err={err:.6f}m {'✅' if err < 0.01 else '❌'}")

for step in range(10):
    scene.simulate(0.002); scene.fetch_results()
    cp = art.get_link_world_pose(1)[0]; jp = art.get_joint_positions()
    if np.any(np.isnan(jp)): print(f"  step {step}: NaN!"); break
    # Expected: child swings down under gravity then PD holds
    print(f"  step {step}: child_z={cp[2]:.4f} joint={jp[0]:.4f} (target=0.5)")

px.release_foundation()

# ═══ Real CoM offsets ═══
print("\n=== Test Y: Non-zero CoM offsets ===")
px.init_foundation()
art = px.Articulation()
art.add_link(-1, "parent")
art.add_link(0, "child")
art.add_joint(0, 1, 1, kp=10.0, kd=1.0, force_limit=10.0)
com_pos = np.array([0,0,0.1, 0,0,0.05], dtype=np.float32)  # non-zero CoM
art.finalize(link_masses=masses, link_diag_inertia=inertias,
             link_local_pos=positions, link_local_quat=quats,
             link_parents=parents, link_com_pos=com_pos, link_com_quat=com_quat,
             joint_axis=axis, joint_lower=lower, joint_upper=upper,
             joint_friction=fric, position_iters=8, velocity_iters=1)

scene2 = px.create_scene(gravity=np.array([0,0,0], dtype=np.float32))
scene2.add_articulation(art)

# Correct parentPose: parent_CoM⁻¹ * child_local
# parent_com = [0,0,0.1], child_local = [0,0,-0.5]
# parent_CoM⁻¹ = R([1,0,0,0]⁻¹) = Identity
# delta = child_local - parent_com = [0,0,-0.6]
# parentPose = R⁻¹ * delta = [0,0,-0.6]

# Fix both parentPose and childPose
art.set_joint_parent_pose(0, np.array([0,0,-0.6], dtype=np.float32), np.array([1,0,0,0], dtype=np.float32))
art.set_joint_child_pose(0, np.array([0,0,0.05], dtype=np.float32), np.array([1,0,0,0], dtype=np.float32))

art.set_root_world_pose(np.array([0,0,3.0], dtype=np.float32), np.array([1,0,0,0], dtype=np.float32))
art.set_joint_positions(np.array([0.0], dtype=np.float32))

# FK chain: child = parent * parent_CoM * parentPose * R * childPose⁻¹ * child_CoM⁻¹
# = [0,0,3] * [0,0,0.1] * [0,0,-0.6] * I * I * [0,0,-0.05]
# = [0,0,3 + 0.1 - 0.6 - 0.05] = [0,0,2.45]
cp = art.get_link_world_pose(1)[0]
expected = np.array([0,0,2.45])
err = np.linalg.norm(np.array(cp) - expected)
print(f"  child={cp} expected={expected} err={err:.6f}m {'✅' if err < 0.01 else '❌'}")

for step in range(10):
    scene2.simulate(0.002); scene2.fetch_results()
    cp = art.get_link_world_pose(1)[0]; jp = art.get_joint_positions()
    if np.any(np.isnan(jp)): print(f"  step {step}: NaN!"); break
    print(f"  step {step}: child_z={cp[2]:.4f} joint={jp[0]:.4f}")

px.release_foundation()
print("\nDone")
