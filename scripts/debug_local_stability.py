"""Test local poses with real CoM offsets — FK correctness + physics stability.

Hypothesis: local poses (correct joint frame) + real CoM (correct mass
distribution) = both FK ✅ and physics ✅.
"""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx")
import physx_core as px, physx_loader as loader, physx_fk
import joblib, glob, os

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

print("Loading data...")
pk = joblib.load(glob.glob(os.path.join(PKL, "**/*.pkl"), recursive=True)[0])
if "dof" not in pk: pk=list(pk.values())[0]
ref_qpos = pk["dof"][500].astype(np.float64)
ref_pos = np.array(pk["root_trans_offset"][500], dtype=np.float64)

# ── Test 1: FK check — verify real CoM doesn't break FK ──
print("\n=== Test 1: FK correctness (real CoM, no physics) ===")
px.init_foundation()
art = loader.load_g1(px, XML)
fk = physx_fk.G1ForwardKinematics("/gear_sonic_deploy/g1/g1_29dof_v17.xml")

art.set_root_world_pose(ref_pos.astype(np.float32),
                        np.array([1,0,0,0], dtype=np.float32))
art.set_joint_positions(ref_qpos.astype(np.float32))

root_pos, root_quat = art.get_root_world_pose()
actual_qpos = art.get_joint_positions()
tracked = fk.get_tracked_poses(root_pos, root_quat, actual_qpos)

max_err = 0.0
for i, (tp, tq) in enumerate(tracked):
    ap, aq = art.get_link_world_pose(i)
    pe = np.linalg.norm(np.array(ap) - tp)
    if pe > max_err: max_err = pe
print(f"  Max FK error across {len(tracked)} bodies: {max_err:.6f}m")
if max_err < 0.01:
    print("  FK ✅ (within tolerance)")
else:
    print(f"  FK ❌ (max error {max_err:.4f}m)")

px.release_foundation()

# ── Test 2: Zero-G, no PD — pure physics stability ──
print("\n=== Test 2: Zero-G, no PD drives ===")
px.init_foundation()
art2 = loader.load_g1(px, XML)
scene2 = px.create_scene(gravity=np.array([0,0,0], dtype=np.float32))
mat2 = scene2.create_material(0.6, 0.5, 0.0)
scene2.add_articulation(art2)

art2.set_root_world_pose(ref_pos.astype(np.float32),
                         np.array([1,0,0,0], dtype=np.float32))
art2.set_joint_positions(ref_qpos.astype(np.float32))
art2.set_joint_velocities(np.zeros(29, dtype=np.float32))

for step in range(20):
    rp = art2.get_root_world_pose()[0]
    jp = art2.get_joint_positions()
    if np.any(np.isnan(jp)) or np.any(np.isnan(rp)):
        print(f"  step {step}: NaN!")
        break
    if step < 3 or step % 5 == 0:
        print(f"  step {step}: root={rp[2]:.4f}, q_range=[{jp.min():.4f}, {jp.max():.4f}]")
    scene2.simulate(0.002)
    scene2.fetch_results()

px.release_foundation()

# ── Test 3: Zero-G with PD drives holding ref ──
print("\n=== Test 3: Zero-G + PD holding ref pose ===")
px.init_foundation()
art3 = loader.load_g1(px, XML)
scene3 = px.create_scene(gravity=np.array([0,0,0], dtype=np.float32))
mat3 = scene3.create_material(0.6, 0.5, 0.0)
scene3.add_articulation(art3)

art3.set_root_world_pose(ref_pos.astype(np.float32),
                         np.array([1,0,0,0], dtype=np.float32))
art3.set_joint_positions(ref_qpos.astype(np.float32))
art3.set_joint_velocities(np.zeros(29, dtype=np.float32))
art3.set_joint_drive_targets(ref_qpos.astype(np.float32))

for step in range(20):
    rp = art3.get_root_world_pose()[0]
    jp = art3.get_joint_positions()
    if np.any(np.isnan(jp)) or np.any(np.isnan(rp)):
        print(f"  step {step}: NaN!")
        break
    if step < 3 or step % 5 == 0:
        print(f"  step {step}: root={rp[2]:.4f}, q_range=[{jp.min():.4f}, {jp.max():.4f}]")
    scene3.simulate(0.002)
    scene3.fetch_results()

px.release_foundation()

# ── Test 4: Full test — gravity + ground + PD ──
print("\n=== Test 4: Gravity + ground + PD holding ref ===")
px.init_foundation()
art4 = loader.load_g1(px, XML)
scene4 = px.create_scene(gravity=np.array([0,0,-9.81], dtype=np.float32))
mat4 = scene4.create_material(0.6, 0.5, 0.0)
scene4.add_ground_plane(mat4, np.array([0,0,1], dtype=np.float32))
scene4.add_articulation(art4)

root_start = np.array([ref_pos[0], ref_pos[1], 0.8], dtype=np.float32)
art4.set_root_world_pose(root_start, np.array([1,0,0,0], dtype=np.float32))
art4.set_joint_positions(ref_qpos.astype(np.float32))
art4.set_joint_velocities(np.zeros(29, dtype=np.float32))
art4.set_joint_drive_targets(ref_qpos.astype(np.float32))

for step in range(20):
    rp = art4.get_root_world_pose()[0]
    jp = art4.get_joint_positions()
    if np.any(np.isnan(jp)) or np.any(np.isnan(rp)):
        print(f"  step {step}: NaN!")
        break
    if step < 3 or step % 5 == 0:
        print(f"  step {step}: root={rp}, q_range=[{jp.min():.4f}, {jp.max():.4f}]")
    scene4.simulate(0.002)
    scene4.fetch_results()

px.release_foundation()

# ── Test 5: Multi-frame motion tracking (subset of ref PD) ──
print("\n=== Test 5: Motion tracking over 10 frames ===")
px.init_foundation()
art5 = loader.load_g1(px, XML)
scene5 = px.create_scene(gravity=np.array([0,0,-9.81], dtype=np.float32))
mat5 = scene5.create_material(0.6, 0.5, 0.0)
scene5.add_ground_plane(mat5, np.array([0,0,1], dtype=np.float32))
scene5.add_articulation(art5)

start_idx = 500
art5.set_root_world_pose(
    np.array(pk["root_trans_offset"][start_idx], dtype=np.float32),
    np.array([1,0,0,0], dtype=np.float32))
art5.set_joint_positions(pk["dof"][start_idx].astype(np.float32))
art5.set_joint_velocities(np.zeros(29, dtype=np.float32))

for step in range(10):
    idx = start_idx + step
    target = pk["dof"][idx].astype(np.float32)
    art5.set_joint_drive_targets(target)

    rp = art5.get_root_world_pose()[0]
    jp = art5.get_joint_positions()
    if np.any(np.isnan(jp)) or np.any(np.isnan(rp)):
        print(f"  step {step}: NaN!")
        break

    jerr = np.linalg.norm(jp - target)
    print(f"  step {step}: root_z={rp[2]:.4f}, joint_err={jerr:.4f}, q_range=[{jp.min():.4f}, {jp.max():.4f}]")
    scene5.simulate(0.002)
    scene5.fetch_results()

px.release_foundation()
print("\nDone.")
