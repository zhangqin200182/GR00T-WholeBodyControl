"""Test: world-pose createLink + setParentPose/childPose fix on G1."""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx")
import physx_core as px, physx_loader, physx_fk
import joblib, glob, os

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"
print("Loading...")
pk = joblib.load(glob.glob(os.path.join(PKL, "**/*.pkl"), recursive=True)[0])
if "dof" not in pk: pk=list(pk.values())[0]
ref_qpos = pk["dof"][500].astype(np.float64)
ref_pos = np.array(pk["root_trans_offset"][500], dtype=np.float64)

# ── Test 1: FK correctness ──
print("\n=== Test 1: FK ===")
px.init_foundation()
art = physx_loader.load_g1(px, XML)
fk = physx_fk.G1ForwardKinematics(XML)

art.set_root_world_pose(ref_pos.astype(np.float32), np.array([1,0,0,0], dtype=np.float32))
art.set_joint_positions(ref_qpos.astype(np.float32))

root_pos, root_quat = art.get_root_world_pose()
actual_qpos = art.get_joint_positions()
tracked = fk.get_tracked_poses(root_pos, root_quat, actual_qpos)

max_err, max_body = 0.0, -1
for i, (tp, tq) in enumerate(tracked):
    ap, aq = art.get_link_world_pose(i)
    pe = np.linalg.norm(np.array(ap) - tp)
    if pe > max_err: max_err = pe; max_body = i
print(f"  Max FK err: {max_err:.5f}m at {fk.TRACKED_BODIES[max_body]} (vs 3.77m baseline)")

for i in range(4):
    ap = np.array(art.get_link_world_pose(i)[0])
    tp = np.array(tracked[i][0])
    print(f"  {fk.TRACKED_BODIES[i]}: PhysX={np.round(ap,3)} FK={np.round(tp,3)} err={np.linalg.norm(ap-tp):.4f}")
px.release_foundation()

# ── Test 2: Zero-G stability ──
print("\n=== Test 2: Zero-G ===")
px.init_foundation()
art2 = physx_loader.load_g1(px, XML)
scene2 = px.create_scene(gravity=np.array([0,0,0], dtype=np.float32))
scene2.add_articulation(art2)

art2.set_root_world_pose(ref_pos.astype(np.float32), np.array([1,0,0,0], dtype=np.float32))
art2.set_joint_positions(ref_qpos.astype(np.float32))
art2.set_joint_velocities(np.zeros(29, dtype=np.float32))

for step in range(20):
    rp = art2.get_root_world_pose()[0]; jp = art2.get_joint_positions()
    if np.any(np.isnan(jp)) or np.any(np.isnan(rp)):
        print(f"  step {step}: NaN!"); break
    if step < 3 or step % 5 == 0:
        print(f"  step {step}: root_z={rp[2]:.4f} q=[{jp.min():.4f},{jp.max():.4f}]")
    scene2.simulate(0.002); scene2.fetch_results()
px.release_foundation()

# ── Test 3: PD hold with gravity ──
print("\n=== Test 3: Gravity + PD hold ===")
px.init_foundation()
art3 = physx_loader.load_g1(px, XML)
scene3 = px.create_scene(gravity=np.array([0,0,-9.81], dtype=np.float32))
mat3 = scene3.create_material(0.6, 0.5, 0.0)
scene3.add_ground_plane(mat3, np.array([0,0,1], dtype=np.float32))
scene3.add_articulation(art3)

root_start = np.array([ref_pos[0], ref_pos[1], 0.8], dtype=np.float32)
art3.set_root_world_pose(root_start, np.array([1,0,0,0], dtype=np.float32))
art3.set_joint_positions(ref_qpos.astype(np.float32))
art3.set_joint_velocities(np.zeros(29, dtype=np.float32))
art3.set_joint_drive_targets(ref_qpos.astype(np.float32))

for step in range(30):
    rp = art3.get_root_world_pose()[0]; jp = art3.get_joint_positions()
    if np.any(np.isnan(jp)) or np.any(np.isnan(rp)):
        print(f"  step {step}: NaN!"); break
    je = np.linalg.norm(jp - ref_qpos)
    if step < 3 or step % 5 == 0:
        print(f"  step {step}: root_z={rp[2]:.4f} jerr={je:.4f}")
    scene3.simulate(0.002); scene3.fetch_results()
px.release_foundation()

# ── Test 4: Motion tracking ──
print("\n=== Test 4: Motion tracking ===")
px.init_foundation()
art4 = physx_loader.load_g1(px, XML)
scene4 = px.create_scene(gravity=np.array([0,0,-9.81], dtype=np.float32))
mat4 = scene4.create_material(0.6, 0.5, 0.0)
scene4.add_ground_plane(mat4, np.array([0,0,1], dtype=np.float32))
scene4.add_articulation(art4)

idx0 = 500
art4.set_root_world_pose(pk["root_trans_offset"][idx0].astype(np.float32), np.array([1,0,0,0], dtype=np.float32))
art4.set_joint_positions(pk["dof"][idx0].astype(np.float32))
art4.set_joint_velocities(np.zeros(29, dtype=np.float32))

for step in range(20):
    ref = pk["dof"][idx0+step].astype(np.float32)
    art4.set_joint_drive_targets(ref)
    rp = art4.get_root_world_pose()[0]; jp = art4.get_joint_positions()
    if np.any(np.isnan(jp)) or np.any(np.isnan(rp)):
        print(f"  step {step}: NaN!"); break
    je = np.linalg.norm(jp - ref)
    print(f"  step {step}: root_z={rp[2]:.4f} jerr={je:.4f}")
    scene4.simulate(0.002); scene4.fetch_results()

px.release_foundation()
print("\nDone")
