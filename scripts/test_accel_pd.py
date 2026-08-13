"""Test PhysX eACCELERATION drive — ref PD tracking."""
import sys, os, numpy as np, glob, joblib, time

_repo = "/root/GR00T-WholeBodyControl"
sys.path.insert(0, _repo)
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx"))
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx", "build"))

import physx_core, physx_loader
load_g1 = physx_loader.load_g1

# Load motion
pkls = [p for p in glob.glob("/sample_data/robot_filtered/**/*.pkl", recursive=True)
        if not os.path.basename(p).startswith("._")]
data = joblib.load(pkls[0])
mname = list(data.keys())[0]; motion = data[mname]
dof_ref = motion["dof"]
root_trans = motion["root_trans_offset"]
root_rot = motion["root_rot"]
print(f"Motion: {len(dof_ref)} frames x {len(dof_ref[0])} DOF")
print(f"Drive type: eACCELERATION")

px = physx_core
px.init_foundation()
art = load_g1(px, "/gear_sonic_deploy/g1/g1_29dof_v17.xml", pos_iters=8, vel_iters=1)
scene = px.create_scene(gravity=np.array([0., 0., -9.81], dtype=np.float32))
mat = scene.create_material(0.6, 0.5, 0.0)
scene.add_ground_plane(mat, np.array([0., 0., 1.], dtype=np.float32))
scene.add_articulation(art)

native_dt = 0.001961
decimation = 17
ctrl_dt = native_dt * decimation
nu = 29

# ── Test 1: Static hold at zero ──
print("\n=== Test 1: Static hold at zero ===")
art.set_root_world_pose(np.array([0., 0., 0.9], dtype=np.float32),
                        np.array([1., 0., 0., 0.], dtype=np.float32))
art.set_joint_positions(np.zeros(nu, dtype=np.float32))
art.set_joint_velocities(np.zeros(nu, dtype=np.float32))
art.set_joint_drive_targets(np.zeros(nu, dtype=np.float32))
for _ in range(100):
    scene.simulate(native_dt)
    scene.fetch_results()
q = art.get_joint_positions()
print(f"  Max |q|: {np.max(np.abs(q)):.6f} rad")

# ── Test 2: Per-DOF perturbation ──
print("\n=== Test 2: Per-DOF 0.1 rad perturbation ===")
for dof_i in range(nu):
    art.set_joint_positions(np.zeros(nu, dtype=np.float32))
    art.set_joint_velocities(np.zeros(nu, dtype=np.float32))
    art.set_joint_drive_targets(np.zeros(nu, dtype=np.float32))
    for _ in range(30):
        scene.simulate(native_dt)
        scene.fetch_results()
    tgt = np.zeros(nu, dtype=np.float32)
    tgt[dof_i] = 0.1
    art.set_joint_drive_targets(tgt)
    for _ in range(50):
        scene.simulate(native_dt)
        scene.fetch_results()
    q = art.get_joint_positions()
    err = float(q[dof_i] - 0.1)
    flag = " <<<" if abs(err) > 0.01 else ""
    if abs(err) > 0.005:
        print(f"  DOF[{dof_i:2d}]: target=0.1 actual={q[dof_i]:.4f} err={err:.4f}{flag}")

# ── Test 3: Sine tracking ──
print("\n=== Test 3: Sine tracking (1 Hz, A=0.1) ===")
art.set_joint_positions(np.zeros(nu, dtype=np.float32))
art.set_joint_velocities(np.zeros(nu, dtype=np.float32))
art.set_joint_drive_targets(np.zeros(nu, dtype=np.float32))
for _ in range(50):
    scene.simulate(native_dt)
    scene.fetch_results()

n_frames = 100
errors = []
t = 0.0
for i in range(n_frames):
    t += ctrl_dt
    ref = np.zeros(nu, dtype=np.float64)
    ref[0] = 0.1 * np.sin(2 * np.pi * 1.0 * t)
    ref[6] = 0.1 * np.sin(2 * np.pi * 1.0 * t + 0.5)
    ref[15] = 0.1 * np.sin(2 * np.pi * 1.0 * t + 1.0)
    art.set_joint_drive_targets(ref.astype(np.float32))
    for _ in range(decimation):
        scene.simulate(native_dt)
        scene.fetch_results()
    errors.append(art.get_joint_positions().astype(np.float64) - ref)

errors = np.array(errors)
alpha = float(np.sqrt(np.mean(errors[-50:] ** 2)))
driven = [0, 6, 15]
alpha_driven = float(np.sqrt(np.mean(errors[-50:, driven] ** 2)))
print(f"  alpha (all): {alpha:.6f} rad")
print(f"  alpha (driven only): {alpha_driven:.6f} rad")

# ── Test 4: Full ref PD tracking (root pinned) ──
print("\n=== Test 4: Ref PD tracking (root pinned) ===")
start_idx = 200
ref_q0 = dof_ref[start_idx]
rr0 = root_rot[start_idx]
root_quat0 = np.array([rr0[3], rr0[0], rr0[1], rr0[2]], dtype=np.float32)
art.set_root_world_pose(root_trans[start_idx].astype(np.float32), root_quat0)
art.set_joint_positions(ref_q0.astype(np.float32))
art.set_joint_velocities(np.zeros(nu, dtype=np.float32))
art.set_joint_drive_targets(ref_q0.astype(np.float32))
for _ in range(100):
    scene.simulate(native_dt)
    scene.fetch_results()

t0 = time.time()
n_frames = 100
errors = []
for i in range(n_frames):
    ref_idx = start_idx + i + 1
    ref_target = dof_ref[ref_idx].astype(np.float32)
    rr = root_rot[ref_idx]
    root_quat = np.array([rr[3], rr[0], rr[1], rr[2]], dtype=np.float32)
    art.set_root_world_pose(root_trans[ref_idx].astype(np.float32), root_quat)
    art.set_joint_drive_targets(ref_target)
    for _ in range(decimation):
        scene.simulate(native_dt)
        scene.fetch_results()
    errors.append(art.get_joint_positions().astype(np.float64) - ref_target.astype(np.float64))

errors = np.array(errors)
alpha = float(np.sqrt(np.mean(errors[-50:] ** 2)))
elapsed = time.time() - t0
print(f"  alpha (ref PD, root pinned): {alpha:.6f} rad")
print(f"  vs Isaac 0.002: {alpha/0.002:.1f}x")

# Per-DOF
print(f"  Per-DOF RMS (last 50 frames):")
leg_rms, arm_rms, waist_rms = [], [], []
for j in range(nu):
    rms = np.sqrt(np.mean(errors[-50:, j] ** 2))
    flag = " ***" if rms > 0.05 else ("  OK" if rms < 0.005 else "")
    print(f"    [{j:2d}] rms={rms:.6f}{flag}")
    if j < 12: leg_rms.append(rms)
    elif 12 <= j <= 14: waist_rms.append(rms)
    else: arm_rms.append(rms)

print(f"  Leg RMS: {np.mean(leg_rms):.6f}, Waist RMS: {np.mean(waist_rms):.6f}, Arm RMS: {np.mean(arm_rms):.6f}")
print(f"  Time: {elapsed:.2f}s ({elapsed/n_frames*1000:.1f} ms/frame)")

# ── Test 5: Ref PD with FREE root (same as real training) ──
print("\n=== Test 5: Ref PD tracking (FREE root, with gravity, no ground) ===")
# Remove ground plane by creating a new scene without it
px2 = physx_core
px2.init_foundation()
art2 = load_g1(px2, "/gear_sonic_deploy/g1/g1_29dof_v17.xml", pos_iters=8, vel_iters=1)
scene2 = px2.create_scene(gravity=np.array([0., 0., -9.81], dtype=np.float32))
mat2 = scene2.create_material(0.6, 0.5, 0.0)
scene2.add_ground_plane(mat2, np.array([0., 0., 1.], dtype=np.float32))
scene2.add_articulation(art2)

start_idx = 200
ref_q0 = dof_ref[start_idx]
rr0 = root_rot[start_idx]
root_quat0 = np.array([rr0[3], rr0[0], rr0[1], rr0[2]], dtype=np.float32)
art2.set_root_world_pose(root_trans[start_idx].astype(np.float32), root_quat0)
art2.set_joint_positions(ref_q0.astype(np.float32))
art2.set_joint_velocities(np.zeros(nu, dtype=np.float32))
art2.set_joint_drive_targets(ref_q0.astype(np.float32))
for _ in range(100):
    scene2.simulate(native_dt)
    scene2.fetch_results()

n_frames = 100
errors = []
for i in range(n_frames):
    ref_idx = start_idx + i + 1
    ref_target = dof_ref[ref_idx].astype(np.float32)
    art2.set_joint_drive_targets(ref_target)
    for _ in range(decimation):
        scene2.simulate(native_dt)
        scene2.fetch_results()
    errors.append(art2.get_joint_positions().astype(np.float64) - ref_target.astype(np.float64))

errors = np.array(errors)
# Check for NaN
if np.any(np.isnan(errors)):
    nan_frame = np.where(np.isnan(errors).any(axis=1))[0][0]
    print(f"  NaN at frame {nan_frame}!")
else:
    alpha = float(np.sqrt(np.mean(errors[-50:] ** 2)))
    print(f"  alpha (free root): {alpha:.6f} rad")
    print(f"  vs Isaac 0.002: {alpha/0.002:.1f}x")

px2.release_foundation()

px.release_foundation()
print("\n*** DONE ***")
