"""Test direct API PD tracking — verify α=0.002 on NPU.
Reference: physx_loader.load_g1 with pos_iters=8, vel_iters=1.
"""
import sys, os, numpy as np, glob, joblib, time

_repo = "/root/GR00T-WholeBodyControl"
sys.path.insert(0, _repo)
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs"))
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx"))
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx", "build"))

import physx_core
import physx_loader
load_g1 = physx_loader.load_g1

print("=== Direct API PD Tracking Test ===")

# physx_core is the module (not a class instance)
px = physx_core
px.init_foundation()
xml = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
art = load_g1(px, xml, pos_iters=8, vel_iters=1)

# Create scene with ground
scene = px.create_scene(gravity=np.array([0, 0, -9.81], dtype=np.float32))
mat = scene.create_material(0.6, 0.5, 0.0)
scene.add_ground_plane(mat, np.array([0, 0, 1], dtype=np.float32))
scene.add_articulation(art)

# Load reference motions
pkl_dir = "/sample_data/robot_filtered"
pkls = [p for p in glob.glob(os.path.join(pkl_dir, "**/*.pkl"), recursive=True)
        if not os.path.basename(p).startswith("._")]
if not pkls:
    print("No PKLs found")
    sys.exit(1)

motion = joblib.load(pkls[0])
if isinstance(motion, dict) and "dof" in motion:
    dof_ref = motion["dof"]
else:
    for key in motion:
        if isinstance(motion[key], dict) and "dof" in motion[key]:
            motion = motion[key]
            dof_ref = motion["dof"]
            break

fps = motion.get("fps", 30.0)
ref_dt = 1.0 / fps
native_dt = 0.002
decimation = 10  # matches PhysXEnv default (50 Hz ctrl)
ctrl_dt = native_dt * decimation

print(f"Motion: {len(dof_ref)} frames @ {fps} FPS")
print(f"Control: {ctrl_dt:.4f}s ({1/ctrl_dt:.1f} Hz)")

# Isaac-style PD action parameters (from PhysXEnv)
jm = np.array([
     0.1745,  1.2217,  0.0000,  1.3963, -0.1745,  0.0000,
     0.1745, -1.2217,  0.0000,  1.3963, -0.1745,  0.0000,
     0.0000,  0.0000,  0.0000,
    -0.2094,  0.3317,  0.0000,  0.5236,  0.0000,  0.0000,  0.0000,
    -0.2094, -0.3317,  0.0000,  0.5236,  0.0000,  0.0000,  0.0000,
], dtype=np.float64)
jh = np.array([
    2.7052, 1.7453, 2.7576, 1.4835, 0.6981, 0.2618,
    2.7052, 1.7453, 2.7576, 1.4835, 0.6981, 0.2618,
    2.6180, 0.5200, 0.5200,
    2.8798, 1.9199, 2.6180, 1.5708, 1.9722, 1.6144, 1.6144,
    2.8798, 1.9199, 2.6180, 1.5708, 1.9722, 1.6144, 1.6144,
], dtype=np.float64)

nu = 29
n_frames = 100
errors_all = []

t0 = time.time()

for ep in range(3):  # 3 episodes
    # Random start position
    start_idx = np.random.randint(0, len(dof_ref) - n_frames - 50)
    ref_q = dof_ref[start_idx].astype(np.float64)

    # Reset articulation
    art.set_root_world_pose(
        np.array([0, 0, 0.828], dtype=np.float32),
        np.array([1, 0, 0, 0], dtype=np.float32))
    art.set_joint_positions(ref_q.astype(np.float32))
    art.set_joint_velocities(np.zeros(nu, dtype=np.float32))
    art.set_joint_drive_targets(ref_q.astype(np.float32))

    # Settle
    for _ in range(20):
        scene.simulate(native_dt)
        scene.fetch_results()

    errors_ep = []
    for i in range(n_frames):
        ref_idx = start_idx + i + 1
        ref_target = dof_ref[ref_idx].astype(np.float64)
        art.set_joint_drive_targets(ref_target.astype(np.float32))

        for _ in range(decimation):
            scene.simulate(native_dt)
            scene.fetch_results()

        q_actual = art.get_joint_positions().astype(np.float64)
        errors_ep.append(q_actual - ref_target)

    errors_ep = np.array(errors_ep)
    alpha = float(np.sqrt(np.mean(errors_ep ** 2)))
    errors_all.append(errors_ep)
    print(f"  Ep {ep}: alpha={alpha:.6f} rad  (start_idx={start_idx})")

elapsed = time.time() - t0
errors_all = np.concatenate(errors_all, axis=0)
alpha_total = float(np.sqrt(np.mean(errors_all ** 2)))
print(f"\nTotal alpha = {alpha_total:.6f} rad")
print(f"vs Isaac target (0.002): {alpha_total/0.002:.1f}x")
print(f"Time: {elapsed:.1f}s for {3*n_frames} steps ({3*n_frames*decimation} physx steps)")
print()

# Per-DOF breakdown
print("Per-DOF RMS error (all episodes):")
for j in range(nu):
    rms = np.sqrt(np.mean(errors_all[:, j] ** 2))
    flag = " ***" if rms > 0.05 else (" !!!" if rms < 0.005 else "")
    print(f"  [{j:2d}] rms={rms:.6f}{flag}")

print("\n*** DONE ***")
