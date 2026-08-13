"""Test direct API PD tracking with root pinned to reference trajectory."""
import sys, os, numpy as np, glob, joblib

_repo = "/root/GR00T-WholeBodyControl"
sys.path.insert(0, _repo)
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx"))
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx", "build"))

import physx_core, physx_loader

# Load motion
pkls = [p for p in glob.glob("/sample_data/robot_filtered/**/*.pkl", recursive=True)
        if not os.path.basename(p).startswith("._")]
data = joblib.load(pkls[0])
mname = list(data.keys())[0]
motion = data[mname]
dof_ref = motion["dof"]
root_trans = motion["root_trans_offset"]
root_rot = motion["root_rot"]
fps = motion["fps"]

print(f"Motion: {len(dof_ref)} frames x {len(dof_ref[0])} DOF")

# Test with different kp multipliers
for mult in [1.0, 5.0, 10.0]:
    # Override PD gains
    original_pd = physx_loader._ISAAC_PD.copy()
    scaled_pd = {k: (v[0] * mult, v[1] * np.sqrt(mult)) for k, v in original_pd.items()}
    physx_loader._ISAAC_PD = scaled_pd

    px = physx_core
    px.init_foundation()
    art = physx_loader.load_g1(px, "/gear_sonic_deploy/g1/g1_29dof_v17.xml", pos_iters=8, vel_iters=1)
    scene = px.create_scene(gravity=np.array([0., 0., -9.81], dtype=np.float32))
    mat = scene.create_material(0.6, 0.5, 0.0)
    scene.add_ground_plane(mat, np.array([0., 0., 1.], dtype=np.float32))
    scene.add_articulation(art)

    native_dt = 0.001961
    decimation = 17
    nu = 29

    start_idx = 200
    ref_q0 = dof_ref[start_idx]
    # Convert root_rot: [x,y,z,w] → [w,x,y,z]
    rr0 = root_rot[start_idx]
    root_quat0 = np.array([rr0[3], rr0[0], rr0[1], rr0[2]], dtype=np.float32)
    root_pos0 = root_trans[start_idx].astype(np.float32)

    art.set_root_world_pose(root_pos0, root_quat0)
    art.set_joint_positions(ref_q0.astype(np.float32))
    art.set_joint_velocities(np.zeros(nu, dtype=np.float32))
    art.set_joint_drive_targets(ref_q0.astype(np.float32))

    # Settle
    for _ in range(100):
        scene.simulate(native_dt)
        scene.fetch_results()

    # Track 80 frames with root pinned to reference
    n_frames = 80
    errors = []
    for i in range(n_frames):
        ref_idx = start_idx + i + 1
        ref_target = dof_ref[ref_idx].astype(np.float32)

        # Pin root to reference
        rr = root_rot[ref_idx]
        root_quat = np.array([rr[3], rr[0], rr[1], rr[2]], dtype=np.float32)
        root_pos = root_trans[ref_idx].astype(np.float32)
        art.set_root_world_pose(root_pos, root_quat)
        art.set_joint_drive_targets(ref_target)

        for _ in range(decimation):
            scene.simulate(native_dt)
            scene.fetch_results()

        q_actual = art.get_joint_positions().astype(np.float64)
        errors.append(q_actual - ref_target.astype(np.float64))

    errors = np.array(errors)
    alpha = float(np.sqrt(np.mean(errors[-40:] ** 2)))

    # Per-DOF breakdown
    leg_rms = np.sqrt(np.mean(errors[-40:, :12] ** 2))
    arm_rms = np.sqrt(np.mean(errors[-40:, 15:] ** 2))

    print(f"  mult={mult:4.1f}x: alpha={alpha:.6f}  leg_rms={leg_rms:.6f}  arm_rms={arm_rms:.6f}")

    # Show worst joints
    per_dof = [np.sqrt(np.mean(errors[-40:, j] ** 2)) for j in range(nu)]
    worst = sorted(range(nu), key=lambda j: -per_dof[j])[:5]
    print(f"    Worst DOFs: {', '.join(f'[{j}]:rms={per_dof[j]:.4f}' for j in worst)}")

    physx_loader._ISAAC_PD = original_pd
    px.release_foundation()

print()
print("*** DONE ***")
