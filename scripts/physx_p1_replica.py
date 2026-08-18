#!/usr/bin/env python3
"""Our-side P1 replica: single-joint drive response with root fixed and
other joints locked — mirrors the Isaac-side isaac_p1_drive.py protocol
(3 joints x step +-0.05/0.1/0.2 rad x sine 0.5/2/5 Hz), 200 Hz sampling.

Used for ③b drive-semantics comparison: same locked state, same target
trajectory, different engine -> compare torque responses.

Env vars: SONIC_PHYSX_DRIVE_ANALYTICAL / _MULT / _KD_MULT / _PD_MEASURED
          select the drive config under test (standard vs measured gains).
Run on NPU: python3 scripts/physx_p1_replica.py --out /tmp/p1_ours
"""
import argparse, os, sys
import numpy as np

_physx_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "gear_sonic", "envs", "physx")
sys.path.insert(0, os.path.join(_physx_dir, "build"))  # compiled physx_core
sys.path.insert(0, _physx_dir)                          # physx_loader source
import physx_core  # noqa: E402
from physx_loader import load_g1  # noqa: E402

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
# deploy default_angles (mujoco order, = _ISAAC_ACT_OFFSET)
Q0 = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
    0.0, 0.0, 0.0,                           # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
], dtype=np.float64)
JOINT_IDX = {"left_hip_pitch": 0, "left_knee": 4, "left_ankle_pitch": 5}
DT = 0.005  # Isaac sim_dt


def run_group(art, scene, j, target_fn, t_end, root0_pos, root0_quat):
    """Drive joint j with target_fn(t) for t_end s; record per substep."""
    q = Q0.copy()
    t = 0.0
    rec = {"t": [], "qpos": [], "qvel": [], "target": [], "torque": []}
    mask_others = np.ones(29, dtype=bool)
    mask_others[j] = False
    art.set_joint_drive_targets(Q0.astype(np.float32))
    while t <= t_end + 1e-6:
        # root fixed: re-write pose + zero velocity (drift check like Isaac side)
        art.set_root_world_pose(root0_pos, np.array([1, 0, 0, 0], dtype=np.float32))
        art.set_root_world_velocity(np.zeros(3, dtype=np.float32),
                                    np.zeros(3, dtype=np.float32))
        # lock other joints (FULL 29-arrays; tested joint keeps its current
        # state, others re-written to default = kinematic lock)
        qcur = art.get_joint_positions()
        vcur = art.get_joint_velocities()
        qfull = Q0.copy(); qfull[j] = qcur[j]
        vfull = np.zeros(29); vfull[j] = vcur[j]
        art.set_joint_positions(qfull.astype(np.float32))
        art.set_joint_velocities(vfull.astype(np.float32))
        # drive target for the tested joint only
        targets = Q0.copy()
        targets[j] = target_fn(t)
        art.set_joint_drive_targets(targets.astype(np.float32))
        art.wake_up()  # re-pinned root + locked joints -> solver sleeps it
        scene.simulate(DT)
        scene.fetch_results()
        qpos = art.get_joint_positions()
        qvel = art.get_joint_velocities()
        rec["t"].append(t)
        rec["qpos"].append(qpos[j])
        rec["qvel"].append(qvel[j])
        rec["target"].append(targets[j])
        rec["torque"].append(art.get_joint_forces()[j])
        t += DT
        q = qpos.copy()
    return {k: np.array(v) for k, v in rec.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/p1_ours")
    parser.add_argument("--joints", nargs="*",
                        default=["left_hip_pitch", "left_knee", "left_ankle_pitch"])
    parser.add_argument("--drive-type", default="ACCELERATION",
                        choices=["ACCELERATION", "FORCE"])
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    physx_core.init_foundation()
    art = load_g1(physx_core, XML, pos_iters=8, vel_iters=1,
                  drive_type=args.drive_type)
    scene = physx_core.create_scene(gravity=np.array([0, 0, -9.81], dtype=np.float32))
    mat = scene.create_material(0.6, 0.5, 0.0)
    scene.add_ground_plane(mat, np.array([0, 0, 1], dtype=np.float32))
    scene.add_articulation(art)
    # stand above ground (like Isaac side: +0.05 m clearance, no contact)
    root0_pos = np.array([0.0, 0.0, 1.05], dtype=np.float32)
    art.set_root_world_pose(root0_pos, np.array([1, 0, 0, 0], dtype=np.float32))
    art.set_joint_positions(Q0.astype(np.float32))
    art.set_joint_velocities(np.zeros(29, dtype=np.float32))

    for jname in args.joints:
        j = JOINT_IDX[jname]
        for a in (0.05, 0.1, 0.2):
            for sign in (+1.0, -1.0):
                tag = f"p1_{jname}_step_{a}{'p' if sign > 0 else 'n'}"
                rec = run_group(art, scene, j, lambda t, a=a, s=sign: Q0[j] + s * a,
                                2.0, root0_pos, np.array([1, 0, 0, 0]))
                np.savez(os.path.join(args.out, f"{tag}.npz"), **rec)
                print(f"{tag}: peak_torque={np.abs(rec['torque']).max():.3f} "
                      f"ss_torque={rec['torque'][-1]:.3f} drift_q="
                      f"{rec['qpos'][-1]-rec['target'][-1]:.4f}", flush=True)
        for f in (0.5, 2.0, 5.0):
            tag = f"p1_{jname}_sine_{f}"
            rec = run_group(art, scene, j,
                            lambda t, f=f: Q0[j] + 0.05 * np.sin(2 * np.pi * f * t),
                            4.0, root0_pos, np.array([1, 0, 0, 0]))
            np.savez(os.path.join(args.out, f"{tag}.npz"), **rec)
            print(f"{tag}: peak_torque={np.abs(rec['torque']).max():.3f}", flush=True)
    print("P1 REPLICA COMPLETE")


if __name__ == "__main__":
    main()
