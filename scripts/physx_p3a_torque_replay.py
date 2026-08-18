#!/usr/bin/env python3
"""③a torque replay: inject Isaac's recorded P1 applied_torque into our
plant (drives zeroed, root fixed, other joints locked, tested joint free)
and compare the early-phase Delta-q against Isaac's trajectory.

The first-100ms Delta-q ratio (torque-dominated phase) is the effective
inertia ratio I_isaac/I_ours — the plant fine-dynamics fingerprint.

Usage (NPU): python3 scripts/physx_p3a_torque_replay.py --isaac-p1 <dir> --out <dir>
"""
import argparse, glob, os, sys
import numpy as np

_physx_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "gear_sonic", "envs", "physx")
sys.path.insert(0, os.path.join(_physx_dir, "build"))
sys.path.insert(0, _physx_dir)
import physx_core  # noqa: E402
from physx_loader import load_g1  # noqa: E402

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
Q0 = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float64)
JOINT_IDX = {"left_hip_pitch": 0, "left_knee": 4, "left_ankle_pitch": 5}
# PhysX cache arrays (jointForce/jointVelocity) follow the INTERNAL DOF
# order = the articulation body-tree order, which equals the isaaclab
# convention (measured 08-19 via cache eVELOCITY probe).  Injection must
# map our XML joint index -> internal slot.
ISAAC_REORDER = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                          4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,
                          20, 27, 21, 28], dtype=np.int64)


def xml_to_internal(j):
    return int(np.where(ISAAC_REORDER == j)[0][0])


# P1-fitted torque-domain gains (Isaac runtime, 08-18). The recorded
# applied_torque field does NOT carry the gravity-supporting component, so
# the injected torque is reconstructed from the fitted drive model instead:
# tau = kp*(target-q) - kd*qdot, using THEIR recorded target/qpos/qvel.
# hip kd not yet dumped by the colleague — CFG fallback (damping is a
# second-order term over the 100ms torque-dominated window).
FIT_GAINS = {"left_hip_pitch": (143.3, 6.3),
             "left_knee": (445.0, 24.7),
             "left_ankle_pitch": (765.0, 44.5)}
DT = 0.005
WINDOW = 20  # substeps = 100 ms torque-dominated phase


def replay_group(art, scene, j, q0_their, tau_their, root_z):
    """Inject tau_their per substep; return our q trajectory (absolute)."""
    q = Q0.copy()
    q[j] = q0_their  # their initial pose for the tested joint
    art.set_joint_positions(q.astype(np.float32))
    art.set_joint_velocities(np.zeros(29, dtype=np.float32))
    art.zero_joint_forces()
    # OUR geometry: root high enough that the crouch pose keeps the feet
    # clearly off the ground (their root_fixed_z is THEIR geometry's value;
    # contact at 1.05 created a ground-support equilibrium at the knee)
    root0_pos = np.array([0.0, 0.0, 1.3], dtype=np.float32)
    # Settle phase: the first simulate after the big pose teleport fires a
    # violent updateKinematic transient (-37 rad/s at the knee).  Hold the
    # tested joint kinematically at q0_their (zero velocity) for 20 substeps
    # to burn it off before the replay starts.
    for _ in range(20):
        art.set_root_world_pose(root0_pos, np.array([1, 0, 0, 0], dtype=np.float32))
        art.set_root_world_velocity(np.zeros(3, dtype=np.float32),
                                    np.zeros(3, dtype=np.float32))
        qfull = Q0.copy(); qfull[j] = q0_their
        art.set_joint_positions(qfull.astype(np.float32))
        art.set_joint_velocities(np.zeros(29, dtype=np.float32))
        art.wake_up()
        scene.simulate(DT)
        scene.fetch_results()
    rec = []
    for k in range(len(tau_their)):
        art.set_root_world_pose(root0_pos, np.array([1, 0, 0, 0], dtype=np.float32))
        art.set_root_world_velocity(np.zeros(3, dtype=np.float32),
                                    np.zeros(3, dtype=np.float32))
        qcur = art.get_joint_positions()
        vcur = art.get_joint_velocities()
        qfull = Q0.copy(); qfull[j] = qcur[j]
        vfull = np.zeros(29); vfull[j] = vcur[j]
        art.set_joint_positions(qfull.astype(np.float32))
        art.set_joint_velocities(vfull.astype(np.float32))
        tau = np.zeros(29, dtype=np.float32)
        tau[xml_to_internal(j)] = float(tau_their[k])
        art.set_joint_forces(tau)  # persistent per-frame (semantics A verified)
        art.wake_up()
        scene.simulate(DT)
        scene.fetch_results()
        rec.append(float(art.get_joint_positions()[j]))
    return np.array(rec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-p1", default="/tmp/isaac_baseline/p1")
    parser.add_argument("--out", default="/tmp/p3a_ours")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    physx_core.init_foundation()
    art = load_g1(physx_core, XML, pos_iters=8, vel_iters=1,
                  drive_type="FORCE")
    scene = physx_core.create_scene(gravity=np.array([0, 0, -9.81], dtype=np.float32))
    mat = scene.create_material(0.6, 0.5, 0.0)
    scene.add_ground_plane(mat, np.array([0, 0, 1], dtype=np.float32))
    scene.add_articulation(art)
    # drives zeroed — pure torque-injection mode
    for i in range(29):
        art.set_joint_drive_params(i, 0.0, 0.0, 100.0, "FORCE")

    for jname, j in JOINT_IDX.items():
        kp, kd = FIT_GAINS[jname]
        for f in sorted(glob.glob(os.path.join(args.isaac_p1, f"p1_{jname}_step_*.npz"))):
            d = np.load(f)
            q0 = float(d["q0"])
            # reconstructed drive torque from THEIR fitted model + THEIR
            # recorded trajectory (includes the gravity-support component)
            tau = kp * (d["target"] - d["qpos"]) - kd * d["qvel"]
            ours_q = replay_group(art, scene, j, q0, tau, 1.05)
            # early-phase delta-q ratio: ours / isaac over the first WINDOW
            dq_o = ours_q[WINDOW] - ours_q[0]
            dq_i = d["qpos"][WINDOW] - d["qpos"][0]
            ratio = dq_o / dq_i if abs(dq_i) > 1e-9 else np.nan
            tag = os.path.basename(f)
            np.savez(os.path.join(args.out, tag.replace(".npz", "_ours.npz")),
                     q=ours_q, tau=tau[:len(ours_q)])
            print(f"{tag}: dq_ours={dq_o:+.5f} dq_isaac={dq_i:+.5f} "
                  f"ratio={ratio:.3f}", flush=True)
    print("P3A REPLAY COMPLETE")


if __name__ == "__main__":
    main()
