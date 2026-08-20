#!/usr/bin/env python3
"""D1b P1 replica: Isaac's EXACT drive law applied manually.

Isaac runtime drive (bit-exact, proven from their applied_torque channel):
    tau[k] = kp * (target - q[k-1]) - kd * v[k-1]
i.e. a one-sample-lagged explicit PD command, held constant over substep k
(injected through the eFORCE cache — NOT the implicit articulation drive).

Protocol differences vs physx_p1_replica.py (Isaac-side semantics):
  - all 29 joints DRIVEN to q0 with their CFG gains (no teleport locks;
    their locked_drift_max ~ 26 mrad shows soft drive holds, not hard locks)
  - root set once at Isaac's root_fixed_z = 0.8196 m, never re-written
    (their root_drift_max = 2.5 mm over the run — free-floating in zero g)
  - articulation drive gains zeroed; torques come only from our injection
  - recorded 'torque' = the drive COMMAND (matches their applied_torque
    act-then-record convention); 'torque_meas' = get_joint_torques

Run: SONIC_PHYSX_ARMATURE=0 python3 scripts/physx_p1_replica_manual.py \
    --out /tmp/p1_d1b
"""
import argparse, os, sys
import numpy as np

_physx_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "gear_sonic", "envs", "physx")
sys.path.insert(0, os.path.join(_physx_dir, "build"))
sys.path.insert(0, _physx_dir)
import physx_core  # noqa: E402
from physx_loader import load_g1, _isaac_pd_gains  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
# deploy default_angles in g1_29dof_v17.xml order (hip_pitch first)
Q0 = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
    0.0, 0.0, 0.0,                           # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
], dtype=np.float64)
JOINT_IDX = {"left_hip_pitch": 0, "left_knee": 3, "left_ankle_pitch": 4}
# Isaac-side P1 q0 (from their npz): hip -0.30229777, knee +0.6686436,
# ankle -0.3639972
ISAAC_Q0_OVERRIDE = {"left_hip_pitch": -0.30229777, "left_knee": 0.6686436,
                     "left_ankle_pitch": -0.3639972}
# isaaclab (BFS) order -> XML order; eFORCE cache is written in internal
# (BFS) order, so f_internal = tau_xml[ISAAC_REORDER]
ISAAC_REORDER = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                          4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,
                          20, 27, 21, 28])
# NOTE: Isaac's root_fixed_z = 0.8196 is THEIR robot geometry; our v17
# stands at root 1.05 (feet clear).  Their P1 M_eff (explicit law: ankle
# 0.0111 vs armature-on 0.0100 within 10%, hip 0.065 vs arm-on free model
# 0.078 within 17%) shows Isaac's mass matrix INCLUDES the armature — the
# earlier "no armature" read was an implicit/explicit interpretation
# artifact.  With armature on, the lagged-explicit loop poles are all
# |z|<1 (stable); without it the free ankle pole is |z|=3.2 (divergent).
# Run WITHOUT SONIC_PHYSX_ARMATURE=0.
ROOT_Z = 1.05
DT = 0.005


def xml_joint_names():
    tree = ET.parse(XML)
    return [j.get("name") for j in tree.getroot().iter("joint")
            if j.get("name") != "floating_base_joint"]


def run_group(art, scene, j, target_fn, t_end, q0, kp, kd):
    t = 0.0
    rec = {"t": [], "qpos": [], "qvel": [], "target": [], "torque": [],
           "torque_meas": []}
    art.set_root_world_pose(np.array([0.0, 0.0, ROOT_Z], dtype=np.float32),
                            np.array([1, 0, 0, 0], dtype=np.float32))
    art.set_root_world_velocity(np.zeros(3, dtype=np.float32),
                                np.zeros(3, dtype=np.float32))
    art.set_joint_positions(q0.astype(np.float32))
    art.set_joint_velocities(np.zeros(29, dtype=np.float32))
    q_prev = q0.copy()
    v_prev = np.zeros(29)
    while t <= t_end + 1e-6:
        targets = q0.copy()
        targets[j] = target_fn(t)
        # Isaac one-sample-lag explicit PD command (XML order)
        tau = kp * (targets - q_prev) - kd * v_prev
        art.set_joint_forces(tau[ISAAC_REORDER].astype(np.float32))
        art.wake_up()
        scene.simulate(DT)
        scene.fetch_results()
        qpos = art.get_joint_positions()
        qvel = art.get_joint_velocities()
        rec["t"].append(t)
        rec["qpos"].append(qpos[j])
        rec["qvel"].append(qvel[j])
        rec["target"].append(targets[j])
        rec["torque"].append(tau[j])
        rec["torque_meas"].append(art.get_joint_torques()[j])
        q_prev = qpos
        v_prev = qvel
        t += DT
    return {k: np.array(v) for k, v in rec.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/p1_d1b")
    parser.add_argument("--joints", nargs="*",
                        default=["left_hip_pitch", "left_knee", "left_ankle_pitch"])
    parser.add_argument("--vel-iters", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = xml_joint_names()
    assert len(names) == 29, f"expected 29 joints, got {len(names)}"
    kp = np.array([_isaac_pd_gains(n)[0] for n in names])
    kd = np.array([_isaac_pd_gains(n)[1] for n in names])

    q0 = Q0.copy()
    for jname, v in ISAAC_Q0_OVERRIDE.items():
        q0[JOINT_IDX[jname]] = v

    physx_core.init_foundation()
    art = load_g1(physx_core, XML, pos_iters=8, vel_iters=args.vel_iters,
                  drive_type="FORCE")
    # zero the articulation drives — torque comes only from our eFORCE
    # injection (explicit application, no implicit TGS drive integration)
    for i in range(29):
        art.set_joint_drive_params(i, 0.0, 0.0, 1e9, "FORCE")
    scene = physx_core.create_scene(gravity=np.zeros(3, dtype=np.float32))
    mat = scene.create_material(0.6, 0.5, 0.0)
    scene.add_ground_plane(mat, np.array([0, 0, 1], dtype=np.float32))
    scene.add_articulation(art)
    root0 = np.array([0.0, 0.0, ROOT_Z], dtype=np.float32)
    art.set_root_world_pose(root0, np.array([1, 0, 0, 0], dtype=np.float32))
    art.set_root_world_velocity(np.zeros(3, dtype=np.float32),
                                np.zeros(3, dtype=np.float32))
    art.set_joint_positions(q0.astype(np.float32))
    art.set_joint_velocities(np.zeros(29, dtype=np.float32))

    for jname in args.joints:
        j = JOINT_IDX[jname]
        for a in (0.05, 0.1, 0.2):
            for sign in (+1.0, -1.0):
                tag = f"p1_{jname}_step_{a}{'p' if sign > 0 else 'n'}"
                rec = run_group(art, scene, j, lambda t, a=a, s=sign: q0[j] + s * a,
                                2.0, q0, kp, kd)
                rec["q0"] = q0[j]
                np.savez(os.path.join(args.out, f"{tag}.npz"), **rec)
                print(f"{tag}: tau0={rec['torque'][0]:.3f} "
                      f"peak={np.abs(rec['torque']).max():.3f} "
                      f"dq1={rec['qpos'][0]-q0[j]:+.5f} v1={rec['qvel'][0]:+.5f}",
                      flush=True)
        for f in (0.5, 2.0, 5.0):
            tag = f"p1_{jname}_sine_{f}"
            rec = run_group(art, scene, j,
                            lambda t, f=f: q0[j] + 0.05 * np.sin(2 * np.pi * f * t),
                            4.0, q0, kp, kd)
            rec["q0"] = q0[j]
            np.savez(os.path.join(args.out, f"{tag}.npz"), **rec)
            print(f"{tag}: peak_tau={np.abs(rec['torque']).max():.3f}", flush=True)
    print("P1 MANUAL REPLICA COMPLETE")


if __name__ == "__main__":
    main()
