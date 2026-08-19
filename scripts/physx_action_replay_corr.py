#!/usr/bin/env python3
"""Action-replay closed-loop correlation: feed Isaac's recorded joint_target
trajectory through our engine (same initial state, same targets, free run)
and compare the resulting qpos trajectories per joint.

This is the measuring stick for the plant closed-loop divergence
(colleague's ③b replay: leg corr 0.67-0.85, ankle_pitch 0.063).

Usage (NPU):
  SONIC_PHYSX_CONTACT_OFFSET=0.005 python3 scripts/physx_action_replay_corr.py \
      --isaac /tmp/isaac_baseline/release/release_walk_sideway_045_loop_003__A033_03.npz \
      --steps 400
"""
import argparse, os, sys
import numpy as np

_physx_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "gear_sonic", "envs", "physx")
sys.path.insert(0, os.path.join(_physx_dir, "build"))
sys.path.insert(0, _physx_dir)
sys.path.insert(0, os.path.join(os.path.dirname(_physx_dir)))
import physx_core  # noqa: E402
from physx_loader import load_g1  # noqa: E402
from mujoco_math import quat_mul, quat_inv  # noqa: E402

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
ISAAC_REORDER = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                          4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,
                          20, 27, 21, 28], dtype=np.int64)


def isaac_to_mujoco(q):
    qm = np.empty(29)
    qm[ISAAC_REORDER] = q
    return qm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--drive", default="FORCE", choices=["FORCE", "ACCELERATION"])
    args = parser.parse_args()

    d = np.load(args.isaac)
    n = min(args.steps, len(d["qpos"]) - 1)

    physx_core.init_foundation()
    art = load_g1(physx_core, XML, pos_iters=8, vel_iters=4,
                  drive_type=args.drive)
    scene = physx_core.create_scene(gravity=np.array([0, 0, -9.81], dtype=np.float32))
    mat = scene.create_material(0.6, 0.5, 0.0)
    scene.add_ground_plane(mat, np.array([0, 0, 1], dtype=np.float32))
    scene.add_articulation(art)

    # Per-step re-seeded replay: at each step, set our state to their
    # recorded state at k-1, apply their target[k], simulate ONE control
    # step, and compare the result with their recorded qpos[k].  This
    # isolates the LOCAL plant response difference (free-running from one
    # initial state amplifies any tiny offset chaotically over hundreds
    # of steps).
    ours = []
    for k in range(1, n):
        art.set_root_world_pose(d["root_pos"][k - 1].astype(np.float32),
                                d["root_quat"][k - 1].astype(np.float32))
        # root velocity: finite-difference from their trajectory (the npz
        # has no root-velocity field; zeroing it breaks the foot-contact
        # phase for a moving robot)
        lin = (d["root_pos"][k - 1] - d["root_pos"][k - 2]) / 0.02
        qd = quat_mul(quat_inv(d["root_quat"][k - 2]), d["root_quat"][k - 1])
        ang = 2.0 * qd[1:] / 0.02
        if qd[0] < 0:
            ang = -ang
        art.set_root_world_velocity(lin.astype(np.float32),
                                    ang.astype(np.float32))
        art.set_joint_positions(isaac_to_mujoco(d["qpos"][k - 1]).astype(np.float32))
        art.set_joint_velocities(isaac_to_mujoco(d["qvel"][k - 1]).astype(np.float32))
        tgt = isaac_to_mujoco(d["joint_target"][k]).astype(np.float32)
        art.set_joint_drive_targets(tgt)
        art.wake_up()
        for _ in range(10):
            scene.simulate(0.001961)
            scene.fetch_results()
        ours.append(isaac_to_mujoco(art.get_joint_positions()))
    ours = np.array(ours)

    theirs = np.stack([isaac_to_mujoco(q) for q in d["qpos"][1:n]])
    # both sides now in mujoco order
    print(f"{'joint':28s} {'corr':>7s} {'rmse':>8s}")
    for j in range(29):
        c = np.corrcoef(ours[:, j], theirs[:, j])[0, 1]
        r = np.sqrt(np.mean((ours[:, j] - theirs[:, j]) ** 2))
        print(f"{j:3d} {'joint_%d' % j:22s} {c:7.3f} {r:8.4f}")
    leg_c = np.mean([np.corrcoef(ours[:, j], theirs[:, j])[0, 1] for j in range(12)])
    print(f"leg mean corr: {leg_c:.3f}")
    np.savez("/tmp/action_replay_ours.npz", ours=ours, theirs=theirs)


if __name__ == "__main__":
    main()
