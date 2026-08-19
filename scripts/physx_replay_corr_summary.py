#!/usr/bin/env python3
"""Corr summary for the release replay (physx_release_replay.py output).

Compares the replay qpos (XML order) against the Isaac recorded qpos
(isaaclab order), per joint, and prints the leg-mean and the key
localization joints (ankle/waist pitch).

Usage: python3 scripts/physx_replay_corr_summary.py \
    --isaac <isaac npz> --replay <replay npz> [--steps N]
"""
import argparse
import numpy as np

ISAAC_REORDER = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                          4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,
                          20, 27, 21, 28], dtype=np.int64)
XML_JOINTS = ["lh_pitch", "lh_roll", "lh_yaw", "lk", "la_pitch", "la_roll",
              "rh_pitch", "rh_roll", "rh_yaw", "rk", "ra_pitch", "ra_roll",
              "waist_yaw", "waist_roll", "waist_pitch", "l_sh_pitch",
              "l_sh_roll", "l_sh_yaw", "l_elb", "l_wr_roll", "l_wr_pitch",
              "l_wr_yaw", "r_sh_pitch", "r_sh_roll", "r_sh_yaw", "r_elb",
              "r_wr_roll", "r_wr_pitch", "r_wr_yaw"]


def isaac_to_mujoco(q):
    qm = np.empty(29)
    qm[ISAAC_REORDER] = q
    return qm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isaac", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()

    isaac = np.load(args.isaac)
    rep = np.load(args.replay)
    n = min(args.steps or len(rep["qpos"]), len(rep["qpos"]), len(isaac["qpos"]) - 1)
    ours = rep["qpos"][:n]                        # XML order
    theirs = np.stack([isaac_to_mujoco(q) for q in isaac["qpos"][1:n + 1]])

    print(f"{'joint':16s} {'corr':>7s} {'rmse':>8s}")
    corrs = {}
    for j in range(29):
        c = np.corrcoef(ours[:, j], theirs[:, j])[0, 1]
        r = np.sqrt(np.mean((ours[:, j] - theirs[:, j]) ** 2))
        corrs[XML_JOINTS[j]] = c
        print(f"{XML_JOINTS[j]:16s} {c:7.3f} {r:8.4f}")
    legs = [corrs[j] for j in XML_JOINTS[:12]]
    print(f"\nleg mean corr: {np.mean(legs):.3f}")
    print(f"key: la_pitch={corrs['la_pitch']:.3f} ra_pitch={corrs['ra_pitch']:.3f} "
          f"waist_pitch={corrs['waist_pitch']:.3f} lk={corrs['lk']:.3f}")


if __name__ == "__main__":
    main()
