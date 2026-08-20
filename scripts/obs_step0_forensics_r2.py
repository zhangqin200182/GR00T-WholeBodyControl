#!/usr/bin/env python3
"""Round-2 obs_step0 forensic — per-episode B3 files vs recorded action_raw.

B1 fix guarantees consumed==recorded clip; the ah-frame9 fingerprint locates
the capture moment in the recorded action_raw sequences, then rebuilds our
obs construction from the same 10-frame state window and diffs per block.

Quat convention: round-2 npz ref_root_quat is wxyz (verified 2026-08-20 —
the earlier xyzw claim was a misreading).  Both conventions are tried and
the consistent one is reported per row (defensive; functional either way).

Usage: python3 scripts/obs_step0_forensics_r2.py
"""
import numpy as np
import os
import glob
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "gear_sonic", "envs"))
from mujoco_math import quat_mul, quat_inv, quat_apply  # noqa: E402

ISAAC_REORDER = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                          4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,
                          20, 27, 21, 28], dtype=np.int64)
ACT_OFFSET = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
    0.0, 0.0, 0.0,                           # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
], dtype=np.float64)
HIST = 10
BLOCKS = {"avh": (0, 3), "jph": (3, 32), "jvh": (32, 61), "ah": (61, 90),
          "gdh": (90, 93)}


def isaac_to_mujoco(q):
    qm = np.empty(29)
    qm[ISAAC_REORDER] = q
    return qm


def build_our_obs(root_quats_wxyz, qpos, qvel, actions, dt=0.02):
    avh = np.zeros((HIST, 3)); jph = np.zeros((HIST, 29))
    jvh = np.zeros((HIST, 29)); ah = np.zeros((HIST, 29)); gdh = np.zeros((HIST, 3))
    for i in range(HIST):
        rq = root_quats_wxyz[i]
        if i > 0:
            qd = quat_mul(quat_inv(root_quats_wxyz[i - 1]), rq)
            w = 2.0 * qd[1:] / dt
            w = np.sign(qd[0] if qd[0] != 0 else 1) * w
            avh[i] = w
        gdh[i] = quat_apply(quat_inv(rq), np.array([0, 0, -1]))
        jph[i] = isaac_to_mujoco(qpos[i]) - ACT_OFFSET
        jvh[i] = isaac_to_mujoco(qvel[i])
        ah[i] = actions[i]
    return np.concatenate([avh.flatten(), jph.flatten(), jvh.flatten(),
                           ah.flatten(), gdh.flatten()]).astype(np.float64)


DATA = "/tmp/isaac_r2/isaac_baseline_round2_20260819/release"


def fingerprint(ah9, clips):
    """Best (err, file, k) for ah frame 9 across all recorded action streams."""
    best = None
    for f in clips:
        ar = np.load(f)["action_raw"]
        for k in range(9, len(ar)):
            e = np.abs(ar[k] - ah9).max()
            if best is None or e < best[0]:
                best = (e, f, k)
    return best


def block_report(our, obs):
    lines = []
    for bname, (s, e) in BLOCKS.items():
        i_s, i_e = s * HIST, e * HIST
        diff = np.abs(our[i_s:i_e] - obs[i_s:i_e])
        dim = e - s
        fm = [np.abs(our[i_s + f * dim:i_s + (f + 1) * dim]
                     - obs[i_s + f * dim:i_s + (f + 1) * dim]).max()
              for f in range(HIST)]
        lines.append(f"    {bname:5s} max={diff.max():8.4f} per-frame: "
                     f"{[round(x, 3) for x in fm]}")
    return lines


def main():
    clips = sorted(glob.glob(os.path.join(DATA, "release_*.npz")))
    obsfiles = sorted(glob.glob(os.path.join(DATA, "obs_step0_release_*_??.npy")))
    if not obsfiles:
        # merged file fallback
        obsfiles = sorted(glob.glob(os.path.join(DATA, "obs_step0_release*.npy")))
    print(f"clips: {len(clips)}, per-episode obs files: {len(obsfiles)}")

    for of in obsfiles:
        base = os.path.basename(of)
        clip_hint = base[len("obs_step0_release_"):-4]
        obs = np.load(of).astype(np.float64).flatten()
        ah9 = obs[61 * HIST:90 * HIST].reshape(HIST, 29)[9]
        e, f, k = fingerprint(ah9, clips)
        # fallback: maybe the ah block is in mujoco/XML order
        if e > 0.05:
            ah9m = np.empty(29)
            ah9m[ISAAC_REORDER] = ah9
            e2, f2, k2 = fingerprint(ah9m, clips)
            if e2 < e:
                e, f, k, note = e2, f2, k2, "(ah block is mujoco-order)"
            else:
                note = "(no good fingerprint)"
        else:
            note = ""
        print(f"\n{base}  hint={clip_hint}")
        print(f"  fingerprint: {os.path.basename(f)} k={k} err={e:.4f} {note}")
        if e > 0.05:
            print("  WARNING: capture moment not found — skip")
            continue
        d = np.load(f)
        sl = slice(k - 9, k + 1)
        rq = d["root_quat"][sl]
        best_conv = None
        for conv, rq_w in [("xyzw->wxyz (r2)", np.roll(rq, 1, axis=-1)),
                           ("wxyz as-is (r1)", rq)]:
            our = build_our_obs(rq_w, d["qpos"][sl], d["qvel"][sl],
                                d["action_raw"][sl])
            # state blocks only (ah trivially matches by construction)
            gdh = np.abs(our[90 * HIST:93 * HIST] - obs[90 * HIST:93 * HIST]).max()
            avh = np.abs(our[0:3 * HIST] - obs[0:3 * HIST]).max()
            print(f"  [{conv}] gdh max={gdh:.4f} avh max={avh:.4f}")
            if best_conv is None or gdh + avh < best_conv[0]:
                best_conv = (gdh + avh, conv, our)
        _, conv, our = best_conv
        print(f"  chosen: {conv}")
        for line in block_report(our, obs):
            print(line)


if __name__ == "__main__":
    main()
