#!/usr/bin/env python3
"""Forensic cross-engine obs comparison.

Locates the capture moment of Isaac's obs_step0 rows by matching the action
history fingerprint against the recorded action_raw sequence, then rebuilds
OUR obs construction from the same 10-frame state window and diffs per block.

Usage: python3 scripts/obs_step0_forensics.py
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "gear_sonic", "envs"))
from mujoco_math import quat_mul, quat_inv, quat_apply  # noqa: E402

ISAAC_REORDER = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                          4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,
                          20, 27, 21, 28], dtype=np.int64)
# deploy default_angles (mujoco order, from physx_env _ISAAC_ACT_OFFSET)
ACT_OFFSET = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
    0.0, 0.0, 0.0,                           # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
], dtype=np.float64)
HIST = 10
BLOCKS = {"avh": (0, 3), "jph": (3, 32), "jvh": (32, 61), "ah": (61, 90), "gdh": (90, 93)}


def isaac_to_mujoco(q):
    qm = np.empty(29)
    qm[ISAAC_REORDER] = q
    return qm


def build_our_obs(root_quats, qpos, qvel, actions, dt=0.02):
    """Rebuild OUR actor obs (block-major [avh,jph,jvh,ah,gdh] x HIST,
    newest last) from a 10-frame state window (isaac-order inputs)."""
    avh = np.zeros((HIST, 3)); jph = np.zeros((HIST, 29))
    jvh = np.zeros((HIST, 29)); ah = np.zeros((HIST, 29)); gdh = np.zeros((HIST, 3))
    for i in range(HIST):
        rq = root_quats[i]
        # angular velocity: finite difference in body frame (world delta rotated)
        if i > 0:
            # quat difference: q_delta = inv(q_prev) * q_cur, omega = 2*vec(q_delta)/dt
            qd = quat_mul(quat_inv(root_quats[i - 1]), rq)
            w = 2.0 * qd[1:] / dt
            w = np.sign(qd[0] if qd[0] != 0 else 1) * w
            avh[i] = w
        gdh[i] = quat_apply(quat_inv(rq), np.array([0, 0, -1]))
        jph[i] = isaac_to_mujoco(qpos[i]) - ACT_OFFSET
        jvh[i] = isaac_to_mujoco(qvel[i])
        ah[i] = actions[i]
    return np.concatenate([avh.flatten(), jph.flatten(), jvh.flatten(),
                           ah.flatten(), gdh.flatten()]).astype(np.float64)


def main():
    isaac_obs = np.load("/tmp/isaac_baseline/release/obs_step0_release.npy")
    # The obs rows were captured from the policy's ACTUAL consumed motion
    # (collection bug: recorded labels != consumed clips) — search ALL 12
    # recorded action_raw sequences for the ah-frame9 fingerprint.
    import glob
    files = sorted(glob.glob("/tmp/isaac_baseline/release/release_*.npz"))
    clips = [np.load(f) for f in files]
    print(f"clip files: {len(clips)}, isaac obs rows: {len(isaac_obs)}")

    for r in range(len(isaac_obs)):
        obs = isaac_obs[r].astype(np.float64)
        ah9 = obs[61 * HIST:90 * HIST].reshape(HIST, 29)[9]
        best = None  # (err, clip_name, k)
        for d, f in zip(clips, files):
            n = len(d["qpos"])
            for k in range(9, n):
                e = np.abs(d["action_raw"][k] - ah9).max()
                if best is None or e < best[0]:
                    best = (e, os.path.basename(f), k)
        e, fname, k = best
        print(f"\nrow {r}: ah-frame9 best match: {fname} k={k} (max|err|={e:.4f})")
        if e > 0.05:
            print("  WARNING: poor match — capture may predate recording or use "
                  "different action convention")
            continue
        d = clips[files.index([f for f in files if fname in f][0])]
        sl = slice(k - 9, k + 1)
        our = build_our_obs(d["root_quat"][sl], d["qpos"][sl],
                            d["qvel"][sl], d["action_raw"][sl])
        print(f"{'block':6s} {'our max|diff| vs isaac':26s} note")
        for bname, (s, e) in BLOCKS.items():
            i_s, i_e = s * HIST, e * HIST
            diff = np.abs(our[i_s:i_e] - obs[i_s:i_e])
            dim = e - s
            fm = [np.abs(our[i_s + f * dim:i_s + (f + 1) * dim]
                         - obs[i_s + f * dim:i_s + (f + 1) * dim]).max()
                  for f in range(HIST)]
            print(f"{bname:6s} max={diff.max():8.4f}  per-frame: "
                  f"{[round(x, 3) for x in fm]}")


if __name__ == "__main__":
    main()
