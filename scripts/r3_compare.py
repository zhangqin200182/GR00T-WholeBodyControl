#!/usr/bin/env python3
"""R3 comparison: our replica vs their recordings, per test.
Steps: rise90/overshoot/settle of the perturbed joint vs q0.
Sines: amplitude ratio/phase at drive freq (pre-truncation window).
Truncation: their truncated flag/time vs our root-pitch divergence time.
"""
import argparse
import glob
import os
import numpy as np

ISAAC_REORDER = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                          4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,
                          20, 27, 21, 28], dtype=np.int64)
def i2m(q):
    qm = np.empty(29)
    qm[ISAAC_REORDER] = q
    return qm

JOINT_IDX = {"left_hip_pitch": 0, "left_hip_roll": 1, "left_knee": 3}


def step_metrics(t, q, q0, step):
    e = (q - q0) * np.sign(step)
    final = e[-1]
    t90 = t[np.argmax(e >= 0.9 * final)] * 1000 if (e >= 0.9 * final).any() else np.nan
    peak = e.max()
    over = (peak - final) / final * 100 if abs(final) > 1e-9 else np.nan
    settled = np.where(np.abs(e - final) <= 0.05 * abs(final))[0]
    ts = t[settled[0]] * 1000 if len(settled) else np.nan
    return t90, over, ts, final


def sine_metrics(t, q, q0, f):
    n = len(q) // 2
    e = (q - q0)[n:]
    t = t[n:]
    s = np.sin(2 * np.pi * f * t)
    c = np.cos(2 * np.pi * f * t)
    A = (e * s).mean() * 2
    B = (e * c).mean() * 2
    amp = np.hypot(A, B)
    phase = np.degrees(np.arctan2(-B, A))
    return amp / 0.05, phase


def root_pitch(q):
    w, x, y, z = q.T
    return np.arctan2(2 * (w * y - x * z), 1 - 2 * (y * y + z * z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isaac", default="/tmp/r3_isaac")
    ap.add_argument("--ours", default="/tmp/r3_ours")
    args = ap.parse_args()

    print(f"{'test':34s} {'rise90':>7s} {'over%':>6s} {'settle5':>7s} | "
          f"{'rise90':>7s} {'over%':>6s} {'settle5':>7s} | {'amp_r':>6s} {'ph(deg)':>7s}")
    for jname, j in JOINT_IDX.items():
        for a in (0.05, 0.1, 0.2):
            for sign, sgn in (("p", +1), ("n", -1)):
                tag = f"r3_{jname}_step_{a}{sign}.npz"
                th = np.load(os.path.join(args.isaac, tag))
                ou = np.load(os.path.join(args.ours, tag))
                q0 = i2m(th["q0"])[j] if "q0" in th.files else th["target"][0, j] - sgn * a
                tm_th = step_metrics(th["t"], i2m_stack(th, j), q0, sgn * a)
                tm_ou = step_metrics(ou["t"], ou["qpos"][:, j], q0, sgn * a)
                print(f"{tag:34s} {tm_th[0]:7.1f} {tm_th[1]:6.1f} {tm_th[2]:7.1f} | "
                      f"{tm_ou[0]:7.1f} {tm_ou[1]:6.1f} {tm_ou[2]:7.1f}")
        for f in (0.5, 2.0, 5.0):
            tag = f"r3_{jname}_sine_{f}.npz"
            th = np.load(os.path.join(args.isaac, tag))
            ou = np.load(os.path.join(args.ours, tag))
            q0 = i2m(th["q0"])[j] if "q0" in th.files else th["target"][0, j]
            n = min(len(th["t"]), len(ou["t"]))
            amp_th, ph_th = sine_metrics(th["t"][:n], i2m_stack(th, j)[:n], q0, f)
            amp_ou, ph_ou = sine_metrics(ou["t"][:n], ou["qpos"][:n, j], q0, f)
            trunc_th = th["t"][-1] if ("truncated" in th.files and th["truncated"]) else np.nan
            pitch_th = root_pitch(i2m_quat(th))[:n]
            pitch_ou = root_pitch(ou["root_quat"])[:n]
            d_th = pitch_th[-1] - pitch_th[0]
            d_ou = pitch_ou[-1] - pitch_ou[0]
            print(f"{tag:34s} {'':7s} {'':6s} {'':7s} | {'':7s} {'':6s} {'':7s} | "
                  f"{amp_th:6.2f}/{amp_ou:6.2f} {ph_th:6.1f}/{ph_ou:6.1f} "
                  f"d_pitch {d_th:.2f}/{d_ou:.2f}")


def i2m_stack(npz, j):
    qs = np.stack([i2m(q) for q in npz["qpos"]])
    return qs[:, j]


def i2m_quat(npz):
    # their root_quat is wxyz already
    return npz["root_quat"]


if __name__ == "__main__":
    main()
