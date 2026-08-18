#!/usr/bin/env python3
"""③b comparison: our P1 replica responses (configs A/B/C) vs Isaac P1.

Metrics: 90% rise time, overshoot, settle time (±5%), 5Hz amplitude ratio
and phase lag. Isaac QA reference for the 0.1 rad step: rise ~40 ms,
overshoot ~10.5%; 5Hz amplitude ratio 0.909.
"""
import numpy as np, glob, os, sys

ISAAC = "/tmp/isaac_baseline/p1"
OURS = "/tmp/p1_ours"


def step_metrics(t, q, q0):
    """t, q absolute; returns (rise90_ms, overshoot_pct, settle5_ms, ss_err)."""
    e = q - q0
    a = e[-1] if abs(e[-1]) > 1e-6 else (e.max() if abs(e.max()) > abs(e.min()) else 1.0)
    sign = 1.0 if a > 0 else -1.0
    en = e * sign  # normalized positive direction
    final = en[-1]
    t90 = t[np.argmax(en >= 0.9 * final)] * 1000 if (en >= 0.9 * final).any() else np.nan
    peak = en.max()
    overshoot = (peak - final) / final * 100 if final > 1e-9 else np.nan
    settled = np.where(np.abs(en - final) <= 0.05 * abs(final))[0]
    t_settle = t[settled[0]] * 1000 if len(settled) and settled[0] > 0 else np.nan
    return t90, overshoot, t_settle, e[-1]


def sine_metrics(t, q, q0):
    """Returns (amplitude_ratio, phase_lag_deg) vs target 0.05 rad sine."""
    e = q - q0
    # steady-state window: last half
    n = len(e) // 2
    e = e[n:]
    tt = t[n:] - t[n]
    f = 0.5  # placeholder; caller passes freq
    return None  # replaced below


def sine_metrics_f(t, q, q0, f):
    n = len(q) // 2
    e = (q - q0)[n:]
    t = t[n:]
    # fit amplitude and phase by projecting onto sin/cos basis
    s = np.sin(2 * np.pi * f * t)
    c = np.cos(2 * np.pi * f * t)
    A = (e * s).mean() * 2
    B = (e * c).mean() * 2
    amp = np.hypot(A, B)
    phase = np.degrees(np.arctan2(-B, A))  # lag positive when B<0 for e = amp*sin(wt+phi)
    return amp / 0.05, phase


def main():
    joints = ["left_hip_pitch", "left_knee", "left_ankle_pitch"]
    print(f"{'group':38s} {'src':6s} {'rise90ms':>9s} {'over%':>7s} {'settle5ms':>10s} {'ss_err':>8s}")
    for j in joints:
        for a in (0.05, 0.1, 0.2):
            for sign in ("p", "n"):
                name = f"p1_{j}_step_{a}{sign}.npz"
                rows = []
                try:
                    d = np.load(os.path.join(ISAAC, name))
                    m = step_metrics(d["t"], d["qpos"], float(d["q0"]))
                    rows.append(("isaac",) + m)
                except FileNotFoundError:
                    pass
                for cfg in ("A", "B", "C"):
                    try:
                        d = np.load(os.path.join(OURS, f"p1_ours_{cfg}", name))
                        m = step_metrics(d["t"], d["qpos"], d["qpos"][0] if False else
                                         (d["target"][0] - (float(a) if sign == "p" else -float(a))))
                        rows.append((cfg,) + m)
                    except FileNotFoundError:
                        pass
                for r in rows:
                    print(f"{name:38s} {r[0]:6s} {r[1]:9.1f} {r[2]:7.1f} {r[3]:10.1f} {r[4]:8.4f}")
        print()
    print("=== 5Hz sine: amplitude ratio / phase lag (deg) ===")
    for j in joints:
        name = f"p1_{j}_sine_5.0.npz"
        for src, path in [("isaac", ISAAC)] + [(c, os.path.join(OURS, f"p1_ours_{c}")) for c in ("A", "B", "C")]:
            try:
                d = np.load(os.path.join(path, name))
                amp, ph = sine_metrics_f(d["t"], d["qpos"],
                                         float(d["q0"]) if "q0" in d.files
                                         else d["target"][0], 5.0)
                print(f"{name:38s} {src:6s} amp_ratio={amp:.3f} phase={ph:6.1f}")
            except FileNotFoundError:
                pass
        print()


if __name__ == "__main__":
    main()
