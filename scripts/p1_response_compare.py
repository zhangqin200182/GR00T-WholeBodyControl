#!/usr/bin/env python3
"""③b comparison: our P1 replica responses vs Isaac P1.

Metrics: 90% rise time, overshoot, settle time (±5%), 5Hz amplitude ratio
and phase lag, plus step torque (tau0 = kp*step readback check).
Isaac QA reference for the 0.1 rad step: rise ~40 ms, overshoot ~10.5%;
5Hz amplitude ratio 0.909.

D1 usage:
  python3 scripts/p1_response_compare.py --ours /tmp/p1_d1 --cfgs ""
"""
import argparse
import numpy as np, os


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


def load(path, name):
    try:
        return np.load(os.path.join(path, name))
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac", default="/tmp/isaac_p1")
    parser.add_argument("--ours", default="/tmp/p1_ours",
                        help="Dir with our replicas; config subdirs p1_ours_<cfg> "
                             "unless --cfgs '' (flat dir)")
    parser.add_argument("--cfgs", default="A,B,C",
                        help="Comma-separated configs; empty string = flat dir")
    args = parser.parse_args()
    cfgs = [c for c in args.cfgs.split(",") if c] or [""]  # "" = flat dir sentinel

    joints = ["left_hip_pitch", "left_knee", "left_ankle_pitch"]
    print(f"{'group':38s} {'src':6s} {'rise90ms':>9s} {'over%':>7s} {'settle5ms':>10s} {'ss_err':>8s}")
    for j in joints:
        for a in (0.05, 0.1, 0.2):
            for sign in ("p", "n"):
                name = f"p1_{j}_step_{a}{sign}.npz"
                rows = []
                d = load(args.isaac, name)
                if d is not None:
                    m = step_metrics(d["t"], d["qpos"], float(d["q0"]))
                    rows.append(("isaac",) + m)
                for cfg in cfgs:
                    path = args.ours if not cfg else os.path.join(args.ours, f"p1_ours_{cfg}")
                    d = load(path, name)
                    if d is None:
                        continue
                    step = float(a) if sign == "p" else -float(a)
                    q0 = d["target"][0] - step
                    m = step_metrics(d["t"], d["qpos"], q0)
                    rows.append((cfg or "ours",) + m)
                for r in rows:
                    print(f"{name:38s} {r[0]:6s} {r[1]:9.1f} {r[2]:7.1f} {r[3]:10.1f} {r[4]:8.4f}")
        print()
    print("=== step torque readback: tau0 (Nm) / peak |tau| (Nm) ===")
    for j in joints:
        for a in (0.05, 0.1, 0.2):
            for sign in ("p", "n"):
                name = f"p1_{j}_step_{a}{sign}.npz"
                out = []
                d = load(args.isaac, name)
                if d is not None:
                    out.append(f"isaac tau0={d['applied_torque'][0]:8.4f} "
                               f"peak={np.abs(d['applied_torque']).max():8.4f}")
                for cfg in cfgs:
                    path = args.ours if not cfg else os.path.join(args.ours, f"p1_ours_{cfg}")
                    d = load(path, name)
                    if d is None:
                        continue
                    out.append(f"{cfg or 'ours':6s} tau0={d['torque'][0]:8.4f} "
                               f"peak={np.abs(d['torque']).max():8.4f}")
                if out:
                    print(f"{name:38s} " + " | ".join(out))
        print()
    print("=== 5Hz sine: amplitude ratio / phase lag (deg) ===")
    for j in joints:
        name = f"p1_{j}_sine_5.0.npz"
        d = load(args.isaac, name)
        if d is not None:
            amp, ph = sine_metrics_f(d["t"], d["qpos"], float(d["q0"]), 5.0)
            print(f"{name:38s} isaac amp_ratio={amp:.3f} phase={ph:6.1f}")
        for cfg in cfgs:
            path = args.ours if not cfg else os.path.join(args.ours, f"p1_ours_{cfg}")
            d = load(path, name)
            if d is None:
                continue
            q0 = float(d["q0"]) if "q0" in d.files else d["target"][0]
            amp, ph = sine_metrics_f(d["t"], d["qpos"], q0, 5.0)
            print(f"{name:38s} {cfg or 'ours':6s} amp_ratio={amp:.3f} phase={ph:6.1f}")
        print()


if __name__ == "__main__":
    main()
