#!/usr/bin/env python3
"""Round-3 motion selection + conversion.

Selects ~36 diverse motions from motion_index.json by category quotas,
converts each via the OFFICIAL converter pipeline (validated bit-exact
against the official walk PKL), applies numeric quality gates, and writes
one PKL per motion into round3_pkls/ (same format the env loader globs).
"""
import json
import os
import shutil
import sys

import joblib
import numpy as np

sys.path.insert(0, "/Users/max/code/ai/embody/gwc-push/gear_sonic/data_process")
from convert_soma_csv_to_motion_lib import (  # noqa: E402
    convert_sequence, downsample_sequence, load_bones_csv,
)

BASE = "/Users/max/code/ai/embody/bones_seed"
OUT = os.path.join(BASE, "round3_pkls")
ORIG_DIR = ("/Users/max/code/ai/embody/GR00T-WholeBodyControl-feature-mujoco-training"
            "/sample_data/robot_filtered/210531")

# (label, regex over family base, count, burst?) — ordered by training priority.
# burst=True: short-action tier (jump/run/lunge ~3-8s), smaller size window,
# min 100 frames; else main tier ~8s+, min 240 frames.
QUOTAS = [
    ("walk_forward", r"^walk_forward_(normal|slow|fast)", 3, False),
    ("walk_variants", r"^walk_(backward|drunk|with)", 2, False),
    ("jog_forward", r"^jog_forward", 2, False),
    ("jog_inplace", r"^jog_in_place", 1, True),
    ("run", r"^run_(loop|like|start)", 2, True),
    ("turn", r"^(turn|walk_turn)", 2, False),
    ("lunge", r"^forward_lunge", 1, True),
    ("jump_light", r"^jump_and_land_light", 2, True),
    ("jump_heavy", r"^jump_and_land_heavy", 1, True),
    ("idle", r"^(idle_stand|neutral_stand|stoop)", 1, False),
    ("clap", r"^(clap|Neutral_clap)", 1, False),
    ("victory", r"^(victory|triumph)", 1, False),
    ("reach", r"^(reach|Neutral_reach)", 1, False),
    ("lift", r"^(lift|Neutral_lift)", 1, False),
    ("body_check", r"^body_check", 1, False),
    ("dance_basic", r"^dance_basic", 2, False),
    ("dance_retro", r"^dance_retro_(twist|swing|disco)", 1, True),
    ("dance_hiphop", r"^dance_hiphop_2_step", 1, True),
    ("crouch", r"^crouch", 1, False),
    ("squat", r"^squat_00", 1, True),
    ("kick", r"^Neutral_kick_trash", 1, True),
]

TARGET_SIZE = 1.5e6        # ~24s of motion
SIZE_RANGE = (0.7e6, 8e6)  # ~11s .. ~2.5min
BURST_RANGE = (0.35e6, 0.7e6)  # ~3.3s .. ~11s
MIRROR_FOR = (r"^walk_forward", r"^jog_forward", r"^run_")  # add _M twin for these


def pick_file(files, mirror=False, burst=False):
    rng = BURST_RANGE if burst else SIZE_RANGE
    cands = [f for f in files if f["mirror"] == mirror and rng[0] <= f["size"] <= rng[1]]
    if not cands:
        return None
    tgt = 0.5e6 if burst else TARGET_SIZE
    return min(cands, key=lambda f: abs(f["size"] - tgt))


def gate(entry, min_frames=240):
    """Return None if OK else failure reason."""
    dof, rt = entry["dof"], entry["root_trans_offset"]
    qn = np.linalg.norm(entry["root_rot"], axis=1)
    if not (0.98 < qn.min() and qn.max() < 1.02):
        return "quat norm"
    z = rt[:, 2]
    if not (0.3 < z.min() and z.max() < 1.4):
        return f"root_z [{z.min():.2f},{z.max():.2f}]"
    if np.abs(dof).max() > 2.8:
        return f"dof range {np.abs(dof).max():.2f}"
    if dof.shape[0] < min_frames:
        return f"too short {dof.shape[0]}f"
    return None


def main():
    idx = json.load(open(os.path.join(BASE, "motion_index.json")))
    import re
    os.makedirs(OUT, exist_ok=True)
    selected, used = [], set()
    for label, pat, n, burst in QUOTAS:
        rx = re.compile(pat)
        fams = sorted(b for b in idx if rx.search(b) and b not in used)
        picked = 0
        for fam in fams:
            if picked >= n:
                break
            f = pick_file(idx[fam], mirror=False, burst=burst)
            if not f:
                continue
            selected.append((label, fam, f, burst))
            used.add(fam)
            picked += 1
            # mirror twin for locomotion families
            if any(re.search(p, fam) for p in MIRROR_FOR):
                fm = pick_file(idx[fam], mirror=True, burst=burst)
                if fm:
                    selected.append((label + "_M", fam + "_M", fm, burst))
        if picked < n:
            print(f"  [warn] {label}: only {picked}/{n} found")

    # convert + gate
    ok, failed = 0, []
    for label, fam, f, burst in selected:
        try:
            seq = load_bones_csv(f["file"])
            e = convert_sequence(seq, 120)
            e = downsample_sequence(e, 120, 30)
            why = gate(e, min_frames=100 if burst else 240)
            if why:
                failed.append((fam, why)); continue
            joblib.dump({fam: e}, os.path.join(OUT, fam + ".pkl"), compress=True)
            ok += 1
            print(f"  [ok] {label:14s} {fam:48s} {e['dof'].shape[0]:5d}f={e['dof'].shape[0]/30:5.1f}s "
                  f"z=[{e['root_trans_offset'][:,2].min():.2f},{e['root_trans_offset'][:,2].max():.2f}]")
        except Exception as ex:  # noqa: BLE001
            failed.append((fam, repr(ex)[:60]))
    # original 2 clips
    for name in ["walk_forward_amateur_001__A001", "walk_forward_amateur_001__A001_M"]:
        shutil.copy(os.path.join(ORIG_DIR, name + ".pkl"), os.path.join(OUT, name + ".pkl"))
    print(f"\nconverted ok: {ok} + 2 originals = {ok + 2}; failed: {len(failed)}")
    for fam, why in failed:
        print(f"  [gate] {fam}: {why}")
    with open(os.path.join(BASE, "PIPELINE_LOG.md"), "a") as log:
        log.write(f"\n[round3-select] {ok} new motions + 2 originals -> round3_pkls/ "
                  f"({len(failed)} gated out)\n")


if __name__ == "__main__":
    main()
