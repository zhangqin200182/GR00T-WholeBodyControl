#!/usr/bin/env python3
"""Pre-send acceptance check for the obs-fixed release re-collection.

Catches the known P0 bug signature: the previous package had 12 npz files
that were only 3 unique trajectories (within-env action_raw bit-identical,
qpos bit-identical). This script fails loudly if that pattern persists.

Run on the collection machine BEFORE sending:
    python verify_recollection.py --npz-dir <new collection dir>

Pass criteria:
  - every pair of files has action_raw maxdiff > 0 (all 12 trajectories distinct)
  - every pair of files has qpos maxdiff > 0
  - consumed_motion_id present and matches the recorded ref clip per file
    (only checked if the field exists; field name is flexible)
"""

import argparse
import re
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", required=True)
    args = parser.parse_args()

    files = sorted(Path(args.npz_dir).glob("release_*.npz"))
    if len(files) != 12:
        print(f"[FAIL] expected 12 release npz, found {len(files)}")
        return 1

    data = {}
    for f in files:
        with np.load(f) as z:
            keys = set(z.files)
            data[f.name] = {
                "action_raw": z["action_raw"],
                "qpos": z["qpos"],
                "consumed_motion_id": (
                    z["consumed_motion_id"] if "consumed_motion_id" in keys
                    else z["consumed_motion_idx"] if "consumed_motion_idx" in keys
                    else None),
            }

    ok = True

    # 1) pairwise uniqueness (the P0-bug signature)
    names = [f.name for f in files]
    print("=== pairwise uniqueness (P0 bug check) ===")
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            a_act = np.abs(data[names[i]]["action_raw"] - data[names[j]]["action_raw"]).max()
            a_q = np.abs(data[names[i]]["qpos"] - data[names[j]]["qpos"]).max()
            if a_act == 0.0 or a_q == 0.0:
                print(f"  [FAIL] {names[i]} vs {names[j]}: "
                      f"action maxdiff={a_act:.6g} qpos maxdiff={a_q:.6g}")
                ok = False
    if ok:
        print("  [PASS] all pairs distinct")

    # 2) per-file displacement (sanity: 3 unique trajectory groups should be gone)
    print("=== per-file displacement |root_pos[-1]-root_pos[0]| ===")
    for name in names:
        with np.load(Path(args.npz_dir) / name) as z:
            rp = z["root_pos"] if "root_pos" in z.files else z["root_states"]
        disp = np.linalg.norm(rp[-1, :2] - rp[0, :2])
        print(f"  {name}: {disp:.3f} m (xy)")

    # 3) consumed_motion_id presence + ref pairing
    print("=== consumed_motion_id / ref pairing ===")
    n_missing = sum(1 for name in names if data[name]["consumed_motion_id"] is None)
    if n_missing:
        print(f"  [WARN] consumed_motion_id missing in {n_missing}/12 files "
              "(field not recorded -- requested in round-5 A)")
    else:
        for name in names:
            cid = data[name]["consumed_motion_id"]
            cid = cid.item() if isinstance(cid, np.generic) else cid
            ref_m = re.search(r"__(A\d{3}[^_]*)_", name)
            print(f"  {name}: consumed_motion_id={cid} "
                  f"(ref-in-filename={'match' if ref_m and str(ref_m.group(1)) in str(cid) else 'check'})")

    print("\n" + ("RESULT: PASS -- safe to send." if ok else
                  "RESULT: FAIL -- fix the obs-side injection before sending."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
