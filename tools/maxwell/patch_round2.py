#!/usr/bin/env python3
"""Round-2 patch for mujoco_env.py: penalize/enforce global pace keeping.
1) terminate when horizontal lag from reference exceeds LAG_METERS
2) boost tracking_anchor_pos weight 0.5 -> 2.0, sigma 0.3 -> 0.2
"""

path = "/data/z00666713/GR00T-WholeBodyControl/gear_sonic/envs/mujoco_env.py"
src = open(path).read()

if "ROUND2" in src:
    print("ALREADY_PATCHED")
    raise SystemExit(0)

# 1) lag termination
anchor_term = """        term = False; h_thresh = 0.75 if ref_h < 0.5 else 0.15
"""
new_term = """        term = False; h_thresh = 0.75 if ref_h < 0.5 else 0.15
        # ROUND2: horizontal lag from the reference is lethal — "marking time
        # with correct poses" must not survive.
        if np.linalg.norm(root_pos[:2] - ref_root_pos[:2]) > 1.0:
            return True, False
"""
count = src.count(anchor_term)
if count != 1:
    print(f"TERM_ANCHOR_COUNT_{count}")
    raise SystemExit(1)
src = src.replace(anchor_term, new_term)

# 2) heavier anchor position reward
old_r1 = """        # 1. tracking_anchor_pos (w=0.5, σ=0.3)
        err = np.linalg.norm(root_pos - ref_root_pos)
        r1 = 0.5 * np.exp(-err**2 / 0.09)
"""
new_r1 = """        # 1. tracking_anchor_pos (ROUND2: w=0.5→2.0, σ=0.3→0.2 — keep up or lose big)
        err = np.linalg.norm(root_pos - ref_root_pos)
        r1 = 2.0 * np.exp(-err**2 / 0.08)
"""
count = src.count(old_r1)
if count != 1:
    print(f"R1_ANCHOR_COUNT_{count}")
    raise SystemExit(1)
src = src.replace(old_r1, new_r1)

open(path, "w").write(src)
print("ROUND2_PATCHED")
