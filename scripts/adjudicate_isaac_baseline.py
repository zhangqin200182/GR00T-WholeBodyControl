#!/usr/bin/env python3
"""Post-hoc termination adjudication on the Isaac baseline package.

Replicates physx_env._check_termination (ori 0.35 / body-h 0.225 /
ankle-pos 0.35 / root-height 0.15) on the recorded Isaac trajectories,
computing per-episode survival steps for the release and PD runs and
comparing against our PhysX-side fixed-12 baselines (Appendix B:
release 25.67 / PD 34.33).

Input: the extracted isaac_baseline_20260818 package.
Pure numpy + local FK — no physx_core needed.
"""
import argparse, glob, os, sys
import numpy as np

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx"))
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs"))
from physx_fk import G1ForwardKinematics  # noqa: E402
from mujoco_math import quat_error_magnitude, quat_mul, quat_inv, quat_apply  # noqa: E402

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "gear_sonic_deploy", "g1", "g1_29dof_v17.xml")

BODY_NAMES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link", "left_shoulder_roll_link", "left_elbow_link",
    "left_wrist_yaw_link", "right_shoulder_roll_link",
    "right_elbow_link", "right_wrist_yaw_link",
)
# ISAAC_REORDER[i] = mujoco index of the joint at isaaclab position i
ISAAC_REORDER = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                          4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,
                          20, 27, 21, 28], dtype=np.int64)

ORI_THRESH = 0.35
ANK_POS_THRESH = 0.35
ANK_H_MULT = 1.5

HEIGHT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link",
                "left_wrist_yaw_link", "right_wrist_yaw_link")
POS_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
# Isaac-side recorded ref_qpos is in mocap joint convention while the robot
# runs in the USD asset convention; upper-body joints (shoulder_yaw/elbow/
# wrist_yaw) carry constant ~10-65deg offsets that the policy never tracks
# (verified: offset constant over 500 steps, legs clean). Wrist-height
# checks are therefore convention-contaminated on Isaac data — drop them.
CONTAMINATED_HEIGHT_LINKS = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def isaac_to_mujoco(q_isaac):
    """Isaac internal order -> our XML (mujoco) order."""
    q_mj = np.empty(29)
    q_mj[ISAAC_REORDER] = q_isaac
    return q_mj


def check_step(fk, root_pos, root_quat, qpos, ref_root_pos, ref_root_quat,
               ref_qpos, t_idx, skip_wrist_h=True):
    """Replicates physx_env._check_termination for one control step.

    qpos / ref_qpos must already be in mujoco (XML) order.

    Returns (terminated, reason_string) or (False, "").
    """
    ref_h = ref_root_pos[t_idx, 2]
    root_h = root_pos[t_idx, 2]
    h_thresh = 0.75 if ref_h < 0.5 else 0.15
    reason = ""

    if abs(ref_h - root_h) > h_thresh:
        return True, f"height({abs(ref_h - root_h):.3f}>{h_thresh:.3f})"

    ori_err = quat_error_magnitude(ref_root_quat[t_idx], root_quat[t_idx]) ** 2
    if ori_err > ORI_THRESH:
        reason += f"ori({ori_err:.3f}>{ORI_THRESH:.3f})"

    actual_body = [t[0] for t in fk.get_tracked_poses(
        root_pos[t_idx], root_quat[t_idx], qpos[t_idx])]
    ref_body = [t[0] for t in fk.get_tracked_poses(
        ref_root_pos[t_idx], ref_root_quat[t_idx], ref_qpos[t_idx])]

    for name in HEIGHT_LINKS:
        if skip_wrist_h and name in CONTAMINATED_HEIGHT_LINKS:
            continue
        idx = BODY_NAMES.index(name)
        err = abs(ref_body[idx][2] - actual_body[idx][2])
        h_limit = h_thresh * ANK_H_MULT
        if err > h_limit:
            return True, reason + f" {name}_h({err:.3f}>{h_limit:.3f})"

    for name in POS_LINKS:
        idx = BODY_NAMES.index(name)
        ref_aligned = ref_body[idx] - ref_root_pos[t_idx] + root_pos[t_idx]
        err = np.linalg.norm(ref_aligned - actual_body[idx])
        if err > ANK_POS_THRESH:
            return True, reason + f" {name}_pos({err:.3f}>{ANK_POS_THRESH:.3f})"

    if reason:
        return True, reason
    return False, ""


def heading_align_ref(root_pos, root_quat, ref_root_pos, ref_root_quat):
    """Rotate the recorded reference into the robot's initial heading frame.

    The Isaac env spawns the robot at the USD asset default orientation and
    never aligns it to the motion's initial heading (the policy tracks
    heading-relative motion — verified: aligned-start turning clips track
    the turn, misaligned-start clips never resolve the offset). Our PhysX
    env resets root = ref heading, so absolute-heading error is a reset
    convention difference, not physics. Aligning the ref reproduces the
    starting condition our env has and measures pure tracking error.
    """
    q_delta = quat_mul(root_quat[0], quat_inv(ref_root_quat[0]))
    ref_q_rot = np.stack([quat_mul(q_delta, q) for q in ref_root_quat])
    origin = ref_root_pos[0]
    ref_p_rot = np.stack([quat_apply(q_delta, p - origin) + origin
                          for p in ref_root_pos])
    return ref_p_rot, ref_q_rot


def adjudicate(npz_path, fk, align_heading=True, grace=0,
               qpos_order="isaac", skip_wrist_h=False):
    d = np.load(npz_path)
    n = len(d["ctrl_step"]) if "ctrl_step" in d.files else len(d["qpos"])
    ref_p, ref_q = d["ref_root_pos"], d["ref_root_quat"]
    if align_heading:
        ref_p, ref_q = heading_align_ref(d["root_pos"], d["root_quat"],
                                         ref_p, ref_q)
    if qpos_order == "isaac":
        qpos = np.stack([isaac_to_mujoco(q) for q in d["qpos"]])
        ref_qpos = np.stack([isaac_to_mujoco(q) for q in d["ref_qpos"]])
    else:
        qpos, ref_qpos = d["qpos"], d["ref_qpos"]
    for k in range(n):
        if k < grace:
            continue
        term, reason = check_step(fk, d["root_pos"], d["root_quat"], qpos,
                                  ref_p, ref_q, ref_qpos, k,
                                  skip_wrist_h=skip_wrist_h)
        if term:
            return k + 1, reason
    return n, "survived"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/tmp/isaac_baseline",
                        help="Extracted Isaac baseline package root")
    parser.add_argument("--grace", type=int, default=0,
                        help="Skip termination checks for the first N steps "
                             "(Isaac side spawns from the asset default pose, "
                             "not ref[0]; the policy converges within ~10 steps)")
    parser.add_argument("--qpos-order", default="isaac", choices=["isaac", "mujoco"],
                        help="Joint order of recorded qpos/ref_qpos: isaac "
                             "(Isaac package, map via ISAAC_REORDER) or mujoco "
                             "(our own save-mode npz, pass through)")
    parser.add_argument("--no-heading-align", action="store_true",
                        help="Skip heading alignment (our own data is already "
                             "aligned at reset)")
    parser.add_argument("--skip-wrist-h", action="store_true",
                        help="Exclude wrist-height checks (convention-"
                             "contaminated on Isaac side only)")
    args = parser.parse_args()

    fk = G1ForwardKinematics(XML)

    results = {}
    for policy in ("release", "pd"):
        files = [f for f in sorted(glob.glob(
            os.path.join(args.data_dir, policy, f"{policy}_*.npz")))
            if "_contacts" not in f]
        if not files:
            continue
        rows = []
        for f in files:
            clip = os.path.basename(f).replace(f"{policy}_", "").replace(".npz", "")
            surv, reason = adjudicate(
                f, fk, grace=args.grace, qpos_order=args.qpos_order,
                align_heading=not args.no_heading_align,
                skip_wrist_h=args.skip_wrist_h)
            rows.append((clip, surv, reason))
        results[policy] = rows

    print(f"{'policy':8s} {'ep':3s} {'clip':45s} {'surv':5s} {'reason'}")
    for policy in ("release", "pd"):
        if policy not in results:
            continue
        for i, (clip, surv, reason) in enumerate(results[policy]):
            print(f"{policy:8s} {i:3d} {clip:45s} {surv:5d} {reason}")
        lens = np.array([r[1] for r in results[policy]], dtype=float)
        print(f"{policy:8s} AGG mean_len={lens.mean():.2f} median={np.median(lens):.1f} "
              f"min={lens.min():.0f} max={lens.max():.0f}")
        print()

    print("=" * 60)
    if "release" in results:
        rl = np.array([r[1] for r in results["release"]], dtype=float)
        print(f"release post-hoc: {rl.mean():.2f} steps")
    if "pd" in results:
        pd = np.array([r[1] for r in results["pd"]], dtype=float)
        print(f"PD      post-hoc: {pd.mean():.2f} steps")
    if "release" in results and "pd" in results:
        print(f"PhysX  release baseline: 25.67 (Appendix B)")
        print(f"PhysX  PD      baseline: 34.33 (Appendix B)")
        print(f"release/PD ratio: {rl.mean()/pd.mean():.2f}")
        print(f"release vs PhysX release: {rl.mean()/25.67:.2f}x")


if __name__ == "__main__":
    main()
