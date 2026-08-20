#!/usr/bin/env python3
"""Ankle divergence onset analysis on the clean replays (4 bad clips + A033
reference).  Finds WHEN the ankle-pitch correlation collapses and what the
plant state looks like at that moment.

Pairing: same-index (our step k state == their qpos[k], state-override
protocol).  Their qpos is isaaclab order -> map to mujoco via ISAAC_REORDER.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "gear_sonic", "envs", "physx"))
from physx_fk import G1ForwardKinematics  # noqa: E402

ISAAC_REORDER = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                          4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,
                          20, 27, 21, 28], dtype=np.int64)
MUJOCO = ["lh_pitch", "lh_roll", "lh_yaw", "lk", "la_pitch", "la_roll",
          "rh_pitch", "rh_roll", "rh_yaw", "rk", "ra_pitch", "ra_roll",
          "waist_yaw", "waist_roll", "waist_pitch", "l_sh_pitch",
          "l_sh_roll", "l_sh_yaw", "l_elb", "l_wr_roll", "l_wr_pitch",
          "l_wr_yaw", "r_sh_pitch", "r_sh_roll", "r_sh_yaw", "r_elb",
          "r_wr_roll", "r_wr_pitch", "r_wr_yaw"]
LA, RA = 4, 10  # mujoco indices
WINDOW = 40
XML = "/Users/kevin/code/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof_v17.xml"

CLIPS = {
    "A476_the_dog": ("/tmp/r2_clean_local/replay_walk_the_dog_ff_180_loop_R_001__A476_01.npz",
                     "/tmp/isaac_r2/isaac_baseline_round2_20260819/release/release_walk_the_dog_ff_180_loop_R_001__A476_01.npz"),
    "A050_ff_loop": ("/tmp/r2_clean_local/replay_walk_ff_loop_180_R_003__A050_00.npz",
                     "/tmp/isaac_r2/isaac_baseline_round2_20260819/release/release_walk_ff_loop_180_R_003__A050_00.npz"),
    "A338_inj_torso": ("/tmp/r2_clean_local/replay_injured_torso_walk_ff_start_225_R_003__A338_02.npz",
                       "/tmp/isaac_r2/isaac_baseline_round2_20260819/release/release_injured_torso_walk_ff_start_225_R_003__A338_02.npz"),
    "A516_crutches": ("/tmp/r2_clean_local/replay_crutches_walk_arc_cw_start_R_001__A516_01.npz",
                      "/tmp/isaac_r2/isaac_baseline_round2_20260819/release/release_crutches_walk_arc_cw_start_R_001__A516_01.npz"),
    "A033_sideway_ref": ("/tmp/r2_clean_local/replay_walk_sideway_045_loop_003__A033_00.npz",
                         "/tmp/isaac_r2/isaac_baseline_round2_20260819/release/release_walk_sideway_045_loop_003__A033_00.npz"),
}


def i2m(q):
    qm = np.empty(29)
    qm[ISAAC_REORDER] = q
    return qm


def rolling_corr(a, b, w):
    n = len(a)
    out = np.full(n, np.nan)
    for k in range(w, n):
        out[k] = np.corrcoef(a[k - w:k], b[k - w:k])[0, 1]
    return out


def main():
    fk = G1ForwardKinematics(XML)
    la_idx = fk.get_link_index("left_ankle_roll_link")
    ra_idx = fk.get_link_index("right_ankle_roll_link")
    print(f"ankle_roll link idx: {la_idx} {ra_idx}")

    for name, (rep_path, isa_path) in CLIPS.items():
        rep = np.load(rep_path)
        isa = np.load(isa_path)
        n = min(len(rep["qpos"]), len(isa["qpos"]) - 1)
        ours = rep["qpos"][:n]
        theirs = np.stack([i2m(q) for q in isa["qpos"][:n]])
        tgt = rep["target"][:n]
        rpos = rep["root_pos"][:n]
        rquat = rep["root_quat"][:n]

        # rolling corr per ankle
        rc_la = rolling_corr(ours[:, LA], theirs[:, LA], WINDOW)
        rc_ra = rolling_corr(ours[:, RA], theirs[:, RA], WINDOW)
        # ankle tracking error (ours: |q - target|) and their drive error proxy
        err_la = np.abs(ours[:, LA] - tgt[:, LA])
        err_ra = np.abs(ours[:, RA] - tgt[:, RA])
        # foot heights via FK (ground clearance)
        foot_h = np.zeros((n, 2))
        for k in range(n):
            poses = fk.compute(rpos[k], rquat[k], ours[k])
            foot_h[k, 0] = poses[la_idx][0][2]
            foot_h[k, 1] = poses[ra_idx][0][2]

        # onset: first step where BOTH ankle rolling corrs stay below 0.3
        # for the remainder (persistent collapse), or min-corr step
        bad = np.nanmin(np.vstack([rc_la, rc_ra]), axis=0)
        onset = None
        for k in range(WINDOW, n):
            if np.all(bad[k:] < 0.3):
                onset = k
                break
        total_lo = np.nansum(bad < 0.3)
        print(f"\n===== {name} (n={n}) =====")
        print(f"  rolling ankle corr: min={np.nanmin(bad):.2f} "
              f"mean={np.nanmean(bad):.2f} | frac<0.3: {total_lo}/{n}")
        if onset is not None:
            print(f"  ONSET k={onset} ({onset * 0.02:.1f}s): corr_la={rc_la[onset]:.2f} "
                  f"corr_ra={rc_ra[onset]:.2f}")
        else:
            onset = int(np.nanargmin(bad))
            print(f"  no persistent collapse; worst k={onset} "
                  f"({onset * 0.02:.1f}s): corr={bad[onset]:.2f}")
        print(f"  @onset: root_z={rpos[onset, 2]:.3f} "
              f"foot_h L/R={foot_h[onset, 0]:.3f}/{foot_h[onset, 1]:.3f} "
              f"err_la={err_la[onset]:.3f} err_ra={err_ra[onset]:.3f} "
              f"tgt_la={tgt[onset, LA]:.3f} tgt_ra={tgt[onset, RA]:.3f}")
        print(f"  mean foot_h: {foot_h[:, 0].mean():.3f}/{foot_h[:, 1].mean():.3f} "
              f"min: {foot_h[:, 0].min():.3f}/{foot_h[:, 1].min():.3f} "
              f"max: {foot_h[:, 0].max():.3f}/{foot_h[:, 1].max():.3f}")
        print(f"  mean err: la={err_la.mean():.3f} ra={err_ra.mean():.3f}")

        # stance-conditioned error: ours vs theirs (their target from npz)
        th_tgt = np.stack([i2m(q) for q in isa["joint_target"][:n]])
        err_th_la = np.abs(theirs[:, LA] - th_tgt[:, LA])
        err_th_ra = np.abs(theirs[:, RA] - th_tgt[:, RA])
        for side, err_o, err_t, fh in (("la", err_la, err_th_la, foot_h[:, 0]),
                                       ("ra", err_ra, err_th_ra, foot_h[:, 1])):
            st = fh < 0.05
            sw = fh > 0.08
            print(f"  {side}: err ours stance={err_o[st].mean():.3f} "
                  f"swing={err_o[sw].mean():.3f} | theirs stance={err_t[st].mean():.3f} "
                  f"swing={err_t[sw].mean():.3f} "
                  f"(stance steps {st.sum()}, swing {sw.sum()})")


if __name__ == "__main__":
    main()
