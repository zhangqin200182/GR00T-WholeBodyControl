#!/usr/bin/env python3
"""Our-side R3 replica — free-root + planted-feet stance response, CFG soft
gains.  Aligned with the ACTUAL Isaac-side delivery (round-3 notes):
  - q0 = default_joint_pos (the walk-sideway ref frame 0 tips over under
    pure PD hold); read from the npz q0 field when present
  - recording at CONTROL steps (dt=0.02, 50Hz) — their physics_context
    cannot sample per-substep
  - targets updated every control step (0.02s), physics at 0.005 x4
  - FREE root, planted feet, non-perturbed joints PD-held at q0
  - perturb hip_pitch / hip_roll / knee: steps +-0.05/0.1/0.2, sines
    0.5/2/5 Hz @ 0.05
"""
import argparse
import os
import sys
import numpy as np

os.environ.setdefault("PHYSX_CONTACT_DEBUG", "1")
_build = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "gear_sonic", "envs", "physx", "build"),
    "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build",
]
for _d in _build:
    if os.path.isdir(_d):
        sys.path.insert(0, _d)
        break
import physx_core  # noqa: E402

NPZ_REF = os.environ.get("R3_REF_NPZ",
    "/tmp/r3_isaac/r3_left_hip_pitch_step_0.05p.npz")
XML_JOINTS = ["lh_pitch", "lh_roll", "lh_yaw", "lk", "la_pitch", "la_roll",
              "rh_pitch", "rh_roll", "rh_yaw", "rk", "ra_pitch", "ra_roll",
              "waist_yaw", "waist_roll", "waist_pitch", "l_sh_pitch",
              "l_sh_roll", "l_sh_yaw", "l_elb", "l_wr_roll", "l_wr_pitch",
              "l_wr_yaw", "r_sh_pitch", "r_sh_roll", "r_sh_yaw", "r_elb",
              "r_wr_roll", "r_wr_pitch", "r_wr_yaw"]
ISAAC_JOINTS = ["lh_pitch", "rh_pitch", "waist_yaw", "lh_roll", "rh_roll",
                "waist_roll", "lh_yaw", "rh_yaw", "waist_pitch", "lk", "rk",
                "l_sh_pitch", "r_sh_pitch", "la_pitch", "ra_pitch",
                "l_sh_roll", "r_sh_roll", "la_roll", "ra_roll", "l_sh_yaw",
                "r_sh_yaw", "l_elb", "r_elb", "l_wr_roll", "r_wr_roll",
                "l_wr_pitch", "r_wr_pitch", "l_wr_yaw", "r_wr_yaw"]
XML2ISAAC = [ISAAC_JOINTS.index(n) for n in XML_JOINTS]
DT = 0.005
ACT_OFFSET_FALLBACK = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0], dtype=np.float32)


def run_group(art, scene, j, target_fn, t_end, q0, record_contacts=True):
    """Free-root stance perturbation.  NO root re-pin, NO joint locking —
    non-perturbed joints are held by their drives at q0."""
    rec = {"t": [], "qpos": [], "qvel": [], "target": [], "tau": [],
           "root_pos": [], "root_quat": [], "n_contacts": []}
    t = 0.0
    while t <= t_end + 1e-6:
        # control step: 4 physics substeps, target updated per control step,
        # recorded at 50Hz (matches the Isaac actual behavior)
        targets = q0.copy()
        targets[j] = target_fn(t)
        art.set_joint_drive_targets(targets.astype(np.float32))
        if record_contacts:
            scene.clear_contacts()
        for _ in range(4):
            scene.simulate(DT)
            scene.fetch_results()
        rp, rq = art.get_root_world_pose()
        rec["t"].append(t)
        rec["qpos"].append(art.get_joint_positions().copy())
        rec["qvel"].append(art.get_joint_velocities().copy())
        rec["target"].append(targets.copy())
        rec["tau"].append(art.get_joint_torques().copy())
        rec["root_pos"].append(rp)
        rec["root_quat"].append(rq)
        rec["n_contacts"].append(len(scene.get_contacts()) if record_contacts else -1)
        t += 0.02
    return {k: np.array(v) for k, v in rec.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/r3_ours")
    ap.add_argument("--joints", nargs="*",
                    default=["left_hip_pitch", "left_hip_roll", "left_knee"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    JOINT_IDX = {"left_hip_pitch": 0, "left_hip_roll": 1, "left_knee": 3,
                 "right_hip_pitch": 6, "right_hip_roll": 7, "right_knee": 9}

    z = np.load(NPZ_REF)
    if "q0" in z.files:
        q0 = z["q0"][XML2ISAAC].astype(np.float32)  # their embedded default pose
        print("q0: from npz q0 field", flush=True)
    else:
        q0 = ACT_OFFSET_FALLBACK.astype(np.float32)
        print("q0: ACT_OFFSET fallback", flush=True)
    root0 = z["root_pos"][0].astype(np.float32)
    rq0 = z["root_quat"][0].astype(np.float32)  # wxyz as-is (their initial, upright)

    physx_core.init_foundation()
    from gear_sonic.envs.physx_env import PhysXEnv
    from omegaconf import OmegaConf
    env = PhysXEnv(physx_core, "/gear_sonic_deploy/g1/g1_29dof_v17.xml",
                   "/sample_data/robot_filtered_fixed12",
                   config=OmegaConf.create({
                       "alive_bonus": 0.0, "ori_thresh": 0.35,
                       "ank_pos_thresh": 0.35, "ank_h_mult": 1.5,
                       "action_trust": 1.0, "isaac_action_space": True,
                       "ignore_terminations": False, "skip_termination": False,
                       "max_episode_length": 500,
                   }),
                   native_dt=0.005, decimation=4, pos_iters=8, vel_iters=4,
                   static_pose=False, root_z_offset=0.04, standing_prob=0.0,
                   drive_type="FORCE")
    env._forced_idx = 3
    env._forced_ref_time = 0.0
    env.reset()
    env.art.set_root_world_pose(root0, rq0)
    env.art.set_joint_positions(q0)
    env.art.set_joint_velocities(np.zeros(29, dtype=np.float32))
    env.art.set_root_world_velocity(np.zeros(3, dtype=np.float32),
                                    np.zeros(3, dtype=np.float32))
    env.art.set_joint_drive_targets(q0)

    # settle 0.5s before each group
    for jname in args.joints:
        j = JOINT_IDX[jname]
        for a in (0.05, 0.1, 0.2):
            for sign in (+1.0, -1.0):
                tag = f"r3_{jname}_step_{a}{'p' if sign > 0 else 'n'}"
                for _ in range(25):
                    env.art.set_joint_drive_targets(q0)
                    for _ in range(4):
                        env.scene.simulate(DT)
                        env.scene.fetch_results()
                rec = run_group(env.art, env.scene, j,
                                lambda t, a=a, s=sign: q0[j] + s * a,
                                2.0, q0)
                np.savez(os.path.join(args.out, f"{tag}.npz"), **rec)
                print(f"{tag}: peak_tau={np.abs(rec['tau'][:, j]).max():.2f} "
                      f"root_z_end={rec['root_pos'][-1, 2]:.3f}", flush=True)
        for f in (0.5, 2.0, 5.0):
            tag = f"r3_{jname}_sine_{f}"
            for _ in range(25):
                env.art.set_joint_drive_targets(q0)
                for _ in range(4):
                    env.scene.simulate(DT)
                    env.scene.fetch_results()
            rec = run_group(env.art, env.scene, j,
                            lambda t, f=f: q0[j] + 0.05 * np.sin(2 * np.pi * f * t),
                            4.0, q0)
            np.savez(os.path.join(args.out, f"{tag}.npz"), **rec)
            print(f"{tag}: root_z_end={rec['root_pos'][-1, 2]:.3f}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
