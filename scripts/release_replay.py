#!/usr/bin/env python3
"""Release 3b replay: feed Isaac-recorded release actions (A033 env0) through
our engine; compare qpos trajectory vs Isaac recorded qpos.

Targets: env.step(a) with isaac_action_space translates a * scale + offset;
their action_raw is isaaclab-ordered — env.step accepts it directly when the
joint-order gate is ON (legacy XML injection otherwise).
Terminations ignored (keep stepping 500).
"""
import argparse, os, sys
import numpy as np

os.environ.setdefault("PHYSX_CONTACT_DEBUG", "1")
_build_candidates = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "gear_sonic", "envs", "physx", "build"),
    "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build",
]
for _d in _build_candidates:
    if os.path.isdir(_d):
        sys.path.insert(0, _d)
        break
import physx_core

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"

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
ISAAC2XML = [XML_JOINTS.index(n) for n in ISAAC_JOINTS]
# action_raw columns are isaaclab order; env.step expects XML order.
# a_xml[j] = ar[isaac column of XML joint j]  (GATHER by XML label)
XML2ISAAC = [ISAAC_JOINTS.index(n) for n in XML_JOINTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Isaac release npz (A033)")
    ap.add_argument("--ep", type=int, default=3, help="forced clip index (A033=3)")
    ap.add_argument("--pkl", default="/sample_data/robot_filtered_fixed12")
    ap.add_argument("--drive-type", default="ACCELERATION",
                    choices=["ACCELERATION", "FORCE"])
    ap.add_argument("--vel-iters", type=int, default=1)
    ap.add_argument("--native-dt", type=float, default=0.001961,
                    help="Physics substep dt (Isaac sim_dt = 0.005)")
    ap.add_argument("--decimation", type=int, default=10,
                    help="Substeps per control step (Isaac decimation = 4)")
    ap.add_argument("--save", required=True)
    ap.add_argument("--legacy-reset", action="store_true",
                    help="Keep the old random-phase reset (A/B vs pre-fix runs). "
                         "Default: reset phase forced to 0 AND plant state "
                         "overridden to the npz ref frame 0.")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    physx_core.init_foundation()

    z = np.load(args.npz)
    ar_isaac = z["action_raw"]  # (T, 29) isaaclab order
    T = len(ar_isaac)
    print(f"replay npz: {args.npz} T={T}", flush=True)

    from gear_sonic.envs.physx_env import PhysXEnv, _ISAAC_JOINT_ORDER
    env = PhysXEnv(physx_core, XML, args.pkl,
                   config=OmegaConf.create({
                       "alive_bonus": 0.0,
                       "ori_thresh": 0.35,
                       "ank_pos_thresh": 0.35,
                       "ank_h_mult": 1.5,
                       "action_trust": 1.0,
                       "isaac_action_space": True,
                       "ignore_terminations": False,
                       "skip_termination": False,
                       "max_episode_length": T,
                   }),
                   native_dt=args.native_dt, decimation=args.decimation,
                   pos_iters=8, vel_iters=args.vel_iters,
                   static_pose=False,
                   root_z_offset=float(os.environ.get(
                       "SONIC_PHYSX_ROOT_Z_OFFSET", "0.04")),
                   standing_prob=0.0,
                   drive_type=args.drive_type)
    env._forced_idx = args.ep
    if not args.legacy_reset:
        env._forced_ref_time = 0.0
    env.reset()
    reset_qpos = env.art.get_joint_positions().copy()
    reset_root_pos, reset_root_quat = env.art.get_root_world_pose()

    if not args.legacy_reset:
        # Override plant state to Isaac's recorded ref frame 0 (pre-step).
        # Their convention: state[k] = post-action[k] state (t[0]=0.02);
        # their pre-step state ~= ref_qpos[0] + 0.067 noise. Our reset
        # samples a random clip phase, which turns their small policy action
        # a[0] into a violent transient (fd vel 24.6 rad/s on step 0 vs
        # their 3.0) — replay then compares garbage from step 0. Quats in
        # their npz are xyzw (isaaclab); PhysX setter takes wxyz.
        try:
            z_ref_q0 = z["ref_qpos"][0][XML2ISAAC]
            rp0 = z["ref_root_pos"][0].astype(np.float32)
            rq0 = z["ref_root_quat"][0].astype(np.float32)  # xyzw
            rq0_wxyz = np.array([rq0[3], rq0[0], rq0[1], rq0[2]],
                                dtype=np.float32)
            env.art.set_joint_positions(z_ref_q0.astype(np.float32))
            env.art.set_joint_velocities(np.zeros(env.nu, dtype=np.float32))
            env.art.set_joint_drive_targets(z_ref_q0.astype(np.float32))
            env.art.set_root_world_pose(rp0, rq0_wxyz)
            env.art.set_root_world_velocity(np.zeros(3, dtype=np.float32),
                                            np.zeros(3, dtype=np.float32))
            print(f"state override: qpos=ref_qpos[0], root={np.round(rp0, 3)}, "
                  f"|reset_qpos - override|="
                  f"{np.abs(reset_qpos - z_ref_q0).max():.3f}", flush=True)
        except KeyError:
            print("WARNING: npz has no ref_qpos/ref_root fields; "
                  "falling back to phase-0 reset without override", flush=True)

    qpos, qvel, tgt, root_pos, root_quat = [], [], [], [], []
    for k in range(T):
        # env.step accepts isaaclab order when the joint-order gate is ON
        # (2026-08-19 obs/action order fix); legacy XML injection otherwise.
        a = ar_isaac[k] if _ISAAC_JOINT_ORDER else ar_isaac[k][XML2ISAAC]
        env.step(a)
        qpos.append(env.art.get_joint_positions().copy())
        qvel.append(env.art.get_joint_velocities().copy())
        tgt.append(env._last_joint_target.copy())
        rp, rq = env.art.get_root_world_pose()
        root_pos.append(rp)
        root_quat.append(rq)
        if (k + 1) % 50 == 0:
            print(f"  step {k+1}/{T}", flush=True)

    os.makedirs(args.save, exist_ok=True)
    out = os.path.join(args.save, f"replay_{os.path.basename(args.npz)}")
    np.savez(out,
             qpos=np.stack(qpos), qvel=np.stack(qvel), target=np.stack(tgt),
             root_pos=np.stack(root_pos), root_quat=np.stack(root_quat),
             reset_qpos=reset_qpos, reset_root_pos=reset_root_pos,
             reset_root_quat=reset_root_quat)
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
