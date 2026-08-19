#!/usr/bin/env python3
"""Release 3b replay: feed Isaac-recorded release actions (A033 env0) through
our engine; compare qpos trajectory vs Isaac recorded qpos.

Targets: env.step(a) with isaac_action_space translates a * scale + offset;
injecting their action_raw (isaaclab order -> XML) reproduces their targets.
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
                    help="native physics dt (Isaac=0.005; ours legacy=0.001961)")
    ap.add_argument("--decimation", type=int, default=10,
                    help="physics steps per control step (Isaac=4 @0.005; legacy=10 @0.001961)")
    ap.add_argument("--save", required=True)
    args = ap.parse_args()

    from omegaconf import OmegaConf
    physx_core.init_foundation()

    z = np.load(args.npz)
    ar_isaac = z["action_raw"]  # (T, 29) isaaclab order
    T = len(ar_isaac)
    print(f"replay npz: {args.npz} T={T}", flush=True)

    from gear_sonic.envs.physx_env import PhysXEnv
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
                   static_pose=False, root_z_offset=0.04, standing_prob=0.0,
                   drive_type=args.drive_type)
    env._forced_idx = args.ep
    env.reset()

    qpos, qvel, tgt, root_pos, root_quat = [], [], [], [], []
    for k in range(T):
        a = ar_isaac[k][XML2ISAAC]
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
             root_pos=np.stack(root_pos), root_quat=np.stack(root_quat))
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
