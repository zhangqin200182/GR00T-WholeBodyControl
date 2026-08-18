#!/usr/bin/env python3
"""Cross-eval: evaluate a checkpoint under specified termination thresholds.

Multi-episode deterministic rollout on single PhysXEnv, reporting survival
length distribution. Used to compare policies trained under different
thresholds under one common (strict) threshold setting.
"""
import argparse, os, sys
from collections import Counter
import numpy as np

# Must import physx_core before torch (fork+import safety)
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

import torch

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

TOKENIZER_OBS_DIMS = {
    "encoder_index": (3,), "command_multi_future_nonflat": (10, 58),
    "command_z_multi_future_nonflat": (10, 1), "motion_anchor_ori_b_mf_nonflat": (10, 6),
    "command_multi_future_lower_body": (240,), "vr_3point_local_target": (9,),
    "vr_3point_local_orn_target": (12,), "motion_anchor_ori_b": (6,), "command_z": (1,),
    "smpl_joints_multi_future_local_nonflat": (10, 72), "smpl_root_ori_b_multi_future": (10, 6),
    "joint_pos_multi_future_wrist_for_smpl": (10, 6),
}


def make_model(ckpt_path):
    from omegaconf import OmegaConf
    from gear_sonic.trl.utils.common import custom_instantiate

    env_config = OmegaConf.create({
        "obs": {"obs_dims": {"actor_obs": 930, "critic_obs": 1645, "tokenizer": 1761},
                "obs_dict": {}, "group_obs_dims": {"tokenizer": TOKENIZER_OBS_DIMS},
                "group_obs_names": {"tokenizer": list(TOKENIZER_OBS_DIMS.keys())}},
        "robot": {"actions_dim": 29, "num_joints": 29,
                   "algo_obs_dim_dict": {"actor_obs": 930, "critic_obs": 1645, "tokenizer": 1761}},
        "rewards": {"num_critics": 1}, "num_envs": 1,
    })

    ckpt_dir = os.path.dirname(ckpt_path)
    cfg_path = os.path.join(ckpt_dir, "config.yaml")
    if os.path.exists(cfg_path):
        cfg = OmegaConf.load(cfg_path)
        algo_cfg = cfg.algo.config
    else:
        algo_cfg = OmegaConf.create({
            "actor": {
                "_target_": "gear_sonic.trl.models.universal_token.backbone.universal_token_backbone.UniversalTokenBackbone",
                "backbone": {
                    "g1_obs_dim": 930, "g1_actions_dim": 29,
                    "aux_loss_coef": {"g1_recon": 0.01},
                    "shared_encoder_cfg": {"encoder_index": {}},
                },
            },
        })

    model = custom_instantiate(algo_cfg.actor, env_config=env_config,
                               algo_config=algo_cfg, _resolve=False).to("cpu")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except AttributeError:
        # Release checkpoint saved by the TRL fork (missing classes in the
        # installed trl) — fall back to the stub loader.
        from gear_sonic.utils.release_ckpt import load_release
        ckpt = load_release(ckpt_path)
    sd = (ckpt.get("actor_model_state_dict") or ckpt.get("policy_state_dict")
          or ckpt.get("model_state_dict") or ckpt.get("policy"))
    if sd is None:
        raise ValueError(f"Cannot find model state in checkpoint: {list(ckpt.keys())}")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise ValueError(f"Checkpoint key mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    model.init_rollout()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--ori", type=float, default=0.30)
    parser.add_argument("--ank", type=float, default=0.30)
    parser.add_argument("--ank_h", type=float, default=1.5)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--motion_seed", type=int, default=0,
                        help="Deterministic reshuffle of env.motions so parallel "
                             "evals sample identical motion clips")
    parser.add_argument("--trust", type=float, default=1.0,
                        help="Action trust: 0.0 = pure ref PD baseline, 1.0 = policy")
    parser.add_argument("--isaac-space", action="store_true",
                        help="Use the Isaac action space (target = offset + scale * action)")
    parser.add_argument("--drive-type", default="ACCELERATION",
                        choices=["ACCELERATION", "FORCE"],
                        help="Articulation drive mode: FORCE = Isaac torque-domain "
                             "PD semantics (raw kp/kd + effort limits)")
    parser.add_argument("--vel-iters", type=int, default=1,
                        help="Solver velocity iterations (Isaac CFG uses 4)")
    parser.add_argument("--native-dt", type=float, default=0.001961,
                        help="Physics substep dt (Isaac sim_dt = 0.005)")
    parser.add_argument("--decimation", type=int, default=10,
                        help="Substeps per control step (Isaac decimation = 4)")
    args = parser.parse_args()

    from omegaconf import OmegaConf

    physx_core.init_foundation()
    print(f"Loading checkpoint: {args.ckpt}", flush=True)
    model = make_model(args.ckpt)

    np.random.seed(args.seed)
    from gear_sonic.envs.physx_env import PhysXEnv
    env = PhysXEnv(physx_core, XML, PKL,
                   config=OmegaConf.create({
                       "alive_bonus": 0.0,
                       "ori_thresh": args.ori,
                       "ank_pos_thresh": args.ank,
                       "ank_h_mult": args.ank_h,
                       "action_trust": args.trust,
                       "isaac_action_space": args.isaac_space,
                       "ignore_terminations": False,
                       "skip_termination": False,
                       "max_episode_length": args.max_steps,
                   }),
                   native_dt=args.native_dt, decimation=args.decimation,
                   pos_iters=8, vel_iters=args.vel_iters,
                   static_pose=False, root_z_offset=0.04, standing_prob=0.0,
                   drive_type=args.drive_type)
    # Deterministic reshuffle (loader itself now uses a fixed seed). Permute
    # the motion-id list alongside so per-episode clip attribution stays aligned.
    order = np.random.RandomState(args.motion_seed).permutation(len(env.motions))
    env.motions = [env.motions[i] for i in order]
    if getattr(env, "_motion_ids", None):
        env._motion_ids = [env._motion_ids[i] for i in order]

    print(f"Eval @ ori={args.ori}, ank_pos={args.ank}, ank_h={args.ank_h}, "
          f"episodes={args.episodes}", flush=True)

    lengths, rewards, reasons = [], [], []
    with torch.no_grad():
        for ep in range(args.episodes):
            obs = env.reset()
            total = 0.0
            reason = "survived"
            for s in range(args.max_steps):
                obs_t = {k: torch.from_numpy(v).float().unsqueeze(0)
                         for k, v in obs.items()}
                a = model.act_inference(obs_t, cur_dones=None)
                obs, r, done, info = env.step(a.squeeze(0).numpy())
                total += r
                if done:
                    reason = info.get("term_reason", "") or "truncated"
                    break
            lengths.append(s + 1)
            rewards.append(total)
            reasons.append(reason)
            print(f"  ep{ep}: survived={s + 1}, reward={total:.1f}, reason={reason}, "
                  f"clip={getattr(env, '_cur_motion_id', '?')}",
                  flush=True)

    lengths = np.array(lengths)
    cats = Counter()
    for r in reasons:
        if r == "survived":
            cats["survived"] += 1
        elif not r:
            cats["truncated"] += 1
        else:
            if "height(" in r: cats["height"] += 1
            if "ori(" in r: cats["ori"] += 1
            if "_h(" in r: cats["body_h"] += 1
            if "_pos(" in r: cats["ank_pos"] += 1
    print(f"RESULT: mean_len={lengths.mean():.2f} median={np.median(lengths):.1f} "
          f"min={lengths.min()} max={lengths.max()} mean_rew={np.mean(rewards):.1f}",
          flush=True)
    print("DEATH MODES: " + ", ".join(f"{k}={v}" for k, v in cats.most_common()),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
