#!/usr/bin/env python3
"""Periodic evaluation & rendering for PhysX training.

Usage (standalone):
  python scripts/physx_eval_render.py --ckpt /path/to/last.pt --output /tmp/eval_001.mp4

Usage (called during training):
  python scripts/physx_eval_render.py --ckpt_dir logs_rl/.../ --ckpt_step 100 --output_dir /tmp/evals/
"""

import argparse, logging, os, sys, glob, json
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("physx_eval")

# Must import physx_core before torch (fork+import safety)
_build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "gear_sonic", "envs", "physx", "build")
sys.path.insert(0, _build_dir)
import physx_core

# Import torch AFTER physx_core
import torch

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"


def make_env(px, static_pose=False):
    from gear_sonic.envs.physx_env import PhysXEnv
    env = PhysXEnv(px, XML, PKL, config=None,
                   native_dt=0.001961, decimation=10, pos_iters=8, vel_iters=1,
                   static_pose=static_pose, root_z_offset=0.0, standing_prob=0.0)
    env.skip_termination = False
    env.alive_bonus = 0.0
    return env


def make_model(ckpt_path):
    from omegaconf import OmegaConf
    from gear_sonic.trl.utils.common import custom_instantiate

    TOKENIZER_OBS_DIMS = {
        "encoder_index": (3,), "command_multi_future_nonflat": (10, 58),
        "command_z_multi_future_nonflat": (10, 1), "motion_anchor_ori_b_mf_nonflat": (10, 6),
        "command_multi_future_lower_body": (240,), "vr_3point_local_target": (9,),
        "vr_3point_local_orn_target": (12,), "motion_anchor_ori_b": (6,), "command_z": (1,),
        "smpl_joints_multi_future_local_nonflat": (10, 72), "smpl_root_ori_b_multi_future": (10, 6),
        "joint_pos_multi_future_wrist_for_smpl": (10, 6),
    }

    env_config = OmegaConf.create({
        "obs": {"obs_dims": {"actor_obs": 930, "critic_obs": 1645, "tokenizer": 1761},
                "obs_dict": {}, "group_obs_dims": {"tokenizer": TOKENIZER_OBS_DIMS},
                "group_obs_names": {"tokenizer": list(TOKENIZER_OBS_DIMS.keys())}},
        "robot": {"actions_dim": 29, "num_joints": 29,
                   "algo_obs_dim_dict": {"actor_obs": 930, "critic_obs": 1645, "tokenizer": 1761}},
        "rewards": {"num_critics": 1}, "num_envs": 1,
    })

    # Find a config.yaml in the ckpt dir or use defaults
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
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("actor_model_state_dict") or ckpt.get("policy_state_dict") or ckpt.get("model_state_dict")
    if sd is None:
        raise ValueError(f"Cannot find model state in checkpoint: {list(ckpt.keys())}")
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.init_rollout()
    return model


def render_frames_to_video(frames, output_path, fps=24):
    """Render frames to MP4 video. Uses ffmpeg if available, otherwise saves as npz."""
    if not frames:
        logger.warning("No frames to render")
        return

    # Try ffmpeg
    import subprocess, tempfile
    try:
        h, w = frames[0].shape[:2]
        # Write frames as raw RGB to ffmpeg
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
            "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", output_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait(timeout=60)
        logger.info(f"Video saved: {output_path} ({len(frames)} frames)")
    except (FileNotFoundError, Exception) as e:
        logger.warning(f"ffmpeg not available ({e}), saving frames as npz")
        np.savez_compressed(output_path.replace(".mp4", ".npz"),
                          frames=np.array(frames, dtype=np.uint8))


def render_rollout(model, env, max_steps=500, record_interval=1):
    """Run a deterministic rollout and record frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs = env.reset()
    obs_t = {k: torch.from_numpy(v).float().unsqueeze(0) for k, v in obs.items()}

    frames = []
    ref_qpos_list = []
    actual_qpos_list = []
    total_reward = 0
    survived = 0

    with torch.no_grad():
        for step in range(max_steps):
            a = model.act_inference(obs_t, cur_dones=None)
            a_np = a.squeeze(0).numpy()

            obs, reward, done, info = env.step(a_np)
            total_reward += reward
            survived += 1

            ref_qpos = env.get_current_ref_qpos()
            ref_qpos_list.append(ref_qpos)
            actual_qpos_list.append(env.art.get_joint_positions())

            if step % record_interval == 0:
                # Render via matplotlib (standalone — no isaac sim)
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                # Plot joint tracking
                ax = axes[0]
                ax.plot(ref_qpos, 'b-', alpha=0.6, label='ref')
                ax.plot(env.art.get_joint_positions(), 'r-', alpha=0.6, label='actual')
                ax.set_title(f'Joint Positions (step {step})')
                ax.set_xlabel('DOF'); ax.set_ylabel('normalized pos')
                ax.legend()
                ax.set_ylim(-1.2, 1.2)

                # Plot reward components
                ax = axes[1]
                ax.bar(range(13), [0]*13, color='lightgray')
                ax.set_title(f'Step {step} | Reward: {reward:.3f} | Survived: {survived}')
                ax.set_xlabel('reward component')

                fig.tight_layout()
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(img)
                plt.close(fig)

            if done:
                break

            obs_t = {k: torch.from_numpy(v).float().unsqueeze(0) for k, v in obs.items()}

    return frames, total_reward, survived


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Single checkpoint path")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Training log directory (find latest ckpt)")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--output_dir", type=str, default="/tmp/physx_evals",
                        help="Output directory for periodic evals")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--static_pose", action="store_true", help="Use static pose instead of motion tracking")
    args = parser.parse_args()

    # Find checkpoint
    if args.ckpt:
        ckpt_path = args.ckpt
    elif args.ckpt_dir:
        ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pt")))
        if not ckpts:
            logger.error(f"No checkpoints found in {args.ckpt_dir}")
            sys.exit(1)
        ckpt_path = ckpts[-1]
    else:
        logger.error("Must provide --ckpt or --ckpt_dir")
        sys.exit(1)

    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Output path
    if args.output:
        output_path = args.output
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        output_path = os.path.join(args.output_dir, f"eval_{ckpt_name}.mp4")

    # Init physx and model
    physx_core.init_foundation()

    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = make_model(ckpt_path)

    logger.info("Creating env...")
    env = make_env(physx_core, static_pose=args.static_pose)

    logger.info(f"Running rollout (max {args.max_steps} steps)...")
    frames, total_reward, survived = render_rollout(model, env, max_steps=args.max_steps)

    logger.info(f"Result: survived={survived}/{args.max_steps}, total_reward={total_reward:.3f}")

    if frames:
        render_frames_to_video(frames, output_path)
    else:
        logger.error("No frames generated!")

    # Clean up
    env.close() if hasattr(env, "close") else None

    return 0 if survived >= 10 else 1


if __name__ == "__main__":
    sys.exit(main())
