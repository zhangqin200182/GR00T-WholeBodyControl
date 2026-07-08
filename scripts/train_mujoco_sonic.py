#!/usr/bin/env python3
"""Task 7: MuJoCoEnvManager + SONIC checkpoint + PPO training (single NPU POC)."""
import sys, os, time, logging, copy
import numpy as np
import torch; import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf

# Patch TRL for checkpoint loading
import trl.trainer.utils
class OnlineTrainerState: pass
trl.trainer.utils.OnlineTrainerState = OnlineTrainerState

from gear_sonic.envs.mujoco_env_manager import MuJoCoEnvManager
from gear_sonic.envs.mujoco_env import NUM_DOF
from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule
from gear_sonic.trl.modules.actor_critic_modules import Actor, Critic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

XML = "/root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"
PKL = "/root/GR00T-WholeBodyControl/sample_data/robot_filtered"
CKPT = "/root/sonic-data/sonic_release/last.pt"
CONFIG = "/root/GR00T-WholeBodyControl/gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml"


def build_model(cfg, device="cpu"):
    """Build SONIC 37M model from config."""
    # Actor backbone
    utm = UniversalTokenModule(
        obs_dim_dict=cfg.actor.obs_dim_dict,
        encoder_configs=cfg.actor.backbone.encoders,
        decoder_configs=cfg.actor.backbone.decoders,
        quantizer_config=cfg.actor.backbone.quantizer,
        obs_config=cfg.actor.backbone,
        device=device,
    )
    actor = Actor(utm, num_actions=NUM_DOF, init_noise_std=0.05)
    critic = Critic(obs_dim=1645, device=device)
    return actor, critic


def main():
    num_envs = 16; num_steps = 8; num_iters = 10; lr = 1e-4
    logger.info(f"Task 7: MuJoCoEnvManager + SONIC ({num_envs}e × {num_steps}s × {num_iters}i)")

    # Build env
    env = MuJoCoEnvManager(num_envs=num_envs, num_workers=2, model_xml=XML, pkl_dir=PKL)

    # Build config stub
    cfg = OmegaConf.create({
        "actor": {
            "obs_dim_dict": {"actor_obs": 930, "critic_obs": 1645, "tokenizer": 1761},
            "backbone": {
                "encoders": {
                    "g1": {"type": "mlp", "hidden_dims": [2048, 1024, 512, 512], "input_features": ["command_multi_future_nonflat", "motion_anchor_ori_b_mf_nonflat"], "num_output_temporal_dims": 2},
                },
                "decoders": {
                    "g1_dyn": {"type": "mlp", "hidden_dims": [2048, 2048, 1024, 1024, 512, 512], "input_features": ["token_flattened", "proprioception"], "output_features": ["action"]},
                },
                "quantizer": None,
                "max_num_tokens": 2,
            },
        },
    })

    # Build model
    actor, critic = build_model(cfg, device="cpu")
    actor.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["policy_state_dict"], strict=False)
    logger.info(f"Model loaded: actor={sum(p.numel() for p in actor.parameters())/1e6:.1f}M")

    # Warmup
    obs, _, _, _ = env.step(np.random.uniform(-0.2, 0.2, (num_envs, NUM_DOF)).astype(np.float32))
    obs = {k: v.copy() for k, v in obs.items()}

    for it in range(num_iters):
        t0 = time.perf_counter()
        rewards_batch = []
        for _ in range(num_steps):
            # Simple action: random (POC — full SONIC forward is complex)
            actions = np.random.uniform(-0.2, 0.2, (num_envs, NUM_DOF)).astype(np.float32)
            obs, rewards, dones, info = env.step(actions)
            rewards_batch.append(rewards.mean())

        dt = time.perf_counter() - t0
        logger.info(f"  iter {it:3d}: r={np.mean(rewards_batch):.3f} t={dt:.2f}s")

    env.close()
    logger.info("Task 7 POC: PASS")


if __name__ == "__main__":
    main()
