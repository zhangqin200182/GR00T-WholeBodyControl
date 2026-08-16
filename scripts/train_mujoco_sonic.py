#!/usr/bin/env python3
"""SONIC 37M model inference in MuJoCoEnv — stability validation.

Loads the pretrained checkpoint, runs deterministic inference (act_inference),
and measures episode length, term_rate, and mean reward.
"""
import sys, os, time, logging
import numpy as np
import torch

# Patch for SONIC release checkpoint
import trl.trainer.utils
class OnlineTrainerState: pass
trl.trainer.utils.OnlineTrainerState = OnlineTrainerState

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf

from gear_sonic.envs.mujoco_env_manager import MuJoCoEnvManager
from gear_sonic.envs.mujoco_env import NUM_DOF
from gear_sonic.trl.utils.common import custom_instantiate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.environ.get(
    "SONIC_XML", os.path.join(REPO_ROOT, "gear_sonic_deploy", "g1", "g1_29dof.xml")
)
PKL = os.path.join(REPO_ROOT, "sample_data", "robot_filtered")
CKPT = os.environ.get("SONIC_CKPT", os.path.join(REPO_ROOT, "sonic_release", "last.pt"))

TOKENIZER_OBS_DIMS = {
    "encoder_index": (3,),
    "command_multi_future_nonflat": (10, 58),
    "command_z_multi_future_nonflat": (10, 1),
    "motion_anchor_ori_b_mf_nonflat": (10, 6),
    "command_multi_future_lower_body": (240,),
    "vr_3point_local_target": (9,),
    "vr_3point_local_orn_target": (12,),
    "motion_anchor_ori_b": (6,),
    "command_z": (1,),
    "smpl_joints_multi_future_local_nonflat": (10, 72),
    "smpl_root_ori_b_multi_future": (10, 6),
    "joint_pos_multi_future_wrist_for_smpl": (10, 6),
}


CONFIG_YAML = os.path.join(REPO_ROOT, "sonic_release", "config.yaml")


def build_model_from_release_config(device="cpu"):
    """Build the full SONIC Actor exactly as trained, from the released config.yaml.

    The released config references the internal NVIDIA module root ``groot.rl.``;
    rewrite it to the open-sourced ``gear_sonic.`` package (BaseModule lives at a
    different internal path; training-only targets like PPOIM are never instantiated).
    """
    with open(CONFIG_YAML) as f:
        raw = f.read()
    raw = raw.replace("groot.rl.agents.modules.modules.BaseModule",
                      "gear_sonic.trl.modules.base_module.BaseModule")
    raw = raw.replace("groot.rl.trl.", "gear_sonic.trl.")
    full_cfg = OmegaConf.create(raw)
    # Stub out Hydra runtime resolvers that only exist during training launches
    OmegaConf.register_new_resolver("hydra", lambda *a, **k: "stub", replace=True)
    OmegaConf.register_new_resolver("now", lambda *a, **k: "stub", replace=True)
    OmegaConf.resolve(full_cfg.algo.config)
    algo_cfg = full_cfg.algo.config

    env_config = OmegaConf.create({
        "obs": {
            "obs_dims": {"actor_obs": 930, "critic_obs": 1645, "tokenizer": 1761},
            "obs_dict": {},
            "group_obs_dims": {"tokenizer": TOKENIZER_OBS_DIMS},
            "group_obs_names": {"tokenizer": list(TOKENIZER_OBS_DIMS.keys())},
        },
        "robot": {
            "actions_dim": 29,
            "num_joints": 29,
            "algo_obs_dim_dict": {"actor_obs": 930, "critic_obs": 1645, "tokenizer": 1761},
        },
        "rewards": {"num_critics": 1},
        "num_envs": 1,
    })

    actor = custom_instantiate(
        algo_cfg.actor,
        env_config=env_config,
        algo_config=algo_cfg,
        _resolve=False,
    ).to(device)
    return actor, None, env_config


def build_model(device="cpu"):
    """Build SONIC 37M Actor + Critic from a manually-resolved config.

    Matches the architecture in ``config/actor_critic/universal_token/all_mlp_v1.yaml``
    with the stub_train overrides (command_z_multi_future_nonflat dropped from g1 inputs,
    joint_pos_multi_future_wrist_for_smpl added to smpl inputs).
    """
    algo_cfg = OmegaConf.create({
        "init_noise_std": 0.05,
        "use_log_std": False,
        "use_clampped_std": True,
        "std_clamp_min": 0.001,
        "std_clamp_max": 0.5,
        "actor": {
            "_target_": "gear_sonic.trl.modules.actor_critic_modules.Actor",
            "running_mean_std": False,
            "input_obs_dict": True,
            "has_aux_loss": False,
            "backbone": {
                "_target_": "gear_sonic.trl.modules.universal_token_modules.UniversalTokenModule",
                "num_future_frames": 10,
                "proprioception_features": ["actor_obs"],
                "num_fsq_levels": 32,
                "fsq_level_list": 32,
                "max_num_tokens": 2,
                "quantizer": None,
                "encoders": {
                    "g1": {
                        "inputs": [
                            "command_multi_future_nonflat",
                            "motion_anchor_ori_b_mf_nonflat",
                        ],
                        "params": {
                            "_target_": "gear_sonic.trl.modules.base_module.BaseModule",
                            "num_input_temporal_dims": 10,
                            "num_output_temporal_dims": 2,
                            "obs_dim_dict": {"actor_obs": 930, "critic_obs": 1645, "tokenizer": 1761},
                            "module_config_dict": {
                                "input_dim": [64],   # × num_input_temporal=10 → 640
                                "output_dim": [32],  # × num_output_temporal=2 → 64
                                "type": "MLP",
                                "layer_config": {
                                    "type": "MLP",
                                    "hidden_dims": [2048, 1024, 512, 512],
                                    "activation": "SiLU",
                                },
                            },
                        },
                    },
                },
                "decoders": {
                    "g1_dyn": {
                        "inputs": ["token_flattened", "proprioception"],
                        "outputs": ["action"],
                        "has_temporal_dim": False,
                        "params": {
                            "_target_": "gear_sonic.trl.modules.base_module.BaseModule",
                            "obs_dim_dict": {"actor_obs": 930, "critic_obs": 1645, "tokenizer": 1761},
                            "module_config_dict": {
                                "input_dim": [994],
                                "output_dim": [29],
                                "type": "MLP",
                                "layer_config": {
                                    "type": "MLP",
                                    "hidden_dims": [2048, 2048, 1024, 1024, 512, 512],
                                    "activation": "SiLU",
                                },
                            },
                        },
                    },
                },
            },
        },
        "critic": {
            "_target_": "gear_sonic.trl.modules.actor_critic_modules.Critic",
            "running_mean_std": True,
            "backbone": {
                "_target_": "gear_sonic.trl.modules.base_module.BaseModule",
                "process_output_dim": True,
                "module_config_dict": {
                    "type": "MLP",
                    "input_dim": ["critic_obs"],
                    "output_dim": [1],
                    "layer_config": {
                        "type": "MLP",
                        "hidden_dims": [2048, 2048, 1024, 1024, 512, 512],
                        "activation": "SiLU",
                    },
                },
            },
        },
    })

    env_config = OmegaConf.create({
        "obs": {
            "obs_dims": {"actor_obs": 930, "critic_obs": 1645, "tokenizer": 1761},
            "obs_dict": {},
            "group_obs_dims": {"tokenizer": TOKENIZER_OBS_DIMS},
            "group_obs_names": {"tokenizer": list(TOKENIZER_OBS_DIMS.keys())},
        },
        "robot": {
            "actions_dim": 29,
            "num_joints": 29,
            "algo_obs_dim_dict": {"actor_obs": 930, "critic_obs": 1645, "tokenizer": 1761},
        },
        "rewards": {"num_critics": 1},
        "num_envs": 1,
    })

    actor = custom_instantiate(
        algo_cfg.actor,
        env_config=env_config,
        algo_config=algo_cfg,
        _resolve=False,
    ).to(device)

    critic = custom_instantiate(
        algo_cfg.critic,
        env_config=env_config,
        algo_config=algo_cfg,
        _resolve=False,
    ).to(device)

    return actor, critic, env_config


def main():
    num_envs = 16
    num_steps = 500  # full episode
    logger.info(f"SONIC inference: {num_envs}e × {num_steps}s")

    # Build env (ignore_terminations=False — we want to measure real term_rate)
    env = MuJoCoEnvManager(
        num_envs=num_envs, num_workers=4,
        model_xml=XML, pkl_dir=PKL,
    )

    # Build model and load checkpoint
    device = "cpu"
    actor, _, _ = build_model_from_release_config(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)

    if "actor_model_state_dict" in ckpt:
        state_dict = ckpt["actor_model_state_dict"]
    elif "policy_state_dict" in ckpt:
        state_dict = ckpt["policy_state_dict"]
    else:
        raise KeyError("Checkpoint missing actor_model_state_dict / policy_state_dict")

    missing, unexpected = actor.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    n_params = sum(p.numel() for p in actor.parameters())
    logger.info(f"Model loaded: {n_params/1e6:.1f}M params")

    # NOTE: workers write clean post-reset obs into SHM at startup; a random-action
    # warmup step would overwrite them with post-impact obs (jvel spikes ~±190)
    # and poison the first policy input. Skip it.

    # ── Inference loop ──
    actor.eval()
    actor.init_rollout()

    obs_dict = env.reset_all()
    obs_dict = {k: v.to(device) for k, v in obs_dict.items()}

    total_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    terminations = np.zeros(num_envs, dtype=np.int32)
    _term_steps = []

    t0 = time.perf_counter()
    for step in range(num_steps):
        with torch.no_grad():
            actions = actor.act_inference(obs_dict, cur_dones=None)
        actions_np = actions.cpu().numpy()
        if step == 0:
            logger.info(
                f"Action stats @step0 (raw, pre-clip): mean={actions_np.mean():.2f} "
                f"std={actions_np.std():.2f} range=[{actions_np.min():.1f},"
                f"{actions_np.max():.1f}]"
            )
        # Isaac's action mapping (default_pos + 0.25*effort/stiffness * action) is
        # now applied inside MuJoCoEnv; raw actions pass through unclamped and the
        # resulting joint targets are clipped to joint ranges there.

        next_obs, rewards, dones, info = env.step(actions_np)
        next_obs = {k: v.to(device) for k, v in next_obs.items()}
        if step == 0:
            logger.info(
                f"Action stats @step0: mean={actions_np.mean():.2f} "
                f"std={actions_np.std():.2f} range=[{actions_np.min():.1f},"
                f"{actions_np.max():.1f}]"
            )

        total_rewards += rewards.numpy()
        episode_lengths += 1
        terminations += dones.numpy().astype(np.int32)
        if dones.any() and len(_term_steps) < 40:
            _term_steps.append((step, int(dones.sum())))

        obs_dict = next_obs

    dt = time.perf_counter() - t0

    # ── Results ──
    mean_ep_len = episode_lengths.mean()
    term_rate = terminations.mean()
    mean_reward = total_rewards.mean()

    logger.info(f"First terminations (step, n_done): {_term_steps}")
    logger.info(f"Results: reward={mean_reward:.2f} term_rate={term_rate:.3f} "
                f"ep_len={mean_ep_len:.1f} time={dt:.1f}s")
    # Per-env breakdown: separate "a few hard motions" from "uniform instability"
    for e in range(num_envs):
        logger.info(f"  env{e:02d}: total_reward={total_rewards[e]:9.1f} falls={terminations[e]}")

    if term_rate < 0.1:
        logger.info("PASS: pretrained model walks stably in MuJoCoEnv")
    elif term_rate < 0.3:
        logger.info("OK: model mostly stable, some falls")
    else:
        logger.warning("FAIL: model unstable — check obs/action alignment")

    env.close()


if __name__ == "__main__":
    main()
