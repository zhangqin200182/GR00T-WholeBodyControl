#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fix sys.path: when running as `python gear_sonic/train_agent_trl.py`, Python adds
# gear_sonic/ to sys.path[0], causing `from trl import ...` to resolve to our local
# gear_sonic/trl/ instead of the HuggingFace trl package. Replace with repo root.
import sys
import os
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# PhysX fork safety: import PhysXEnvManager BEFORE torch/HCCL
# so child processes don't inherit Ascend runtime state.
if os.environ.get("SONIC_PHYSX_ENV"):
    from gear_sonic.envs.physx_env_manager import PhysXEnvManager  # noqa: F401

try:
    import isaaclab  # noqa: F401
except ImportError:
    if not os.environ.get("SONIC_STUB_ENV") and not os.environ.get("SONIC_MUJOCO_ENV") and not os.environ.get("SONIC_PHYSX_ENV"):
        print(
            "\n"
            "ERROR: Isaac Lab is required for training but not installed.\n"
            "\n"
            "Isaac Lab is not a pip dependency — it must be installed separately.\n"
            "Follow the official guide:\n"
            "  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html\n"
            "\n"
            "After installing, activate the Isaac Lab conda/venv environment\n"
            "before running this script.\n"
            "\n"
            "To train without Isaac Sim (stub mode), set SONIC_STUB_ENV=1\n"
        )
        sys.exit(1)

import glob
import logging
import os
from pathlib import Path
import re
import sys

from filelock import FileLock
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import wandb
import yaml

from gear_sonic.trl.utils.common import (
    custom_instantiate,
    get_filtered_state_dict,
    materialize_lazy_params,
    wandb_run_exists,
)
from gear_sonic.utils.common import seeding
from gear_sonic.utils.config_utils import register_rl_resolvers
from gear_sonic.utils.obs_utils import get_group_term_obs_shape

register_rl_resolvers()


def resume_training(config):
    if config.get("checkpoint", None) is not None:
        last_existing_checkpoint = config.checkpoint
    elif config.get("experiment_dir", None) is not None:
        last_existing_checkpoint = os.path.join(config.experiment_dir, "last.pt")
    else:
        # Use experiment_dir to find the checkpoint, rather than reconstructing
        # from config.project_name which can differ from the actual filesystem path.
        experiment_dir_base = re.sub(r"-\d{8}_\d{6}$", "", config.experiment_dir)
        checkpoints = sorted(glob.glob(os.path.join(f"{experiment_dir_base}-*", "last.pt")))
        if not checkpoints:
            print(f"No checkpoint found matching {experiment_dir_base}-*/last.pt, starting fresh")
            return
        last_existing_checkpoint = checkpoints[-1]
    experiment_dir = os.path.dirname(last_existing_checkpoint)
    config.experiment_dir = experiment_dir
    config.checkpoint = last_existing_checkpoint
    print(f"Resuming training from {last_existing_checkpoint}")


def resume_checkpoint(config):
    config.checkpoint = config.checkpoint


def create_manager_env(config, device, args_cli):

    # import wandb

    from isaaclab.envs import (
        ManagerBasedRLEnv,
    )

    from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper

    env_instance_cfg = custom_instantiate(config.manager_env)

    # Iteratively check the difference in attribute of env_instance_cfg1 and env_instance_cfg, print out the difference
    def compare_attrs(obj1, obj2, prefix=""):
        # Only compare attributes that do not start with '__' and are not methods
        attrs1 = set(dir(obj1))
        attrs2 = set(dir(obj2))
        common_attrs = attrs1 & attrs2
        for attr in sorted(common_attrs):
            if (
                attr.startswith("__")
                or callable(getattr(obj1, attr))
                or callable(getattr(obj2, attr))
            ):
                continue
            try:
                val1 = getattr(obj1, attr)
                val2 = getattr(obj2, attr)
            except Exception:
                continue
            # Recursively compare if both are objects with __dict__ or are dicts
            if isinstance(val1, dict | DictConfig) and isinstance(val2, dict | DictConfig):
                compare_attrs(val1, val2, prefix + attr + ".")
            elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
                compare_attrs(val1, val2, prefix + attr + ".")
            else:
                if isinstance(val1, list):
                    val1 = tuple(val1)
                if isinstance(val2, list):
                    val2 = tuple(val2)
                if val1 != val2:
                    print(
                        f"\nDifference found at '{prefix}{attr}':\n"
                        f"  - env_instance_cfg1: {val1!r}\n"
                        f"  - env_instance_cfg : {val2!r}\n"
                    )

    env_instance_cfg.seed = config.seed
    env_instance_cfg.sim.device = device
    env_instance_cfg.config["headless"] = args_cli.headless
    env = ManagerBasedRLEnv(
        cfg=env_instance_cfg, render_mode="rgb_array" if not args_cli.headless else None
    )

    env = ManagerEnvWrapper(env, env_instance_cfg.config)
    return env


@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    if os.environ.get("SONIC_MUJOCO_ENV"): simulator_type = "MuJoCo"
    elif os.environ.get("SONIC_STUB_ENV"): simulator_type = "Stub"
    elif os.environ.get("SONIC_PHYSX_ENV"): simulator_type = "PhysX"
    else: simulator_type = "IsaacSim"
    env_config = config.manager_env
    from transformers import HfArgumentParser
    from trl import ModelConfig, PPOConfig, ScriptArguments

    # Setup model components
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))

    if config.get("resume", False):
        resume_training(config)
    elif config.get("checkpoint", None) is not None:
        resume_checkpoint(config)

    config.algo.trl.output_dir = str(Path(config.experiment_dir))

    script_args, training_args, model_args = parser.parse_dict(config.algo.trl)

    # Add exp_name from main config to training_args
    training_args.exp_name = config.experiment_name

    from datetime import timedelta

    from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
    import torch  # noqa: E402

    # NPU detection — must happen before Accelerator creation
    _is_npu = False
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            _is_npu = True
            training_args.bf16 = False
            training_args.fp16 = False
            logger.info("NPU detected — forcing fp32 (NPU doesn't support bf16 torch.normal)")
    except ImportError:
        pass

    _mixed_precision = "no" if _is_npu else None
    # When encoders or decoders are frozen (ENC_ADAPT / BC_ONLY), DDP must
    # tolerate parameters that produce no gradient in the forward/backward pass.
    _freeze_parts = bool(os.environ.get("SONIC_PHYSX_ENC_ADAPT") or os.environ.get("SONIC_PHYSX_BC_ONLY") or os.environ.get("SONIC_PHYSX_FREEZE_ENCODER"))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=_freeze_parts)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=6000))
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs, kwargs],
        mixed_precision=_mixed_precision,
    )

    device = str(accelerator.device)
    if device == "cuda":
        device = "cuda:0"

    if _is_npu:
        device = f"npu:{torch.npu.current_device()}"
        logger.info(f"Using NPU device: {device}")

    _use_stub_env = bool(os.environ.get("SONIC_STUB_ENV"))
    config.multi_gpu = accelerator.num_processes > 1
    if config.multi_gpu:
        config.global_rank = accelerator.process_index
        config.seed += accelerator.process_index
        config.algo.config.global_rank = accelerator.process_index
        config.algo.config.world_size = accelerator.num_processes
    seeding(config.seed)

    meta_path = Path(config.experiment_dir) / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(open(meta_path))
        config.wandb.wandb_id = meta["wandb_run"]
        print(f"resume wandb from run: {config.wandb.wandb_id}")

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    if config.use_wandb and accelerator.is_main_process:
        project_name = f"{config.project_name}"
        run_name = config.experiment_dir.replace(f"{config.base_dir}/{project_name}/", "")
        wandb_dir = Path(config.wandb.wandb_dir)
        wandb_dir.mkdir(exist_ok=True, parents=True)
        wandb_group = None if config.wandb.wandb_id is not None else config.wandb.wandb_group
        logger.info(f"Saving wandb logs to {wandb_dir}")
        wandb.init(
            project=project_name,
            entity=config.wandb.wandb_entity,
            name=run_name,
            sync_tensorboard=True,
            config=unresolved_conf,
            dir=wandb_dir,
            id=config.wandb.wandb_id,
            group=wandb_group,
            resume="allow",
        )

    # Setup simulator similar to train_agent.py

    if simulator_type == "IsaacSim":
        try:
            with open("./rl/simulator/isaacsim/.isaacsim_version", encoding="utf-8") as f:
                DEFAULT_ISAACSIM_VERSION = f.read().strip()
        except FileNotFoundError:
            DEFAULT_ISAACSIM_VERSION = "4.5"

        if DEFAULT_ISAACSIM_VERSION == "4.5":
            from isaaclab.app import AppLauncher
        elif DEFAULT_ISAACSIM_VERSION == "4.2":
            logger.warning("Using IsaacSim 4.2, replacing isaaclab with omni.isaac.lab")
            from omni.isaac.lab.app import AppLauncher  # 4.2

            # from isaaclab.app import AppLauncher # not working
            # from omni.isaac.lab.app import AppLauncher

        import argparse

        parser = argparse.ArgumentParser(description="Train an RL agent with TRL.")
        AppLauncher.add_app_launcher_args(parser)

        ######################################################### ZL: fix isaacsim 4.5 rendering #########################################################
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = env_config.config.env_spacing  # config.env_spacing
        args_cli.output_dir = config.output_dir
        # Enable cameras if enable_cameras, render_results, render_ego, or overview_camera is True
        args_cli.enable_cameras = (
            env_config.config.get("enable_cameras", False)
            or env_config.config.get("render_results", False)
            or env_config.config.get("render_ego", False)
            or env_config.config.get("overview_camera", False)
        )
        args_cli.headless = config.headless
        args_cli.multi_gpu = config.multi_gpu
        args_cli.distributed = config.multi_gpu
        args_cli.device = device

        # Base kit args (quiet logs)
        args_cli.kit_args = (
            "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
        )

        # AppLauncher can't handle multiple processes creating it at the same time so we need a lock
        _lock_path = "/tmp/isaaclab_app_launcher.lock"
        _local_rank = int(os.environ.get("LOCAL_RANK", 0))
        with FileLock(_lock_path):
            app_launcher = AppLauncher(args_cli)

        simulation_app = app_launcher.app

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

    from gear_sonic.utils.logging import HydraLoggerBridge

    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    # Setup wandb if enabled
    os.chdir(hydra.utils.get_original_cwd())

    # Save config and meta BEFORE env creation so eval jobs can postprocess
    # checkpoint configs even if training crashes during env init.
    experiment_save_dir = Path(config.experiment_dir)
    if accelerator.is_main_process:
        experiment_save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving config file to {experiment_save_dir}")
        with open(experiment_save_dir / "config.yaml", "w") as file:
            OmegaConf.save(unresolved_conf, file)
        meta = {"wandb_run": wandb.run.id if wandb_run_exists() else None}
        meta["max_train_steps"] = config.algo.config.num_learning_iterations
        yaml.safe_dump(meta, open(meta_path, "w"))
        print("saved meta:", meta)

    # Initialize environment
    env_config.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_training")
    env_config.config.experiment_dir = str(Path(config.experiment_dir))

    # SONIC tokenizer observation dimensions (12 sub-fields → 1761D total)
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

    if os.environ.get("SONIC_MUJOCO_ENV"):
        from gear_sonic.envs.mujoco_env_manager import MuJoCoEnvManager
        logger.info("Using MuJoCoEnvManager (CPU physics)")
        # MuJoCo-specific exploration/noise adaptation (previously leaked into
        # the global ppo_im_phc.yaml / sonic_release.yaml defaults).
        OmegaConf.update(config.algo.config, "init_noise_std", 0.15, force_add=True)
        OmegaConf.update(config.algo.config, "std_clamp_min", 0.05, force_add=True)
        OmegaConf.update(config.algo.config, "std_clamp_max", 1.0, force_add=True)
        # DDP: each rank gets num_envs / world_size envs
        local_envs = config.num_envs // accelerator.num_processes
        env = MuJoCoEnvManager(
            num_envs=local_envs,
            num_workers=getattr(config, "mujoco_workers", 160) // accelerator.num_processes,
            model_xml="/gear_sonic_deploy/g1/g1_29dof_v17.xml",
            pkl_dir="/sample_data/robot_filtered",
            env_config=OmegaConf.create({"alive_bonus": 0.0}),
        )
        # Build config with all keys the model init needs
        config_dict = OmegaConf.to_container(env_config, resolve=True)
        config_dict.setdefault("obs", {}).setdefault("obs_dims", {})
        config_dict.setdefault("obs", {})["obs_dict"] = {}
        config_dict["obs"]["obs_dims"] = {"actor_obs": 930, "critic_obs": 1645, "tokenizer": 1761}
        config_dict["obs"]["group_obs_dims"] = {"tokenizer": TOKENIZER_OBS_DIMS}
        config_dict["obs"]["group_obs_names"] = {"tokenizer": list(TOKENIZER_OBS_DIMS.keys())}
        config_dict.setdefault("robot", {})["actions_dim"] = 29
        config_dict["num_envs"] = config.num_envs
        config_dict.setdefault("robot", {}).setdefault("algo_obs_dim_dict", {})
        # Convert back to OmegaConf — now all keys exist, struct mode is satisfied
        env.config = OmegaConf.create(config_dict, flags={"allow_objects": True})
    elif os.environ.get("SONIC_PHYSX_ENV"):
        from gear_sonic.envs.physx_env_manager import PhysXEnvManager
        logger.info("Using PhysXEnvManager (PhysX 5 Direct API CPU physics)")
        local_envs = config.num_envs // accelerator.num_processes
        # Termination mode:
        #   skip_termination: episode never ends (env._check_termination → False)
        #     Used for encoder adaptation — bad for PPO (GAE bootstrap broken)
        #   ignore_terminations: termination triggers env reset, but trainer
        #     sees done=False. Finite episodes → GAE bootstraps normally.
        #     Used for BC warmup / joint training — policy survives rollout
        #     boundaries while still getting bounded-episode advantages.
        is_enc_adapt = bool(os.environ.get("SONIC_PHYSX_ENC_ADAPT"))
        skip_term = bool(os.environ.get("SONIC_PHYSX_SKIP_TERM"))
        ignore_term = bool(os.environ.get("SONIC_PHYSX_IGNORE_TERM"))
        env_config_dict = {"alive_bonus": 0.0}
        # Gradual threshold tightening: overridable via env vars
        if os.environ.get("SONIC_PHYSX_ORI_THRESH"):
            env_config_dict["ori_thresh"] = float(os.environ["SONIC_PHYSX_ORI_THRESH"])
        if os.environ.get("SONIC_PHYSX_ANK_POS_THRESH"):
            env_config_dict["ank_pos_thresh"] = float(os.environ["SONIC_PHYSX_ANK_POS_THRESH"])
        if os.environ.get("SONIC_PHYSX_ANK_H_MULT"):
            env_config_dict["ank_h_mult"] = float(os.environ["SONIC_PHYSX_ANK_H_MULT"])
        if os.environ.get("SONIC_PHYSX_ACTION_TRUST"):
            env_config_dict["action_trust"] = float(os.environ["SONIC_PHYSX_ACTION_TRUST"])
        if os.environ.get("SONIC_PHYSX_HEIGHT_HINGE"):
            env_config_dict["height_hinge_weight"] = float(os.environ["SONIC_PHYSX_HEIGHT_HINGE"])
        if os.environ.get("SONIC_PHYSX_ANKLE_HINGE"):
            env_config_dict["ankle_hinge_weight"] = float(os.environ["SONIC_PHYSX_ANKLE_HINGE"])
        # Root height offset: with the corrected link frames the mocap foot
        # hover (25-35mm) would start the foot boxes inside the ground and
        # eject the robot.  0.04 is the scanned optimum (P0 fix, 08-14).
        root_z_off = float(os.environ.get("SONIC_PHYSX_ROOT_Z_OFFSET", "0.04"))
        if is_enc_adapt or skip_term:
            env_config_dict["skip_termination"] = True
            logger.info(f"PhysX skip_termination=True (enc_adapt={is_enc_adapt}, skip_term={skip_term})")
        elif ignore_term:
            env_config_dict["ignore_terminations"] = True
            logger.info("PhysX ignore_terminations=True — env resets, trainer ignores done")
        logger.info(f"PhysX thresholds: ori={env_config_dict.get('ori_thresh', 0.2)}, "
                    f"ank_pos={env_config_dict.get('ank_pos_thresh', 0.2)}, "
                    f"ank_h_mult={env_config_dict.get('ank_h_mult', 1.0)}, "
                    f"action_trust={env_config_dict.get('action_trust', 1.0)}, "
                    f"height_hinge={env_config_dict.get('height_hinge_weight', 0.0)}")
        env = PhysXEnvManager(
            num_envs=local_envs,
            num_workers=int(os.environ.get("SONIC_PHYSX_WORKERS", "1024")) // accelerator.num_processes,
            model_xml="/gear_sonic_deploy/g1/g1_29dof_v17.xml",
            pkl_dir="/sample_data/robot_filtered",
            env_config=OmegaConf.create(env_config_dict),
            root_z_offset=root_z_off,
        )
        # Build config with all keys the model init needs
        config_dict = OmegaConf.to_container(env_config, resolve=True)
        config_dict.setdefault("obs", {}).setdefault("obs_dims", {})
        config_dict.setdefault("obs", {})["obs_dict"] = {}
        config_dict["obs"]["obs_dims"] = {"actor_obs": 930, "critic_obs": 1645, "tokenizer": 1761, "ref_action": 29}
        config_dict["obs"]["group_obs_dims"] = {"tokenizer": TOKENIZER_OBS_DIMS}
        config_dict["obs"]["group_obs_names"] = {"tokenizer": list(TOKENIZER_OBS_DIMS.keys())}
        config_dict.setdefault("robot", {})["actions_dim"] = 29
        config_dict["num_envs"] = config.num_envs
        config_dict.setdefault("robot", {}).setdefault("algo_obs_dim_dict", {})
        env.config = OmegaConf.create(config_dict, flags={"allow_objects": True})
        # BC loss: MSE(policy action_mean, reference qpos) — direct behavioral supervision for decoder
        if os.environ.get("SONIC_PHYSX_BC_COEF"):
            bc_coef = float(os.environ["SONIC_PHYSX_BC_COEF"])
            OmegaConf.update(config.algo.config, "compute_bc_loss", True, force_add=True)
            OmegaConf.update(config.algo.config, "bc_loss_coef", bc_coef, force_add=True)
            logger.info(f"PhysX BC loss ENABLED: coef={bc_coef}")
    elif _use_stub_env:
        from gear_sonic.envs.stub_env import StubEnv
        logger.info("Using StubEnv (no physics simulation)")
        env = StubEnv(config, env_config, device)
    else:
        env = create_manager_env(config, device, args_cli)
    if config.get("replay", False):
        _save_video_path = config.get("replay_save_video", None)
        env.run_replay(
            start_time_step=-1,
            loop=config.get("replay_loop_num", True),
            save_video_path=_save_video_path,
            grid_spacing=config.get("replay_grid_spacing", 2.0),
        )
        os._exit(0)
    if config.get("vplanner_replay", False):
        vplanner_checkpoint = config.get("vplanner_checkpoint", None)
        if vplanner_checkpoint is None:
            raise ValueError("vplanner_checkpoint must be specified for vplanner_replay")
        env.run_vplanner_replay(
            checkpoint_path=vplanner_checkpoint,
            max_frames=config.get("vplanner_max_frames", 500),
            replan_interval=config.get("vplanner_replan_interval", 0),
            speed=config.get("vplanner_speed", 1.0),
            loop=config.get("vplanner_loop", True),
            save_images=config.get("vplanner_save_images", False),
            output_dir=config.get("vplanner_output_dir", None),
            dof_noise=config.get("vplanner_dof_noise", 0.0),
            dof_vel_noise=config.get("vplanner_dof_vel_noise", 0.0),
            quat_noise=config.get("vplanner_quat_noise", 0.0),
        )
        os._exit(0)

    ref_model = None
    value_model = None
    disc_model = None
    # import ipdb; ipdb.set_trace()

    if config.algo.config.get("use_new_actor_critic", False):
        module_dim_dict = getattr(config.algo.config, "module_dim", {})
        policy_backbone_kwargs = {}
        critic_backbone_kwargs = {}
        # Populate env.config from observation_space (works for Isaac Sim, StubEnv, MuJoCo)
        env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
        env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
        env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space[
            "policy"
        ].shape[-1]
        env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space[
            "critic"
        ].shape[-1]
        if os.environ.get("SONIC_MUJOCO_ENV") or os.environ.get("SONIC_PHYSX_ENV"):
            # MuJoCo/PhysX: no group obs (tokenizer is flat 1761D)
            env.config["obs"]["obs_dims"]["tokenizer"] = 1761
            env.config["robot"]["algo_obs_dim_dict"]["tokenizer"] = 1761
            env.config["robot"]["actions_dim"] = 29
            if os.environ.get("SONIC_PHYSX_ENV"):
                # ref_action for BC loss: MSE(policy action_mean, reference qpos)
                env.config["obs"]["obs_dims"]["ref_action"] = 29
                env.config["robot"]["algo_obs_dim_dict"]["ref_action"] = 29
        else:
            example_obs = env.reset(flatten_dict_obs=False)
            for key in env.env.observation_space:
                if key not in ["policy", "critic"]:
                    group_obs_dims, group_obs_names, group_obs_total_dim = get_group_term_obs_shape(
                        example_obs, key
                    )
                    env.config["obs"]["group_obs_dims"][key] = group_obs_dims
                    env.config["obs"]["group_obs_names"][key] = group_obs_names
                    env.config["obs"]["obs_dims"][key] = group_obs_total_dim
                    env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim
            if config.manager_env.config.get("meta_action_dim", None) is not None:
                env.config["robot"]["actions_dim"] = config.manager_env.config.meta_action_dim
            else:
                env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]

        # --- PhysX g1_recon coef override (general, independent of mode) ---
        if os.environ.get("SONIC_PHYSX_G1_RECON_COEF"):
            g1_recon_coef = float(os.environ["SONIC_PHYSX_G1_RECON_COEF"])
            OmegaConf.update(config.algo.config, "actor.backbone.aux_loss_coef.g1_recon", g1_recon_coef, force_add=True)
            logger.info(f"PhysX g1_recon_coef override: {g1_recon_coef}")

        # --- PhysX encoder adaptation (Step 1): freeze decoder, max g1_recon ---
        if os.environ.get("SONIC_PHYSX_ENC_ADAPT"):
            OmegaConf.update(config.algo.config, "actor.backbone.aux_loss_coef.g1_recon", 1.0, force_add=True)
            OmegaConf.update(config.algo.config, "ppo_loss_coef", 0, force_add=True)
            policy_backbone_kwargs["freeze_decoders"] = True
            logger.info("PhysX encoder adaptation ENABLED: g1_recon->1.0, ppo_loss_coef=0, freeze_decoders=True")

        # --- PhysX BC-only (Step 2a): freeze encoder, decoder learns via BC loss ---
        # Action noise must be tiny — large noise causes immediate termination even
        # though action_mean matches reference qpos (BC loss ~0.005).
        if os.environ.get("SONIC_PHYSX_BC_ONLY"):
            OmegaConf.update(config.algo.config, "ppo_loss_coef", 0, force_add=True)
            OmegaConf.update(config.algo.config, "init_noise_std", 0.01, force_add=True)
            OmegaConf.update(config.algo.config, "deterministic_rollout", True, force_add=True)
            policy_backbone_kwargs["freeze_encoders"] = True
            logger.info("PhysX BC-only ENABLED: ppo=0, deterministic_rollout=True, init_noise_std=0.01, freeze_encoders=True")

        # --- PhysX PPO noise control: lower init noise + freeze + clamp ---
        # Without this, PPO learns to inflate action_std (entropy bonus) and
        # per-step tracking reward degrades.
        #
        # Controlled by SONIC_PHYSX_NOISE_CTL=1 (default: on for non-BC_ONLY)
        # Disable with SONIC_PHYSX_NOISE_CTL=0 to match pre-control code path.
        #
        # Env var tuning knobs (when enabled):
        #   SONIC_PHYSX_NOISE_INIT     — init_noise_std (default 0.05)
        #   SONIC_PHYSX_MAX_NOISE_STD  — clamp ceiling (default 0.10)
        #   SONIC_PHYSX_ENTROPY_COEF   — override entropy_coef (default: unset)
        _noise_ctl = os.environ.get("SONIC_PHYSX_NOISE_CTL", "1")
        if (not os.environ.get("SONIC_PHYSX_BC_ONLY") and _noise_ctl == "1"):
            noise_init = float(os.environ.get("SONIC_PHYSX_NOISE_INIT", "0.05"))
            max_noise = float(os.environ.get("SONIC_PHYSX_MAX_NOISE_STD", "0.10"))
            OmegaConf.update(config.algo.config, "init_noise_std", noise_init, force_add=True)
            OmegaConf.update(config.algo.config, "freeze_noise_std", True, force_add=True)
            OmegaConf.update(config.algo.config, "clamp_noise_std", True, force_add=True)
            OmegaConf.update(config.algo.config, "max_noise_std", max_noise, force_add=True)
            if os.environ.get("SONIC_PHYSX_ENTROPY_COEF"):
                ent_coef = float(os.environ["SONIC_PHYSX_ENTROPY_COEF"])
                OmegaConf.update(config.algo.config, "entropy_coef", ent_coef, force_add=True)
                logger.info(f"PhysX entropy_coef override: {ent_coef}")
            logger.info(f"PhysX PPO noise control: init_noise_std={noise_init}, "
                        f"freeze_noise_std=True, clamp_noise_std=True, max_noise_std={max_noise}")

        # --- PhysX freeze encoder (generic, e.g. to protect encoder during PPO) ---
        if os.environ.get("SONIC_PHYSX_FREEZE_ENCODER"):
            policy_backbone_kwargs["freeze_encoders"] = True
            logger.info("PhysX freeze_encoders=True (generic)")

        policy = custom_instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            backbone_kwargs=policy_backbone_kwargs,
            _resolve=False,
        ).to(device)

        if getattr(config.algo.config, "use_dagger", False):
            # Get teacher input key from config or default to "teacher"
            teacher_input_key = config.algo.config.get("teacher_input_key", "teacher")
            ref_model = custom_instantiate(
                config.algo.config.teacher_actor,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _resolve=False,
                input_key=teacher_input_key,
            ).to(device)
        if not getattr(config.algo.config, "distill_only", False):
            value_model = custom_instantiate(
                config.algo.config.critic,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                backbone_kwargs=critic_backbone_kwargs,
                _resolve=False,
            ).to(device)
        if config.algo.config.get("use_amp", False):
            disc_model = custom_instantiate(
                config.algo.config.disc,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _resolve=False,
            ).to(device)
    else:
        raise ValueError("No longer supported")

    materialize_lazy_params(policy, env)

    # PhysX BC checkpoint loading (env-var triggered)
    # Load policy weights from a checkpoint saved by a previous BC/PPO run.
    # Triggered by SONIC_PHYSX_BC_CHECKPOINT=/path/to/checkpoint.pt
    _bc_ckpt = os.environ.get("SONIC_PHYSX_BC_CHECKPOINT", "")
    if _bc_ckpt:
        import torch as _torch
        _sd = _torch.load(_bc_ckpt, map_location=device, weights_only=False)
        # Fix (08-14): checkpoints store the actor under "policy_state_dict"
        # (same key physx_cross_eval.py reads).  The old "policy" lookup fell
        # back to the whole checkpoint dict — keys never matched, the BC
        # weights were silently skipped (missing=55) and PPO started from the
        # untrained SONIC release weights (near-zero actions, step-1 death).
        _policy_sd = (_sd.get("actor_model_state_dict")
                      or _sd.get("policy_state_dict")
                      or _sd.get("policy", _sd))
        missing, unexpected = policy.load_state_dict(_policy_sd, strict=False)
        logger.info(f"PhysX BC checkpoint loaded from {_bc_ckpt}: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            logger.info(f"  Missing keys: {missing[:5]}...")
        if unexpected:
            logger.info(f"  Unexpected keys: {unexpected[:5]}...")
        # Fail-fast: our checkpoints are same-architecture, so a healthy load
        # has zero missing/unexpected keys.  A silent partial load is exactly
        # the failure mode that caused the step-1-death bug.
        if missing or unexpected:
            raise RuntimeError(
                f"PhysX BC checkpoint load mismatch: missing={len(missing)}, "
                f"unexpected={len(unexpected)}. Checkpoint: {_bc_ckpt}"
            )

    if config.algo.config.get("pretrained_model", None) is not None:
        pretrained_cfg = config.algo.config.pretrained_model
        sd_key = pretrained_cfg.get("state_dict_key", "state_dict")
        strict = pretrained_cfg.get("strict", True)
        state_dict = torch.load(pretrained_cfg.path, map_location=device, weights_only=False)[
            sd_key
        ]
        for (
            module_name,
            state_dict_key,
        ) in pretrained_cfg.module_mapping.items():
            module = eval(module_name)
            filtered_state_dict = get_filtered_state_dict(state_dict, state_dict_key)
            missing, unexpected = module.load_state_dict(filtered_state_dict, strict=strict)
            if missing:
                logger.info(f"Pretrained loading '{module_name}': missing keys: {missing}")
            if unexpected:
                logger.info(f"Pretrained loading '{module_name}': unexpected keys: {unexpected}")

    # ── v14/v14b experimental blocks REMOVED (08-14) ─────────────────
    # g1_dyn_reinit (hardcoded layer indices) and g1_dyn_freeze_backbone
    # were v14/v14b experiments; both routes were terminated (see
    # v14-postmortem-v14b-design).  Removed per pipeline audit.

    accelerator.wait_for_everyone()

    callbacks = []
    for callback in config.callbacks.values():
        callbacks.append(instantiate(callback))

    ################
    # Training
    ################
    trainer = custom_instantiate(
        config.trainer,
        args=training_args,
        config=config.algo.config,
        env=env,
        model=policy,
        disc_model=disc_model,
        value_model=value_model,
        ref_model=ref_model,
        use_ref_model=getattr(config.algo.config, "use_dagger", False),
        train_dataset=None,
        eval_dataset=None,
        callbacks=callbacks,
        checkpoint=config.checkpoint,
        resume=config.get("resume", False),
        local_seed=config.seed,
        log_dir=experiment_save_dir,
        accelerator=accelerator,
        _resolve=False,
    )

    # PhysX BC-only: reset action noise AFTER checkpoint load (which restores 0.38)
    if os.environ.get("SONIC_PHYSX_BC_ONLY"):
        unwrapped = accelerator.unwrap_model(trainer.model)
        actor = unwrapped.policy
        if hasattr(actor, "log_std"):
            with torch.no_grad():
                actor.log_std.data = torch.log(0.01 * torch.ones_like(actor.log_std))
                actor.log_std.requires_grad = False
            logger.info("PhysX BC-only: log_std reset to log(0.01) (after checkpoint load)")
        elif hasattr(actor, "std"):
            with torch.no_grad():
                actor.std.data.fill_(0.01)
                actor.std.requires_grad = False
            logger.info("PhysX BC-only: std reset to 0.01 (after checkpoint load)")
        else:
            logger.warning("PhysX BC-only: no std/log_std on actor — noise NOT reset")

    # Training loop
    trainer.train()

    if simulator_type == "IsaacSim":
        os._exit(0)


if __name__ == "__main__":

    main()
