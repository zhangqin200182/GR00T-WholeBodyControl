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

try:
    import isaaclab  # noqa: F401
except ImportError:
    print(
        "\n"
        "ERROR: Isaac Lab is required for evaluation but not installed.\n"
        "\n"
        "Isaac Lab is not a pip dependency — it must be installed separately.\n"
        "Follow the official guide:\n"
        "  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html\n"
        "\n"
        "After installing, activate the Isaac Lab conda/venv environment\n"
        "before running this script.\n"
    )
    import sys
    sys.exit(1)

import filelock  # noqa: I001
import json
import os
import shutil
import subprocess
import sys

sys.path.append(os.getcwd())
import logging
from pathlib import Path

import easydict
import hydra
from hydra import utils
from hydra.core import hydra_config
from loguru import logger
import omegaconf
import yaml

from gear_sonic import train_agent_trl
from gear_sonic.trl.utils import common as trl_utils_common
from gear_sonic.trl.utils import scheduler
from gear_sonic.utils import common as rl_utils_common
from gear_sonic.utils import config_utils, obs_utils

config_utils.register_rl_resolvers()


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: omegaconf.OmegaConf):

    hydra_log_path = os.path.join(hydra_config.HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    from gear_sonic.utils import logging as utils_logging

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(utils_logging.HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                raw = file.read()
            # Backward compatibility: rewrite internal repo module paths to release repo paths
            raw = raw.replace("groot.rl.trl.", "gear_sonic.trl.")
            raw = raw.replace("groot.rl.envs.", "gear_sonic.envs.")
            raw = raw.replace("groot.rl.utils.", "gear_sonic.utils.")
            raw = raw.replace("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module.")
            raw = raw.replace("groot.rl.agents.", "gear_sonic.trl.")
            raw = raw.replace("groot/rl/data/", "gear_sonic/data/")
            raw = raw.replace("assets/bm/unitree_description/", "assets/robot_description/")
            raw = raw.replace("1215_bones_seed_filtered", "bones_seed_smpl")
            import io
            train_config = omegaconf.OmegaConf.load(io.StringIO(raw))

            if train_config.eval_overrides is not None:
                train_config = omegaconf.OmegaConf.merge(train_config, train_config.eval_overrides)

            config = omegaconf.OmegaConf.merge(train_config, override_config)
        else:
            config = override_config

        config.experiment_dir = checkpoint.parent
    elif override_config.eval_overrides is not None:
        config = override_config.copy()
        eval_overrides = omegaconf.OmegaConf.to_container(config.eval_overrides, resolve=True)
        for arg in sys.argv[1:]:
            if not arg.startswith("+"):
                key = arg.split("=")[0]
                if key in eval_overrides:
                    del eval_overrides[key]
        config.eval_overrides = omegaconf.OmegaConf.create(eval_overrides)
        config = omegaconf.OmegaConf.merge(config, eval_overrides)
    else:
        config = override_config

    meta_path = Path(config.experiment_dir) / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(open(meta_path))  # noqa: SIM115
        if config.get("wandb", None) is not None and meta.get("wandb_run"):
            config.wandb.wandb_id = meta["wandb_run"]
            print(f"resume wandb from run: {config.wandb.wandb_id}")  # noqa: T201

    # ===== [COLLECT] P0 baseline protocol: extend episode, empty recorders =====
    with omegaconf.open_dict(config):
        try:
            config.manager_env.env.episode_length_s = 12.0
        except Exception as _e:
            print(f"[COLLECT] episode_length_s override skipped: {_e}")
        print("[COLLECT] recorders will use empty config (no render buffers)")
        print("[COLLECT] terminations will be neutralized at runtime (not config level)")
    # ===== [/COLLECT] config =====

    with omegaconf.open_dict(config):
        for event in config.manager_env.config.get("train_only_events", []):
            if event in config.manager_env.events:
                config.manager_env.events.pop(event)
            remove_schedule_keys = []
            for key in config.trainer.get("schedule_dict", {}):
                if event in key:
                    remove_schedule_keys.append(key)
            for key in remove_schedule_keys:
                config.trainer.schedule_dict.pop(key)

        for termination in config.manager_env.config.get("train_only_terminations", []):
            if termination in config.manager_env.terminations:
                config.manager_env.terminations.pop(termination)

    use_encoder = config.get("use_encoder", None)
    if use_encoder is not None:
        encoder_sample_probs = config.manager_env.commands.motion.encoder_sample_probs
        if encoder_sample_probs is not None:
            for encoder in encoder_sample_probs:
                if encoder != use_encoder:
                    encoder_sample_probs[encoder] = 0.0
            print(f"Using encoder: {use_encoder}")  # noqa: T201
            print(f"Encoder sample probs: {encoder_sample_probs}")  # noqa: T201

    simulator_type = "IsaacSim"
    env_config = config.manager_env

    import datetime as dt

    import accelerate
    import torch  # noqa: E402, RUF100

    kwargs = accelerate.InitProcessGroupKwargs(timeout=dt.timedelta(seconds=6000))
    accelerator = accelerate.Accelerator(kwargs_handlers=[kwargs])

    device = str(accelerator.device)
    if accelerator.device.type == "cuda":
        try:
            torch.cuda.set_device(accelerator.local_process_index)
        except Exception:  # noqa: S110, BLE001
            pass

    device = str(accelerator.device)
    config.multi_gpu = accelerator.num_processes > 1
    if config.multi_gpu:
        config.global_rank = accelerator.process_index
        config.seed += accelerator.process_index
        config.algo.config.global_rank = accelerator.process_index
        config.algo.config.world_size = accelerator.num_processes
    rl_utils_common.seeding(config.seed)

    def _pick_display_gpu_index(default_idx: int = 0) -> int:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,display_active,name", "--format=csv,noheader"],
                text=True,
            )
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    idx, active = int(parts[0]), parts[1].lower()
                    if active.startswith("enabled") or active.startswith("on"):
                        return idx
        except Exception:  # noqa: S110, BLE001
            pass
        return default_idx

    render_gpu_idx = _pick_display_gpu_index(default_idx=0)

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
        import argparse

        parser = argparse.ArgumentParser(description="Evaluate an RL agent with TRL.")
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args  # noqa: RUF005
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = env_config.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.enable_cameras = env_config.config.get(
            "render_results", False
        ) or env_config.config.get("enable_cameras", False)

        args_cli.headless = config.headless
        args_cli.multi_gpu = config.multi_gpu
        args_cli.distributed = config.multi_gpu
        args_cli.device = device

        base_kit_args = (
            "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
        )
        if args_cli.headless:
            args_cli.kit_args = base_kit_args + " --no-window"
        else:
            args_cli.kit_args = base_kit_args + f" --/renderer/activeGpu={render_gpu_idx}"

        _lock_path = "/tmp/isaaclab_app_launcher.lock"  # noqa: S108
        with filelock.FileLock(_lock_path):
            app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app  # noqa: F841
        # Enable the URDF importer extension (required by UrdfFileCfg spawners)
        import omni.kit.app
        _ext_mgr = omni.kit.app.get_app_interface().get_extension_manager()
        _ext_mgr.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    unresolved_conf = omegaconf.OmegaConf.to_container(config, resolve=False)  # noqa: F841
    os.chdir(hydra.utils.get_original_cwd())

    ckpt_num = config.checkpoint.split("/")[-1].split("_")[-1].split(".")[0]

    if env_config.config.get("save_rendering_dir", None) is None:
        env_config.config.save_rendering_dir = str(
            checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}"
        )

    metrics_file = config.get("metrics_file", None)
    if metrics_file is not None:
        metrics_file = Path(metrics_file)
        assert metrics_file.exists(), f"Metrics file {metrics_file} does not exist"
        if metrics_file.exists():
            metrics = json.load(open(metrics_file))  # noqa: SIM115
            all_dict = metrics["eval/all_metrics_dict"]

            # Check if this is grab evaluation (has success_lift)
            has_obj_metrics = "obj_pos_error" in all_dict
            if "success_lift" in all_dict:
                # Grab evaluation: prioritize failed grasps (not lifted) and terminated trajectories
                motion_keys = all_dict["motion_keys"]
                terminated = all_dict["terminated"]
                success_lift = all_dict["success_lift"]
                progress = all_dict.get("progress", [1.0] * len(motion_keys))
                obj_pos_errors = all_dict.get("obj_pos_error", [0.0] * len(motion_keys))

                pairs = []
                for i in range(len(motion_keys)):
                    term = bool(terminated[i]) if i < len(terminated) else False
                    lifted = bool(success_lift[i]) if i < len(success_lift) else False
                    prog = progress[i] if i < len(progress) else 1.0
                    obj_err = obj_pos_errors[i] if i < len(obj_pos_errors) else 0.0
                    priority = 0 if not lifted else (1 if term else 2)
                    pairs.append((motion_keys[i], term, lifted, prog, obj_err, priority))

                pairs_sorted = sorted(pairs, key=lambda x: (x[5], x[3]))
                if len(pairs_sorted) > config.num_envs:
                    pairs_sorted = pairs_sorted[: config.num_envs]

                render_info = []
                for pair in pairs_sorted:
                    motion_key, term, lifted, prog, obj_err, _ = pair
                    status = "FAILED" if not lifted else ("TERMINATED" if term else "SUCCESS")
                    info = [
                        f"{motion_key}",
                        f"lifted: {lifted}",
                        f"progress: {prog:.3f}",
                        f"status: {status}",
                    ]
                    if has_obj_metrics:
                        info.append(f"obj_pos_err: {obj_err:.4f}m")
                    render_info.append(tuple(info))

                filter_keys = [pair[0] for pair in pairs_sorted]

                with omegaconf.open_dict(env_config.config):
                    env_config.config.render_info = render_info
                    env_config.config.max_render_envs = len(render_info)
                with omegaconf.open_dict(env_config.commands.motion):
                    env_config.commands.motion.filter_motion_keys = filter_keys
                    if "motion_lib_cfg" in env_config.commands.motion:
                        env_config.commands.motion.motion_lib_cfg.filter_motion_keys = filter_keys
            else:
                # Imitation evaluation: use MPJPE-based sorting
                obj_pos_errors = all_dict.get("obj_pos_error", None)
                success_pair = [
                    (
                        all_dict["motion_keys"][i],
                        all_dict["mpjpe_l"][i],
                        all_dict["mpjpe_g"][i],
                        True,
                        obj_pos_errors[i] if obj_pos_errors is not None else 0.0,
                    )
                    for i in range(len(all_dict["motion_keys"]))
                    if not all_dict["terminated"][i]
                ]
                render_sort_by = config.get("render_sort_by", "mpjpe_l")
                sort_idx = 4 if render_sort_by == "obj_pos_error" else 1
                success_pair_sorted = sorted(success_pair, key=lambda x: x[sort_idx], reverse=True)
                failed_pair = [
                    (
                        all_dict["motion_keys"][i],
                        all_dict["mpjpe_l"][i],
                        all_dict["mpjpe_g"][i],
                        False,
                        obj_pos_errors[i] if obj_pos_errors is not None else 0.0,
                    )
                    for i in range(len(all_dict["motion_keys"]))
                    if all_dict["terminated"][i]
                ]
                failed_pair_sorted = sorted(failed_pair, key=lambda x: x[sort_idx], reverse=True)
                all_pair = failed_pair_sorted + success_pair_sorted
                if len(all_pair) > config.num_envs:
                    all_pair = all_pair[: config.num_envs]
                render_info = []
                for pair in all_pair:
                    info = [
                        f"{pair[0]}",
                        f"mpjpe_l: {pair[1]:.2f}",
                        f"mpjpe_g: {pair[2]:.2f}",
                        f"success: {pair[3]}",
                    ]
                    if has_obj_metrics:
                        info.append(f"obj_pos_err: {pair[4]:.4f}m")
                    render_info.append(tuple(info))
                with omegaconf.open_dict(env_config.config):
                    env_config.config.render_info = render_info
                    env_config.config.max_render_envs = len(all_pair)
                filter_keys = [pair[0] for pair in all_pair]
                with omegaconf.open_dict(env_config.commands.motion):
                    env_config.commands.motion.filter_motion_keys = filter_keys
                    if "motion_lib_cfg" in env_config.commands.motion:
                        env_config.commands.motion.motion_lib_cfg.filter_motion_keys = filter_keys

    env = train_agent_trl.create_manager_env(config, device, args_cli)

    module_dim_dict = getattr(config.algo.config, "module_dim", {})
    policy_backbone_kwargs = {}
    critic_backbone_kwargs = {}
    env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
    env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space[
        "policy"
    ].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space[
        "critic"
    ].shape[-1]
    example_obs = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key not in ["policy", "critic"]:
            group_obs_dims, group_obs_names, group_obs_total_dim = (
                obs_utils.get_group_term_obs_shape(example_obs, key)
            )
            env.config["obs"]["group_obs_dims"][key] = group_obs_dims
            env.config["obs"]["group_obs_names"][key] = group_obs_names
            env.config["obs"]["obs_dims"][key] = group_obs_total_dim
            env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim

    meta_action_dim = env.config.get("meta_action_dim", None)
    if meta_action_dim is not None and meta_action_dim > 0:
        env.config["robot"]["actions_dim"] = meta_action_dim
    else:
        env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]

    policy = trl_utils_common.custom_instantiate(
        config.algo.config.actor,
        env_config=env.config,
        algo_config=config.algo.config,
        module_dim_dict=module_dim_dict,
        backbone_kwargs=policy_backbone_kwargs,
        _resolve=False,
    ).to(device)

    if not getattr(config.algo.config, "distill_only", False):
        value_model = trl_utils_common.custom_instantiate(
            config.algo.config.critic,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            backbone_kwargs=critic_backbone_kwargs,
            _resolve=False,
        ).to(device)

    accelerator.wait_for_everyone()

    args = easydict.EasyDict()
    args.is_main_process = accelerator.is_main_process
    args.global_rank = accelerator.process_index
    args.world_size = accelerator.num_processes
    state = easydict.EasyDict()

    from gear_sonic.trl.trainer import ppo_trainer

    model = ppo_trainer.PolicyAndValueWrapper(policy, value_model)

    checkpoint_path = str(config.checkpoint)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device, weights_only=False)

    # Load policy state dict with backward compatibility for std/log_std
    if "actor_model_state_dict" in checkpoint:
        state_dict = checkpoint["actor_model_state_dict"]
    elif "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    else:
        state_dict = None

    if state_dict is not None:
        model_uses_std = "std" in model.policy.state_dict()
        checkpoint_has_std = "std" in state_dict
        checkpoint_has_log_std = "log_std" in state_dict

        logger.info(f"Model parameterization: {'std' if model_uses_std else 'log_std'}")
        logger.info(
            f"Checkpoint parameterization: {'std' if checkpoint_has_std else 'log_std' if checkpoint_has_log_std else 'unknown'}"  # noqa: E501
        )

        if model_uses_std and checkpoint_has_log_std and not checkpoint_has_std:
            logger.info("Transforming 'log_std' -> 'std' (applying exp) for backward compatibility")
            state_dict["std"] = torch.exp(state_dict.pop("log_std"))
        elif not model_uses_std and checkpoint_has_std and not checkpoint_has_log_std:
            logger.info("Transforming 'std' -> 'log_std' (applying log) for backward compatibility")
            state_dict["log_std"] = torch.log(state_dict.pop("std"))

        model.policy.load_state_dict(state_dict)
        logger.info("Successfully loaded policy state dict")

    state.global_step = checkpoint["state"].global_step

    schedule_wrapper = easydict.EasyDict(env=env, model=model)
    if "schedule_dict" in config.trainer:
        scheduled_params_dict = scheduler.update_scheduled_params(  # noqa: F841
            schedule_wrapper, config.trainer.schedule_dict, state.global_step
        )
    env.reinit_dr()

    global_step = checkpoint["state"].global_step
    exported_policy_path = os.path.join(config.experiment_dir, "exported")
    os.makedirs(exported_policy_path, exist_ok=True)
    exported_onnx_name = f"model_step_{global_step:06d}.onnx"
    new_cp_path = f"{os.path.dirname(config.checkpoint)}/model_step_{global_step:06d}.pt"
    if not os.path.exists(new_cp_path):
        shutil.copy(checkpoint_path, new_cp_path)

    if config.get("export_onnx_only", False):

        def get_example_obs():
            obs_dict = env.reset_all()
            for k in obs_dict:
                obs_dict[k] = obs_dict[k].cpu()
            return obs_dict

        assert config.num_envs == 1, "num_envs must be 1 for exporting onnx"
        from gear_sonic.utils import inference_helpers

        example_obs_dict = get_example_obs()

        # Check if actor has universal-token encoder structure
        has_actor_module = hasattr(model.policy, "actor_module")
        has_encoders = has_actor_module and hasattr(
            model.policy.actor_module, "encoders_to_iterate"
        )

        if "tokenizer" in example_obs_dict and has_encoders:

            inference_helpers.export_universal_token_module_as_onnx(
                model.policy.actor_module,
                encoder_name="smpl",
                decoder_name="g1_dyn",
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_smpl.onnx"),
                batch_size=1,
            )
            inference_helpers.export_universal_token_module_as_onnx(
                model.policy.actor_module,
                encoder_name="g1",
                decoder_name="g1_dyn",
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_g1.onnx"),
                batch_size=1,
            )
            inference_helpers.export_universal_token_module_as_onnx(
                model.policy.actor_module,
                encoder_name="teleop",
                decoder_name="g1_dyn",
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_teleop.onnx"),
                batch_size=1,
            )

            inference_helpers.export_universal_token_encoders_as_onnx(
                model.policy.actor_module,
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_encoder.onnx"),
                batch_size=1,
            )
            inference_helpers.export_universal_token_decoder_as_onnx(
                model.policy.actor_module,
                decoder_name="g1_dyn",
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_decoder.onnx"),
                batch_size=1,
            )
            print(  # noqa: T201
                f'Exported encoders ONNX to {os.path.join(exported_policy_path, exported_onnx_name.replace(".onnx", "_encoder.onnx"))}'  # noqa: E501
            )
            print(  # noqa: T201
                f'Exported decoder ONNX to {os.path.join(exported_policy_path, exported_onnx_name.replace(".onnx", "_decoder.onnx"))}'  # noqa: E501
            )

        else:
            inference_helpers.export_policy_as_onnx(
                {"actor": model.policy}, exported_policy_path, exported_onnx_name, example_obs_dict
            )

        logger.info(f"Exported policy as onnx to: {os.path.join(exported_policy_path)}")

        # Export configs to YAML
        export_config = {
            "env_config": omegaconf.OmegaConf.to_container(env.config, resolve=True),
            "algo_config": omegaconf.OmegaConf.to_container(config.algo.config, resolve=True),
        }
        config_yaml_path = os.path.join(os.path.dirname(config.checkpoint), "model_config.yaml")
        with open(config_yaml_path, "w") as f:
            yaml.dump(export_config, f, default_flow_style=False)
        logger.info(f"Exported config to: {config_yaml_path}")
        exit()  # noqa: PLR1722

    eval_callbacks = config.get("eval_callbacks", [])
    if isinstance(eval_callbacks, str):
        eval_callbacks = [eval_callbacks]

    callbacks = {}
    for callback_name in eval_callbacks:
        if callback_name == "im_eval":
            with omegaconf.open_dict(config.callbacks.im_eval):
                config.callbacks.im_eval.eval_only = True
                config.callbacks.im_eval.eval_frequency = 1
                config.callbacks.im_eval.output_dir = config.get("eval_output_dir", None)
                config.callbacks.im_eval.log_keys = config.get("log_keys", None)
        if callback_name not in config.callbacks:
            raise ValueError(f"Callback {callback_name} not found")
        callbacks[callback_name] = utils.instantiate(config.callbacks[callback_name])

    for callback_name, callback in callbacks.items():  # noqa: B007
        if hasattr(callback, "model") and callback.model is None:
            callback.model = model

    for callback_name, callback in callbacks.items():  # noqa: B007
        callback.on_step_end(args, state, None, env=env, model=model, accelerator=accelerator)

    # ===== [R3] free-root + planted-feet stance response (round-3 protocol) =====
    # -- unwrap to the ManagerBasedRLEnv --
    real_env = env
    for _ in range(5):
        if hasattr(real_env, "termination_manager"):
            break
        real_env = getattr(real_env, "env", None) or getattr(real_env, "_env", None)
    assert hasattr(real_env, "termination_manager"), f"[R3] unwrap failed: {type(real_env)}"

    import math as _math
    import numpy as _np1

    sim = real_env.sim
    robot = real_env.scene["robot"]
    _dt = sim.get_physics_dt()  # 0.005 physics substep
    R3_OUT = os.environ.get("R3_OUT", "/root/isaac_r3_out")
    os.makedirs(R3_OUT, exist_ok=True)
    print(f"[R3] physics_dt={_dt:.4f} ({1/_dt:.0f} Hz substep), out={R3_OUT}", flush=True)

    # -- neutralize randomization events (we never call env.reset) --
    try:
        _em = real_env.event_manager
        _nev = 0
        for _mode in ("startup", "reset", "interval"):
            try:
                _names = list(_em.get_term_names(_mode))
            except Exception:
                continue
            for _n in _names:
                try:
                    _c = _em.get_term_cfg(_n)
                    _c.func = lambda _env, **_kw: None
                    _em.set_term_cfg(_n, _c)
                    _nev += 1
                except Exception as _e:
                    print(f"[R3] event neutralize failed {_mode}/{_n}: {_e}")
        print(f"[R3] events neutralized: {_nev}")
    except Exception as _e:
        print(f"[R3] event manager unavailable (ok if never reset): {_e}")

    _jn = list(robot.joint_names)
    JOINTS = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint"]
    for _j in JOINTS:
        assert _j in _jn, f"[R3] joint {_j} not in articulation"
    _jidx = {j: _jn.index(j) for j in JOINTS}
    # replica naming (no _joint suffix): left_hip_pitch / left_hip_roll / left_knee
    _jtag = {j: j.replace("_joint", "") for j in JOINTS}
    print(f"[R3] test joints: {JOINTS}")

    # -- contact sensor (P0 pattern) --
    _contact_sensor = None
    _foot_body_idx = {}
    try:
        for _sn, _ss in real_env.scene._sensors.items():
            from isaaclab.sensors import ContactSensor
            if isinstance(_ss, ContactSensor):
                _contact_sensor = _ss
                _bn = list(_ss.body_names)
                if "left_ankle_roll_link" in _bn and "right_ankle_roll_link" in _bn:
                    _foot_body_idx["left"] = _bn.index("left_ankle_roll_link")
                    _foot_body_idx["right"] = _bn.index("right_ankle_roll_link")
                break
        print(f"[R3] contact sensor: {_contact_sensor is not None} foot_idx={_foot_body_idx}")
    except Exception as _e:
        print(f"[R3] contact sensor unavailable: {_e}")

    # -- q0: DEFAULT standing pose (URDF, balanced, feet together), per README
    #    "默认站立 ref 姿态、双脚并立".  (walk first-frame q0 has pre-bent ankles
    #    -0.35..-0.47 rad and topples under pure PD hold; default pose is stable.)
    _default_pos = robot.data.default_joint_pos.clone()
    q0 = _default_pos[0].cpu().numpy().astype(_np1.float32)
    print(f"[R3] q0 = default standing pose (balanced, feet together)", flush=True)

    _env_ids = torch.arange(1, device=robot.device)
    _zero6 = torch.zeros(1, 6, device=robot.device)

    # anchor: default standing root (feet on ground) captured once, never re-derived
    # from a possibly-toppled current state
    _anchor_root_pos = robot.data.root_pos_w.clone()
    _anchor_root_quat = robot.data.root_quat_w.clone()
    print(f"[R3] anchor root_pos={_anchor_root_pos[0].cpu().tolist()} "
          f"quat={_anchor_root_quat[0].cpu().tolist()}", flush=True)

    def _contacts_active(_nf=None):
        if _contact_sensor is None:
            return True
        if _nf is None:
            _nf = _contact_sensor.data.net_forces_w
        if _nf.dim() == 3 and _nf.shape[0] == 1:
            _fl = _nf[0, _foot_body_idx["left"], :]
            _fr = _nf[0, _foot_body_idx["right"], :]
        elif _nf.dim() == 3:
            _fl = _nf[_foot_body_idx["left"], 0, :]
            _fr = _nf[_foot_body_idx["right"], 0, :]
        else:
            return True
        return float(_fl.norm()) > 0.5 and float(_fr.norm()) > 0.5

    def _settle(_n_steps, _tgt):
        for _ in range(_n_steps):
            robot.set_joint_position_target(_tgt)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(_dt)
            if _contact_sensor is not None:
                _contact_sensor.update(_dt)

    def _initial_pose():
        """Standing: ref pose (q0 + ref_root_quat, replica-style), z adaptive so
        feet contact, then 5mm pre-penetration, PD-held at q0. Always re-derives
        from the reference, never from current (possibly toppled) state."""
        _q0t = torch.tensor(q0, dtype=torch.float32, device=robot.device).unsqueeze(0)
        _zv = torch.zeros_like(_q0t)
        _rq = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32,
                           device=robot.device)  # upright
        _rp = _anchor_root_pos.clone()  # xy from env origin, z adaptive below
        _zcur = _rp[:, 2].clone()  # anchor z = default standing (feet on ground)
        for _it in range(40):
            _rp2 = _rp.clone()
            _rp2[:, 2] = _zcur - _it * 0.002
            robot.write_root_pose_to_sim(torch.cat([_rp2, _rq], dim=1), _env_ids)
            robot.write_root_velocity_to_sim(_zero6, _env_ids)
            robot.write_joint_state_to_sim(_q0t, _zv)
            robot.write_data_to_sim()
            _settle(5, _q0t)
            if _contacts_active():
                break
        # 5mm pre-penetration then settle 0.5s
        _rp3 = robot.data.root_pos_w.clone()
        _rp3[:, 2] -= 0.005
        robot.write_root_pose_to_sim(torch.cat([_rp3, robot.data.root_quat_w.clone()], dim=1), _env_ids)
        robot.write_root_velocity_to_sim(_zero6, _env_ids)
        robot.write_joint_state_to_sim(_q0t, _zv)
        robot.write_data_to_sim()
        _settle(int(0.5 / _dt), _q0t)
        _ok = _contacts_active()
        print(f"[R3] initial pose: contact_active={_ok} root_z={_rp3[:, 2].item():.4f}", flush=True)
        return _ok

    def _run_test(jname, kind, param, sign=1.0):
        j = _jidx[jname]
        _q0t = torch.tensor(q0, dtype=torch.float32, device=robot.device).unsqueeze(0)
        # measured: one sim.step() advances 0.02s (physics_context substeps=4).
        # Control dt = record dt = 0.02, same as P1 real behavior. README "200Hz"
        # is nominal; actual 50Hz (documented in protocol_notes).
        _DT_CTRL = 0.02
        _T_END = 2.0 if kind == "step" else 4.0
        _n_ctrl = int(_T_END / _DT_CTRL)
        _rec = {_k: [] for _k in ("t", "qpos", "qvel", "target", "applied_torque",
                                  "root_pos", "root_quat", "contact_force_left",
                                  "contact_force_right", "n_contacts")}
        _truncated = False
        _trunc_reason = "none"
        _airborne = 0
        _initial_pose()  # standard start pose for every test (free-root, soft PD)
        for _kc in range(_n_ctrl):
            _tt = _kc * _DT_CTRL
            if kind == "step":
                _tgtv = q0[j] + sign * param
            else:
                _tgtv = q0[j] + 0.05 * _math.sin(2.0 * _math.pi * param * _tt)
            _tg = _q0t.clone()
            _tg[0, j] = _tgtv
            robot.set_joint_position_target(_tg)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(_dt)
            if _contact_sensor is not None:
                _contact_sensor.update(_dt)
            _tf = _tt + _DT_CTRL
            _rec["t"].append(_tf)
            _rec["qpos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
            _rec["qvel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
            _rec["target"].append(_tg[0].cpu().numpy().copy())
            _rec["applied_torque"].append(robot.data.applied_torque[0].cpu().numpy().copy())
            _rec["root_pos"].append(robot.data.root_pos_w[0].cpu().numpy().copy())
            _rec["root_quat"].append(robot.data.root_quat_w[0].cpu().numpy().copy())
            _fl = _fr = None
            if _contact_sensor is not None:
                _nf = _contact_sensor.data.net_forces_w
                if _nf.dim() == 3 and _nf.shape[0] == 1:
                    _fl = _nf[0, _foot_body_idx["left"], :]
                    _fr = _nf[0, _foot_body_idx["right"], :]
                elif _nf.dim() == 3:
                    _fl = _nf[_foot_body_idx["left"], 0, :]
                    _fr = _nf[_foot_body_idx["right"], 0, :]
            if _fl is not None:
                _rec["contact_force_left"].append(_fl.cpu().numpy().copy())
                _rec["contact_force_right"].append(_fr.cpu().numpy().copy())
                _rec["n_contacts"].append(float(_fl.norm()) > 0.5 and float(_fr.norm()) > 0.5)
            else:
                _rec["contact_force_left"].append(_np1.zeros(3, _np1.float32))
                _rec["contact_force_right"].append(_np1.zeros(3, _np1.float32))
                _rec["n_contacts"].append(-1.0)
            # imbalance rule: root roll/pitch > 0.35 rad (replica ori_thresh)
            _rq = robot.data.root_quat_w[0].cpu().numpy()  # wxyz
            _w, _x, _y, _z = _rq
            _roll = _math.atan2(2.0 * (_w * _x + _y * _z), 1.0 - 2.0 * (_x * _x + _y * _y))
            _pitch = _math.asin(max(-1.0, min(1.0, 2.0 * (_w * _y - _z * _x))))
            if abs(_roll) > 0.35 or abs(_pitch) > 0.35:
                _truncated = True
                _trunc_reason = f"root_roll={_roll:.2f}_pitch={_pitch:.2f}"
                break
            if _fl is not None:
                if float(_fl.norm()) < 1.0 and float(_fr.norm()) < 1.0:
                    _airborne += 1
                else:
                    _airborne = 0
                if _airborne >= 10:  # 200ms both-feet airborne (10 x 0.02)
                    _truncated = True
                    _trunc_reason = "both_feet_airborne"
                    break
        # settle back 0.5s before next test
        _settle(int(0.5 / _dt), _q0t)
        return _rec, _truncated, _trunc_reason

    def _save(path, jname, kind, param, sign, rec, truncated, trunc_reason, retries):
        _np1.savez(
            path,
            t=_np1.asarray(rec["t"], dtype=_np1.float32),
            qpos=_np1.asarray(rec["qpos"], dtype=_np1.float32),
            qvel=_np1.asarray(rec["qvel"], dtype=_np1.float32),
            target=_np1.asarray(rec["target"], dtype=_np1.float32),
            applied_torque=_np1.asarray(rec["applied_torque"], dtype=_np1.float32),
            root_pos=_np1.asarray(rec["root_pos"], dtype=_np1.float32),
            root_quat=_np1.asarray(rec["root_quat"], dtype=_np1.float32),
            contact_force_left=_np1.asarray(rec["contact_force_left"], dtype=_np1.float32),
            contact_force_right=_np1.asarray(rec["contact_force_right"], dtype=_np1.float32),
            n_contacts=_np1.asarray(rec["n_contacts"], dtype=_np1.float32),
            quat_format=_np1.array("wxyz"),
            control_dt=_np1.float32(0.02),
            record_dt=_np1.float32(0.02),
            record_hz_nominal=_np1.float32(200.0),
            record_hz_actual=_np1.float32(50.0),
            drive_cfg=_np1.array(
                "CFG: kp=armature*(2*pi*10)^2, zeta=2.0; g1.py G1_CYLINDER_MODEL_12_DEX_CFG"),
            joint_name=jname, kind=kind, param=_np1.float32(param), sign=_np1.float32(sign),
            q0=_np1.asarray(q0, dtype=_np1.float32),
            truncated=_np1.bool_(truncated), trunc_reason=_np1.array(trunc_reason),
            retries=_np1.int_(retries),
        )

    _manifest = []
    _initial_pose()

    _smoke = os.environ.get("R3_SMOKE", "0") == "1"
    _smoke_tests = os.environ.get("R3_SMOKE_TESTS", "left_hip_pitch_joint/step/0.05/+1.0")
    if _smoke:
        print(f"[R3] SMOKE MODE: only {_smoke_tests}", flush=True)

    # 3 joints x (6 steps + 3 sines) = 27 tests, free-root with retry-once on imbalance
    for _jname in JOINTS:
        if _smoke:
            _want = []
            for _p in _smoke_tests.split("|"):
                _flds = _p.split("/")
                _a, _b, _c = _flds[0], _flds[1], float(_flds[2])
                _d = float(_flds[3]) if len(_flds) > 3 else 0.0
                _want.append((_a, _b, _c, _d))
        else:
            _want = None
        if _want is not None and not any(a == _jname for a, _, _, _ in _want):
            continue
        for _amp in (0.05, 0.1, 0.2):
            for _sgn in (1.0, -1.0):
                if _want is not None and not any(
                        a == _jname and k == "step" and abs(p - _amp) < 1e-9 and s == _sgn
                        for a, k, p, s in _want):
                    continue
                _tag = f"{_amp}{'p' if _sgn > 0 else 'n'}"
                _base = f"r3_{_jtag[_jname]}_step_{_tag}"
                _rec, _tr, _trr = _run_test(_jname, "step", _amp, _sgn)
                _retries = 0
                if _tr:
                    # keep truncated data (usable before touchdown), retry once
                    _save(f"{R3_OUT}/{_base}_truncated.npz", _jname, "step", _amp, _sgn,
                          _rec, _tr, _trr, 0)
                    _initial_pose()
                    _rec2, _tr2, _trr2 = _run_test(_jname, "step", _amp, _sgn)
                    _retries = 1
                    _rec, _tr, _trr = _rec2, _tr2, _trr2
                _save(f"{R3_OUT}/{_base}.npz", _jname, "step", _amp, _sgn,
                      _rec, _tr, _trr, _retries)
                _manifest.append((_base, _jname, "step", _amp, _sgn, len(_rec["t"]),
                                  _tr, _trr, _retries))
                print(f"[R3] {_base}: steps={len(_rec['t'])} truncated={_tr} "
                      f"retries={_retries} {_trr}", flush=True)
        for _f in (0.5, 2.0, 5.0):
            if _want is not None and not any(
                    a == _jname and k == "sine" and abs(p - _f) < 1e-9
                    for a, k, p, _ in _want):
                continue
            _base = f"r3_{_jtag[_jname]}_sine_{_f}"
            _rec, _tr, _trr = _run_test(_jname, "sine", _f)
            _retries = 0
            if _tr:
                _save(f"{R3_OUT}/{_base}_truncated.npz", _jname, "sine", _f, 0.0,
                      _rec, _tr, _trr, 0)
                _initial_pose()
                _rec2, _tr2, _trr2 = _run_test(_jname, "sine", _f)
                _retries = 1
                _rec, _tr, _trr = _rec2, _tr2, _trr2
            _save(f"{R3_OUT}/{_base}.npz", _jname, "sine", _f, 0.0,
                  _rec, _tr, _trr, _retries)
            _manifest.append((_base, _jname, "sine", _f, 0.0, len(_rec["t"]),
                              _tr, _trr, _retries))
            print(f"[R3] {_base}: steps={len(_rec['t'])} truncated={_tr} "
                  f"retries={_retries} {_trr}", flush=True)

    with open(f"{R3_OUT}/r3_manifest.txt", "w") as _mf:
        _mf.write("# file | joint | kind | param | sign | n_substeps | truncated | reason | retries\n")
        for row in _manifest:
            _mf.write(" | ".join(str(x) for x in row) + "\n")
    _np1.savetxt(f"{R3_OUT}/joint_names.txt", _np1.array(_jn), fmt="%s")
    print(f"[R3] ALL_DONE tests={len(_manifest)}", flush=True)
    import os as _os
    _os._exit(0)
    # ===== [/R3] =====

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
        import argparse

        parser = argparse.ArgumentParser(description="Evaluate an RL agent with TRL.")
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args  # noqa: RUF005
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = env_config.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.enable_cameras = env_config.config.get(
            "render_results", False
        ) or env_config.config.get("enable_cameras", False)

        args_cli.headless = config.headless
        args_cli.multi_gpu = config.multi_gpu
        args_cli.distributed = config.multi_gpu
        args_cli.device = device

        base_kit_args = (
            "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
        )
        if args_cli.headless:
            args_cli.kit_args = base_kit_args + " --no-window"
        else:
            args_cli.kit_args = base_kit_args + f" --/renderer/activeGpu={render_gpu_idx}"

        _lock_path = "/tmp/isaaclab_app_launcher.lock"  # noqa: S108
        with filelock.FileLock(_lock_path):
            app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app  # noqa: F841
        # Enable the URDF importer extension (required by UrdfFileCfg spawners)
        import omni.kit.app
        _ext_mgr = omni.kit.app.get_app_interface().get_extension_manager()
        _ext_mgr.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    unresolved_conf = omegaconf.OmegaConf.to_container(config, resolve=False)  # noqa: F841
    os.chdir(hydra.utils.get_original_cwd())

    ckpt_num = config.checkpoint.split("/")[-1].split("_")[-1].split(".")[0]

    if env_config.config.get("save_rendering_dir", None) is None:
        env_config.config.save_rendering_dir = str(
            checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}"
        )

    metrics_file = config.get("metrics_file", None)
    if metrics_file is not None:
        metrics_file = Path(metrics_file)
        assert metrics_file.exists(), f"Metrics file {metrics_file} does not exist"
        if metrics_file.exists():
            metrics = json.load(open(metrics_file))  # noqa: SIM115
            all_dict = metrics["eval/all_metrics_dict"]

            # Check if this is grab evaluation (has success_lift)
            has_obj_metrics = "obj_pos_error" in all_dict
            if "success_lift" in all_dict:
                # Grab evaluation: prioritize failed grasps (not lifted) and terminated trajectories
                motion_keys = all_dict["motion_keys"]
                terminated = all_dict["terminated"]
                success_lift = all_dict["success_lift"]
                progress = all_dict.get("progress", [1.0] * len(motion_keys))
                obj_pos_errors = all_dict.get("obj_pos_error", [0.0] * len(motion_keys))

                pairs = []
                for i in range(len(motion_keys)):
                    term = bool(terminated[i]) if i < len(terminated) else False
                    lifted = bool(success_lift[i]) if i < len(success_lift) else False
                    prog = progress[i] if i < len(progress) else 1.0
                    obj_err = obj_pos_errors[i] if i < len(obj_pos_errors) else 0.0
                    priority = 0 if not lifted else (1 if term else 2)
                    pairs.append((motion_keys[i], term, lifted, prog, obj_err, priority))

                pairs_sorted = sorted(pairs, key=lambda x: (x[5], x[3]))
                if len(pairs_sorted) > config.num_envs:
                    pairs_sorted = pairs_sorted[: config.num_envs]

                render_info = []
                for pair in pairs_sorted:
                    motion_key, term, lifted, prog, obj_err, _ = pair
                    status = "FAILED" if not lifted else ("TERMINATED" if term else "SUCCESS")
                    info = [
                        f"{motion_key}",
                        f"lifted: {lifted}",
                        f"progress: {prog:.3f}",
                        f"status: {status}",
                    ]
                    if has_obj_metrics:
                        info.append(f"obj_pos_err: {obj_err:.4f}m")
                    render_info.append(tuple(info))

                filter_keys = [pair[0] for pair in pairs_sorted]

                with omegaconf.open_dict(env_config.config):
                    env_config.config.render_info = render_info
                    env_config.config.max_render_envs = len(render_info)
                with omegaconf.open_dict(env_config.commands.motion):
                    env_config.commands.motion.filter_motion_keys = filter_keys
                    if "motion_lib_cfg" in env_config.commands.motion:
                        env_config.commands.motion.motion_lib_cfg.filter_motion_keys = filter_keys
            else:
                # Imitation evaluation: use MPJPE-based sorting
                obj_pos_errors = all_dict.get("obj_pos_error", None)
                success_pair = [
                    (
                        all_dict["motion_keys"][i],
                        all_dict["mpjpe_l"][i],
                        all_dict["mpjpe_g"][i],
                        True,
                        obj_pos_errors[i] if obj_pos_errors is not None else 0.0,
                    )
                    for i in range(len(all_dict["motion_keys"]))
                    if not all_dict["terminated"][i]
                ]
                render_sort_by = config.get("render_sort_by", "mpjpe_l")
                sort_idx = 4 if render_sort_by == "obj_pos_error" else 1
                success_pair_sorted = sorted(success_pair, key=lambda x: x[sort_idx], reverse=True)
                failed_pair = [
                    (
                        all_dict["motion_keys"][i],
                        all_dict["mpjpe_l"][i],
                        all_dict["mpjpe_g"][i],
                        False,
                        obj_pos_errors[i] if obj_pos_errors is not None else 0.0,
                    )
                    for i in range(len(all_dict["motion_keys"]))
                    if all_dict["terminated"][i]
                ]
                failed_pair_sorted = sorted(failed_pair, key=lambda x: x[sort_idx], reverse=True)
                all_pair = failed_pair_sorted + success_pair_sorted
                if len(all_pair) > config.num_envs:
                    all_pair = all_pair[: config.num_envs]
                render_info = []
                for pair in all_pair:
                    info = [
                        f"{pair[0]}",
                        f"mpjpe_l: {pair[1]:.2f}",
                        f"mpjpe_g: {pair[2]:.2f}",
                        f"success: {pair[3]}",
                    ]
                    if has_obj_metrics:
                        info.append(f"obj_pos_err: {pair[4]:.4f}m")
                    render_info.append(tuple(info))
                with omegaconf.open_dict(env_config.config):
                    env_config.config.render_info = render_info
                    env_config.config.max_render_envs = len(all_pair)
                filter_keys = [pair[0] for pair in all_pair]
                with omegaconf.open_dict(env_config.commands.motion):
                    env_config.commands.motion.filter_motion_keys = filter_keys
                    if "motion_lib_cfg" in env_config.commands.motion:
                        env_config.commands.motion.motion_lib_cfg.filter_motion_keys = filter_keys

    env = train_agent_trl.create_manager_env(config, device, args_cli)

    module_dim_dict = getattr(config.algo.config, "module_dim", {})
    policy_backbone_kwargs = {}
    critic_backbone_kwargs = {}
    env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
    env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space[
        "policy"
    ].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space[
        "critic"
    ].shape[-1]
    example_obs = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key not in ["policy", "critic"]:
            group_obs_dims, group_obs_names, group_obs_total_dim = (
                obs_utils.get_group_term_obs_shape(example_obs, key)
            )
            env.config["obs"]["group_obs_dims"][key] = group_obs_dims
            env.config["obs"]["group_obs_names"][key] = group_obs_names
            env.config["obs"]["obs_dims"][key] = group_obs_total_dim
            env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim

    meta_action_dim = env.config.get("meta_action_dim", None)
    if meta_action_dim is not None and meta_action_dim > 0:
        env.config["robot"]["actions_dim"] = meta_action_dim
    else:
        env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]

    policy = trl_utils_common.custom_instantiate(
        config.algo.config.actor,
        env_config=env.config,
        algo_config=config.algo.config,
        module_dim_dict=module_dim_dict,
        backbone_kwargs=policy_backbone_kwargs,
        _resolve=False,
    ).to(device)

    if not getattr(config.algo.config, "distill_only", False):
        value_model = trl_utils_common.custom_instantiate(
            config.algo.config.critic,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            backbone_kwargs=critic_backbone_kwargs,
            _resolve=False,
        ).to(device)

    accelerator.wait_for_everyone()

    args = easydict.EasyDict()
    args.is_main_process = accelerator.is_main_process
    args.global_rank = accelerator.process_index
    args.world_size = accelerator.num_processes
    state = easydict.EasyDict()

    from gear_sonic.trl.trainer import ppo_trainer

    model = ppo_trainer.PolicyAndValueWrapper(policy, value_model)

    checkpoint_path = str(config.checkpoint)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device, weights_only=False)

    # Load policy state dict with backward compatibility for std/log_std
    if "actor_model_state_dict" in checkpoint:
        state_dict = checkpoint["actor_model_state_dict"]
    elif "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    else:
        state_dict = None

    if state_dict is not None:
        model_uses_std = "std" in model.policy.state_dict()
        checkpoint_has_std = "std" in state_dict
        checkpoint_has_log_std = "log_std" in state_dict

        logger.info(f"Model parameterization: {'std' if model_uses_std else 'log_std'}")
        logger.info(
            f"Checkpoint parameterization: {'std' if checkpoint_has_std else 'log_std' if checkpoint_has_log_std else 'unknown'}"  # noqa: E501
        )

        if model_uses_std and checkpoint_has_log_std and not checkpoint_has_std:
            logger.info("Transforming 'log_std' -> 'std' (applying exp) for backward compatibility")
            state_dict["std"] = torch.exp(state_dict.pop("log_std"))
        elif not model_uses_std and checkpoint_has_std and not checkpoint_has_log_std:
            logger.info("Transforming 'std' -> 'log_std' (applying log) for backward compatibility")
            state_dict["log_std"] = torch.log(state_dict.pop("std"))

        model.policy.load_state_dict(state_dict)
        logger.info("Successfully loaded policy state dict")

    state.global_step = checkpoint["state"].global_step

    schedule_wrapper = easydict.EasyDict(env=env, model=model)
    if "schedule_dict" in config.trainer:
        scheduled_params_dict = scheduler.update_scheduled_params(  # noqa: F841
            schedule_wrapper, config.trainer.schedule_dict, state.global_step
        )
    env.reinit_dr()

    global_step = checkpoint["state"].global_step
    exported_policy_path = os.path.join(config.experiment_dir, "exported")
    os.makedirs(exported_policy_path, exist_ok=True)
    exported_onnx_name = f"model_step_{global_step:06d}.onnx"
    new_cp_path = f"{os.path.dirname(config.checkpoint)}/model_step_{global_step:06d}.pt"
    if not os.path.exists(new_cp_path):
        shutil.copy(checkpoint_path, new_cp_path)

    if config.get("export_onnx_only", False):

        def get_example_obs():
            obs_dict = env.reset_all()
            for k in obs_dict:
                obs_dict[k] = obs_dict[k].cpu()
            return obs_dict

        assert config.num_envs == 1, "num_envs must be 1 for exporting onnx"
        from gear_sonic.utils import inference_helpers

        example_obs_dict = get_example_obs()

        # Check if actor has universal-token encoder structure
        has_actor_module = hasattr(model.policy, "actor_module")
        has_encoders = has_actor_module and hasattr(
            model.policy.actor_module, "encoders_to_iterate"
        )

        if "tokenizer" in example_obs_dict and has_encoders:

            inference_helpers.export_universal_token_module_as_onnx(
                model.policy.actor_module,
                encoder_name="smpl",
                decoder_name="g1_dyn",
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_smpl.onnx"),
                batch_size=1,
            )
            inference_helpers.export_universal_token_module_as_onnx(
                model.policy.actor_module,
                encoder_name="g1",
                decoder_name="g1_dyn",
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_g1.onnx"),
                batch_size=1,
            )
            inference_helpers.export_universal_token_module_as_onnx(
                model.policy.actor_module,
                encoder_name="teleop",
                decoder_name="g1_dyn",
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_teleop.onnx"),
                batch_size=1,
            )

            inference_helpers.export_universal_token_encoders_as_onnx(
                model.policy.actor_module,
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_encoder.onnx"),
                batch_size=1,
            )
            inference_helpers.export_universal_token_decoder_as_onnx(
                model.policy.actor_module,
                decoder_name="g1_dyn",
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_decoder.onnx"),
                batch_size=1,
            )
            print(  # noqa: T201
                f'Exported encoders ONNX to {os.path.join(exported_policy_path, exported_onnx_name.replace(".onnx", "_encoder.onnx"))}'  # noqa: E501
            )
            print(  # noqa: T201
                f'Exported decoder ONNX to {os.path.join(exported_policy_path, exported_onnx_name.replace(".onnx", "_decoder.onnx"))}'  # noqa: E501
            )

        else:
            inference_helpers.export_policy_as_onnx(
                {"actor": model.policy}, exported_policy_path, exported_onnx_name, example_obs_dict
            )

        logger.info(f"Exported policy as onnx to: {os.path.join(exported_policy_path)}")

        # Export configs to YAML
        export_config = {
            "env_config": omegaconf.OmegaConf.to_container(env.config, resolve=True),
            "algo_config": omegaconf.OmegaConf.to_container(config.algo.config, resolve=True),
        }
        config_yaml_path = os.path.join(os.path.dirname(config.checkpoint), "model_config.yaml")
        with open(config_yaml_path, "w") as f:
            yaml.dump(export_config, f, default_flow_style=False)
        logger.info(f"Exported config to: {config_yaml_path}")
        exit()  # noqa: PLR1722

    eval_callbacks = config.get("eval_callbacks", [])
    if isinstance(eval_callbacks, str):
        eval_callbacks = [eval_callbacks]

    callbacks = {}
    for callback_name in eval_callbacks:
        if callback_name == "im_eval":
            with omegaconf.open_dict(config.callbacks.im_eval):
                config.callbacks.im_eval.eval_only = True
                config.callbacks.im_eval.eval_frequency = 1
                config.callbacks.im_eval.output_dir = config.get("eval_output_dir", None)
                config.callbacks.im_eval.log_keys = config.get("log_keys", None)
        if callback_name not in config.callbacks:
            raise ValueError(f"Callback {callback_name} not found")
        callbacks[callback_name] = utils.instantiate(config.callbacks[callback_name])

    for callback_name, callback in callbacks.items():  # noqa: B007
        if hasattr(callback, "model") and callback.model is None:
            callback.model = model

    for callback_name, callback in callbacks.items():  # noqa: B007
        callback.on_step_end(args, state, None, env=env, model=model, accelerator=accelerator)

    if config.get("run_eval_loop", True):
        env.set_is_evaluating(True)
        obs_dict = env.reset_all()
        model.eval()
        for obs_key in obs_dict:
            obs_dict[obs_key] = obs_dict[obs_key].to(device)

        eval_step_callbacks = {
            name: cb
            for name, cb in callbacks.items()
            if hasattr(cb, "eval_step") and callable(getattr(cb, "eval_step"))  # noqa: B009
        }
        if eval_step_callbacks:
            logger.info(f"Eval step callbacks enabled: {list(eval_step_callbacks.keys())}")

        step_count = 0
        max_render_steps = config.get("max_render_steps", 0)

        run_once = config.get("run_once", False)
        envs_completed = torch.zeros(config.num_envs, dtype=torch.bool, device=device)

        with torch.no_grad():
            while True:
                policy_model = model.policy
                value_model = model.value_model
                policy_model.init_rollout()

                actor_state = {}
                actions = policy_model.rollout(obs_dict=obs_dict)
                actor_state["actions"] = policy_model.action_mean.detach()
                actor_state["obs_dict"] = actions["obs_dict"]

                step_count += 1

                if max_render_steps > 0 and step_count >= max_render_steps:
                    logger.info(f"Reached max_render_steps={max_render_steps}. Exiting.")
                    if hasattr(env, "end_render_results"):
                        env.end_render_results()
                    break

                results = env.step(actor_state)
                obs_dict, rewards, dones, infos = (
                    results[0],
                    results[1],
                    results[2],
                    results[3],
                )  # noqa: F841

                if eval_step_callbacks:
                    all_want_exit = all(
                        cb.eval_step(env, results) for cb in eval_step_callbacks.values()
                    )
                    if all_want_exit:
                        logger.info("All eval step callbacks signaled exit. Exiting evaluation loop.")
                        break

                if run_once:
                    envs_completed = (
                        envs_completed | dones.squeeze(-1)
                        if dones.dim() > 1
                        else envs_completed | dones
                    )
                    if envs_completed.all():
                        logger.info("All environments completed one episode. Exiting (run_once=True).")
                        if hasattr(env, "end_render_results"):
                            env.end_render_results()
                        break

                for obs_key in obs_dict.keys():  # noqa: SIM118
                    obs_dict[obs_key] = obs_dict[obs_key].to(device)

    if simulator_type == "IsaacSim":
        os._exit(0)


if __name__ == "__main__":
    main()
