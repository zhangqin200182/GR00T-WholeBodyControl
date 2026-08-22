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

    # ===== [COLLECT] P0 baseline collection loop (protocol 2026-08-18) =====
    import json as _json
    import numpy as _np
    POLICY_MODE = os.environ.get("COLLECT_POLICY", "release")   # release | pd
    OUT_DIR = os.environ.get("COLLECT_OUT", "/root/isaac_baseline_out")
    MAX_STEPS = int(os.environ.get("COLLECT_STEPS", "500"))
    CLIP_LIST = _json.loads(os.environ.get("COLLECT_CLIPS", "[]"))
    os.makedirs(OUT_DIR, exist_ok=True)

    # -- unwrap to the ManagerBasedRLEnv --
    real_env = env
    for _ in range(5):
        if hasattr(real_env, "termination_manager"):
            break
        real_env = getattr(real_env, "env", None) or getattr(real_env, "_env", None)
    assert hasattr(real_env, "termination_manager"), f"[COLLECT] unwrap failed: {type(real_env)}"

    # -- neutralize terminations at runtime too (belt & braces) --
    _tm = real_env.termination_manager
    _N = real_env.num_envs
    for _tn in list(_tm.active_terms):
        if str(_tn).startswith("_"):
            continue
        try:
            _c = _tm.get_term_cfg(_tn)
            _c.func = lambda _env, _n=_N, **_kw: torch.zeros(_n, dtype=torch.bool, device=_env.device)
            _tm.set_term_cfg(_tn, _c)
        except Exception as _e:
            print(f"[COLLECT] term neutralize failed {_tn}: {_e}")
    print("[COLLECT] terminations neutralized:", list(_tm.active_terms))

    # -- find the motion tracking command term --
    _track = None
    for _cn in real_env.command_manager.active_terms:
        _t = real_env.command_manager.get_term(_cn)
        if hasattr(_t, "motion_lib"):
            _track = _t
            break
    assert _track is not None, "[COLLECT] no motion command term found"
    _ml = _track.motion_lib
    _num_m = len(_ml._motion_data_keys)  # all loaded clips, not just num_envs
    _load_names = [str(k) for k in _ml._motion_data_keys[:_num_m]]
    print("[COLLECT] motion load order:", _load_names)

    if CLIP_LIST:
        _name2idx = {n: i for i, n in enumerate(_load_names)}
        _missing = [n for n in CLIP_LIST if n not in _name2idx]
        assert not _missing, f"[COLLECT] clips missing from motion lib: {_missing}"
        _desired = torch.tensor([_name2idx[n] for n in CLIP_LIST], device=device)
        _ml.sample_motions = lambda n, _d=_desired: _d[:n].clone()
        _ml.sample_time_steps = lambda ids, truncate_time=None: torch.zeros_like(ids)
        # prevent eval/training reload that would shrink motion set to num_envs
        _ml.load_motions_for_evaluation = lambda start_idx=0: None
        _ml.load_motions_for_training = lambda max_num_seqs=None: False
        _ml.load_all_motions = lambda: None
        # ---- B1 (round2): hijack the command term's episode-reset sampler ----
        # Root cause of the round-1 bug: in _resample_command, the is_evaluating /
        # use_paired branches assign motion_ids = arange(num_envs) % _num_motions
        # (library load order), never calling motion_lib.sample_motions -> the
        # policy consumed lib clips [0,1,2] every batch. Patch the method source
        # so both branches index our forced ids instead (modulo is identity for
        # ids < num_motions).
        import inspect as _inspect
        import importlib as _importlib
        from textwrap import dedent as _dedent
        _track._collect_forced_ids = _desired
        _cls = type(_track)
        _rsrc = _inspect.getsource(_cls._resample_command)
        _frag = "torch.arange(self.num_envs).to(self.device)"
        assert _frag in _rsrc, "[COLLECT] B1: source fragment not found"
        _rsrc2 = _rsrc.replace(_frag, "self._collect_forced_ids.long().to(self.device)")
        _ns = dict(vars(_importlib.import_module(_cls.__module__)))
        exec(_dedent(_rsrc2), _ns)  # noqa: S102
        _track._resample_command = _ns["_resample_command"].__get__(_track, _cls)
        print(f"[COLLECT] B1: _resample_command hijacked; ep i -> clip {CLIP_LIST} @ t=0")
    else:
        CLIP_LIST = _load_names[:_N]

    # -- articulation data & contact sensors introspection --
    _robot = real_env.scene["robot"]
    _ad = _robot.data
    _torque_attr = next((k for k in ("applied_torque", "computed_torque", "joint_effort", "joint_torque")
                         if hasattr(_ad, k) and getattr(_ad, k) is not None), None)
    print(f"[COLLECT] torque field available: {_torque_attr}")
    # -- action term scale/offset (for joint_target recompute + PD mode) --
    _am = real_env.action_manager
    _act_scale = _act_offset = None
    for _an in _am.active_terms:
        _at = _am.get_term(_an)
        _s = getattr(_at, "scale", None)
        _o = getattr(_at, "offset", None)
        _jn = getattr(_at, "joint_names", None) or getattr(getattr(_at, "cfg", None), "joint_names", None)
        print(f"[COLLECT] action term '{_an}' type={type(_at).__name__} "
              f"joints={len(_jn) if _jn else 0} scale={'Y' if _s is not None else 'N'} "
              f"offset={'Y' if _o is not None else 'N'}")
        if _s is not None and _jn and 20 <= len(_jn) <= 35:
            _act_scale = _s if torch.is_tensor(_s) else torch.tensor(_s, device=device)
            _act_offset = _o if torch.is_tensor(_o) else torch.tensor(_o, device=device)
            if _act_scale.numel() > 1:
                _act_scale = _act_scale.flatten()
                _act_offset = _act_offset.flatten()
            print(f"[COLLECT] scale/offset captured (scale[0]={_act_scale[0]:.5f}, offset[0]={_act_offset[0]:.5f})")
            break
    if _act_scale is None:
        # fallback: read from hydra action cfg (reliable source)
        try:
            for _an, _acfg in config.manager_env.actions.items():
                if str(_an).startswith("_"):
                    continue
                _s = _acfg.get("scale", None)
                _udo = _acfg.get("use_default_offset", False)
                _o = _acfg.get("offset", None)
                print(f"[COLLECT] action cfg '{_an}' scale={_s} use_default_offset={_udo} offset={'Y' if _o is not None else 'N'}")
                if _s is not None:
                    _act_scale = (_s if torch.is_tensor(_s) else torch.tensor(_s, device=device)).flatten().to(device)
                    if _act_scale.numel() == 1:
                        _act_scale = _act_scale.repeat(29)
                if _udo:
                    _act_offset = _robot.data.default_joint_pos[0].clone().flatten()
                elif _o is not None:
                    _act_offset = (_o if torch.is_tensor(_o) else torch.tensor(_o, device=device)).flatten().to(device)
                    if _act_offset.numel() == 1:
                        _act_offset = _act_offset.repeat(29)
                if _act_scale is not None:
                    print(f"[COLLECT] cfg-based scale/offset OK (scale[0]={_act_scale[0]:.5f}, offset[0]={_act_offset[0]:.5f})")
                break
        except Exception as _e:
            print(f"[COLLECT] action cfg read ERR: {_e}")
    if _act_scale is None:
        print("[COLLECT] WARN: no dof action scale/offset found; PD/joint_target recompute disabled")


    _contact_sensor = None
    for _sn, _ss in real_env.scene.sensors.items():
        if "contact" in type(_ss).__name__.lower():
            _contact_sensor = _ss
            _bn = getattr(getattr(_ss, "cfg", None), "body_names", None) or []
            try:
                _nf = _ss.data.net_forces_w
                print(f"[COLLECT] contact sensor '{_sn}' bodies={_bn} net_forces_w.shape={tuple(_nf.shape)}")
            except Exception as _e:
                print(f"[COLLECT] contact sensor '{_sn}' bodies={_bn} net_forces ERR {_e}")
    _foot_body_idx = {}
    try:
        _sb_names = list(getattr(_contact_sensor, "body_names", []) or [])
        print(f"[COLLECT] sensor body names ({len(_sb_names)}): {_sb_names}")
        _foot_body_idx["left"] = _sb_names.index("left_ankle_roll_link")
        _foot_body_idx["right"] = _sb_names.index("right_ankle_roll_link")
    except Exception as _e:
        print(f"[COLLECT] sensor body-name index ERR: {_e}; fallback robot idx")
        for _bn2 in ["left_ankle_roll_link", "right_ankle_roll_link"]:
            try:
                _foot_body_idx["left" if "left" in _bn2 else "right"] = int(
                    _robot.find_bodies(_bn2)[0][0])
            except Exception:
                pass
    print(f"[COLLECT] foot body idx (sensor order): {_foot_body_idx}")

    _FIELDS = ["ctrl_step", "t", "root_pos", "root_quat", "qpos", "qvel",
               "ref_qpos", "ref_root_pos", "ref_root_quat", "action_raw",
               "joint_target", "applied_torque", "contact_force_left",
               "contact_force_right", "term_reason", "survived_steps"]
    _rec = {f: [] for f in _FIELDS}
    _obs0_saved = False

    env.set_is_evaluating(True)
    obs_dict = env.reset_all()
    model.eval()
    for obs_key in obs_dict:
        obs_dict[obs_key] = obs_dict[obs_key].to(device)

    # ---- B2 (round2): record the actually-consumed motion per env ----
    if CLIP_LIST:
        _consumed = [_load_names[int(i)] for i in _track.motion_ids[:_N].tolist()]
        print(f"[COLLECT] B2 consumed motions: {_consumed}")
        assert _consumed == list(CLIP_LIST[:_N]), (
            f"B1 FAILED: consumed {_consumed} != intended {CLIP_LIST}")
        with open(f"{OUT_DIR}/consumed_motions_{POLICY_MODE}.txt", "a") as _cf:
            for _e2, _n2 in enumerate(_consumed):
                _cf.write(f"{POLICY_MODE}\tep{_e2:02d}\t{_n2}\n")
        # ---- B3 (round2): per-episode obs_step0 at a defined moment ----
        for _ok, _ov in obs_dict.items():
            try:
                if hasattr(_ov, "shape") and _ov.shape[-1] == 930:
                    for _e2 in range(min(_N, len(CLIP_LIST))):
                        _np.save(f"{OUT_DIR}/obs_step0_{POLICY_MODE}_{CLIP_LIST[_e2]}_{_e2:02d}.npy",
                                 _ov[_e2].cpu().numpy())
                    print("[COLLECT] B3: per-episode obs_step0 saved (post-reset, pre-action)")
                    break
            except Exception as _e:
                print(f"[COLLECT] B3 obs capture skip {_ok}: {type(_e).__name__}")

        # ---- R5-A: friction sample (actual material props at episode start;
        #      randomization events are neutralized in this script => defaults) ----
        try:
            _mp = _robot.root_physx_view.get_material_properties()
            _fric_all = _mp[0].cpu().numpy() if hasattr(_mp[0], "cpu") else _np.asarray(_mp[0])
            print(f"[COLLECT] R5 friction props shape={_fric_all.shape} "
                  f"row0={_fric_all[0].round(4).tolist()}")
        except Exception as _e:
            _fric_all = None
            print(f"[COLLECT] R5 friction read ERR: {_e}")

    step_count = 0
    with torch.no_grad():
        while step_count < MAX_STEPS:
            policy_model = model.policy
            policy_model.init_rollout()
            actor_state = {}
            actions = policy_model.rollout(obs_dict=obs_dict)
            raw_actions = policy_model.action_mean.detach()

            # -- PD mode: drive target = ref_qpos via same action translation --
            if POLICY_MODE == "pd":
                _times = torch.full((_N,), step_count * 0.02, device=device)
                _ref_dof = _ml.get_motion_state(_desired[:_N], _times)["dof_pos"]
                # Isaac action manager: target = default_joint_pos + 1.0 * action
                raw_actions = (_ref_dof - _robot.data.default_joint_pos).reshape(_N, -1)

            actor_state["actions"] = raw_actions
            actor_state["obs_dict"] = actions["obs_dict"]
            step_count += 1

            _jtarget = None
            if _am.active_terms:
                _at = _am.get_term(_am.active_terms[0])
                _jtarget = getattr(_at, "_processed_actions", None)
            if _jtarget is None and _act_scale is not None:
                _jtarget = _act_offset + _act_scale * raw_actions.reshape(_N, -1)

            results = env.step(actor_state)
            obs_dict, rewards, dones, infos = results[0], results[1], results[2], results[3]

            # ---- record (batched, split later) ----
            _times = torch.full((_N,), step_count * 0.02, device=device)
            try:
                _ms = _ml.get_motion_state(_desired[:_N], _times)
                _ref_dof, _ref_root_p, _ref_root_q = _ms["dof_pos"], _ms["root_pos"], _ms["root_rot"]
            except Exception as _e:
                if step_count <= 2:
                    print(f"[COLLECT] get_motion_state failed: {type(_e).__name__} {_e}")
                _ref_dof = _ref_root_p = _ref_root_q = None
            _tau = getattr(_ad, _torque_attr) if _torque_attr else None
            _fl = _fr = None
            if _contact_sensor is not None and _foot_body_idx:
                try:
                    _nf = _contact_sensor.data.net_forces_w  # (envs, bodies, 3) or (bodies, envs, 3)
                    if _nf.dim() == 3 and _nf.shape[0] == _N:
                        _fl = _nf[:, _foot_body_idx["left"], :]
                        _fr = _nf[:, _foot_body_idx["right"], :]
                    elif _nf.dim() == 3:
                        _fl = _nf[_foot_body_idx["left"], :, :].T
                        _fr = _nf[_foot_body_idx["right"], :, :].T
                except Exception as _e:
                    if step_count == 1:
                        print(f"[COLLECT] contact force extract ERR: {_e}")

            _rec["ctrl_step"].append(_np.full(_N, step_count - 1))
            _rec["t"].append(_np.full(_N, step_count * 0.02))
            _rec["root_pos"].append(_ad.root_pos_w.cpu().numpy())
            _rec["root_quat"].append(_ad.root_quat_w.cpu().numpy())
            _rec["qpos"].append(_ad.joint_pos.cpu().numpy())
            _rec["qvel"].append(_ad.joint_vel.cpu().numpy())
            _rec["ref_qpos"].append(_ref_dof.cpu().numpy() if _ref_dof is not None else _np.zeros((_N, 29), _np.float32))
            _rec["ref_root_pos"].append(_ref_root_p.cpu().numpy() if _ref_root_p is not None else _np.zeros((_N, 3), _np.float32))
            _rec["ref_root_quat"].append(_ref_root_q.cpu().numpy() if _ref_root_q is not None else _np.zeros((_N, 4), _np.float32))
            _rec["action_raw"].append(raw_actions.cpu().numpy())
            _rec["joint_target"].append(_jtarget.cpu().numpy() if _jtarget is not None else _np.zeros((_N, 29), _np.float32))
            _rec["applied_torque"].append(_tau.cpu().numpy() if _tau is not None else _np.zeros((_N, 29), _np.float32))
            _rec["contact_force_left"].append(_fl.cpu().numpy() if _fl is not None else _np.zeros((_N, 3), _np.float32))
            _rec["contact_force_right"].append(_fr.cpu().numpy() if _fr is not None else _np.zeros((_N, 3), _np.float32))
            _rec["term_reason"].append(_np.array(["none"] * _N))
            if step_count == 1:
                try:
                    _eo = real_env.scene.env_origins.cpu().numpy()
                    _np.save(f"{OUT_DIR}/env_origins_{POLICY_MODE}.npy", _eo)
                    print(f"[COLLECT] env_origins saved: {_eo[:, :2].tolist()}")
                except Exception as _e:
                    print(f"[COLLECT] env_origins ERR: {_e}")
                if _contact_sensor is not None:
                    try:
                        _nfs = _contact_sensor.data.net_forces_w
                        print(f"[COLLECT] contact diag: total|F|={_nfs.abs().sum().item():.1f} "
                              f"body18={_nfs[:, 18, :].abs().sum().item():.1f} "
                              f"body19={_nfs[:, 19, :].abs().sum().item():.1f}")
                    except Exception as _e:
                        print(f"[COLLECT] contact diag ERR: {_e}")
            _rec["survived_steps"].append(_np.full(_N, step_count))
            if step_count == 1 and not _obs0_saved:
                for _ok, _ov in obs_dict.items():
                    try:
                        if _ov.shape[-1] == 930:
                            _np.save(f"{OUT_DIR}/obs_step0_{POLICY_MODE}.npy", _ov.cpu().numpy())
                            _obs0_saved = True
                            break
                    except Exception:
                        pass
            if step_count % 100 == 0:
                print(f"[COLLECT] step {step_count}/{MAX_STEPS}")

    # ---- dump per-env npz ----
    _n_steps = _rec["ctrl_step"] and len(_rec["ctrl_step"])
    for _e in range(min(_N, len(CLIP_LIST))):
        _data = {}
        for _f in _FIELDS:
            _arr = _np.stack([_s[_e] if _s.ndim > (0 if _f in ("term_reason",) else 0) else _s
                              for _s in _np.array(_rec[_f])]) if False else None
        # simpler: build per-field stack then index env
        for _f in _FIELDS:
            _stack = _np.stack(_rec[_f])       # (T, N, ...) or (T, N)
            _data[_f] = _stack[:, _e]
        _data["consumed_motion_id"] = _np.int64(int(_track.motion_ids[_e].item()))
        _data["consumed_motion_name"] = _np.array(CLIP_LIST[_e])
        _data["ref_clip_name"] = _np.array(CLIP_LIST[_e])
        if _fric_all is not None:
            _data["friction_sample_all"] = _fric_all.astype(_np.float32)
        _fname = f"{OUT_DIR}/{POLICY_MODE}_{CLIP_LIST[_e]}_{_e:02d}.npz"
        _np.savez(_fname, **_data)
        print(f"[COLLECT] saved {_fname} ({_n_steps} steps)")
    _np.savetxt(f"{OUT_DIR}/joint_names.txt", _np.array(list(_robot.joint_names)), fmt="%s")
    print("[COLLECT] ALL_DONE")
    import os as _os
    _os._exit(0)  # ALL_DONE_HARD_EXIT: kit teardown hangs otherwise
    # ===== [/COLLECT] =====

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
