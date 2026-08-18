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

    # ===== [PROBE] V6/V7/V8 physics reference data (protocol: E15 hard gates) =====
    # -- unwrap to the ManagerBasedRLEnv --
    real_env = env
    for _ in range(5):
        if hasattr(real_env, "termination_manager"):
            break
        real_env = getattr(real_env, "env", None) or getattr(real_env, "_env", None)
    assert hasattr(real_env, "termination_manager"), f"[PROBE] unwrap failed: {type(real_env)}"

    import numpy as _np1

    sim = real_env.sim
    robot = real_env.scene["robot"]
    _dt = sim.get_physics_dt()  # 0.005 -> 200 Hz
    PROBE_OUT = os.environ.get("PROBE_OUT", "/root/isaac_baseline_out/physics_probe")
    os.makedirs(PROBE_OUT, exist_ok=True)
    print(f"[PROBE] dt={_dt:.4f} ({1/_dt:.0f} Hz), out={PROBE_OUT}", flush=True)

    # -- belt & braces: neutralize events (never call env.reset) --
    try:
        _em = real_env.event_manager
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
                except Exception:
                    pass
        print("[PROBE] events neutralized")
    except Exception as _e:
        print(f"[PROBE] event manager unavailable (ok if never reset): {_e}")

    # -- contact sensor (sensor-order foot indexing, same as P0) --
    _contact_sensor = None
    for _sn, _ss in real_env.scene.sensors.items():
        if "contact" in type(_ss).__name__.lower():
            _contact_sensor = _ss
            break
    _foot_idx = {}
    try:
        _sb = list(getattr(_contact_sensor, "body_names", []) or [])
        _foot_idx["left"] = _sb.index("left_ankle_roll_link")
        _foot_idx["right"] = _sb.index("right_ankle_roll_link")
        print(f"[PROBE] foot sensor idx: {_foot_idx} (of {len(_sb)} bodies)")
    except Exception as _e:
        print(f"[PROBE] sensor idx ERR: {_e}")

    def _foot_forces():
        try:
            _nf = _contact_sensor.data.net_forces_w
            if _nf.dim() == 3 and _nf.shape[0] == 1:
                return _nf[0, _foot_idx["left"], :].cpu().numpy().copy(), _nf[0, _foot_idx["right"], :].cpu().numpy().copy()
            if _nf.dim() == 3:
                return _nf[_foot_idx["left"], :, 0].cpu().numpy().copy(), _nf[_foot_idx["right"], :, 0].cpu().numpy().copy()
        except Exception:
            pass
        return _np1.zeros(3), _np1.zeros(3)

    _jn = list(robot.joint_names)
    _default_pos = robot.data.default_joint_pos.clone()
    _zero_jvel = torch.zeros_like(_default_pos)
    _env_ids = torch.arange(1, device=robot.device)
    _root_zero_vel = torch.zeros(1, 6, device=robot.device)
    _mass_kg = 0.0
    for _attr in ("body_mass", "body_mass_w"):
        try:
            _mass_kg = float(getattr(robot.data, _attr).sum().item())
            break
        except Exception:
            pass
    if _mass_kg == 0.0:
        try:
            _mass_kg = float(robot.root_physx_view.get_link_masses().sum().item())
        except Exception:
            pass
    print(f"[PROBE] total mass: {_mass_kg:.2f} kg", flush=True)

    _FIELDS = ["t", "root_pos", "root_quat", "root_linvel", "root_angvel",
               "qpos", "contact_left", "contact_right"]

    # discover velocity attribute names for this IsaacLab version
    _lv = next((a for a in ("root_lin_vel_w", "root_linvel_w", "root_vel_w") if hasattr(robot.data, a)), None)
    _av = next((a for a in ("root_ang_vel_w", "root_angvel_w", "root_gyro_w") if hasattr(robot.data, a)), None)
    print(f"[PROBE] vel attrs: lin={_lv} ang={_av}", flush=True)
    assert _lv and _av, f"[PROBE] no vel attrs of {dir(robot.data)}"

    def _rec_dict():
        return {k: [] for k in _FIELDS}

    def _refresh_sensors():
        try:
            _contact_sensor.update(_dt)
        except Exception:
            pass

    def _snap(rec, tt):
        _refresh_sensors()
        rec["t"].append(tt)
        rec["root_pos"].append(robot.data.root_pos_w[0].cpu().numpy().copy())
        rec["root_quat"].append(robot.data.root_quat_w[0].cpu().numpy().copy())
        rec["root_linvel"].append(getattr(robot.data, _lv)[0].cpu().numpy().copy())
        rec["root_angvel"].append(getattr(robot.data, _av)[0].cpu().numpy().copy())
        rec["qpos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        _fl, _fr = _foot_forces()
        rec["contact_left"].append(_fl)
        rec["contact_right"].append(_fr)

    def _save(rec, fname, meta):
        data = {}
        for k in _FIELDS:
            data[k] = _np1.asarray(rec[k], dtype=_np1.float32)
        _np1.savez(f"{PROBE_OUT}/{fname}", dt=_np1.float32(DT), dt_physics_nominal=_np1.float32(_dt),
                   joint_names=_np1.array(_jn), total_mass_kg=_np1.float32(_mass_kg),
                   **{f"meta_{k}": _np1.array(str(v)) for k, v in meta.items()}, **data)
        n = len(rec["t"])
        print(f"[PROBE] saved {fname} ({n} substeps)", flush=True)

    def _lock_joints():
        robot.write_joint_state_to_sim(_default_pos, _zero_jvel)
        robot.set_joint_position_target(_default_pos)
        robot.write_data_to_sim()

    # capture the spawn standing pose as reference (robot spawns standing)
    _stand_pos = robot.data.root_pos_w.clone()
    _stand_quat = robot.data.root_quat_w.clone()
    _stand_state = torch.cat([_stand_pos, _stand_quat], dim=1)
    print(f"[PROBE] standing ref: pos={_stand_pos[0].cpu().tolist()}", flush=True)

    _root_zero6 = torch.zeros(1, 6, device=robot.device)

    def _pin_root_stand():
        robot.write_root_pose_to_sim(_stand_state, _env_ids)
        robot.write_root_velocity_to_sim(_root_zero6, _env_ids)

    def _settle_pinned(sec=2.0):
        for _ in range(int(sec / _dt)):
            _pin_root_stand()
            _lock_joints()
            sim.step()
            robot.update(_dt)

    # settle with root pinned at standing (free-root PD cannot hold stance here)
    _settle_pinned()
    print("[PROBE] settle done (root pinned)", flush=True)

    # ===== dt calibration: introspect + vertical toss =====
    print("[PROBE] sim dt attrs: " + str({a: getattr(sim, a) for a in dir(sim) if "dt" in a.lower() and not a.startswith("_")}), flush=True)
    _settle_pinned(1.0)
    _rp = _stand_pos.clone(); _rq = _stand_quat.clone()
    _rp[0, 2] = _stand_pos[0, 2].item() + 0.3
    robot.write_root_pose_to_sim(torch.cat([_rp, _rq], dim=1), _env_ids)
    _rv = torch.zeros(1, 6, device=robot.device); _rv[0, 2] = 1.5
    robot.write_root_velocity_to_sim(_rv, _env_ids)
    robot.write_joint_state_to_sim(_default_pos, _zero_jvel)
    robot.set_joint_position_target(_default_pos)
    robot.write_data_to_sim()
    robot.update(_dt)
    _z_launch = robot.data.root_pos_w[0, 2].item()
    _n_updown = 0
    _apex_passed = False
    for _ in range(2000):
        robot.set_joint_position_target(_default_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(_dt)
        _n_updown += 1
        _z_now = robot.data.root_pos_w[0, 2].item()
        if _z_now > _z_launch + 0.02:
            _apex_passed = True
        if _apex_passed and _z_now <= _z_launch:
            break
    _dt_step = 0.3058 / max(_n_updown, 1)
    print(f"[PROBE] toss calibration: N={_n_updown} steps -> dt_step={_dt_step:.5f}s "
          f"(nominal physics_dt={_dt}, ratio={_dt_step/_dt:.2f}x)", flush=True)
    DT = 0.02  # exact step dt (toss calibration verifies; g-fit vs theory 0.1%)

    # ===== V8: parabola launches (joints locked, free flight) =====
    for (vx, vz) in [(1.0, 2.0), (2.0, 2.5), (0.5, 3.0)]:
        _settle_pinned(1.0)
        rec = _rec_dict()
        # launch from pinned upright stance: raise +0.3, keep upright quat, set velocity, then FREE root
        _rp = _stand_pos.clone(); _rq = _stand_quat.clone()
        _rp[0, 2] = _stand_pos[0, 2].item() + 0.3
        robot.write_root_pose_to_sim(torch.cat([_rp, _rq], dim=1), _env_ids)
        _rv = torch.zeros(1, 6, device=robot.device)
        _rv[0, 0] = vx; _rv[0, 2] = vz
        robot.write_root_velocity_to_sim(_rv, _env_ids)
        # one-shot joint lock at launch; during flight only PD targets (no state writes,
        # no root writes) so the free root is not perturbed by kinematic re-pinning
        robot.write_joint_state_to_sim(_default_pos, _zero_jvel)
        robot.set_joint_position_target(_default_pos)
        robot.write_data_to_sim()
        robot.update(_dt)
        _snap(rec, 0.0)  # launch baseline
        z0 = _rp[0, 2].item()
        t0 = 0.0
        for _ in range(int(3.0 / _dt)):
            robot.set_joint_position_target(_default_pos)
            robot.write_data_to_sim()
            sim.step()
            robot.update(_dt)
            t0 += DT
            _snap(rec, t0)
            if robot.data.root_pos_w[0, 2].item() < 0.15:
                break
        _save(rec, f"v8_parabola_vx{vx}_vz{vz}.npz",
              {"vx": vx, "vz": vz, "kind": "parabola", "z0": z0})

    # ===== V7: drops from two heights (energy dissipation) =====
    for h in [0.30, 0.15]:
        _settle_pinned(1.0)
        rec = _rec_dict()
        _rp = _stand_pos.clone(); _rq = _stand_quat.clone()
        _rp[0, 2] = _stand_pos[0, 2].item() + h
        robot.write_root_pose_to_sim(torch.cat([_rp, _rq], dim=1), _env_ids)
        robot.write_root_velocity_to_sim(_root_zero6, _env_ids)
        robot.write_joint_state_to_sim(_default_pos, _zero_jvel)
        robot.set_joint_position_target(_default_pos)
        robot.write_data_to_sim()
        robot.update(_dt)
        _snap(rec, 0.0)
        z0 = _rp[0, 2].item()
        tt = 0.0
        for _ in range(int(6.0 / _dt)):
            robot.set_joint_position_target(_default_pos)
            robot.write_data_to_sim()
            sim.step()
            robot.update(_dt)
            tt += DT
            _snap(rec, tt)
        _save(rec, f"v7_drop_h{h}.npz", {"h": h, "kind": "drop", "z0": z0})

    # ===== V6: GRF stance + in-place marching (free root after pinned settle) =====
    _settle_pinned(2.0)
    _hipL = _jn.index("left_hip_pitch_joint"); _hipR = _jn.index("right_hip_pitch_joint")
    _kneeL = _jn.index("left_knee_joint"); _kneeR = _jn.index("right_knee_joint")

    # V6a static stance: unpin root, native PD to default, record 6 s
    rec = _rec_dict()
    tt = 0.0
    for _ in range(int(6.0 / _dt)):
        tt += DT
        robot.set_joint_position_target(_default_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(_dt)
        _snap(rec, tt)
    _save(rec, "v6a_grf_stance.npz", {"kind": "grf_stance", "dur_s": 6.0})

    # V6b marching: continue free-root, 1 Hz alternating steps, 10 s
    rec = _rec_dict()
    tt = 0.0
    for _ in range(int(10.0 / _dt)):
        tt += DT
        _ph = 2 * 3.14159265 * 1.0 * tt
        _amp = 0.35
        _tg = _default_pos.clone()
        if _np1.sin(_ph) > 0:  # left swings
            _tg[0, _hipL] += _amp * _np1.sin(_ph)
            _tg[0, _kneeL] += 0.6 * _amp * _np1.sin(_ph)
        else:                  # right swings
            _tg[0, _hipR] += -_amp * _np1.sin(_ph)
            _tg[0, _kneeR] += -0.6 * _amp * _np1.sin(_ph)
        robot.set_joint_position_target(_tg)
        robot.write_data_to_sim()
        sim.step()
        robot.update(_dt)
        _snap(rec, tt)
    _save(rec, "v6b_grf_march.npz", {"kind": "grf_march", "dur_s": 10.0, "freq_hz": 1.0, "amp_rad": 0.35})

    print(f"[PROBE] ALL_DONE mass={_mass_kg:.2f}", flush=True)
    import os as _os
    _os._exit(0)
    # ===== [/PROBE] =====

if __name__ == "__main__":
    main()
