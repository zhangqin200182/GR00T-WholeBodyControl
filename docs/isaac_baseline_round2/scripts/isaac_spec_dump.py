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

    # ===== [SPEC DUMP] round-2 item A (A1-A3) =====
    # -- unwrap to the ManagerBasedRLEnv --
    real_env = env
    for _ in range(5):
        if hasattr(real_env, "termination_manager"):
            break
        real_env = getattr(real_env, "env", None) or getattr(real_env, "_env", None)
    assert hasattr(real_env, "termination_manager"), f"[SPEC] unwrap failed: {type(real_env)}"

    import json as _json
    import numpy as _np1

    sim = real_env.sim
    robot = real_env.scene["robot"]
    specs = {}

    def _rep(sec, data):
        specs[sec] = data
        print(f"=== {sec} ===")
        print(_json.dumps(data, indent=2, default=str))
        print(flush=True)

    # ---- robot prim path ----
    _prim_path = None
    for _cand in (getattr(robot, "prim_path", None),
                  getattr(getattr(robot, "cfg", None), "prim_path", None)):
        if _cand:
            _prim_path = _cand
            break
    print(f"[SPEC] robot prim path: {_prim_path}", flush=True)

    import omni.usd as _ousd
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    _stage = _ousd.get_context().get_stage()

    # ---- A1: enumerate ALL collision shapes of the robot (focus feet) ----
    def _shape_geom(prim):
        """Return (type, dims, local_pos, local_quat) for a geometry prim."""
        if prim.IsA(UsdGeom.Cube):
            g = UsdGeom.Cube(prim)
            sz = g.GetSizeAttr().Get()
            return ("box", [sz/2, sz/2, sz/2])
        if prim.IsA(UsdGeom.Cylinder):
            g = UsdGeom.Cylinder(prim)
            return ("cylinder", {"radius": g.GetRadiusAttr().Get(),
                                 "half_height": g.GetHalfHeightAttr().Get(),
                                 "axis": str(g.GetAxisAttr().Get())})
        if prim.IsA(UsdGeom.Capsule):
            g = UsdGeom.Capsule(prim)
            return ("capsule", {"radius": g.GetRadiusAttr().Get(),
                                "half_height": g.GetHalfHeightAttr().Get(),
                                "axis": str(g.GetAxisAttr().Get())})
        if prim.IsA(UsdGeom.Sphere):
            g = UsdGeom.Sphere(prim)
            return ("sphere", {"radius": g.GetRadiusAttr().Get()})
        if prim.IsA(UsdGeom.Mesh):
            g = UsdGeom.Mesh(prim)
            ext = g.GetExtentAttr().Get()
            lo = ext[0]; hi = ext[1] if ext else (Gf.Vec3f(0), Gf.Vec3f(0))
            return ("mesh(convex)", {"bbox_min": list(lo), "bbox_max": list(hi),
                     "points": len(g.GetPointsAttr().Get() or [])})
        return ("unknown", None)

    a1 = {"note": "geoms with CollisionAPI under ankle_roll links (env_0)"}
    a1["diag_ankle_prims"] = []
    for p in _stage.Traverse():
        s = str(p.GetPath())
        if "ankle_roll" in s:
            a1["diag_ankle_prims"].append(s)
            if len(a1["diag_ankle_prims"]) > 40:
                break
    _foot_root = "/World/envs/env_0/Robot/left_ankle_roll_link"
    _root_prim = _stage.GetPrimAtPath(_foot_root)
    if _root_prim and _root_prim.IsValid():
        _stack = [_root_prim]
        while _stack:
            p = _stack.pop()
            s = str(p.GetPath())
            try:
                has_col = p.HasAPI(UsdPhysics.CollisionAPI)
            except Exception:
                has_col = False
            typ, dims = _shape_geom(p) if (p.IsA(UsdGeom.Gprim) or has_col) else ("scope", None)
            entry = {"prim_type": p.GetTypeName(), "geom": typ, "dims": dims, "has_collision_api": bool(has_col)}
            try:
                xf = UsdGeom.Xformable(p)
                m = xf.GetLocalTransformation()
                q = m.ExtractRotationQuat()
                entry["local_pos"] = list(m.ExtractTranslation())
                entry["local_quat_xyzw"] = [q.imaginary[0], q.imaginary[1], q.imaginary[2], q.real]
            except Exception:
                pass
            if has_col:
                try:
                    ca = UsdPhysics.CollisionAPI(p, False)
                    entry["contactOffset"] = ca.GetContactOffsetAttr().Get()
                    entry["restOffset"] = ca.GetRestOffsetAttr().Get()
                except Exception:
                    pass
            a1["tree:" + s.replace("/World/envs/env_0", "...")] = entry
            for c in p.GetChildren():
                _stack.append(c)
    # USD-side capsule/cylinder gprims anywhere under env_0 Robot (佐证 capsules 转换)
    _caps = {}
    for p in _stage.Traverse():
        s = str(p.GetPath())
        if "/env_0/Robot" not in s:
            continue
        if p.IsA(UsdGeom.Capsule):
            g = UsdGeom.Capsule(p)
            try:
                m = UsdGeom.Xformable(p).GetLocalTransformation()
                q = m.ExtractRotationQuat()
                _caps[s[-70:]] = {
                    "radius": g.GetRadiusAttr().Get(),
                    "half_height": g.GetHalfHeightAttr().Get(),
                    "axis": str(g.GetAxisAttr().Get()),
                    "local_pos": list(m.ExtractTranslation()),
                    "local_quat_xyzw": [q.imaginary[0], q.imaginary[1], q.imaginary[2], q.real],
                }
            except Exception:
                pass
        elif p.IsA(UsdGeom.Cylinder):
            g = UsdGeom.Cylinder(p)
            _caps.setdefault("_raw_cylinders", []).append(s[-70:])
    a1["usd_capsules_found"] = len([k for k in _caps if not k.startswith("_")])
    a1["usd_capsule_details"] = _caps
    _rep("A1_foot_colliders", a1)

    # ---- A2: friction (ground + robot shapes, runtime) ----
    a2 = {}
    try:
        mp = robot.root_physx_view.get_material_properties()
        a2["robot_material_properties_shape"] = {
            "shape": list(mp.shape),
            "env0_shape0": [float(x) for x in mp.reshape(-1, mp.shape[-1])[0]],
            "env0_all_shapes_first3": [[float(x) for x in r] for r in mp.reshape(-1, mp.shape[-1])[:3]],
            "legend": "per-shape [static_friction, dynamic_friction, restitution, ...]",
        }
    except Exception as e:
        a2["robot_material_properties_err"] = str(e)
    # ground plane material prim
    a2["ground"] = {}
    for p in _stage.Traverse():
        s = str(p.GetPath()).lower()
        if ("material" in s or "physics" in s) and ("ground" in s or "terrain" in s or "plane" in s):
            found = {}
            for attr in ("staticFriction", "dynamicFriction", "restitution"):
                for ns in ("physxMaterial:", "physics:"):
                    v = p.GetAttribute(ns + attr).Get()
                    if v is not None:
                        found[attr] = float(v)
            if found:
                a2["ground"][str(p.GetPath())[-50:]] = found
    # terrain cfg physics material (from hydra config if reachable)
    try:
        a2["terrain_cfg_material"] = str(config.manager_env.scene.terrain.physics_material)
    except Exception as e:
        a2["terrain_cfg_material_err"] = str(e)
    _rep("A2_friction", a2)

    # ---- A3: scene contact params ----
    a3 = {}
    _ps = None
    for p in _stage.Traverse():
        if p.IsA(UsdPhysics.Scene) or p.GetTypeName() == "PhysicsScene":
            _ps = p
            break
    if _ps is None:
        _ps = _stage.GetPrimAtPath("/PhysicsScene")
    if _ps and _ps.IsValid():
        for attr in _ps.GetAttributes():
            nm = attr.GetName()
            if any(k in nm.lower() for k in ("bounce", "friction", "offset", "solver",
                                             "contact", "rest", "iteration", "substep")):
                try:
                    v = attr.Get()
                    a3["PhysicsScene." + nm] = str(v)
                except Exception:
                    pass
    else:
        a3["PhysicsScene"] = "prim not found"
    # sim dt / substeps (round2 C 相关佐证)
    try:
        a3["sim_render_dt"] = str(sim.get_rendering_dt())
        a3["sim_physics_dt"] = str(sim.get_physics_dt())
    except Exception:
        pass
    _rep("A3_scene_contact", a3)

    with open("/root/isaac_baseline_out/specs.txt", "w") as f:
        f.write("# Isaac-side specs — round2 A1-A3 (runtime extracted)\n")
        for k, v in specs.items():
            f.write(f"\n=== {k} ===\n{_json.dumps(v, indent=2, default=str)}\n")
    print("[SPEC] saved /root/isaac_baseline_out/specs.txt", flush=True)
    print("[SPEC] ALL_DONE", flush=True)
    import os as _os
    _os._exit(0)
    # ===== [/SPEC DUMP] =====

if __name__ == "__main__":
    main()
