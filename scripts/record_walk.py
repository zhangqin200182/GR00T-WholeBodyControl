#!/usr/bin/env python3
"""CPU 策略评估 + OSMesa 离屏渲染（不占 NPU）。

复用训练同款的 hydra 配置与 Actor 构建代码（+exp=stub_train），
在单进程 MuJoCoEnv 上跑确定性 rollout（action_mean），渲染 mp4 并打印
每个 episode 的 reward / length / 结束原因（即 §3 的评估指标）。

用法（容器内）：
  MUJOCO_GL=osmesa python3 scripts/record_walk.py \
    --ckpt logs_rl/TRL_G1_Stub/<run>/last.pt --out /data/sonic/renders/walk.mp4

注意：
- 服务器无图形栈，默认 MUJOCO_GL=osmesa（CPU 软件渲染）；不要试 EGL
  （昇腾 NPU 机器没有 EGL GL 栈）。Mac 本地跑可用 MUJOCO_GL=cgl。
- 官方 release 权重（sonic_release/last.pt）含旧版 TRL fork 的
  OnlineTrainerState 对象，走 gear_sonic.utils.release_ckpt.load_release
  的 stub 注入加载，两种 checkpoint（官方/自训）都兼容。
"""
import argparse
import os
import sys

# 必须在 import gear_sonic / mujoco 之前设置
os.environ.setdefault("SONIC_MUJOCO_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "osmesa")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
from omegaconf import OmegaConf

# SONIC tokenizer 观测的 12 个子字段（→ 1761D），与 train_agent_trl.py 一致
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


def build_config():
    """hydra compose 训练同款配置（+exp=stub_train），不启动训练入口。"""
    from hydra import compose, initialize_config_dir

    from gear_sonic.utils.config_utils import register_rl_resolvers

    register_rl_resolvers()
    with initialize_config_dir(
        config_dir=os.path.join(_repo_root, "gear_sonic", "config"), version_base="1.1"
    ):
        config = compose(config_name="base", overrides=["+exp=stub_train"])
    return config


def build_policy(config):
    """按 train_agent_trl.py 的 MuJoCo 分支构建 env_config 并实例化 Actor。"""
    from gear_sonic.trl.utils.common import custom_instantiate

    env_config = config.manager_env
    config_dict = OmegaConf.to_container(env_config, resolve=True)
    config_dict.setdefault("obs", {}).setdefault("obs_dims", {})
    config_dict["obs"]["obs_dict"] = {}
    config_dict["obs"]["obs_dims"] = {"actor_obs": 930, "critic_obs": 1645, "tokenizer": 1761}
    config_dict["obs"]["group_obs_dims"] = {"tokenizer": TOKENIZER_OBS_DIMS}
    config_dict["obs"]["group_obs_names"] = {"tokenizer": list(TOKENIZER_OBS_DIMS.keys())}
    config_dict.setdefault("robot", {})["actions_dim"] = 29
    config_dict["num_envs"] = 1
    config_dict["robot"].setdefault("algo_obs_dim_dict", {})
    # 与 train_agent_trl.py SONIC_MUJOCO_ENV 分支一致（无 group obs，tokenizer 为 flat 1761D）
    config_dict["robot"]["algo_obs_dim_dict"]["actor_obs"] = 930
    config_dict["robot"]["algo_obs_dim_dict"]["critic_obs"] = 1645
    config_dict["robot"]["algo_obs_dim_dict"]["tokenizer"] = 1761
    env_cfg = OmegaConf.create(config_dict, flags={"allow_objects": True})

    module_dim_dict = getattr(config.algo.config, "module_dim", {})
    policy = custom_instantiate(
        config.algo.config.actor,
        env_config=env_cfg,
        algo_config=config.algo.config,
        module_dim_dict=module_dim_dict,
        backbone_kwargs={},
        _resolve=False,
    ).to("cpu")
    return policy


def load_checkpoint(policy, ckpt_path):
    """加载 actor 权重。官方 release 与自训 checkpoint 均兼容。"""
    from gear_sonic.utils.release_ckpt import load_release

    ckpt = load_release(ckpt_path)
    sd = (
        ckpt.get("actor_model_state_dict")
        or ckpt.get("policy_state_dict")
        or ckpt.get("policy", ckpt)
    )
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    print(f"checkpoint: {ckpt_path}")
    print(f"  loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint 与当前模型结构不匹配（silent partial load 会导致立即倒地），"
            "请确认 checkpoint 来源与 +exp=stub_train 配置一致"
        )
    step = ckpt.get("global_step") or ckpt.get("step")
    if step is not None:
        print(f"  train step: {step}")


def to_torch(obs):
    return {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0) for k, v in obs.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint 路径（官方 release 或 logs_rl 产物）")
    p.add_argument("--out", default="walk.mp4", help="输出 mp4 路径")
    p.add_argument("--episodes", type=int, default=3, help="渲染/评估的 episode 数")
    p.add_argument("--max-steps", type=int, default=500, help="单 episode 最大步数（与训练 max_episode_length 一致）")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=0, help="默认 0 = 按 ctrl_dt 实时（50）")
    p.add_argument("--stochastic", action="store_true", help="采样动作而非 action_mean（默认确定性）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-xml", default="/gear_sonic_deploy/g1/g1_29dof_v17.xml")
    p.add_argument("--pkl-dir", default="/sample_data/robot_filtered")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    import av
    import mujoco

    from gear_sonic.envs.mujoco_env import MuJoCoEnv

    config = build_config()
    policy = build_policy(config)
    policy.eval()

    # alive_bonus=0.0 与训练 env_config 一致，评估口径对齐训练 reward
    env = MuJoCoEnv(args.model_xml, args.pkl_dir,
                    config=OmegaConf.create({"alive_bonus": 0.0}))
    # XML 默认离屏帧缓冲 640x480，高分辨率渲染需调大
    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, args.width)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, args.height)

    # Lazy 模块需要先跑一次 dummy forward 再 load_state_dict
    obs0 = to_torch(env.reset())
    import torch.nn as nn
    if any(isinstance(m, (nn.LazyLinear, nn.LazyConv2d)) for m in policy.modules()):
        with torch.no_grad():
            policy.act(obs0)
    load_checkpoint(policy, args.ckpt)

    fps = args.fps or round(1.0 / env.ctrl_dt)
    renderer = mujoco.Renderer(env.model, args.height, args.width)
    camera = mujoco.MjvCamera()
    camera.type = 1  # tracking
    camera.trackbodyid = env._body_idx["pelvis"]
    camera.distance = 3.0
    camera.elevation = -15
    camera.azimuth = 90

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    ep_rewards, ep_lengths = [], []
    if hasattr(policy, "init_rollout"):
        policy.init_rollout()

    for ep in range(args.episodes):
        container = av.open(args.out if args.episodes == 1
                            else args.out.replace(".mp4", f"_ep{ep + 1}.mp4"), mode="w")
        stream = container.add_stream("h264", rate=fps)
        stream.width, stream.height = args.width, args.height
        stream.pix_fmt = "yuv420p"

        obs = to_torch(env.reset())
        dones = torch.zeros(1, dtype=torch.bool)
        ep_rew, steps, end_reason = 0.0, 0, "max-steps"

        for step in range(args.max_steps):
            with torch.no_grad():
                psd = policy.rollout(obs_dict=obs, episode_attnmask=None, cur_dones=dones)
            act = psd["actions" if args.stochastic else "action_mean"][0].cpu().numpy()

            obs_np, rew, done, info = env.step(act)
            ep_rew += rew
            steps = step + 1

            renderer.update_scene(env.data, camera=camera)
            frame = av.VideoFrame.from_ndarray(renderer.render(), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

            dones = torch.tensor([done], dtype=torch.bool)
            obs = to_torch(obs_np)
            if done:
                end_reason = "truncated" if info.get("time_outs") else "terminated"
                break

        for packet in stream.encode():
            container.mux(packet)
        container.close()
        ep_rewards.append(ep_rew)
        ep_lengths.append(steps)
        print(f"episode {ep + 1}: reward={ep_rew:.1f} length={steps} end={end_reason}")

    renderer.close()
    if hasattr(policy, "clear_rollout"):
        policy.clear_rollout()

    print(f"\nmean reward={np.mean(ep_rewards):.1f}  mean length={np.mean(ep_lengths):.0f}  "
          f"({args.episodes} episodes, {'stochastic' if args.stochastic else 'deterministic'})")
    print(f"video: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
