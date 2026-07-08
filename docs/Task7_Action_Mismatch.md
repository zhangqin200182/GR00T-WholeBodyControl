# Task 7: Pretrained Policy → MuJoCo Action Space Mismatch

## 结论

SONIC pretrained checkpoint 在 MuJoCo 里全链路可跑，但输出的 action 范围与 MuJoCoEnv 的 PD 控制器不兼容，原因在于 **Isaac Sim 和 MuJoCo 使用了不同的 joint 归一化参数**。

## 验证过程

### Step 1: 确认全链路已通

在 sonic-train 容器 (`SONIC_MUJOCO_ENV=1`) 跑通了完整 PPO 训练循环：

```
env 创建 → 模型 init → obs 计算 → policy forward → env.step → reward → GAE → update
                                                              ↑
                                              checkpoint saved to last.pt ✅
```

`ppo_trainer.py` 没有改动。只改了 `train_agent_trl.py`（加 `SONIC_MUJOCO_ENV=1` 环境变量分支）和 `mujoco_env_manager.py`（TensorDict → numpy 转换）。

### Step 2: 加载 pretrained checkpoint，跑 inference

直接用 checkpoint 做 inference（不训练），看策略在 MuJoCo 里能不能站稳。

**模型构建**：从 `config/actor_critic/universal_token/all_mlp_v1.yaml` 手动解析出 OmegaConf 结构，用 `custom_instantiate` 构建 Actor。关键参数：

```python
# Encoder (G1): input_dim=[64], num_input_temporal=10 → 640
#   inputs: command_multi_future_nonflat(10,58) + motion_anchor_ori_b_mf_nonflat(10,6)
#   per-frame dim = 58 + 6 = 64, temporal ×10 = 640

# Decoder (g1_dyn): input_dim=[994], output_dim=[29]
#   inputs: token_flattened(64) + proprioception(930) = 994
```

Checkpoint 加载: `torch.load(last.pt, weights_only=False)["policy_state_dict"]` → `actor.load_state_dict(state, strict=False)`。Shape 验证通过，`missing/unexpected` keys 仅 FSQ 编码表（不影响 G1 推理）。

### Step 3: 对比 `act_inference` vs `rollout`

```python
obs_t = {k: torch.from_numpy(v).float().unsqueeze(0) for k,v in obs_np.items()}

with torch.no_grad():
    out_inf = actor.act_inference(obs_t)           # 直接输出 mean
    out_rol = actor.rollout(obs_dict=obs_t)         # Normal(mean,std) 采样

a_inf = out_inf.squeeze().cpu().numpy()            # [29]
a_rol = out_rol["actions"].squeeze().cpu().numpy() # [29]
```

**结果**：

```
act_inference: mean=0.47  std=1.59  range=[-2.1, +4.8]   clipped(>1.0)=51%
rollout:       mean=0.51  std=1.66  range=[-1.7, +5.0]   clipped(>1.0)=51%
```

两者分布几乎相同，排除"采样 vs 均值"的差异。

### Step 4: 这些 action 在 MuJoCo 里的行为

```python
env.step(action)：action * jh + jm → target_joint_pos → PD torque → mj_step
```

_action 超出 [-1,1] 时，target_joint_pos 被推到接近关节限位 → PD 产生巨大力矩 → 关节撞击限位 → QACC 不稳定 → 机器人 1 步倒地。_

即使 `np.clip(action, -1, 1)` 只是延迟了 1-2 步，因为 51% 的关节被 clamp 后 action 信息大量丢失。

### Step 5: 排除 MuJoCo 本身的问题

同一 MuJoCoEnv，用 PD oracle（action 直接指向 reference joint pos）可以站立 200 步且 tracking error < 0.02 rad。说明**物理仿真、PD 控制器都没有问题**。

## 根因分析

```
┌─────────────────────────────────────────────────────────────┐
│  SONIC 训练时 (Isaac Sim)                                   │
│                                                             │
│  model → action_mean[29]                                    │
│    ↓                                                        │
│  action = action_mean + noise * std  (noise_std≈0.05~0.5)  │
│    ↓                                                        │
│  target_qpos = action * IsaacSim_joint_half                 │
│                      + IsaacSim_joint_mid                   │
│    ↓                                                        │
│  Isaac Sim PD / position controller → PhysX                 │
│                                                             │
│  模型训练时学到: action ∈ [≈-1, ≈+1] 范围内分布             │
├─────────────────────────────────────────────────────────────┤
│  MuJoCoEnv (现在)                                           │
│                                                             │
│  model → action_mean[29]  (值范围 [-2.1, +4.8])              │
│    ↓                                                        │
│  target_qpos = action * MuJoCo_jh + MuJoCo_jm               │
│    ↓                                                        │
│  MuJoCo PD → mj_step                                        │
│                                                             │
│  MuJoCo_jh/jm ≠ IsaacSim_joint_half/mid                     │
│  → 同一 action 映射到完全不同的关节角 → 仿真爆炸              │
└─────────────────────────────────────────────────────────────┘
```

**核心矛盾**：SONIC checkpoint 输出的 action 分布是基于 Isaac Sim 的 joint 归一化学习的，MuJoCoEnv 用 MuJoCo XML 的 `jnt_range` 算 `jh/jm`，两边参数不同。

## 需要的信息

为使 pretrained policy 直接适用，需要知道 Isaac Sim 训练时的 joint 归一化参数：

1. **Isaac Sim G1 的 joint range 是什么？**（`joint_mid` / `joint_half` 或 `default_joint_angles` / `action_scale`）
2. 这些参数在哪里定义？可能路径：
   - `gear_sonic/config/manager_env/` 下的 robot config
   - Isaac Lab 的 `g1_model_12_dex` 定义文件
   - `sonic_release.yaml` 中的 `manager_env.config.robot` 配置
3. 或者在 SONIC checkpoint 中是否保存了 action 空间的参数？

## 验证脚本位置

所有验证脚本在服务器 `sonic-train` 容器内可复现：
- 训练: `SONIC_MUJOCO_ENV=1 python3 gear_sonic/train_agent_trl.py +exp=stub_train num_envs=16`
- 推理: `python3 scripts/train_mujoco_sonic.py`
- Action 分布: `cat /tmp/check_rollout.py | python3 /dev/stdin`
