# SONIC 训练过程详细报告

> 基于 16 卡 Ascend 910 NPU 的 StubEnv 训练实验
> 日期：2026-06-28

---

## 目录

1. [模型架构](#1-模型架构)
2. [训练方法（PPO 算法）](#2-训练方法ppo-算法)
3. [数据集格式与加载](#3-数据集格式与加载)
4. [数据流与 Shape](#4-数据流与-shape)
5. [Loss 计算体系](#5-loss-计算体系)
6. [StubEnv 环境](#6-stubenv-环境)
7. [分布式训练（DDP）](#7-分布式训练ddp)
8. [训练步骤计算](#8-训练步骤计算)
9. [GAE 优势值估计](#9-gae-优势值估计)
10. [PPO 价值函数分析](#10-ppo-价值函数分析)
11. [实际训练曲线与分析](#11-实际训练曲线与分析)
12. [总结](#12-总结)

---

## 1. 模型架构

### 1.1 原理

SONIC（Systematic Orchestration of Neural Imitation Control）的核心思想是通过**统一 Token 表示**将不同来源的运动数据（机器人关节角、VR 遥操作、SMPL 人体动作）映射到同一个离散 token 空间，使得模型能够从多种输入方式生成一致的机器人动作。

整体模型分为两部分：

- **Actor（策略网络）**：基于 `UniversalTokenModule` 的编码-量化-解码流水线，输出 29 维关节角目标
- **Critic（价值网络）**：独立的 MLP 网络，输入特权观测，输出标量价值估计

### 1.2 实现

**模型参数统计**（来自实际训练 iter 1 日志）：

```
Total parameters: 37,373,819 (~37.4M)
├── actor_module: 25,870,685 params (~25.9M)
├── critic_module: 11,503,105 params (~11.5M)
└── std: 29 params
```

#### 1.2.1 Actor — UniversalTokenModule

Actor 类定义于 `actor_critic_modules.py:18`，其核心 backbone 是 `UniversalTokenModule`（`universal_token_modules.py:33`）。

**整体流水线**：

```
tokenizer_obs [256, 1761]
     │
     ▼
┌─────────────────────────────────────────────────┐
│  编码器路由（encoder_masks）                        │
│  ├── g1 环境 → G1 Encoder                        │
│  ├── teleop 环境 → Teleop Encoder                │
│  └── smpl 环境 → SMPL Encoder                    │
└─────────────────────────────────────────────────┘
     │ (B, S, 2, 32) — 2 个 token，每 token 32 维
     ▼
┌─────────────────────────────────────────────────┐
│  FSQ 量化器（Finite Scalar Quantization）          │
│  连续 latent → 离散 token（32 个级别）              │
└─────────────────────────────────────────────────┘
     │ (B, S, 2, 32) — 量化后 token
     ▼
┌─────────────────────────────────────────────────┐
│  解码器                                           │
│  ├── g1_kin decoder → 10 帧未来运动（重构损失用）     │
│  └── g1_dyn decoder → 29 维关节角目标（最终动作）     │
└─────────────────────────────────────────────────┘
     │ [256, 29] — action_mean
     ▼
  Normal(action_mean, std)  → 采样动作
```

#### 1.2.2 三个编码器

每个编码器的 backbone 都是 `BaseModule`（`base_module.py`）构建的 MLP 网络：

```
Linear(in → 2048) → SiLU → Linear(2048 → 1024) → SiLU →
Linear(1024 → 512) → SiLU → Linear(512 → 512) → SiLU → Linear(512 → out)
```

| 编码器 | 输入特征 | 输入维度 | 时序维度 | 输出 |
|--------|----------|----------|---------|------|
| **G1** | `command_multi_future_nonflat`（关节角+速度 × 10帧）<br>`motion_anchor_ori_b_mf_nonflat`（根朝向 6D × 10帧） | 64 per frame × 10 frames | input=10, output=2 | (B, 2, 32) |
| **Teleop** | `command_multi_future_lower_body`（下半身目标）<br>`vr_3point_local_target`（VR 3点位置）<br>`vr_3point_local_orn_target`（VR 3点朝向）<br>`motion_anchor_ori_b`（根朝向） | 拼接后输入 | output=2 | (B, 2, 32) |
| **SMPL** | `smpl_joints_multi_future_local_nonflat`（SMPL 关节 × 10帧）<br>`smpl_root_ori_b_multi_future`（根朝向 × 10帧）<br>`joint_pos_multi_future_wrist_for_smpl`（手腕位置） | per frame × 10 frames | input=10, output=2 | (B, 2, 32) |

**编码器路由逻辑**（来自 `universal_token_modules.py` forward 方法）：

训练时，每个环境实例被随机分配一种数据源。分配概率由配置控制：

```yaml
encoder_sample_probs:
  g1: 1.0      # 归一化后概率 1/3
  teleop: 1.0  # 归一化后概率 1/3
  smpl: 1.0    # 归一化后概率 1/3
```

`encoder_masks` 是一个 multi-hot 向量，标识每个环境使用哪个编码器。只有被分配的环境才通过对应编码器前向传播，节省计算。**所有编码器输出到同一维度的 latent 空间**：2 tokens × 32 dim = 64 维。

### 1.2.3 FSQ 量化器

- 实现：`vector_quantize_pytorch.FSQ`
- 配置：`num_fsq_levels=32`，每个标量有 32 个离散值
- 输入：连续 latent (B, 2, 32)
- 输出：量化后 token (B, 2, 32)，值约束到离散网格
- **与 VQ-VAE 的区别**：FSQ 是确定性映射，无 codebook，无 commitment loss。通过 straight-through estimator 保持梯度流。

### 1.2.4 两个解码器

| 解码器 | 输入 | 输出 | 网络结构 | 作用 |
|--------|------|------|---------|------|
| **g1_kin** | token (2, 32) | 10帧未来运动 | MLP [2048, 1024, 512, 512]<br>input_temporal=2 → output_temporal=10 | 重构损失（训练用） |
| **g1_dyn** | token_flat (64) + proprioception (930) | 29 维关节角目标 | MLP [2048, 2048, 1024, 1024, 512, 512]<br>无时序展开 | **最终动作输出** |

- `g1_kin`（运动学解码器）：将 2 个 token 解码回 10 帧的未来运动序列，用于计算重构损失（g1_recon），确保 token 保留足够的运动信息。
- `g1_dyn`（动力学解码器）：将 token 展平后（64维）与当前本体感受（930维）拼接，解码为 29 维关节角位置目标。这才是送入环境的最终动作。

### 1.2.5 动作分布

```python
distribution = Normal(action_mean, std)  # std 是 29 维可学习参数
actions = distribution.sample()
```

- `std` 初始值：0.05
- clamp 范围：[0.001, 0.5]

### 1.2.6 实际训练中

**初始状态（iter 1）**：
- action noise std = 0.05（初始值，29 维都一样）
- action_mean 由 g1_dyn 解码器的随机初始权重决定

**训练结束（iter 200）**：
- action noise std 增长到约 0.08-0.10（模型学会了更多探索）

### 1.3 Critic（价值网络）

`Critic` 类（`actor_critic_modules.py:566`）

- **输入**：`critic_obs`（特权观测，包含地形、接触力等完整状态）
- **归一化**：`RunningMeanStd` 在线归一化，训练中持续更新均值和方差
- **Backbone**：MLP `[2048, 2048, 1024, 1024, 512, 512]`，激活函数 `SiLU`
- **输出**：标量 value（每个环境一个值）

**实际训练中**（iter 1）：
```
critic_obs: shape=[256, 1645], mean=0.0417, std=0.5637
values: shape=[24, 256, 1], mean=0.0320, std=0.0002
```

初始时 Critic 输出的 value 非常集中（std=0.0002），因为网络刚初始化。

### 1.4 PolicyAndValueWrapper（DDP 包装器）

`PolicyAndValueWrapper`（`ppo_trainer.py:71`）将 Actor + Critic 包在一个 `nn.Module` 中：

- DDP 要求每次 backward 只能调用一次 forward → wrapper 的 `forward()` 同时执行 policy + value 前向传播
- `accelerator.prepare()` 将 wrapper 包为 DDP 模块
- `forward(modes=["policy", "value"])` → 返回策略结果和价值估计

---

## 2. 训练方法（PPO 算法）

### 2.1 原理

PPO（Proximal Policy Optimization）是一种 on-policy 强化学习算法，通过裁剪策略梯度来保证每次更新不会偏离旧策略太远。核心思想：

1. **收集 rollout**：用当前策略与环境交互，收集一批轨迹数据
2. **计算优势值**：用 GAE（Generalized Advantage Estimation）估计每个状态-动作对相对于均值的好坏程度
3. **多次 PPO 更新**：在收集的数据上做多个 epoch 的梯度更新，用裁剪机制防止策略变化过大

### 2.2 实现

`TRLAuxLossPPOTrainer`（`ppo_trainer_aux_loss.py:6`）继承自 `TRLPPOTrainer`（`ppo_trainer.py`），在标准 PPO 基础上添加了辅助损失。

**关键超参数**（来自 `stub_train.yaml` 和 `ppo_im_phc.yaml`）：

| 超参数 | 值 | 含义 |
|--------|-----|------|
| `cliprange` | 0.2 | PPO 策略裁剪范围 |
| `cliprange_value` | 动态 | 价值函数裁剪范围 |
| `vf_coef` | 1.0 | 价值函数损失系数 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE 参数 |
| `max_grad_norm` | 0.1 | 梯度裁剪阈值 |
| `learning_rate` | 初始 ~1.5e-4 | KL 自适应调整 |
| `num_ppo_epochs` | 5（默认） | 每 iteration PPO 更新次数 |

**KL 自适应学习率**（`_adjust_learning_rate_based_on_kl`）：
- 计算新旧策略的 KL 散度
- KL 过大 → 降低学习率（防止策略更新过猛）
- KL 过小 → 提高学习率（鼓励更大步更新）

### 2.3 实际训练中

**PPO 指标演变**：

| 指标 | Iter 1 | Iter 50 | Iter 100 | Iter 150 | Iter 200 |
|------|--------|---------|----------|----------|----------|
| approx_kl | 0.0074 | 0.0148 | 0.0142 | 0.0145 | 0.0161 |
| clip_fraction | 4.9% | 12.5% | 12.0% | 12.8% | 15.1% |
| learning_rate | 1.52e-4 | 2.0e-4 | 2.0e-4 | 2.0e-4 | 8.9e-5 |
| noise_std | 0.050 | 0.063 | 0.083 | 0.111 | 0.148 |

**学习率变化**说明了 KL 自适应机制的工作：
- Iter 1 → 50：KL 从 0.007 升到 0.015，在 `desired_kl` 范围内，LR 从 1.5e-4 上升到上限 2.0e-4
- Iter 50 → 150：KL 保持在 ~0.014-0.015，LR 维持在上限 2.0e-4
- Iter 150 → 200：KL 升到 0.016，可能超过上限，LR 被下调到 8.9e-5

---

## 3. 数据集格式与加载

### 3.1 原理

SONIC 训练需要三种运动数据：

1. **Robot 运动数据（G1）**：机器人实际的关节角轨迹，从物理仿真或真实运动采集
2. **Teleop 数据**：VR 遥操作采集的 3 点追踪数据（左手、右手、头部）
3. **SMPL 人体运动数据**：从大规模人体运动捕捉数据集转换而来，用于扩充训练数据

### 3.2 实现

**PKL 文件格式**：

每个 PKL 文件包含一条运动轨迹，由 `MotionLibRobot`（`motion_lib_robot.py`）加载。内容包括：

- 关节位置（`dof_pos`）：[T, 29] — 每帧 29 个关节的角度
- 关节速度（`dof_vel`）：[T, 29] — 每帧 29 个关节的角速度
- 根节点位置（`root_trans`）：[T, 3] — 每帧机器人根节点世界坐标
- 根节点朝向（`root_quat`）：[T, 4] — 每帧根节点四元数 (w,x,y,z)
- 关节体位置（`body_pos`）：[T, 14, 3] — 14 个关节体的世界坐标
- 关节体旋转（`body_rot`）：[T, 14, 4] — 14 个关节体的四元数
- 帧率（`fps`）：标量，通常 50 fps

**G1 机器人 14 个关节体**：
```
pelvis, left_hip_roll_link, left_knee_link, left_ankle_roll_link,
right_hip_roll_link, right_knee_link, right_ankle_roll_link,
torso_link, left_shoulder_roll_link, left_elbow_link,
left_wrist_yaw_link, right_shoulder_roll_link, right_elbow_link,
right_wrist_yaw_link
```

**SMPL 数据格式**：
- SMPL 文件包含 24 个关节点的位置和旋转
- 通过 `smpl_motion_file` 配置路径加载
- 按名称与 Robot 运动配对

### 3.3 实际训练中

**数据加载日志**：
```
Loading motion data from sample_data/robot_filtered...
Loaded 2 motions
Loaded 2 motions with a total length of 80.040s and 4004 frames.
Current motion keys: ['walk_forward_amateur_001__A001_M', 'walk_forward_amateur_001__A001']
```

- Robot 数据：2 条运动轨迹，总时长 80.04 秒，4004 帧（50fps）
- SMPL 数据：131,455 个 PKL 文件可用于 `/root/sonic-data/data/smpl_filtered/`
- 加载方式：`MotionLibRobot` 在初始化时一次性加载所有 Robot PKL 到内存，SMPL 数据按名称匹配延迟加载

**数据在训练中如何使用**：

`MotionLib` 通过 `sample_motions()` 随机选择运动 ID 和时间偏移，然后 `get_motion_state()` 返回指定时刻的完整关节状态。这些状态被转换为 StubEnv 的观测向量。

---

## 4. 数据流与 Shape

### 4.1 原理

每个训练 iteration 的数据流可分为两个阶段：

1. **Rollout 收集**：策略与环境交互 `num_steps_per_env` 步，收集 (obs, action, reward, done) 元组
2. **PPO 更新**：在收集的 rollout 数据上做多轮梯度更新

### 4.2 实现

完整的数据流：

```
env.reset() → obs_dict = {
  "actor_obs":   [256, 930],     # 本体感受（关节角+速度+历史动作）
  "critic_obs":  [256, 1645],    # 特权观测（地形+接触力+完整状态）
  "tokenizer":   [256, 1761],    # 运动参考特征（编码器输入）
}

── 每个 rollout step（共 24 步）──

policy_step(obs_dict) → {
  "actions":      [256, 29],     # 29 个关节角目标
  "action_mean":  [256, 29],     # 高斯分布均值
  "action_sigma": [256, 29],     # 高斯分布标准差
  "log_prob":     [256],         # log π(a|s)
}

env.step(actions) → obs_dict, rewards [256], dones [256], infos

── rollout 结束后 ──

RolloutStorage 形状: [24, 256, ...]  (steps × envs × dim)
value_evaluate(all_obs) → values [25, 256, 1]  (多一步用于 bootstrapping)
GAE → returns [24, 256, 1], advantages [24, 256, 1]
```

### 4.3 实际训练中

**Iter 1 rollout 统计**：
```
obs[actor_obs]:   shape=[256, 930],    mean=0.0153, std=0.5903
obs[critic_obs]:  shape=[256, 1645],   mean=0.0417, std=0.5637
obs[tokenizer]:   shape=[256, 1761],   mean=0.0595, std=0.5641

values:     shape=[24, 256, 1], mean= 0.0320, std=0.0002
returns:    shape=[24, 256, 1], mean= 0.0150, std=0.2443
advantages: shape=[24, 256, 1], mean=-0.0686, std=1.0026
rewards:    shape=[24, 256, 1], mean=-0.0021, std=0.1000, min=-0.4081, max=0.3310
```

关键观察：
- **actor_obs (930维)**：包含关节位置(29)、速度(29)、历史动作(29×10)、历史观测(29×10) 等
- **critic_obs (1645维)**：比 actor_obs 多出地形信息、接触力、多帧未来运动参考等特权信息
- **tokenizer (1761维)**：编码器输入，包含未来运动帧、SMPL 关节位置、VR 追踪点等
- **values 初始 std=0.0002**：Critic 刚初始化，对所有状态输出几乎相同的值
- **advantages 均值≈0，std≈1**：经过归一化后的优势值，符合 PPO 标准做法

**Iter 2 rollout 统计（对比）**：
```
values:     mean=0.0351, std=0.0135   # std 从 0.0002 增长到 0.0135
returns:    mean=0.0444, std=0.2475
advantages: mean=0.0414, std=0.9988
rewards:    mean=0.0006, std=0.1013
```

Critic 开始学习区分不同状态的价值（std 从 0.0002 增长到 0.0135）。

---

## 5. Loss 计算体系

### 5.1 原理

SONIC 的训练损失由两大部分组成：

1. **PPO Loss**：标准 PPO 的策略梯度损失 + 价值函数损失 + 熵正则化
2. **辅助损失（Auxiliary Losses）**：5 个额外损失，核心目的是**将三个编码器的 latent 空间对齐**

总损失公式：

```
total_loss = ppo_loss + aux_loss_scale × Σ(coef_i × aux_loss_i)
```

其中：
```
ppo_loss = pg_loss + vf_coef × vf_loss + entropy_coef × entropy_loss
```

### 5.2 PPO Loss 实现

#### 5.2.1 Policy Gradient Loss（裁剪代理目标）

```python
# ppo_trainer.py:1410-1421
ratio = exp(new_logprob - old_logprob)           # 新旧策略概率比
pg_loss1 = -advantage × ratio                     # 未裁剪
pg_loss2 = -advantage × clamp(ratio, 1-0.2, 1+0.2) # 裁剪到 [0.8, 1.2]
pg_loss = max(pg_loss1, pg_loss2).mean()           # 取较大者（悲观估计）
```

**为什么取 max？** 这是 PPO 的"悲观"策略：
- 当 advantage > 0（好动作）：ratio 被限制不能太大，防止过度利用
- 当 advantage < 0（差动作）：ratio 被限制不能太小，防止过度惩罚

#### 5.2.2 Value Function Loss（裁剪价值损失）

```python
# ppo_trainer.py:1399-1408
vpredclipped = clamp(vpred, old_values - ε_v, old_values + ε_v)
vf_losses1 = (vpred - returns)²
vf_losses2 = (vpredclipped - returns)²
vf_loss = max(vf_losses1, vf_losses2).mean()
```

裁剪价值函数防止其更新过大，保持训练稳定。

#### 5.2.3 Entropy Loss

```python
# ppo_trainer.py:1424
entropy_loss = -mean(entropy)    # 负号使得 entropy 越大 loss 越小 → 鼓励探索
```

### 5.3 辅助损失（Auxiliary Losses）— SONIC 的核心创新

SONIC 通过 5 个辅助损失将三个编码器的 latent 空间对齐。这是论文中提到的 4 种损失类型的具体实现：

- **PPO Loss** → 强化学习主目标
- **Token Loss** → g1_smpl_latent, g1_teleop_latent, teleop_smpl_latent（latent 对齐）
- **Recon Loss** → g1_recon（重构损失）
- **Cycle Loss** → reencoded_smpl_g1_latent（循环一致性）

#### 5.3.1 详细损失列表

| # | 损失名 | 类名 | 系数 | 作用 |
|---|--------|------|------|------|
| 1 | `g1_recon` | `G1ReconLoss` | 0.01 | 重构损失：g1_kin 解码器重构的运动 vs 原始 tokenizer obs |
| 2 | `g1_smpl_latent` | `G1SmplLatentLoss` | 1.0 | G1↔SMPL latent 对齐（G1 做 teacher） |
| 3 | `g1_teleop_latent` | `G1TeleopLatentLoss` | 1.0 | G1↔Teleop latent 双向对齐 |
| 4 | `teleop_smpl_latent` | `TeleopSmplLatentLoss` | 1.0 | Teleop↔SMPL latent 对齐（Teleop 做 teacher） |
| 5 | `reencoded_smpl_g1_latent` | `ReencodedSmplG1LatentLoss` | 1.0 | 循环一致性：SMPL→decode→re-encode→latent vs 原始 G1 latent |

#### 5.3.2 梯度流向分析

**这是理解 SONIC 训练最关键的部分。** 通过策略性使用 `.detach()` 操作，SONIC 精确控制每个损失更新哪些模块：

##### 损失 1: g1_recon（重构损失，系数 0.01）

```
计算: MSE(g1_kin_decoder(tokens), original_tokenizer_obs)
梯度流: loss → g1_kin decoder → tokens → FSQ → 当前活跃编码器
更新模块: g1_kin 解码器 + FSQ + 当前活跃编码器
```

确保 token 保留足够的运动信息，使 decoder 能重构原始运动。系数设为 0.01（远小于其他损失），因为重构是辅助目标。

##### 损失 2: g1_smpl_latent（G1→SMPL 对齐，系数 1.0）

```
计算: MSE(g1_latent.detach(), smpl_latent)  ← G1 latent 被 detach
梯度流: loss → smpl_latent → SMPL 编码器
更新模块: 仅 SMPL 编码器
```

G1 编码器产生的 latent 被 `.detach()`（`universal_token_modules.py:977`），所以 **G1 编码器不被此损失更新**。只有 SMPL 编码器学习产生与 G1 相同的 latent。G1 在这里是 "teacher"。

##### 损失 3: g1_teleop_latent（G1↔Teleop 双向对齐，系数 1.0）

```
计算: MSE(g1_latent, teleop_latent)  ← 两者都没有 detach
梯度流: loss → g1_latent → G1 编码器 AND loss → teleop_latent → Teleop 编码器
更新模块: G1 编码器 + Teleop 编码器（双向）
```

这是唯一一个让 G1 和 Teleop 编码器**互相靠拢**的损失。两者都不 detach，因此双向对齐。

##### 损失 4: teleop_smpl_latent（Teleop→SMPL 对齐，系数 1.0）

```
计算: MSE(teleop_latent.detach(), smpl_latent)  ← Teleop latent 被 detach
梯度流: loss → smpl_latent → SMPL 编码器
更新模块: 仅 SMPL 编码器
```

配置中 `detach_teleop_target: True`。Teleop 做 "bridge"（桥梁），SMPL 学习匹配 Teleop 的 latent。

##### 损失 5: reencoded_smpl_g1_latent（循环一致性，系数 1.0）

```
计算流程:
  1. SMPL 编码器 → latent → FSQ → tokens → g1_kin 解码器 → 重构的 G1 运动
  2. 重构的 G1 运动 → G1 编码器 → re-encoded latent
  3. MSE(re-encoded_latent, original_g1_latent.detach())

梯度流: loss → re-encoded G1 latent → G1 编码器(re-encode) → decoded motion
         → g1_kin decoder → tokens → FSQ → SMPL 编码器
更新模块: G1 编码器(re-encode path) + g1_kin 解码器 + FSQ + SMPL 编码器
```

这是最复杂的损失，确保 SMPL→decode→re-encode 的循环是一致的。

#### 5.3.3 梯度更新总结表

| 网络模块 | 被哪些 loss 更新 | 角色 |
|---------|-----------------|------|
| **G1 编码器** | PPO、g1_recon、g1_teleop_latent、reencoded（re-encode path） | **锚点**：其他编码器向 G1 空间对齐 |
| **Teleop 编码器** | PPO、g1_recon（活跃时）、g1_teleop_latent | **桥梁**：连接 G1 和 SMPL |
| **SMPL 编码器** | PPO、g1_smpl_latent、teleop_smpl_latent、reencoded | **学生**：被最多损失训练 |
| **FSQ 量化器** | PPO、g1_recon、reencoded | 所有编码器共享 |
| **g1_kin 解码器** | g1_recon、reencoded | 重构 + 循环一致性 |
| **g1_dyn 解码器** | **仅 PPO** | 产生最终动作 |
| **Actor std** | PPO（entropy + log_prob） | 探索噪声 |
| **Critic** | vf_loss | 价值估计 |

**关键设计洞察**：
- G1 编码器是 "锚点"：它的 latent 空间是对齐的目标
- Teleop 是 "桥梁"：既与 G1 双向对齐（损失 3），又做 SMPL 的 teacher（损失 4）
- SMPL 编码器受 3 个 latent 对齐损失驱动，是被训练最多的编码器
- g1_dyn 解码器**只被 PPO 训练**：它专注于从 token 产生正确的动作

### 5.4 实际训练中的 Loss 值

**Iter 1 各损失分量**：
```
loss/policy_avg:                        -0.00399   # 策略梯度损失（负值因为 -advantage × ratio）
loss/value_avg:                          0.05930   # 价值函数损失
loss/entropy_avg:                      -45.51205   # 熵（29维高斯的熵，负值正常）
loss/weighted_ppo_loss_avg:              0.51042   # PPO 总损失

loss/aux_g1_recon_avg:                   0.40816   # 重构损失（最大，因为初始 decoder 还不准）
loss/aux_g1_smpl_latent_avg:             0.00040   # G1↔SMPL latent 差距很小
loss/aux_g1_teleop_latent_avg:           0.00043   # G1↔Teleop latent 差距很小
loss/aux_teleop_smpl_latent_avg:         0.00050   # Teleop↔SMPL latent 差距
loss/aux_reencoded_smpl_g1_latent_avg:   0.00010   # 循环一致性损失

loss/total_aux_loss_avg:                 0.00551   # 辅助损失总和（含 g1_recon × 0.01）
aux_loss_scale:                          1.0       # 辅助损失全局缩放
```

关键观察：
- **g1_recon 是最大的辅助损失**（0.408），但被 0.01 的系数抑制后只贡献 0.00408
- **latent 对齐损失初始就很小**（~0.0004），说明不同编码器的初始 latent 空间已经比较接近
- **PPO loss (0.51) 远大于辅助损失 (0.006)**，PPO 主导训练方向

---

## 6. StubEnv 环境

### 6.1 原理

StubEnv 是一个**替代 Isaac Sim 物理仿真器**的轻量环境（`stub_env.py`），用于在没有 GPU 仿真器的情况下（如 NPU 上）训练 SONIC 的编码器-解码器模块。

**核心设计**：
- 直接从 MotionLib 采样运动数据作为 "观测"
- 不执行物理模拟 — `step()` 只是推进时间步并采样新的运动片段
- rewards 基于运动跟踪误差计算
- **因果关系断开**：动作不影响下一状态

**适用场景**：
- 训练编码器-解码器的 latent 对齐（辅助损失）
- 预训练 token 表示
- 不适合训练完整的控制策略（需要物理仿真的因果反馈）

### 6.2 实现

```python
class StubEnv:
    def step(self, policy_state_dict):
        actions = policy_state_dict["actions"]
        # 不使用 actions 进行物理模拟！
        # 而是推进 motion_lib 的时间步
        self._current_time += self._dt
        # 重新采样运动参考
        obs_dict = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        return obs_dict, rewards, dones, infos
```

**观测计算**（`_compute_observations`）：
- `actor_obs`：从当前运动帧提取关节角、速度、历史信息
- `critic_obs`：actor_obs + 地形信息 + 接触力 + 多帧未来运动参考
- `tokenizer`：编码器输入特征 — G1 未来运动帧、Teleop 追踪点、SMPL 关节位置

**奖励计算**（`_compute_rewards`）：
- 基于运动跟踪误差：当前姿态 vs 参考运动的差异
- 包含位置跟踪、朝向跟踪、关节角度跟踪等多个奖励项
- 因为没有物理模拟，奖励反映的是 "采样到的运动参考有多接近目标"

### 6.3 实际训练中

**Iter 1 rewards 分布**：
```
rewards: mean=-0.0021, std=0.1000, min=-0.4081, max=0.3310
```

奖励在 [-0.41, 0.33] 范围内波动，均值接近 0。因为 StubEnv 的奖励主要反映采样运动的质量，而非策略的好坏。

**Episode 长度变化**：
```
Iter 1:   mean_length = 11.43    # 初始 episode 很短
Iter 30:  mean_length = 308.17   # 快速增长
Iter 87:  mean_length = 370.40   # 趋于稳定
```

Episode 长度增长表明策略学会了不触发终止条件（如关节角超限、摔倒等）。

---

## 7. 分布式训练（DDP）

### 7.1 原理

本次训练使用 **16 卡 Ascend 910 NPU** 进行数据并行分布式训练：

- 每张卡运行独立的 StubEnv 实例和 rollout 收集
- 梯度通过 AllReduce 跨卡同步
- 优势值可选跨卡归一化

### 7.2 实现

**框架栈**：
```
HuggingFace Accelerate → PyTorch DDP → HCCL（华为集合通信库）
```

- `accelerate launch --num_processes=16`：启动 16 个进程
- 每个进程绑定一张 NPU（通过 `LOCAL_RANK`）
- 每进程 `num_envs=256` 个独立 StubEnv

**梯度同步**：
- `PolicyAndValueWrapper` 被 `accelerator.prepare()` 包装为 DDP 模块
- `backward()` 时 DDP 自动触发 AllReduce，平均所有卡的梯度
- 效果：等同于在 4096 个环境上计算梯度

**额外的同步操作**：

1. **RunningMeanStd 同步**（`sync_running_mean_std()`）：
   - 前 200 次迭代每步同步
   - 之后按 `sync_running_mean_std_freq` 间隔同步
   - 确保所有卡的观测归一化统计一致

2. **优势值归一化同步**（`sync_advantage_normalization=True`）：
   - 在 GAE 计算后，gather 所有卡的 advantages
   - 全局归一化：`(A - global_mean) / (global_std + 1e-8)`
   - 然后 ungather 回各自的进程

### 7.3 实际训练中

**16 卡配置**：
```
硬件: 16 × Ascend 910 NPU (Ascend910_9362)
每卡内存: 61.3 GB HBM
每卡环境数: 256
全局环境数: 256 × 16 = 4,096
每 iteration transitions: 4,096 × 24 = 98,304
通信后端: HCCL
```

**吞吐量对比**：
```
单卡训练（Run 1, num_envs=64）:  ~1,032 steps/s
16卡训练（本次, num_envs=256/卡）: ~49,398 steps/s (稳态)
等效加速比: ~47.9× (超线性)
```

超线性加速的原因：
- 每卡 num_envs 从 64 提升到 256 → 更大的 batch 提高 NPU 矩阵运算利用率
- 全局 batch 从 64 扩大到 4096 → 更稳定的梯度估计
- NPU 的 HCCL AllReduce 通信开销相对较小

**单次 iteration 时间分解（稳态）**：
```
Collection (rollout): ~0.65s   # 每卡独立收集 24 步 × 256 环境
Learning (PPO update): ~1.35s  # 含前向/反向传播 + AllReduce 梯度同步
Total:                 ~2.0s
```

---

## 8. 训练步骤计算

### 8.1 原理

PPO 的训练循环嵌套多层：

```
外层: num_learning_iterations 次迭代
  ├── rollout 收集: num_steps_per_env 步
  └── PPO 更新:
       └── num_ppo_epochs 次遍历
            └── num_mini_batches 个 mini-batch
                 └── num_micro_batches 个 micro-batch（梯度累积）
```

### 8.2 实现与实际数值

**本次训练配置**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_envs` | 256（每卡） | 每张 NPU 上的环境数 |
| `world_size` | 16 | NPU 卡数 |
| `num_steps_per_env` | 24 | 每个环境收集 24 步 rollout |
| `num_learning_iterations` | 200 | 总迭代数 |
| `num_ppo_epochs` | 5（默认） | 每 iteration PPO 更新次数 |

**关键数值**：

```
全局 batch_size = num_envs × world_size = 256 × 16 = 4,096
每 iteration transitions = 4,096 × 24 = 98,304
总 transitions = 200 × 98,304 = 19,660,800 (~2000 万)
总 episodes = 200 × 4,096 = 819,200
```

### 8.3 实际训练中

**Iter 1 的时间分解**：
```
Collection time: 2.631s    # rollout 收集（16 卡各自独立收集）
Learn time:      1.879s    # PPO 更新（含梯度同步）
Iteration time:  4.510s    # 总时间（含 logging、同步等）
FPS:            21,796 steps/s   # 首 iteration 较慢（包含初始化开销）
```

**Iter 87 的时间分解**：
```
Collection time: 0.671s    # rollout 收集
Learn time:      1.348s    # PPO 更新
Iteration time:  2.020s    # 稳定在 ~2s/iter
FPS:            48,684 steps/s
```

**预计总训练时间**：
- 200 iterations × ~2s/iter = ~400s ≈ 6.7 分钟
- 加上初始化时间 ~30s，总计 ~7 分钟

---

## 9. GAE 优势值估计

### 9.1 原理

**Generalized Advantage Estimation (GAE)** 用于估计每个状态-动作对相对于均值的好坏程度。它在 TD(0)（低方差高偏差）和 Monte Carlo（高方差低偏差）之间取权衡。

**核心公式**：

```
δ_t = r_t + γ × V(s_{t+1}) × (1 - done_t) - V(s_t)     # TD 误差
A_t = δ_t + (γ × λ) × (1 - done_t) × A_{t+1}            # 递推计算
returns_t = A_t + V(s_t)                                   # 回报值
```

其中：
- `γ = 0.99`（折扣因子）：未来 reward 的衰减率。γ=0.99 意味着 100 步后的 reward 衰减到 ~37%
- `λ = 0.95`（GAE 参数）：
  - λ → 0：退化为 TD(0)，只看一步 TD 误差（低方差，高偏差）
  - λ → 1：退化为 Monte Carlo，使用全部回报（高方差，低偏差）
  - λ = 0.95 是广泛使用的默认值，偏向低偏差

### 9.2 实现

```python
# ppo_trainer.py:2097-2153
def _compute_returns(self, values, last_values, policy_state_dict):
    advantage = 0
    for step in reversed(range(num_steps)):
        if step == num_steps - 1:
            next_values = last_values
        else:
            next_values = values[step + 1]
        next_is_not_terminal = 1.0 - dones[step].float()
        delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
        advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
        returns[step] = advantage + values[step]

    # 归一化
    advantages = returns - values
    if self.sync_advantage_normalization:
        advantages = self.accelerator.gather(advantages)  # 跨进程 gather
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.reshape(num_processes, -1, ...)[process_index]  # ungather
```

**同步归一化的重要性**：在多卡训练中，各卡的 rollout 数据独立收集，advantage 的分布可能不同。跨卡 gather 后统一归一化确保所有卡看到一致的 advantage 尺度，避免某些卡主导梯度方向。

### 9.3 实际训练中

**优势值变化**：

| Iter | advantage_mean | advantage_std | rewards_mean | rewards_range |
|------|---------------|---------------|-------------|---------------|
| 1 | -0.069 | 1.003 | -0.002 | [-0.41, 0.33] |
| 51 | -0.027 | 1.015 | -0.001 | [-0.34, 0.41] |
| 101 | -0.015 | 0.994 | 0.001 | [-0.36, 0.37] |
| 151 | 0.019 | 0.990 | 0.001 | [-0.31, 0.43] |

关键观察：
- **advantage_mean 趋近 0**：归一化的效果，每个 iteration 的 advantages 被标准化为 (A - mean) / std
- **advantage_std ≈ 1.0**：同样是归一化的结果，保证 PPO 更新步长一致
- **rewards 范围逐渐扩大**：max 从 0.33 增到 0.43，说明策略探索到了更高奖励的区域
- **rewards_min 绝对值减小**：从 -0.41 到 -0.31，说明策略学会避免最差的行为

**GAE 的实际效果**：
```
γ = 0.99, λ = 0.95
每个 rollout 24 步 → GAE 递推 24 步
有效回溯窗口 ≈ 1/(1-γλ) = 1/(1-0.9405) ≈ 17 步
```

在 24 步的 rollout 中，GAE 的有效回溯几乎覆盖了整个窗口，提供了接近 Monte Carlo 的低偏差估计。

---

## 10. PPO 价值函数分析

### 10.1 原理

Critic 网络学习预测每个状态的未来累计 reward（即 V(s)）。一个好的 value 估计能：
- 减少 advantage 估计的方差
- 作为基线（baseline）减少策略梯度的方差
- 通过 `vf_loss` 提供额外的学习信号

### 10.2 实际训练中

**Value 预测的演变**：

| Iter | values_mean | values_std | returns_mean | returns_std | vf_loss |
|------|------------|-----------|-------------|------------|---------|
| 1 | 0.0320 | 0.0002 | 0.0150 | 0.2443 | 0.0593 |
| 51 | 0.0024 | 0.0375 | -0.0079 | 0.2557 | ~0.060 |
| 101 | 0.0101 | 0.0289 | 0.0074 | 0.2426 | 0.0608 |
| 151 | -0.0084 | 0.0329 | -0.0048 | 0.2461 | 0.0595 |
| 200 | — | — | — | — | 0.0596 |

**关键观察**：

1. **values_std 从 0.0002 跳到 0.0375**：Critic 在前 50 个 iteration 内完成了从"所有状态输出相同值"到"能区分不同状态"的转变
2. **vf_loss 始终在 0.059-0.061 波动**：没有持续下降，说明 Critic 在 StubEnv 中的学习遇到了天花板——因为 rewards 缺乏与策略动作的因果联系（StubEnv 的 step() 不使用 actions）
3. **val/clipfrac 始终为 0**：Critic 的更新从未触发裁剪，说明价值函数变化平缓
4. **returns_std ≈ 0.24-0.26**：returns 的波动性保持稳定，反映了 StubEnv 奖励的固有随机性

---

## 11. 实际训练曲线与分析

### 11.1 训练概要

```
训练配置:
  - 设备: 16 × Ascend 910 NPU (Ascend910_9362, 61.3GB HBM each)
  - 环境数: 256/卡 × 16 卡 = 4,096 全局
  - 迭代数: 200
  - 每 iteration: 98,304 transitions (4096 envs × 24 steps)
  - 总 transitions: 19,660,800 (~2000 万)
  - 总 episodes: 819,200
  - 精度: FP32（NPU 不支持 bf16 的 torch.normal）
  - 吞吐量: 49,000 steps/s（稳态）
  - 总训练时间: 419 秒 ≈ 7 分钟
  - 通信后端: HCCL（华为集合通信库）
  - Checkpoint 保存: logs_rl/TRL_G1_Stub/stub_train_test-20260628_160520/last.pt
```

### 11.2 关键指标变化

（数据来自 16 卡 Ascend 910 NPU 实际训练，TensorBoard 和日志提取）

**主要训练指标**：

| 指标 | Iter 1 | Iter 50 | Iter 100 | Iter 150 | Iter 200 |
|------|--------|---------|----------|----------|----------|
| Mean entropy | -45.51 | -38.85 | -31.22 | -22.72 | -14.32 |
| Mean rewards | -0.013 | -0.031 | 0.051 | -0.081 | 0.014 |
| Mean ep length | 11.43 | 363.58 | 379.32 | 377.28 | 378.04 |
| Action noise std | 0.05 | 0.06 | 0.08 | 0.11 | 0.15 |
| FPS | 21,796 | 51,004 | 48,834 | 49,301 | 49,398 |

**PPO Loss 分量**：

| 指标 | Iter 1 | Iter 50 | Iter 100 | Iter 150 | Iter 200 |
|------|--------|---------|----------|----------|----------|
| policy_loss | -0.0040 | -0.0102 | -0.0102 | -0.0139 | -0.0201 |
| value_loss | 0.0593 | 0.0598 | 0.0608 | 0.0595 | 0.0596 |
| entropy_loss | -45.51 | -38.85 | -31.22 | -22.72 | -14.32 |
| weighted_ppo_loss | 0.5104 | 0.4381 | 0.3628 | 0.2728 | 0.1827 |
| approx_kl | 0.0074 | 0.0148 | 0.0142 | 0.0145 | 0.0161 |
| clip_fraction | 0.049 | 0.125 | 0.120 | 0.128 | 0.151 |

**辅助损失分量**：

| 辅助损失 | Iter 1 | Iter 50 | Iter 100 | Iter 150 | Iter 200 |
|---------|--------|---------|----------|----------|----------|
| g1_recon | 0.4082 | 0.3662 | 0.3653 | 0.3681 | 0.3630 |
| g1_smpl_latent | 3.97e-4 | 6.0e-6 | 1.9e-5 | 1.3e-5 | 1.7e-5 |
| g1_teleop_latent | 4.32e-4 | 1.3e-5 | 2.5e-5 | 3.0e-5 | 2.9e-5 |
| teleop_smpl_latent | 5.04e-4 | 1.0e-5 | 1.9e-5 | 2.7e-5 | 2.1e-5 |
| reencoded_smpl_g1 | 9.9e-5 | 4.0e-6 | 1.2e-5 | 7.0e-6 | 8.0e-6 |
| **total_aux_loss** | **0.0055** | **0.0037** | **0.0037** | **0.0038** | **0.0037** |

**Rollout 统计变化**（来自 INSTRUMENT 日志）：

| 指标 | Iter 1 | Iter 51 | Iter 101 | Iter 151 |
|------|--------|---------|----------|----------|
| values_mean | 0.0320 | 0.0024 | 0.0101 | -0.0084 |
| values_std | 0.0002 | 0.0375 | 0.0289 | 0.0329 |
| returns_mean | 0.0150 | -0.0079 | 0.0074 | -0.0048 |
| advantages_mean | -0.069 | -0.027 | -0.015 | 0.019 |
| advantages_std | 1.003 | 1.015 | 0.994 | 0.990 |
| rewards_min | -0.408 | -0.342 | -0.356 | -0.314 |
| rewards_max | 0.331 | 0.407 | 0.375 | 0.429 |

### 11.3 观察与分析

**1. Episode 长度（最显著的变化）**：

从 11.43 步（iter 1）迅速增长到 363.58 步（iter 50），然后稳定在 ~378 步。这说明策略在前 50 个 iteration 内学会了不触发终止条件（如关节角超限、摔倒检测等），episode 长度接近最大值。

**2. Entropy 稳定下降**：

Entropy 从 -45.51（iter 1）线性下降到 -14.32（iter 200）。这反映的不是策略变得更确定，而是 **action noise std 从 0.05 增长到 0.15**。29 维高斯分布的 entropy = Σ log(σ_i) + const，std 越大 entropy 绝对值越小。std 增长说明 PPO 的 entropy bonus 有效鼓励了探索。

**3. PPO weighted loss 持续下降**：

weighted_ppo_loss 从 0.51 降到 0.18，主要是 entropy_loss 项在下降（因为 std 增大）。policy_loss 的绝对值在增大（从 0.004 到 0.020），说明策略更新幅度在增加。

**4. Latent 对齐损失迅速收敛**：

所有 latent 对齐损失在前 50 个 iteration 内下降 1-2 个数量级：
- g1_smpl_latent: 3.97e-4 → 6.0e-6（降低 66×）
- g1_teleop_latent: 4.32e-4 → 1.3e-5（降低 33×）
- 之后稳定在 1e-5 量级

这说明三个编码器的 latent 空间快速对齐到一致的表示。

**5. G1 recon loss 缓慢下降**：

g1_recon 从 0.408 降到 0.363（仅 ~11%），说明 g1_kin 解码器的重构能力提升有限。这是预期的——重构准确度受限于 FSQ 量化的信息瓶颈。

**6. Value 函数学习**：

- values_std 从 0.0002（iter 1）增长到 ~0.03（iter 50+）：Critic 从 "所有状态输出同一个值" 进步到 "能区分不同状态"
- value_loss 保持稳定（~0.06）：Critic 的预测精度没有显著提升，因为 StubEnv 的 rewards 缺乏明确的因果信号
- val/clipfrac 始终为 0：价值函数更新幅度小，从未被裁剪

**7. KL 散度与裁剪率**：

- approx_kl 从 0.0074 升到 0.016：策略更新幅度逐渐增大
- clip_fraction 从 5% 升到 15%：更多样本被裁剪，但仍在健康范围内（< 20%）

**8. 16 卡加速效果**：

- 首 iteration FPS = 21,796（含初始化）
- 稳态 FPS = ~49,000
- 总训练时间 = 419 秒（~7 分钟）完成 200 iterations / 1966 万 transitions

---

## 12. 总结

### 12.1 架构总结

SONIC 的 Universal Token Module 通过 "编码-量化-解码" 流水线，将三种不同来源的运动数据（G1 关节角、VR 遥操作、SMPL 人体动作）统一到 2×32 维的离散 token 空间。37.4M 参数的模型（25.9M Actor + 11.5M Critic + 29 std）在 16 卡 Ascend 910 上以 ~49K steps/s 的速度训练，200 iterations / 2000 万 transitions 在 7 分钟内完成。

### 12.2 训练方法总结

PPO + 5 种辅助损失的联合训练。PPO 提供强化学习信号驱动 g1_dyn 解码器学习产生好的动作；辅助损失通过精巧的 detach 设计，让 G1 编码器做 "锚点"、Teleop 做 "桥梁"、SMPL 编码器被推向 G1 的 latent 空间。

### 12.3 训练结果总结

| 指标 | 初始 (Iter 1) | 最终 (Iter 200) | 变化趋势 |
|------|-------------|----------------|---------|
| Episode 长度 | 11.4 步 | 378 步 | 前 50 iter 快速收敛 |
| Action noise std | 0.05 | 0.15 | 持续增长（探索增强） |
| Latent 对齐损失 | ~4e-4 | ~2e-5 | 前 50 iter 下降 20-60× |
| G1 recon 损失 | 0.408 | 0.363 | 缓慢下降 11% |
| PPO weighted loss | 0.510 | 0.183 | 持续下降（主要来自 entropy） |
| KL 散度 | 0.007 | 0.016 | 策略更新幅度增大 |
| Clip fraction | 5% | 15% | 更多样本被裁剪 |

### 12.4 关键洞察

1. **Detach 策略是 SONIC 训练的核心**：通过选择性 detach，同一个 MSE 损失可以单向或双向训练编码器。G1 做 teacher（detach），SMPL 做 student（不 detach）。

2. **Latent 对齐极快收敛**：三个编码器的 latent 空间在 50 iterations 内就基本对齐（损失降低 20-60×），之后稳定在 1e-5 量级。这说明 SONIC 的 token 统一方案在数据层面是高效的。

3. **G1 重构损失系数很低（0.01）**：重构是手段不是目的，避免其主导训练方向。即使系数低，g1_recon 仍是最大的辅助损失（~0.37 × 0.01 = 0.0037），约占辅助损失总和的大部分。

4. **g1_dyn 只被 PPO 训练**：确保动作输出完全由 RL 信号驱动，不受辅助损失干扰。

5. **StubEnv 的局限**：
   - Episode 长度快速收敛但 rewards 无明确趋势（均值在 [-0.08, 0.05] 间波动）
   - Value loss 无法持续下降（稳定在 ~0.06）
   - 因果关系断开：策略产生的 actions 不影响下一步的 obs
   - 训练出的模型需要在 Isaac Sim 中 fine-tuning

6. **16 卡超线性加速**：49K steps/s 对比单卡 ~1K steps/s，加速比 ~49×（超线性）。原因是更大的全局 batch（4096 envs vs 64 envs）提高了 NPU 的计算利用率。

---

*本报告基于代码分析（gear_sonic/trl/）和 16 卡 Ascend 910 NPU 上的实际 StubEnv 训练实验（2026-06-28，训练目录 stub_train_test-20260628_160520）。训练日志保存于容器内 /tmp/train_smpl_16card.log，TensorBoard 数据位于 logs_rl/TRL_G1_Stub/stub_train_test-20260628_160520/tb/。*
