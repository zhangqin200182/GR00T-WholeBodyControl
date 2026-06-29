# SONIC 训练原理与实验报告

---

# 上篇：原理与流程

---

## 第 0 章：全景概览

### SONIC 是什么

SONIC（Systematic Orchestration of Neural Imitation Control）是一个人形机器人全身控制系统。它的核心创新是**统一 Token 表示**：将三种不同来源的运动数据——机器人关节回放（G1）、VR 遥操作（Teleop）、人体动作捕捉（SMPL）——映射到同一个离散 token 空间，使得一个解码器就能从任意输入生成一致的机器人动作。

### 端到端训练流程

```
┌─────────────────────────────── 一个 Iteration ────────────────────────────────┐
│                                                                               │
│  ┌─── Rollout 收集（24 步 × 4096 环境 = 98,304 条数据）────────────────────┐   │
│  │                                                                         │   │
│  │  MotionLib（PKL 运动数据）                                               │   │
│  │       │                                                                 │   │
│  │       ▼                                                                 │   │
│  │  ┌─────────┐    obs_dict     ┌─────────────────────────┐   action[29]  │   │
│  │  │ StubEnv │ ──────────────→ │ Actor                   │ ───────────┐  │   │
│  │  │         │                 │ 编码器→FSQ→解码器→Normal │            │  │   │
│  │  │         │ ←───────────────│                         │ ←──────────┘  │   │
│  │  │         │  reward, done   └─────────────────────────┘   env.step()  │   │
│  │  └─────────┘                                                           │   │
│  │       │                                                                 │   │
│  │       │ obs_dict                                                        │   │
│  │       ▼                                                                 │   │
│  │  ┌──────────┐                                                           │   │
│  │  │ Critic   │ → V(s)                                                    │   │
│  │  └──────────┘                                                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌─── 优势值计算 ──────────────────────────────────────────────────────────┐   │
│  │  rewards + V(s) → GAE 反向递推 → advantages（归一化）                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌─── PPO 更新（5 epochs × 4 mini-batches = 20 次梯度更新）───────────────┐   │
│  │                                                                         │   │
│  │  total_loss = PPO loss（策略 + 价值 + 熵）                               │   │
│  │             + 辅助 loss（latent 对齐 + 重构 + 循环一致性）                │   │
│  │                                                                         │   │
│  │  → 梯度裁剪 → Adam 优化器更新 → AllReduce 跨卡同步                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                              ↓ 重复 200 次
                         训练完成（~7 分钟）
```

### 一个 Iteration 的关键数字

```
全局环境数:        256 envs/卡 × 16 卡 = 4,096
每 iteration 数据:  4,096 × 24 步 = 98,304 条 transitions
PPO 更新:          5 epochs × 4 mini-batches = 20 次梯度更新
总训练量:          200 iterations × 98,304 = 19,660,800 transitions (~2000 万)
```

---

## 第 1 章：数据 — 运动从哪来

SONIC 训练使用三种运动数据源，它们格式不同但最终都被统一到 token 空间。

### 1.1 PKL 文件格式

每个 PKL 文件包含一条运动轨迹，使用 `joblib` 序列化（带 zlib 压缩）：

```python
motion_data = joblib.load(pkl_path)  # motion_lib_base.py:367
```

**Robot G1 数据**（`robot_filtered/` 目录）：

```python
{
    'walk_forward_amateur_001__A001_M': {
        'dof':               (1202, 29),    # 29 个关节角度，[-1.69, 1.62] 弧度
        'pose_aa':           (1202, 30, 3), # 轴角姿态
        'root_rot':          (1202, 4),     # 根节点四元数 (w,x,y,z)
        'root_trans_offset': (1202, 3),     # 根节点位移
        'fps':               30,
    }
}
# 1202 帧 @ 30fps = 40.1 秒的行走动作
```

**SMPL 人体数据**（`smpl_filtered/` 目录，131,455 个文件，~31GB）：

```python
{
    'fps':         50,
    'pose_aa':     (N, 72),      # 24 关节 × 3 轴角 = 72 维
    'smpl_joints': (N, 24, 3),   # 24 关节的 3D 位置
    'transl':      (N, 3),       # 全局位移
}
# N = 144~6527 帧，总计 ~1324 小时人体运动
```

### 1.2 MotionLib 加载与运行时采样

`MotionLibRobot`（`motion_lib_robot.py`）在初始化时一次性加载所有 PKL 到内存。运行时：

1. `sample_motions()` — 随机选择运动 ID 和起始时间
2. `get_motion_state()` — 返回指定时刻的完整关节状态（位置、速度、朝向等）

### 1.3 自适应采样

MotionLib 支持基于失败率的自适应采样，让策略更多练习难的动作片段：

```
每条运动按 50 帧一个 bin 切分
  → 追踪每个 bin 的失败率（terminated 次数 / 采样次数）
  → 采样概率 ∝ failure_rate(bin)
  → 防过拟合：单 bin 最大概率 ≤ 50 × mean_prob，与均匀分布混合
```

采样过程（`sample_motion_ids_and_time_steps`, line 2813）：按概率采样 bin → bin 内均匀采样起始帧 → `torch.multinomial` 加权采样。

---

## 第 2 章：模型架构

### 2.1 整体结构

SONIC 模型由 **Actor（策略网络）** 和 **Critic（价值网络）** 组成，共 37.4M 参数：

| 组件 | 参数量 | 部署大小 |
|------|--------|---------|
| Actor (`actor_module`) | 25.9M | 98.7 MB |
| Critic (`critic_module`) | 11.5M | 43.9 MB |
| 动作噪声 (`std`) | 29 | — |
| **总计** | **37.4M** | **428 MB**（含 Adam 优化器 285MB） |

部署时只需 Actor 权重（98.7MB）。Adam 优化器为每个参数维护一阶动量 m 和二阶动量 v，所以优化器 ≈ 2 × (98.7+43.9) = 285MB。

### 2.2 Actor — 编码-量化-解码流水线

Actor 的核心是 `UniversalTokenModule`（`universal_token_modules.py:33`），一条完整的编码-量化-解码流水线：

```
tokenizer_obs [B, 1761]
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│  编码器路由（根据 encoder_masks 选择）                       │
│  ├── G1 Encoder    ← 机器人关节回放数据                     │
│  ├── Teleop Encoder ← VR 遥操作数据                        │
│  └── SMPL Encoder  ← 人体动捕数据                          │
│                                                            │
│  每个编码器: MLP [in→2048→1024→512→512→out], 激活 SiLU      │
│  输出统一维度: (B, 2, 32) — 2 个 token，每 token 32 维      │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│  FSQ 量化器（Finite Scalar Quantization）                   │
│  连续 latent → 离散 token（32 个级别/标量）                  │
│  确定性映射，无 codebook，straight-through estimator         │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│  两个解码器                                                 │
│  ├── g1_kin decoder → 10 帧未来运动（训练用，计算重构损失）    │
│  │   MLP [2048,1024,512,512], input_temporal=2→output=10   │
│  │                                                         │
│  └── g1_dyn decoder → 29 维关节角目标（最终动作）             │
│      MLP [2048,2048,1024,1024,512,512]                     │
│      输入: token_flat(64) + proprioception(930) = 994 维    │
└──────────────────────────────────────────────────────────┘
     │
     ▼ action_mean [B, 29]
  Normal(action_mean, std) → 采样动作 [B, 29]
```

**三个编码器的输入**：

| 编码器 | 输入内容 | 时序 |
|--------|---------|------|
| **G1** | 未来 10 帧关节角+速度 + 根朝向 6D | input=10 → output=2 |
| **Teleop** | 下半身目标 + VR 3点位置/朝向 + 根朝向 | output=2 |
| **SMPL** | 未来 10 帧 SMPL 关节位置 + 根朝向 + 手腕位置 | input=10 → output=2 |

**编码器路由**：训练时每个环境实例被随机分配一种数据源（概率各 1/3）。`encoder_masks` 是 multi-hot 向量——SMPL 被选中时 G1 编码器总是同时激活（用于 latent 对齐），Teleop 有 50% 概率也激活。**所有编码器输出到同一 latent 空间**：2 tokens × 32 dim = 64 维。

### 2.3 Critic — 价值网络

`Critic`（`actor_critic_modules.py:566`）是独立的 MLP：

- **输入**：`critic_obs`（1645 维特权观测，含地形、接触力等完整状态）
- **归一化**：`RunningMeanStd` 在线更新均值和方差
- **Backbone**：MLP `[2048, 2048, 1024, 1024, 512, 512]`，SiLU 激活
- **输出**：标量 V(s)

### 2.4 可学习动作噪声 std

```python
# actor_critic_modules.py:120
self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))
# 29 维可学习参数，初始值 0.05，clamp [0.001, 0.5]
```

`std` 不依赖状态——所有状态共享同一个 std 向量。两个力量在博弈：

- **Entropy loss**（`-mean(entropy)`，coef > 0）→ 鼓励 std 增大，增加探索
- **PPO policy loss**（通过 log_prob 梯度）→ 倾向 std 减小，动作更确定

高斯 entropy = `0.5 × log(2πeσ²)`，std 越大 entropy 越大，entropy_loss 越小。

### 2.5 DDP 包装器

`PolicyAndValueWrapper`（`ppo_trainer.py:71`）将 Actor + Critic 包在一个 `nn.Module` 中，因为 DDP 要求每次 backward 只能调一次 forward。`accelerator.prepare()` 将其包装为 DDP 模块。

---

## 第 3 章：一个 Iteration 的完整生命周期

> 这是报告的核心章节。我们用 **env[42]**（4096 个并行环境中的第 42 个）的视角，完整走一遍从数据采样到梯度更新的全过程。

### 3.1 Rollout 收集 — 24 步与环境的交互

Trainer 的 rollout 循环（`ppo_trainer.py:918-984`）极其简洁：

```python
for step in range(24):                        # num_steps_per_env = 24
    actions = policy_step(obs_dict)            # 策略推理
    obs_dict, rewards, dones, infos = env.step(actions)  # 环境执行
    storage.store(obs, actions, rewards, dones, ...)
```

Trainer 只做两件事：调用策略、调用 env.step()。所有 reset 逻辑都在 env.step() 内部完成。

#### 端到端数据流追踪

以 env[42] 被分配到 SMPL 编码器为例，追踪一条运动数据从磁盘到最终动作的完整路径：

```
Step 1: 数据加载
  smpl_filtered/dance_001.pkl → joblib.load() → {pose_aa: (500,72), smpl_joints: (500,24,3), ...}

Step 2: MotionLib 采样
  sample_motions() → motion_id=42, start_frame=100
  get_motion_state(motion_id=42, t=100..109) → 10 帧未来运动状态

Step 3: 环境构造观测（StubEnv._compute_observations）
  ┌──────────────────────────────────────────────────────────────┐
  │ tokenizer_obs [1761 维]                                      │
  │   ├── encoder_index: [2]          (SMPL=2)                   │
  │   ├── smpl_joints_future: [10×24×3=720]  (10帧×24关节×xyz)   │
  │   ├── smpl_root_ori: [10×6=60]    (10帧×6D旋转)              │
  │   ├── command_multi_future: [10×29=290]  (G1编码器同时需要)   │
  │   ├── motion_anchor_ori: [10×6=60]                           │
  │   └── ... 其他字段                                           │
  │                                                              │
  │ actor_obs [930 维] — 10 帧历史滑动窗口                        │
  │   ├── joint_pos_history:  10×29 = 290                        │
  │   ├── joint_vel_history:  10×29 = 290                        │
  │   ├── gravity_dir_history: 10×3 = 30                         │
  │   ├── ang_vel_history:     10×3 = 30                         │
  │   └── action_history:     10×29 = 290                        │
  │                                                              │
  │ critic_obs [1645 维] — actor_obs + 特权信息                   │
  │   ├── actor_obs 内容: 930                                    │
  │   ├── 运动目标（多帧未来参考）: ~700                          │
  │   └── 地形/接触力等: ~15                                     │
  └──────────────────────────────────────────────────────────────┘

Step 4: Actor 前向传播
  tokenizer_obs[1761]
       │
       ▼ encoder_index=2 → 选择 SMPL Encoder
  SMPL Encoder(smpl_joints + root_ori + wrist)
       │
       ▼ latent [2, 32] — 2 个 token，每 token 32 维连续值
  FSQ 量化
       │
       ▼ quantized_token [2, 32] — 每个标量被映射到 32 个离散级别之一
  g1_dyn Decoder(token_flat[64] ⊕ actor_obs[930] = [994])
       │
       ▼ action_mean [29] — 29 个关节的目标角度均值
  Normal(action_mean, std=[0.05]*29)
       │
       ▼ action [29] — 采样得到的关节角目标
         例如: [0.12, -0.34, 0.08, ..., 0.21]（29 个弧度值）

Step 5: env.step(action)
  StubEnv: 不执行物理模拟，只推进 MotionLib 时间步
  → 计算 reward（运动跟踪误差）
  → 检查终止条件
  → 返回新的 obs_dict, reward, done
```

#### History Buffer 滑动窗口

环境维护每个 env 的 10 帧观测历史：

```python
# 每步更新：向左滑动 1 格，写入最新帧
self._joint_pos_history = torch.roll(self._joint_pos_history, shifts=-1, dims=1)
self._joint_pos_history[:, -1] = current_joint_pos
```

Reset 时全部清零，防止旧 episode 信息泄漏。

### 3.2 Episode 生命周期

一个 episode = 从 reset 到 done=True 的完整交互序列。Episode 长度可变，与 24 步 rollout 窗口独立。

以 env[42] 的 24 步窗口为例：

```
步骤:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
done:  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0
t_out: 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0
       ├──── Episode A (8步) ────┤  ├────────── Episode B (12步) ──────────┤  ├─ C ─┤
              terminated(失败)                              truncated(播完)    继续中
```

- **Episode A (步 0-7)**：策略表现差，第 7 步触发终止条件（如关节角超限） → done=1, time_outs=0
- **Episode B (步 8-19)**：策略表现好，运动片段正常播完 → done=1, time_outs=1
- **Episode C (步 20-23)**：新 episode 开始，窗口结束时未完成

**Auto-reset 机制**：reset 完全由 env 内部管理。`env.step()` 返回的 `obs_dict` 对 done 的 env 已经是**新 episode 的第一个 obs**——trainer 无感知。

#### Terminated vs Truncated

**Terminated（终止 — 失败）**：物理条件检测到策略表现太差。

| 终止条件 | 阈值 | 含义 |
|---------|------|------|
| `exceeded_anchor_pos` | 0.15m | 根节点位置偏差 > 15cm |
| `exceeded_anchor_ori` | 0.2 | 根节点朝向偏差过大 |
| `exceeded_foot_pos` | 0.2m | 脚部位置偏差 > 20cm |
| `exceeded_ee_body_pos` | 0.15m | 末端执行器偏差 > 15cm |
| `cumulative_error` | 连续 N 步 | 跟踪误差连续超标 |

**Truncated（截断 — 正常结束）**：运动片段播放完毕（`elapsed_frames >= total_frames`），不是策略失败。

两者的关键区别在 GAE 计算中：
- Terminated：未来价值 = 0（合理，失败了）
- Truncated：需要 reward 修正来补偿（详见 3.4）

### 3.3 Reward 计算

每步 reward = 当前状态与参考运动帧的对比，加权求和。

#### 六个跟踪奖励（Gaussian kernel）

```
reward_i = exp(-error² / std²) × weight
```

误差 = 0 → reward = 1.0（完美）；误差 >> std → reward → 0。

| # | 奖励项 | 权重 | std | 计算内容 |
|---|--------|------|-----|---------|
| 1 | `tracking_anchor_pos` | 0.5 | 0.3m | 根节点位置误差 |
| 2 | `tracking_anchor_ori` | 0.5 | 0.4 | 根节点朝向误差 |
| 3 | `tracking_relative_body_pos` | 1.0 | 0.3m | 14 个关节相对位置误差 |
| 4 | `tracking_relative_body_ori` | 1.0 | 0.4 | 14 个关节相对朝向误差 |
| 5 | `tracking_body_linvel` | 1.0 | 1.0 m/s | 关节线速度误差 |
| 6 | `tracking_body_angvel` | 1.0 | 3.14 rad/s | 关节角速度误差 |

#### 三个惩罚项

| # | 惩罚项 | 权重 | 内容 |
|---|--------|------|------|
| 7 | `action_rate_l2` | -0.1 | 相邻两步动作差 L2（惩罚抖动） |
| 8 | `joint_limit` | -10.0 | 超关节限制（最严厉） |
| 9 | `undesired_contacts` | -0.1 | 不期望的身体接触 |

#### 数值示例

**完美跟踪**（所有误差=0）：

```
6 个跟踪项: 0.5+0.5+1.0+1.0+1.0+1.0 = 5.0
3 个惩罚项: 0
总计: 5.0（理论最大值）
```

**典型训练中**（有跟踪误差）：

```
anchor_pos 误差 0.05m:  exp(-0.028) × 0.5 = 0.486
anchor_ori 误差 0.1:    exp(-0.063) × 0.5 = 0.470
body_pos 误差 0.08m:    exp(-0.071) × 1.0 = 0.931
body_ori 误差 0.15:     exp(-0.141) × 1.0 = 0.869
linvel 误差 0.3m/s:     exp(-0.090) × 1.0 = 0.914
angvel 误差 0.5rad/s:   exp(-0.025) × 1.0 = 0.975
action_rate: -0.001,  joint_limit: 0,  contacts: 0
──────────────────────────────────────────────
总计: ~4.64
```

### 3.4 GAE 优势值估计

**GAE（Generalized Advantage Estimation）** 估计每个状态-动作对相对于均值的好坏程度。

核心公式：

```
δ_t = r_t + γ × V(s_{t+1}) × (1-done_t) - V(s_t)     # TD 误差
A_t = δ_t + (γ × λ) × (1-done_t) × A_{t+1}            # 递推计算
returns_t = A_t + V(s_t)                                 # 回报值
```

- `γ = 0.99`：未来 reward 的衰减率。100 步后衰减到 ~37%
- `λ = 0.95`：在 TD(0)（低方差高偏差）和 Monte Carlo（高方差低偏差）之间取权衡
- 有效回溯窗口 ≈ 1/(1-γλ) = 1/(1-0.9405) ≈ 17 步

#### Reward 修正（Timeout Compensation）

**在计算 GAE 之前**，对 truncated episode 的最后一步 reward 做修正：

```python
new_rewards = rewards + gamma * time_outs * values  # ppo_trainer.py:1006-1010
```

**为什么？** GAE 中 `done=1` 让 `next_is_not_terminal = 0`：

```
delta = r + 0 × γ × V(s_next) - V(s)
      = r - V(s)
```

对 terminated（失败）：合理，未来价值确实是 0。
对 truncated（正常结束）：**错误的负信号**。

```
例: 动作播完时 r=0.4, V(s)=2.8
未修正: delta = 0.4 - 2.8 = -2.4    ← 错误！策略并没有做错
修正后: r' = 0.4 + 0.99×2.8 = 3.172
        delta = 3.172 - 2.8 = 0.372  ← 合理的正信号
```

#### env[42] 的 GAE 完整 Walkthrough

沿用 3.2 的 3 个 episode，从 step 23 反向递推（γ=0.99, λ=0.95）：

```
── Episode C (步 20-23, 未完成) ──

step 23: done=0, 使用 last_values bootstrap
  δ_23 = r_23 + 0.99 × V(s_24) - V(s_23)
  A_23 = δ_23

step 22: done=0
  δ_22 = r_22 + 0.99 × V(s_23) - V(s_22)
  A_22 = δ_22 + 0.9405 × A_23              ← γλ = 0.99×0.95 = 0.9405

step 21, 20: 类似递推

── Episode B 的边界 (步 19, done=1, truncated) ──

step 19: done=1 → next_is_not_terminal = 0
  reward 已修正: r_19 = 0.4 + 0.99×2.8 = 3.172
  δ_19 = 3.172 + 0×γ×V(s_20) - V(s_19)
       = 3.172 - 2.8 = 0.372
  A_19 = 0.372
  ★ done=1 截断优势传播 — A_19 不会传递到 step 20（不同 episode）

step 18: done=0
  δ_18 = r_18 + 0.99 × V(s_19) - V(s_18)
  A_18 = δ_18 + 0.9405 × A_19

...（步 8-17 正常递推）...

── Episode A 的边界 (步 7, done=1, terminated) ──

step 7: done=1, time_outs=0 → reward 不修正
  δ_7 = r_7 + 0×γ×V(s_8) - V(s_7)
      = r_7 - V(s_7)                       ← 未来价值=0，失败了
  A_7 = δ_7
  ★ 优势传播被截断

...（步 0-6 正常递推）...
```

**关键点**：
- `done=1` 同时截断 δ 中的 V(next) 和 A 的递推传播
- Terminated (step 7)：未来价值 = 0 合理
- Truncated (step 19)：reward 修正后 δ 是正的
- Episode C (steps 20-23)：依赖 `last_values` bootstrap

#### 同步归一化

多卡训练中，各卡 rollout 独立收集，advantage 分布可能不同：

```python
advantages = self.accelerator.gather(advantages)           # 跨卡 gather
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 全局归一化
advantages = advantages.reshape(...)[process_index]         # ungather
```

### 3.5 PPO 更新

收集完 98,304 条数据后，进行 5 epochs × 4 mini-batches = 20 次梯度更新。

#### Policy Gradient Loss（裁剪代理目标）

```python
ratio = exp(new_logprob - old_logprob)           # 新旧策略概率比
pg_loss1 = -advantage × ratio                     # 未裁剪
pg_loss2 = -advantage × clamp(ratio, 0.8, 1.2)   # 裁剪到 [0.8, 1.2]
pg_loss = max(pg_loss1, pg_loss2).mean()           # 取较大者（悲观估计）
```

取 max 是 PPO 的"悲观"策略：好动作不过度利用，差动作不过度惩罚。

#### Value Function Loss

```python
vpredclipped = clamp(vpred, old_values - ε_v, old_values + ε_v)
vf_loss = max((vpred-returns)², (vpredclipped-returns)²).mean()
```

#### Entropy Loss

```python
entropy_loss = -mean(entropy)  # entropy 越大 loss 越小 → 鼓励探索
```

#### 关键超参数

| 超参数 | 值 | 含义 |
|--------|-----|------|
| `cliprange` | 0.2 | 策略裁剪范围 [0.8, 1.2] |
| `vf_coef` | 1.0 | 价值函数损失系数 |
| `max_grad_norm` | 0.1 | 梯度裁剪阈值 |
| `learning_rate` | ~1.5e-4 | KL 自适应调整 |
| `num_ppo_epochs` | 5 | PPO 更新次数 |
| `num_mini_batches` | 4 | 每 epoch mini-batch 数 |

#### 多 Epoch 复用与 Clipping

同一批 98,304 条数据被遍历 5 次。每次策略已微变，ratio 不再是 1.0。Clipping 防止过度利用——ratio 偏离 [0.8, 1.2] 后梯度被截断。

---

## 第 4 章：Loss 体系与梯度流

### 4.1 总损失公式

```
total_loss = ppo_loss + aux_loss_scale × Σ(coef_i × aux_loss_i)

其中:
ppo_loss = pg_loss + vf_coef × vf_loss + entropy_coef × entropy_loss
```

### 4.2 五个辅助损失 — SONIC 的核心创新

SONIC 通过 5 个辅助损失将三个编码器的 latent 空间对齐。关键技术：**选择性 `.detach()` 控制梯度流向**。

#### 损失 1: g1_recon（重构损失，系数 0.01）

```
计算: MSE(g1_kin_decoder(tokens), original_tokenizer_obs)
梯度: loss → g1_kin decoder → tokens → FSQ → 当前活跃编码器
更新: g1_kin 解码器 + FSQ + 当前编码器
```

确保 token 保留足够运动信息。系数 0.01（远小于其他损失），重构是辅助目标。

#### 损失 2: g1_smpl_latent（G1→SMPL 对齐，系数 1.0）

```python
g1_latents = self.encode("g1", ...).detach()  # ← G1 latent 被 detach！
loss = MSE(g1_latents, smpl_latents)
```

```
梯度: loss → smpl_latent → SMPL 编码器（G1 不更新）
更新: 仅 SMPL 编码器
```

G1 做 teacher，SMPL 做 student。

#### 损失 3: g1_teleop_latent（G1↔Teleop 双向对齐，系数 1.0）

```python
# 两者都没有 detach
loss = MSE(g1_latents, teleop_latents)
```

```
梯度: 同时流向 G1 编码器和 Teleop 编码器
更新: 双向对齐
```

#### 损失 4: teleop_smpl_latent（Teleop→SMPL 对齐，系数 1.0）

```python
teleop_latents = self.encode("teleop", ...).detach()  # ← Teleop 被 detach
loss = MSE(teleop_latents, smpl_latents)
```

```
梯度: loss → SMPL 编码器（Teleop 不更新）
更新: 仅 SMPL 编码器
```

#### 损失 5: reencoded_smpl_g1_latent（循环一致性，系数 1.0）

```
流程: SMPL Encoder → latent → FSQ → g1_kin decoder → 重构 G1 运动
      → G1 Encoder(re-encode) → re-encoded latent
      → MSE(re-encoded, original_g1_latent.detach())

更新: G1 编码器(re-encode path) + g1_kin 解码器 + FSQ + SMPL 编码器
```

最复杂的损失——确保 SMPL→decode→re-encode 循环一致。

### 4.3 梯度更新总结

| 网络模块 | 被哪些 loss 更新 | 角色 |
|---------|-----------------|------|
| **G1 编码器** | PPO, g1_recon, g1_teleop_latent, reencoded | **锚点** |
| **Teleop 编码器** | PPO, g1_recon(活跃时), g1_teleop_latent | **桥梁** |
| **SMPL 编码器** | PPO, g1_smpl_latent, teleop_smpl_latent, reencoded | **学生** |
| **FSQ 量化器** | PPO, g1_recon, reencoded | 共享 |
| **g1_kin 解码器** | g1_recon, reencoded | 重构 |
| **g1_dyn 解码器** | **仅 PPO** | 最终动作 |
| **Critic** | vf_loss | 价值估计 |

### 4.4 Teacher-Student 层级

```
        PPO 训练出好 token
              ↓
         G1 编码器 (锚点/teacher)
          ↗        ↘
   双向对齐         单向教 (detach)
        ↗                ↘
  Teleop 编码器 ─────→ SMPL 编码器
    (桥梁/teacher)    单向教 (detach)
```

**为什么 G1 latent 要 detach？** G1 已被 PPO 训练好，产生的 token 能控制机器人。不 detach 的话，对齐 loss 会拉偏 G1 的 latent 空间，破坏已学好的动作能力。

### 4.5 信用分配：为什么需要辅助 loss

**问题**：PPO 梯度从 action 一路传回编码器和解码器，怎么知道是谁的问题？

```
SMPL编码器 → FSQ → g1_dyn解码器 → action → reward 差
                                              ↓
                              PPO 梯度同时更新 SMPL编码器 和 g1_dyn解码器
```

纯 PPO 确实分不清。链式法则不做"归因"——它只给每个参数算梯度，同时更新。

**SONIC 的解法**：辅助 loss 给每个组件**独立的考试**：

- **g1_dyn 解码器的考试**：给你正确的 token（G1 编码器产生的），你能输出正确 action 吗？→ PPO loss 直接评判
- **SMPL 编码器的考试**：给你 SMPL 数据，你的 latent 和 G1 的一样吗？→ latent 对齐 loss 直接评判，跟解码器无关

各考各的，锅分得清楚。这是 SONIC 设计 5 个辅助 loss 的根本原因——解决 PPO 单独搞不定的信用分配。

---

## 第 5 章：梯度如何改变策略

> 这一章建立直觉：随机采样的动作，如何通过梯度改进策略的均值？

### 5.1 log_prob 计算

29 维高斯分布，各维独立：

```
log P(a|μ,σ) = Σ_{i=1}^{29} [-0.5 × ((a_i-μ_i)/σ_i)² - log(σ_i) - 0.5×log(2π)]
```

数值示例（单个关节 μ=0.3, σ=0.05）：

```
a=0.32（接近均值）: log P = -0.08+3.0-0.92 = 2.0    ← 高概率
a=0.50（远离均值）: log P = -8.0+3.0-0.92  = -5.92   ← 极低概率
```

### 5.2 策略梯度的四种情况

PPO loss（简化）：`L = -advantage × log π(a|s)`

log π 对 μ 的导数：`∂ log π / ∂μ = (a - μ) / σ²`

Loss 对 μ 的导数：`∂L/∂μ = -advantage × (a - μ) / σ²`

```
                        advantage > 0           advantage < 0
                        (好动作)                (坏动作)

a > μ (采样偏右)        ∂L/∂μ < 0               ∂L/∂μ > 0
                        梯度下降: μ 增大          梯度下降: μ 减小
                        → μ 靠近 a ✓             → μ 远离 a ✓

a < μ (采样偏左)        ∂L/∂μ > 0               ∂L/∂μ < 0
                        梯度下降: μ 减小          梯度下降: μ 增大
                        → μ 靠近 a ✓             → μ 远离 a ✓
```

**一句话：advantage 正就靠近 a，advantage 负就远离 a。**

### 5.3 数值示例

```
某关节: μ = 0.3, σ = 0.05, 采样到 a = 0.35

情况1: advantage = +2.0 (好动作)
  ∂L/∂μ = -2.0 × (0.35-0.3) / 0.0025 = -40
  梯度下降: μ_new = 0.3 + lr×40 → μ 增大，往 0.35 移动（靠近好动作）

情况2: advantage = -2.0 (坏动作)
  ∂L/∂μ = +40
  梯度下降: μ_new = 0.3 - lr×40 → μ 减小，远离 0.35（远离坏动作）
```

### 5.4 梯度改权重，不是直接改 μ

**μ 不是参数**，它是网络的输出：`μ = f(s; W)`

真正被更新的是权重 W，通过链式法则：

```
∂L/∂W = ∂L/∂μ × ∂μ/∂W
         ↑          ↑
     方向信号      执行路径
     (-40)       (PyTorch autograd)
```

效果：更新前 `network(W_old, s) → μ = 0.30`，更新后 `network(W_new, s) → μ ≈ 0.31`。

### 5.5 单个幸运样本的偏差与三层消噪

**问题**：如果 a=0.48 碰巧拿了高 reward（纯运气），梯度不会"强化"错误方向吗？

**澄清**：梯度推的是 μ 往 a 方向移动（0.3→0.48），不是"强化" μ=0.3。但如果 a=0.48 本身也不是真正好的动作，确实会产生有偏梯度。PPO 靠三层机制消噪：

**第一层：V(s) baseline**

```
advantage = Q(s,a) - V(s) = "这个动作的回报" - "这个状态下的平均回报"
```

只有比同状态平均水平更好，advantage 才是正的。削弱 90% 的"运气"影响。

**第二层：98,304 条样本的统计平均**

```
同一状态 s 附近的多次采样：
  a=0.48, adv=+0.5  → 推 μ 往右
  a=0.52, adv=+0.8  → 推 μ 往右（力度大）
  a=0.25, adv=-0.6  → 推 μ 往右（远离坏动作）
  a=0.45, adv=-0.1  → 推 μ 往左 ← 噪声！被淹没
```

**第三层：极小的学习率**

```
lr = 2e-5
即使梯度完全错误: Δμ = lr × 40 = 0.0008
μ 从 0.3 变成 0.3008 — 几乎没动
```

**这也是 RL 比监督学习慢数量级的根本原因**：监督学习有确定标签，RL 只有带噪声的 reward，必须靠海量样本把正确方向"冲"出来。

---

## 第 6 章：分布式训练（DDP）

### 6.1 框架栈

```
HuggingFace Accelerate → PyTorch DDP → HCCL（华为集合通信库）
```

`accelerate launch --num_processes=16` 启动 16 个进程，每个绑定一张 NPU。

### 6.2 三层同步

**1. 梯度同步**：DDP 在 `backward()` 时自动 AllReduce，平均所有卡的梯度。效果 = 在 4096 个环境上计算梯度。

**2. RunningMeanStd 同步**：
- 前 200 次迭代每步同步
- 之后按间隔同步
- 确保所有卡的观测归一化统计一致

**3. 优势值归一化同步**：
- gather 所有卡的 advantages
- 全局 `(A - mean) / (std + 1e-8)`
- ungather 回各自进程

### 6.3 KL 自适应学习率

```python
# _adjust_learning_rate_based_on_kl
KL 过大 → 降低学习率（防止策略更新过猛）
KL 过小 → 提高学习率（鼓励更大步更新）
```

---

## 第 7 章：StubEnv — 无物理仿真的训练

### 7.1 设计思路

StubEnv（`stub_env.py`）替代 Isaac Sim 物理仿真器，用于在 NPU（无 CUDA）上训练：

```python
class StubEnv:
    def step(self, policy_state_dict):
        actions = policy_state_dict["actions"]
        # 不使用 actions 进行物理模拟！
        self._current_time += self._dt        # 推进时间步
        dones = self._compute_dones()         # 检查终止
        self._resample_motion(done_env_ids)   # 重新采样
        obs_dict = self._compute_observations()
        rewards = self._compute_rewards()
        return obs_dict, rewards, dones, infos
```

### 7.2 与 Isaac Sim 的区别

| | StubEnv | Isaac Sim |
|---|---------|-----------|
| 物理模拟 | 无 | 完整刚体+接触 |
| 因果关系 | **断开**（action 不影响下一状态） | 完整 |
| isaaclab 依赖 | 无 | 需要 |
| 适用硬件 | CPU/NPU/GPU | 仅 NVIDIA GPU |
| 训练目标 | 验证 pipeline + 预训练编码器 | 训练完整控制策略 |

### 7.3 局限性

- 动作不影响下一状态，rewards 缺乏因果联系
- Value loss 无法持续下降
- 训练出的模型需要在 Isaac Sim 中 fine-tuning

---
---

# 下篇：16 卡 NPU 训练实验分析

---

## 第 8 章：实验配置

### 8.1 硬件环境

```
设备:     16 × Ascend 910 NPU (Ascend910_9362)
每卡内存:  61.3 GB HBM
通信后端:  HCCL（华为集合通信库）
精度:     FP32（NPU 不支持 bf16 的 torch.normal）
```

### 8.2 训练配置

| 参数 | 值 |
|------|-----|
| 每卡环境数 (`num_envs`) | 256 |
| NPU 卡数 (`world_size`) | 16 |
| 全局环境数 | 4,096 |
| 每环境步数 (`num_steps_per_env`) | 24 |
| 总迭代数 (`num_learning_iterations`) | 200 |
| PPO epochs | 5 |
| Mini-batches | 4 |
| 每 iteration transitions | 98,304 |
| 总 transitions | 19,660,800 (~2000 万) |
| 总 episodes | 819,200 |
| 数据 | SMPL: 131,455 PKL + Robot: 2 PKL |

### 8.3 与 SONIC 原始训练规模对比

```
                    本次实验              SONIC 原始
环境数:             4,096                4,096
迭代数:             200                  100,000
总 transitions:     ~2000 万             ~98 亿（500×）
硬件:              16× Ascend 910       A100/RTX 4090
时间:              ~7 分钟              2-3 天
环境:              StubEnv（无物理）      Isaac Sim（完整仿真）
```

### 8.4 Checkpoint

```
目录: logs_rl/TRL_G1_Stub/stub_train_test-20260628_160520/
文件: last.pt (428MB)
内容: policy_state_dict + value_state_dict + optimizer_state_dict
      + lr_scheduler_state_dict + state + env_state_dict
日志: /tmp/train_smpl_16card.log (506KB)
TB:   logs_rl/TRL_G1_Stub/stub_train_test-20260628_160520/tb/
```

---

## 第 9 章：训练曲线与指标分析

### 9.1 主要训练指标

| 指标 | Iter 1 | Iter 50 | Iter 100 | Iter 150 | Iter 200 |
|------|--------|---------|----------|----------|----------|
| Mean entropy | -45.51 | -38.85 | -31.22 | -22.72 | -14.32 |
| Mean rewards | -0.013 | -0.031 | 0.051 | -0.081 | 0.014 |
| Mean ep length | 11.43 | 363.58 | 379.32 | 377.28 | 378.04 |
| Action noise std | 0.05 | 0.06 | 0.08 | 0.11 | 0.15 |
| FPS | 21,796 | 51,004 | 48,834 | 49,301 | 49,398 |

### 9.2 PPO Loss 分量

| 指标 | Iter 1 | Iter 50 | Iter 100 | Iter 150 | Iter 200 |
|------|--------|---------|----------|----------|----------|
| policy_loss | -0.0040 | -0.0102 | -0.0102 | -0.0139 | -0.0201 |
| value_loss | 0.0593 | 0.0598 | 0.0608 | 0.0595 | 0.0596 |
| entropy_loss | -45.51 | -38.85 | -31.22 | -22.72 | -14.32 |
| weighted_ppo_loss | 0.5104 | 0.4381 | 0.3628 | 0.2728 | 0.1827 |
| approx_kl | 0.0074 | 0.0148 | 0.0142 | 0.0145 | 0.0161 |
| clip_fraction | 0.049 | 0.125 | 0.120 | 0.128 | 0.151 |

### 9.3 辅助损失分量

| 辅助损失 | Iter 1 | Iter 50 | Iter 100 | Iter 150 | Iter 200 |
|---------|--------|---------|----------|----------|----------|
| g1_recon | 0.4082 | 0.3662 | 0.3653 | 0.3681 | 0.3630 |
| g1_smpl_latent | 3.97e-4 | 6.0e-6 | 1.9e-5 | 1.3e-5 | 1.7e-5 |
| g1_teleop_latent | 4.32e-4 | 1.3e-5 | 2.5e-5 | 3.0e-5 | 2.9e-5 |
| teleop_smpl_latent | 5.04e-4 | 1.0e-5 | 1.9e-5 | 2.7e-5 | 2.1e-5 |
| reencoded_smpl_g1 | 9.9e-5 | 4.0e-6 | 1.2e-5 | 7.0e-6 | 8.0e-6 |
| **total_aux_loss** | **0.0055** | **0.0037** | **0.0037** | **0.0038** | **0.0037** |

### 9.4 Value 函数

| Iter | values_mean | values_std | returns_mean | returns_std | vf_loss |
|------|------------|-----------|-------------|------------|---------|
| 1 | 0.0320 | 0.0002 | 0.0150 | 0.2443 | 0.0593 |
| 51 | 0.0024 | 0.0375 | -0.0079 | 0.2557 | ~0.060 |
| 101 | 0.0101 | 0.0289 | 0.0074 | 0.2426 | 0.0608 |
| 151 | -0.0084 | 0.0329 | -0.0048 | 0.2461 | 0.0595 |

### 9.5 详细分析

#### 1. Episode 长度 — 最显著的变化

从 11.43 步（iter 1）迅速增长到 363.58 步（iter 50），然后稳定在 ~378 步。策略在前 50 个 iteration 内学会了不触发终止条件（如关节角超限）。

#### 2. Entropy 与 Action Noise

Entropy 从 -45.51 线性变化到 -14.32，反映 action noise std 从 0.05 增长到 0.15。29 维高斯 entropy = Σ log(σ_i) + const，std 越大 entropy 绝对值越小。说明 PPO 的 entropy bonus 有效鼓励了探索。

#### 3. PPO Weighted Loss

从 0.51 降到 0.18，主要来自 entropy_loss 下降（std 增大）。policy_loss 绝对值增大（0.004→0.020），策略更新幅度在增加。

#### 4. Latent 对齐损失 — 迅速收敛

所有 latent 对齐损失在前 50 个 iteration 内下降 1-2 个数量级，之后稳定在 1e-5 量级：

| 辅助损失 | Iter 1 → Iter 50 | 下降倍数 |
|---------|-------------------|---------|
| g1_smpl_latent | 3.97e-4 → 6.0e-6 | 66× |
| g1_teleop_latent | 4.32e-4 → 1.3e-5 | 33× |
| teleop_smpl_latent | 5.04e-4 → 1.0e-5 | 50× |
| reencoded_smpl_g1 | 9.9e-5 → 4.0e-6 | 25× |

三个编码器的 latent 空间快速对齐到一致的表示。

#### 5. G1 Recon Loss

从 0.408 降到 0.363（仅 ~11%），重构准确度受限于 FSQ 量化的信息瓶颈。

#### 6. Value 函数学习

- values_std 从 0.0002 增长到 ~0.03：Critic 学会区分不同状态
- value_loss 保持稳定（~0.06）：预测精度受限于 StubEnv 因果断开
- val/clipfrac 始终为 0：Critic 更新从未触发裁剪

#### 7. KL 散度与裁剪率

- approx_kl 从 0.0074 升到 0.016：策略更新幅度逐渐增大
- clip_fraction 从 5% 升到 15%：更多样本被裁剪，但仍在健康范围（< 20%）

#### 8. 学习率变化

```
Iter 1→50:   KL 0.007→0.015，在 desired_kl 范围内，LR 1.5e-4 → 2.0e-4（上限）
Iter 50→150: KL 保持 ~0.014-0.015，LR 维持 2.0e-4
Iter 150→200: KL 升到 0.016，超过上限，LR 下调到 8.9e-5
```

#### 9. 吞吐量

```
单卡 (num_envs=64):       ~1,032 steps/s
16 卡 (num_envs=256/卡):  ~49,398 steps/s（稳态）
等效加速比:               ~47.9×（超线性）
```

超线性加速原因：每卡 num_envs 从 64→256 提高矩阵运算利用率 + HCCL 通信开销小。

#### 10. 单次 Iteration 时间

```
Collection (rollout): ~0.65s
Learning (PPO update): ~1.35s（含前向/反向 + AllReduce）
Total: ~2.0s
总训练: 200 × ~2s + ~30s 初始化 = ~419s ≈ 7 分钟
```

---

## 第 10 章：关键洞察与总结

### 10.1 训练结果总结

| 指标 | 初始 (Iter 1) | 最终 (Iter 200) | 变化 |
|------|-------------|----------------|------|
| Episode 长度 | 11.4 步 | 378 步 | 前 50 iter 快速收敛 |
| Action noise std | 0.05 | 0.15 | 持续增长（探索增强） |
| Latent 对齐损失 | ~4e-4 | ~2e-5 | 前 50 iter 下降 20-60× |
| G1 recon 损失 | 0.408 | 0.363 | 缓慢下降 11% |
| PPO weighted loss | 0.510 | 0.183 | 持续下降 |
| KL 散度 | 0.007 | 0.016 | 策略更新幅度增大 |
| Clip fraction | 5% | 15% | 健康范围内 |

### 10.2 关键洞察

1. **Detach 策略是 SONIC 训练的核心**：通过选择性 detach，同一个 MSE 损失可以单向或双向训练编码器。G1 做 teacher（detach），SMPL 做 student。

2. **辅助 loss 解决信用分配**：纯 PPO 无法区分编码器和解码器的误差来源，5 个辅助 loss 给每个组件独立的监督信号。

3. **Latent 对齐极快收敛**：三个编码器的 latent 空间在 50 iterations 内基本对齐（下降 25-66×），说明 SONIC 的 token 统一方案高效。

4. **PPO 梯度的本质**：不是直接改 μ，而是通过链式法则改权重 W，间接让 μ 往好动作方向移动。单个样本有噪声，靠 V(s) baseline、大 batch、小 lr 三层消噪。

5. **Episode 生命周期完全由 env 管理**：PPO trainer 只做 policy_step + env.step 循环，所有 reset、motion 重分配、terminated/truncated 区分在 env 内部完成。

6. **Reward 修正是必要的妥协**：truncated 的 done=1 产生假负信号，reward 修正补偿了这个问题，但用 V(s_t) 近似 V(s_{t+1}) 有近似误差。

7. **StubEnv 的局限**：动作不影响下一状态（因果断开），value loss 无法持续下降，rewards 无明确趋势。模型需在 Isaac Sim 中 fine-tuning。

8. **16 卡超线性加速**：49K steps/s vs 单卡 ~1K steps/s，加速比 ~49×。更大全局 batch（4096 vs 64 envs）提高了 NPU 计算利用率。

---

*本报告基于代码分析（gear_sonic/trl/）和 16 卡 Ascend 910 NPU 上的 StubEnv 训练实验（2026-06-28，目录 stub_train_test-20260628_160520）。训练日志：/tmp/train_smpl_16card.log，TensorBoard：logs_rl/.../tb/。*
