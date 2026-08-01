# PhysX (ovphysx) 引擎切换设计方案

## 1. 背景与动机

### 1.1 15 轮实验的结论

经过 v6-v18, E0-E2 共 15+ 轮实验，定位了 MuJoCo 上 SONIC G1 训练的根因：

```
策略能力:    BC warmup = 20.6 步 ≈ 纯 ref PD = 21.2 步
             ↓ 策略已经达到物理天花板

物理天花板:  参数推顶后 ~28 步 (implicitfast + cone=1 + solref/damping 最优)
             ↓ 所有可调参数已确认最优

引擎天花板:  28 步 → 100+ 步的 gap
             ↓ 硬约束接触 + 力-接触解耦 + pyramidal 摩擦锥
             ↓ MuJoCo C++ 源码控制的，参数够不到
```

### 1.2 6.5× 精度差距

| | per-step drift α | ref PD 存活 (ANK=0.2) |
|---|---|---|
| MuJoCo (参数推顶) | ~0.010 m/步 | ~28 步 |
| **Isaac Sim (PhysX)** | **< 0.002 m/步** | **100+ 步** |

差距不在参数层——在 MuJoCo 的硬约束接触模型和关节体力-接触解耦。这两项都是引擎 C++ 源码控制的。

### 1.3 为什么选中 ovphysx

2025 年 NVIDIA 发布了 `ovphysx`——PhysX 5 的独立 Python 绑定：

- `pip install ovphysx`（开箱即用）
- CPU 可用（不强制 GPU）→ NPU 服务器能跑
- Linux aarch64 支持 → 华为昇腾 ARM 服务器能跑
- 和 Isaac Sim 使用**完全相同的物理引擎**
- BSD-3 开源协议
- DLPack 零拷贝对接 PyTorch/NumPy

**换 ovphysx = 保留全部训练架构 + 物理引擎换成和 Isaac 一样的。**

## 2. 架构对比

### 2.1 当前架构 (MuJoCo)

```
ppo_trainer (复用) → MuJoCoEnvManager (自写, multiprocessing)
                       → MuJoCoEnv (自写, 520行)
                         → mujoco.mj_step (C 引擎, 硬约束)
```

### 2.2 目标架构 (ovphysx)

```
ppo_trainer (复用) → PhysXEnvManager (新写, multiprocessing)
                       → PhysXEnv (新写, 移植物理层)
                         → physx.step (C++ 引擎, PGS 软约束)
```

### 2.3 改动清单

| 模块 | 当前 | 目标 | 改动量 | 说明 |
|---|---|---|---|---|
| **物理引擎** | `mujoco.mj_step` | `physx.step` | 新写 | ovphysx API |
| **PD 控制** | `_pd_control()` + `_physics_step()` | PhysX Articulation Drive | 新写 | 和 Isaac 行为一致 |
| **观测构建** | `_build_tokenizer()` 1761D | 同逻辑，读 PhysX state | 移植 | 公式不变，数据源换 API |
| **奖励计算** | `_compute_reward()` 13 项 | 同逻辑，读 PhysX state | 移植 | 公式不变 |
| **终止判断** | `_check_termination()` 4 条件 | 同逻辑 | 移植 | 公式不变 |
| **并行管理** | `MuJoCoEnvManager` | `PhysXEnvManager` | 新写 | 复用 SHM+Barrier 模式 |
| **训练器** | `ppo_trainer` | **不改** | 0 | 观测/奖励格式兼容即可 |
| **模型** | UniversalTokenModule | **不改** | 0 | 输入输出维度不变 |
| **G1 模型** | MJCF XML | USD (ovphysx 原生加载) | 转换 | MJCF→USD 转换器 |
| **渲染** | OSMesa | 可选 | 新写 | ovphysx 无内置渲染 |

**总计**：~1500 行新代码 + G1 模型格式转换。ppo_trainer、模型架构、训练流程全不变。

## 3. 实现方案

### Phase 1：最小可行验证（1 周）

**目标**：G1 模型在 ovphysx 里能站稳 + 跑 ref PD。

1. G1 MJCF XML → USD 转换（或用 ovphysx API 手动搭建刚体链）
2. 单 env ref PD tracking：`physx.step(dt)` 替代 `mj_step`
3. 测 α——验证 < 0.002
4. 如果 ref PD 存活 > 50 步 → 物理达标，继续 Phase 2

**判据**：ref PD 存活 > 50 步（当前 MuJoCo：21 步）。

### Phase 2：MDP 层移植（1-2 周）

**目标**：`PhysXEnv` 实现完整的 step/reset/obs/reward/termination。

1. 移植 `_compute_reward()`——13 项数学公式不变，数据源从 `data.xpos` 换成 PhysX state API
2. 移植 `_check_termination()`——4 条件不变
3. 移植 `_build_tokenizer()`——1761D 拼接逻辑不变
4. 移植 PD 控制——换成 PhysX Articulation Drive API（`drive.set_target()`）

**判据**：单 env BC warmup 权重 deterministic 推理 > 20 步。

### Phase 3：并行 + 训练联调（1 周）

**目标**：完整的训练管线跑通。

1. `PhysXEnvManager`——复用 SHM+Barrier 模式
2. 对接 `ppo_trainer`——观测/奖励格式验证
3. BC warmup smoke test（100 iter）
4. PPO smoke test（100 iter）

**判据**：训练不 crash、指标正常。

## 4. 目标与验收

| 指标 | MuJoCo 当前 | ovphysx 目标 | 验收方法 |
|---|---|---|---|
| ref PD α | 0.013 | **< 0.002** | 100 episode ref PD 统计 |
| ref PD 存活 (ANK=0.2) | 21 | **> 80** | 同上 |
| BC warmup length | 20.6 | **> 60** | 100 episode deterministic |
| PPO 训练收敛 | entropy 不收敛 | **entropy 正常收敛** | TB 监控 |
| 训练速度 | 15K fps | **> 5K fps**（可接受降速） | TB fps 指标 |

**最终验收**：ovphysx 上训练的 BC+PPO 策略，在 Isaac Sim 上 deterministic 推理存活 > 80 步（如果能访问 Isaac 做验证）。

## 5. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| ovphysx aarch64 不兼容或 API 不稳定 | 中 | Phase 1 第一天验证 |
| G1 MJCF→USD 转换失败或模型不对 | 中 | 备选：ovphysx API 手动搭建刚体链 |
| PhysX Articulation Drive 和 Isaac 行为不一致 | 低 | 用 Isaac Sim 相同的 kp/kd/damping 参数 |
| 训练速度显著下降（CPU PhysX 比 MuJoCo 慢） | 中 | ovphysx 宣称支持 CPU 并行；如果太慢考虑减少 env 数 |
| ovphysx 社区支持不足 | 高 | 以 MuJoCo 实现为备份，ovphysx 出问题随时切回 |

## 6. 不做的事

- 改 MuJoCo C++ 源码（投入产出比差，ovphysx 直接解决）
- Isaac Sim / GPU 依赖（NPU 服务器没有 GPU）
- 换其他引擎 (RaiSim/Bullet)（ovphysx = Isaac Sim 同引擎，天然最优）
- 改训练算法或模型架构

## 7. 相关文档

- [[mujoco-vs-isaac-precision-analysis]] — 架构对比与精度分析完整报告
- [[status-and-next-steps]] — 项目当前状态
- [[training-retrospective-report]] — 15 轮实验回顾
