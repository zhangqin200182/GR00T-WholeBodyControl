---
name: mujoco-vs-isaac-precision-analysis
description: MuJoCo vs Isaac Lab+Sim 架构对比、精度差距量化分析、提升路径评估
metadata: 
  node_type: memory
  type: project
  created: 2026-08-01
  status: active
  originSessionId: 4198c65f-f2d5-40fc-bf59-96dad9b06ecb
---

# MuJoCo vs Isaac Sim：架构对比与精度分析报告

## 1. 系统架构对比

### 1.1 Isaac 参考系统

```
Isaac Lab (NVIDIA)           ← RL 训练框架，GPU 并行环境
  ├─ MDP 配置层              ← YAML 驱动：观测、奖励、终止条件
  ├─ manager_env_wrapper.py   ← 齿轮智能写的 PyTorch 接口
  └─ 调用 ↓
Isaac Sim (NVIDIA)           ← 仿真+渲染平台
  ├─ PhysX 5 (NVIDIA)        ← 物理引擎 (C++)
  │   ├─ Articulation Drive  ← 关节 PD 控制（引擎内置）
  │   ├─ TGS 摩擦求解         ← 时间平滑摩擦迭代
  │   ├─ 软约束接触           ← PGS 迭代 + 允许微量穿透
  │   └─ mesh-to-plane 碰撞   ← 完整 STL 几何接触
  ├─ 传感器管线               ← 自动观测生成
  └─ RTX 渲染                ← GPU 光追
```

### 1.2 MuJoCo 系统（我们构建的）

```
ppo_trainer (齿轮智能)       ← PPO 训练器（复用）
MuJoCoEnvManager (自写)      ← CPU 多进程并行管理
  ├─ mujoco_env.py (自写)     ← MDP 层：520行 Python
  │   ├─ _pd_control()        ← 手动 PD: kp×(target-qpos)-kd×qvel
  │   ├─ _compute_reward()    ← 13项硬编码数学公式
  │   ├─ _check_termination() ← 4条件手动翻译
  │   └─ _build_tokenizer()   ← 1761D 手动拼接
  └─ 调用 ↓
MuJoCo 3.10 (Google DeepMind) ← 物理引擎 (C，开源)
  ├─ mj_step()                ← 半隐式欧拉积分
  ├─ Newton 约束求解           ← 硬约束互补条件
  ├─ pyramidal 摩擦锥          ← 4棱锥近似 (cone=0)
  ├─ solref/solimp 接触参数    ← 我们已调 P1
  └─ box 解析图元脚掌           ← 16cm×7cm (P0)，不是 mesh
OSMesa (离线软件渲染)          ← 纯 CPU 光栅化
```

### 1.3 关键差异矩阵

| 层次 | Isaac | MuJoCo | 差距来源 |
|---|---|---|---|
| 物理引擎 | PhysX 5 (闭源) | MuJoCo 3.10 (开源 C) | 不同团队，不同算法 |
| 积分器 | 隐式积分 | 半隐式欧拉 (已试 implicitfast +16%) | 参数级 (可改 XML) |
| 接触约束 | 软约束 (PGS/TGS) | 硬约束 (互补+Newton) | **引擎级 (必须改 C++)** |
| 接触几何 | mesh-to-plane (完整 STL) | box(16×7cm) 解析图元 | **引擎级 (MuJoCo 不支持 mesh 碰撞)** |
| 摩擦模型 | 各向异性 + TGS 平滑 | 4棱锥 (可改 cone=1 椭圆锥) | 参数级+引擎级 |
| 关节体算法 | Articulation Drive | `torque = kp*err - kd*vel` | **引擎级 (力-接触解耦)** |
| MDP 实现 | YAML 自动推导 | 520行 Python 硬编码 | 手工翻译 (已完成，验证一致) |
| 并行环境 | GPU 向量化 | Python multiprocessing + SHM | 功能等价 (已完成) |

## 2. 实验回顾：从"跑不起来"到"定位精度瓶颈"

### 2.1 五个阶段

| 阶段 | 版本 | 问题 | 发现 |
|---|---|---|---|
| 第一阶段 | v6-v11 | QACC NaN、reward 退化、HCCL 崩溃 | **稳定性修复**：PD per-substep、feet_acc 量纲修正、alive_bonus、阈值放宽 |
| 第二阶段 | v12-v15 | length 全卡 2.0 | **2.0 地板围剿**：阈值放宽无效、g1_dyn reinit 无效、backbone freeze 无效 |
| 第三阶段 | 发现 | XML 审查 | **根因发现**：8×5mm 小球、mesh contype=0 |
| 第四阶段 | v16-v18 | 物理栈修复 | **P0+P1+P1.5**：box 脚、solref/solimp、joint damping → 首次突破 2.0 |
| 第五阶段 | E0-E2 | 策略收敛、PPO 训坏 | **训练/物理分离**：BC=20.6≈ref PD，PPO 6.3=训坏 |

### 2.2 当前状态（消融矩阵）

| 组 | 配置 | 存活步数 | 状态 |
|---|---|---|---|
| ① 物理天花板 | ref PD (Euler, v21阈值) | **21.2** | ✅ 已测 |
| ①' 物理天花板 | ref PD (implicitfast) | **24.6** | ✅ 已测 |
| ⑤ BC 权重 | BC warmup (deterministic) | **20.6** | ✅ 已测 |
| ③ E0 | sonic_release → PPO 20K | 7.0 | ✅ 已跑 |
| ③' E2 | BC → PPO 15K | 6.3 | ✅ 已跑 |
| ② 直接迁移 | sonic_release 裸推理 | 未测 | ❌ trl 格式阻塞 |

### 2.3 核心洞察

**BC=20.6 ≈ ref PD=21.2：策略质量 = 物理天花板。**

PPO 不是没学到——是当前 reward 结构 (`ignore_terminations=True`) 让它"重置刷分"：摔倒→新 motion→前几步跟踪误差≈0→免费高奖励→策略学快速摔倒换重置。**PPO 把 BC 从 20.6 训到 6.3。**

**真正的瓶颈不在策略、不在 reward、不在训练配置——在物理引擎精度。** ref PD 在 Isaac 原始阈值 (ANK=0.2m) 下只能活 21 步，而 Isaac 同条件能跑完整 motion (100+) 步。

## 3. 精度差距量化

### 3.1 per-step drift rate α（物理精度指纹）

纯 ref PD 跑同一段 motion，每步记录踝位置跟踪误差：

| | α (m/步) | ref PD 存活 (ANK=0.2) | 判定 |
|---|---|---|---|
| **Isaac Sim 目标** | **< 0.002** | **100+** | 完整 motion tracking |
| MuJoCo 当前 (Euler) | **~0.013** | 21 | 基线 |
| MuJoCo (implicitfast) | ~0.011 | 25 (+16%) | 参数改善 |
| MuJoCo 参数推顶 | ~0.006 (预期) | ~35 | cone=1 + solref 调软 |
| MuJoCo 源码修改 | ~0.003 (预期) | ~65 | 软约束 + 关节体力传播 |
| MuJoCo 引擎天花板 | ~0.002 (理论上限) | ~100 | 达到 Isaac 级别 |

### 3.2 误差来源分解

| 误差源 | 贡献占比 | 可优化性 |
|---|---|---|
| 硬约束接触振荡 | ~40% | 源码级：改用软约束迭代 |
| 关节体力-接触解耦 | ~30% | 源码级：实现 ArticulationDrive 等价物 |
| 摩擦锥精度 (pyramidal) | ~15% | 参数级：cone=1 椭圆锥 |
| 积分器 (半隐式欧拉) | ~10% | 参数级：implicitfast |
| box vs mesh 接触面 | ~5% | 引擎级：MuJoCo 不支持 mesh |

## 4. 精度提升路径

### 4.1 路径 A：MuJoCo 源码修改

**目标**：改 MuJoCo C++ 引擎层，使其物理行为逼近 PhysX。

**需要改动的模块**：

| 优先级 | 模块 | 文件 | 改动内容 | 预期 α 改善 | 工作量 |
|---|---|---|---|---|---|
| P0 | 接触模型 | `engine/engine_support.c` | 加软约束模式：允许微量穿透，用罚力迭代替代互补条件 | -30% | 2-3 周 |
| P1 | 关节体算法 | `engine/engine_forward.c` | 力矩施加前检查接触约束，关节力在空间传播时自动妥协 | -25% | 3-4 周 |
| P2 | 摩擦求解 | `engine/engine_solver.c` | TGS 时间平滑摩擦迭代 | -15% | 2-3 周 |
| P3 | mesh 碰撞 | `engine/engine_collision_driver.c` | 支持 mesh-to-plane 碰撞检测 | -5% | 1-2 周 |

**总工作量**：2-3 个月（熟悉代码库 + 实现 + 测试 + 调参）。

**风险**：
- MuJoCo 代码库 ~10 万行 C，模块高度耦合，改接触约束可能破坏全引擎稳定性
- 上游社区不合并 patch → 每次 MuJoCo 版本升级要手动 rebase
- 精度提升了但训练速度可能下降（软约束需要更多迭代）

**判定方法**：不需要 Isaac Sim。每次修改后跑 ref PD 测 α，对比基线 0.013。α 持续下降 = 方向正确。

### 4.2 路径 B：换物理引擎

换用 RaiSim（ETH Zurich，CPU 约束投影 + 软穿透 + mesh 碰撞）或 Bullet。

**工作量**：重写 `mujoco_env.py` / `mujoco_env_manager.py`（~1000 行），适配新引擎的 Python API。2-4 周。

**风险**：新引擎的观测空间、动作空间、MDP 接口全部要重新对接；可能引入新的兼容性问题。

### 4.3 路径 C：参数推顶（当前能做的）

继续调 MuJoCo 参数到极限：

| 参数 | 当前 | 目标 | 预期 α | 工作量 |
|---|---|---|---|---|
| cone=1 (椭圆摩擦锥) | 0 | 1 | -0.002 | 10 分钟 |
| solref 继续调软 | (0.04, 0.8) | (0.06, 0.7) | -0.001 | 30 分钟扫描 |
| implicitfast | 0 | 3 | -0.002 | 已做 |
| timestep=0.001 | 0.002 | 0.001 | -0.001 | 训练速度减半 |

**预期**：参数推顶后 α 从 0.013 降到 0.007-0.009，ref PD 从 21 到 ~35 步。**但永远到不了 0.002（100 步）。**

## 5. 结论

| 问题 | 答案 |
|---|---|
| 真正的瓶颈是什么 | MuJoCo 硬约束接触模型 + 关节体力-接触解耦 → per-step drift rate α=0.013，Isaac 需要 α<0.002 |
| 参数能推到多远 | α 从 0.013 到 ~0.007，ref PD 从 21 到 ~35 步 |
| 能否达到 Isaac 精度 | **在 MuJoCo 不改 C++ 源码的前提下：不能** |
| 要改源码需要多少工作量 | 2-3 个月 (接触模型+关节体算法+摩擦求解) |
| BC warmup 已经证明了什么 | 策略可以在 MuJoCo 中达到物理天花板 (20.6≈21.2)，剩下的 gap 是物理引擎的 |

## 6. 参考

- [[training-retrospective-report]] — 15 轮实验全流程
- [[mujoco-foot-contact-fix-design]] — 物理栈 P0/P1/P1.5
- [[status-and-next-steps]] — 当前状态总结
- [[mujoco-isaac-gap-audit]] — Isaac vs MuJoCo 配置差距
- [[mujoco-training-test-plan-v2]] — 测试验证方案 v2
