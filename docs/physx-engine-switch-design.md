# PhysX 5 引擎切换设计方案 v2

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
| Isaac Sim (PhysX) | < 0.002 m/步 | 100+ 步 |

### 1.3 路径选择：裸 PhysX 5 SDK + MJCF（不走 ovphysx USD 路径）

| 路径 | aarch64 状态 | 结论 |
|---|---|---|
| ovphysx + USD | ❌ pip wheel 缺 libovstage.so / libov_usd_ms.so | 阻塞（NVIDIA 未开源这些模块） |
| **裸 PhysX 5 C++ API + MJCF** | **✅ 已编译验证通过** | **可行** |

OpenUSD 的 200 万行代码 70% 给了渲染（Hydra, MaterialX, 灯光/相机/UI）——我们只需要 Physics 那 10%。MJCF XML 已经包含了 PhysX 需要的全部物理参数（质量、惯性、关节、碰撞），OpenUSD 是多余的。

## 2. 架构对比

### 2.1 当前架构 (MuJoCo)

```
ppo_trainer → MuJoCoEnvManager (multiprocessing, SHM)
                → MuJoCoEnv (520行 Python)
                  → mujoco.mj_step (C 引擎, 硬约束)
                  → MJCF XML 加载 (mujoco 原生)
```

### 2.2 目标架构 (PhysX 5)

```
ppo_trainer → PhysXEnvManager (复用 SHM+Barrier 模式)
                → PhysXEnv (~500行 Python, pybind11 绑定)
                  → PhysX 5 PxScene::simulate (C++ 引擎, PGS 软约束)
                  → MJCF→PhysX 转换器 (~300行 Python)
```

### 2.3 改动清单

| 模块 | 当前 | 目标 | 改动量 |
|---|---|---|---|
| **物理引擎** | `mujoco.mj_step` | `PxScene::simulate` | 新写 pybind11 封装 |
| **PD 控制** | `_pd_control()` | `PxArticulationDrive` | 移植 API |
| **模型加载** | `mujoco.MjModel.from_xml_path` | **MJCF→PhysX 转换器** | ~300 行 Python |
| **观测构建** | `_build_tokenizer()` | 同逻辑，读 PhysX state | 移植（公式不变） |
| **奖励计算** | `_compute_reward()` 13 项 | 同逻辑 | 移植（公式不变） |
| **终止判断** | `_check_termination()` | 同逻辑 | 移植（公式不变） |
| **并行管理** | MuJoCoEnvManager | PhysXEnvManager | 复用 SHM+Barrier |
| **训练器/模型** | ppo_trainer, UniversalTokenModule | **不改** | 0 |

## 3. 实现方案

### Phase 0：aarch64 原生编译 ✅（已完成）

```
PhysX 5 SDK → cmake linux-aarch64-gcc-cpu-only → make -j64
→ bin/linux.aarch64/release/libPhysX_static_64.a (静态库)
→ bin/linux.aarch64/release/libPhysX_64.so     (动态库)
→ 成功编译，0 error
```

### Phase 1：pybind11 封装 + 模型加载（2 周）

1. pybind11 封装核心 PhysX API：`PxCreateScene`, `PxArticulation`, `PxArticulationDrive`, `PxScene::simulate`, `PxScene::fetchResults`
2. MJCF→PhysX 转换器：解析 G1 XML → 创建 `PxArticulationReducedCoordinate` + 29 个 `PxArticulationJointReducedCoordinate` + box 碰撞几何
3. 单 env 验证：加载 G1 模型 → 跑一步 `simulate(dt=0.002)` → 读出 qpos/qvel 验证正确

### Phase 2：MDP 层移植 + ref PD 验证（1 周）

1. 移植 `_pd_control()` → PhysX `PxArticulationDrive`
2. 移植 `_compute_reward()`（13 项数学不变）
3. 移植 `_check_termination()`（4 条件不变）
4. 移植 `_build_tokenizer()`（1761D 拼接不变）
5. ref PD 验证：测 α，预期 < 0.002

### Phase 3：并行 + 训练联调（1 周）

1. PhysXEnvManager（SHM+Barrier 复用）
2. BC warmup smoke test
3. PPO smoke test

## 4. 目标与验收

| 指标 | MuJoCo 当前 | PhysX 目标 | 验收方法 |
|---|---|---|---|
| aarch64 编译 | N/A | **✅ 已通过** | Phase 0 |
| ref PD α | 0.013 | **< 0.002** | 100 episode |
| ref PD 存活 (ANK=0.2) | 21 | **> 80** | 同上 |
| BC warmup length | 20.6 | **> 60** | 100 episode deterministic |
| 训练速度 | 15K fps | > 5K fps（可接受降速） | TB 指标 |

## 5. MJCF→PhysX 转换器设计

PhysX 5 的 `PxArticulationReducedCoordinate` 是专为机器人设计的 reduced-coordinate 关节体——和 MuJoCo 的运动学模型完全对应：

```
MJCF <joint name="left_knee" axis="0 1 0" range="..." damping="2.0" frictionloss="3.0"/>
  ↓ 直接映射
PhysX: PxArticulationJointReducedCoordinate
  parentLink = left_hip_yaw
  childLink = left_knee
  jointType = eREVOLUTE
  axis = (0, 1, 0)
  lowerLimit = -0.087
  upperLimit = 2.880
  friction = 3.0
  damping = 2.0
```

无需 OpenUSD。MJCF XML 里的全部物理参数已经足够构建 PhysX 模型。

## 6. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| PhysX 浮基精度达不到 Isaac 水平 | 低 | PhysX 5 和 Isaac Sim 是同一个引擎——接触模型、积分器、摩擦求解完全一致 |
| aarch64 性能不足 | 中 | 使用 CPU-only 预设，复用现有 SHM 架构；如果太慢考虑减少 env 数 |
| pybind11 封装工作量超预期 | 中 | 只封装最小 API 集（~20 个函数），不追求完整绑定 |
| PhysX API 行为与 MuJoCo 不同 | 低 | Phase 2 用 ref PD 验证——数据对比 α，直接量化差距 |

## 7. 参考

- PhysX 5 GitHub: https://github.com/NVIDIA-Omniverse/PhysX
- PhysX 5 SDK 文档: https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/
- [[mujoco-vs-isaac-precision-analysis]] — 架构对比与精度分析完整报告
- [[status-and-next-steps]] — 项目当前状态
