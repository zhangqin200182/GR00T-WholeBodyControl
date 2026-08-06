# PhysX 训练管线：现状与计划

> 2026-08-06 | eACCELERATION 突破：α=0.0016-0.0055，直接匹配/超越 Isaac 0.002

## 1. 突破：eACCELERATION（2026-08-06）

### 1.1 问题与根因

裸 PhysX 5 C++ API 的 `PxArticulationDriveType::eFORCE`（力级 PD）在 G1 上只能达到 α=0.014，与 MuJoCo α=0.013 同一水平。多次尝试提 kp、增 solver 迭代、重力补偿等手段均无法突破。

Isaac Sim 使用 `eACCELERATION`（加速度级 PD），我们的 physx_bindings.cpp 具备切换到 eACCELERATION 的能力，但初测用 Isaac 原始 kp 值（14-99）得到 α=0.217 —— 比 eFORCE 更差，一度误判 eACCELERATION "不可用"。

**根因**：Isaac 的 omni.physx 在 eACCELERATION 内部对 stiffness 做了惯性归一化（`τ = I × kp × ε`），kp 保持 Nm/rad 的物理意义。裸 PhysX 5 的 eACCELERATION 将 kp 直接作为加速度增益（`q̈ = kp × ε`），单位是 1/s²。对于 G1 关节惯性 I≈0.001-0.05 kg·m² 的小惯量关节，Isaac 的 kp 值在加速度域中太小（差 200-200,000×）。

### 1.2 解决方案

```cpp
// physx_bindings.cpp:329,395
drive.driveType = PxArticulationDriveType::eACCELERATION;  // was eFORCE
```

PD 参数分为三组缩放：

| 关节组 | Isaac base kp | 缩放倍数 | 有效 kp (1/s²) | kd = √kp × 0.4 |
|--------|--------------|---------|-----------------|-----------------|
| Legs (hip/knee/ankle) | 28.5 - 99.1 | × 10,000 | 285k - 991k | 214 - 400 |
| Waist | 28.5 - 40.2 | × 10,000 | 285k - 402k | 214 - 254 |
| Arms (shoulder/elbow/wrist) | 14.3 - 16.8 | × 200,000 | 2.86M - 3.36M | 676 - 733 |

TGS solver, pos_iters=8, vel_iters=1。更高迭代数无帮助。

### 1.3 结果

| 运动段 | 幅度 | α (rad) | vs Isaac 0.002 |
|--------|------|---------|-----------------|
| Easy (frame 200-300) | max_abs=0.85 rad | **0.0016** | 0.8× — 击败 Isaac |
| Hard (frame 400-500) | max_abs=1.37 rad | **0.0055** | 2.7× — 比 MuJoCo 0.013 好 2.4× |

路径对比：

| | Isaac Sim | MuJoCo | ovphysx | Direct API + eACCELERATION |
|---|---|---|---|---|
| α | 0.002 | 0.013 | 0.014 | **0.0016-0.0055** |
| 驱动类型 | eACCELERATION | 隐式位置约束 | eFORCE (硬编码) | eACCELERATION (可控) |
| Python 绑定 | omni.physx (闭源) | mujoco (开源) | ovphysx pip (闭源) | physx_bindings.cpp (自有) |
| 可行性 | GPU 需要 | NPU 上可行 | 无法改 drive type | ✅ 已验证 |

## 2. 原分析修正

2026-08-04 时误判裸 PhysX SDK 有 `createLink` 的不可修复结构缺陷（joint frame / parentPose / CoM 耦合），声称 α≈0.02 是 "hard plateau"。实际上：

1. **eACCELERATION 的 kp 缩放才是根因**——不是 createLink 的问题。eFORCE α=0.014 与 Isaac 0.002 的差距来自驱动类型，不是 joint frame。
2. **createLink 的 joint frame 没有问题**——world-accumulated 路径下物理稳定 100+ 步，TGS solver 内部正确处理了 constraint frame。
3. **α=0.0016 证明**——createLink 的 joint frame 精度足以支持 α < 0.002。之前的 "hard plateau" 只是 eFORCE 极限。

撤销 2026-08-04 的建议：ovphysx 路径已废弃（无法改 drive type，α=0.014）。

## 3. 技术栈确定

```
MJCF XML → physx_loader.py → physx_bindings.cpp (pybind11) → PhysX 5.4 C++ SDK
    ↓                              ↓
  XML 解析                   eACCELERATION 驱动
  createLink                 kp × 10k-200k
  Python FK                   TGS 8/1 iter
```

不依赖 ovphysx，不依赖 USD。所有组件都在自己的代码里，可任意修改。

### 关键文件

| 文件 | 用途 |
|------|------|
| `gear_sonic/envs/physx/physx_bindings.cpp` | PhysX 5 pybind11 wrapper，eACCELERATION |
| `gear_sonic/envs/physx/physx_loader.py` | MJCF XML → PhysX articulation 转换器 |
| `gear_sonic/envs/physx/physx_fk.py` | 纯 Python forward kinematics |
| `gear_sonic/envs/physx_env.py` | PhysX 单环境（Direct API） |
| `scripts/sweep_accel_kp.py` | kp 扫描脚本 |
| `scripts/test_accel_ref_pd.py` | ref PD 跟踪验证 |

## 4. 已完成的工作

| 任务 | 状态 |
|------|------|
| T1: Per-joint PD gains | ✅ Isaac 配置 |
| T2: Python FK 统一观测 | ✅ 替代所有 getGlobalPose |
| T3: Joint frame fix | ✅ 不需要修复 — createLink 是对的 |
| T4: Ref PD 验证 | ✅ α=0.0016-0.0055 |
| Multi-process | ✅ fork+import 方案 |
| Collision shapes | ✅ MJCF geom → PhysX shapes |
| eACCELERATION | ✅ 已提交 |

## 5. 待办

| 任务 | 优先级 | 说明 |
|------|--------|------|
| kp 缩放集成到 physx_env.py | P0 | 实现 per-joint-type 缩放逻辑 |
| 碰撞 smoke test | P0 | 确认 geom shapes 在 step 中正常工作 |
| 多进程 eACCELERATION 验证 | P0 | 确认 fork+import + eACCELERATION 稳定 |
| 端到端 PPO smoke test | P1 | BC warmup + PPO 100 iter |
| 配置对齐 Isaac (alive_bonus, thresholds) | P1 | 见 mujoco-isaac-gap-audit |

## 6. 历史文档

- `docs/physx-alpha-root-cause-analysis.md` — 完整实验矩阵与分析（2026-08-06 更新版）
- `docs/physx-stability-analysis.md` — createLink 时序分析（历史参考）
- `docs/physx-tracking-tasks.md` — T1-T6 任务拆解（历史参考）
- `docs/physx-next-steps.md` — 早期 ovphysx 方向（已废弃）
- `docs/physx-engine-switch-implementation.md` — 原 ovphysx 实现计划（已废弃）
- `docs/ovphysx-next-verification-plan.md` — ovphysx 验证计划（已废弃）
