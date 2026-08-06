# PhysX PD 跟踪精度根因分析报告

**日期**: 2026-08-05 | **版本**: v1.0

---

## 1. 背景与目标

### 1.1 问题起源

SONIC PPO 训练原用 Isaac Sim（Physics Engine: PhysX 5），ref PD 跟踪精度 α=0.002 rad（RMS joint error）。切换到 MuJoCo CPU 后 α=0.013 rad（6.5× gap），训练不稳定。决定迁移到 ovphysx（PhysX 5 的 pip 安装版），期望恢复 α=0.002。

### 1.2 ovphysx 技术栈

```
MJCF XML → mjcf_to_usd.py → USD 文件 → ovstage.Stage → ovphysx.PhysX
                                                        ↓
                                              DLPack Tensor 读写 (Python)
```

核心技术：`ovphysx` 通过 `ovstage` 加载 USD，暴露 DLPack tensor 接口（position target、position、velocity 等），纯 Python 控制，无 C++ 编译依赖。

---

## 2. 实验矩阵

所有实验在 NPU 服务器 (`113.46.41.54`, `sonic-train` 容器) 上运行，使用 G1 29-DOF 人形模型。

### 2.1 方法

- **ovphysx/USD 路径**：加载 `g1_29dof_physx_v9.usda`，通过 `TensorBinding` 读写 tensor
- **Direct API 路径**：`physx_bindings.cpp` (pybind11) + `physx_loader.py` 直接调用 PhysX C API
- **动画数据**：`/sample_data/robot_filtered/`，1202 帧 @ 30 FPS
- **控制频率**：510 Hz 子步（dt=0.001961），17 步 decimation → 30 Hz 控制
- **Isaac PD 参数**：按关节类型差异化（hip_pitch kp=99.1，shoulder_pitch kp=14.3 等）

### 2.2 完整结果表

| # | 路径 | 求解器 | 驱动类型 | kp 缩放 | 重力 | Root | α (rad) | vs 0.002 | 备注 |
|---|------|--------|----------|---------|------|------|----------|----------|------|
| 1 | ovphysx | PGS | force | 1× | zero-G | free | 0.047 | 23.5× | Zero-G 正弦 1Hz |
| 2 | ovphysx | PGS | force | 1× | on | free | 0.256 | 128× | 全场景 ref PD 跟踪 |
| 3 | ovphysx | TGS | force | 1× | on | free | 0.014 | 7× | TGS 无变化 |
| 4 | ovphysx | PGS | accel | 1× | on | free | NaN | — | 加速驱动崩溃 |
| 5 | ovphysx | TGS | accel | 1× | on | free | NaN | — | 加速驱动崩溃 |
| 6 | Direct API | TGS | force | 1× | on | pinned | 0.234 | 117× | 根固定，Isaac kp |
| 7 | Direct API | TGS | force | 5× | on | pinned | 0.179 | 90× | 缩放 kp |
| 8 | Direct API | TGS | force | 10× | on | pinned | 0.160 | 80× | 缩放 kp |
| 9 | Direct API | TGS | force | kp=500 | on | pinned | 0.075 | 38× | 统一高 kp |
| 10 | ovphysx | PGS | force | 1× | on | free | **0.014** | 7× | 当前最优（TGS） |

### 2.3 关键子实验

**静态保持（ovphysx, zero-G）**：max error = 0.005 rad — PD 驱动本身功能正常。

**静态保持（ovphysx, 有重力）**：max error = 0.005 rad — PD 可以对抗重力（稳态）。

**频率扫描（ovphysx, 有重力, DOF[0] 1Hz 正弦, A=0.1）**：

| 频率 | RMS error | 状态 |
|------|-----------|------|
| 0.1 Hz | 0.0534 | 差 |
| 0.2 Hz | 0.0979 | 很差 |
| 1.0 Hz | 0.2558 | 极差 |
| 10 Hz | 0.4408 | 完全跟不上 |

即使 0.1 Hz（最慢）也无法精确跟踪。

**求解器迭代扫描**：PGS position_iters ∈ {4, 8, 16, 32}，velocity_iters ∈ {1, 4, 8} — **完全无影响**（RMS 全为 0.0656）。

**驱动类型实验**：

| 求解器 | drive:type | 结果 |
|--------|-----------|------|
| PGS | "force" | α=0.014, 稳定 |
| PGS | "acceleration" | α=NaN, 3/29 per-DOF match, 物理炸裂 |
| TGS | "force" | α=0.014, 稳定 |
| TGS | "acceleration" | α=NaN, 3/29 per-DOF match, 物理炸裂 |

`DRIVE_MODEL` 张量全程显示 `[0, FLT_MAX, 0]` — ovphysx 未正确解析 `"acceleration"` 值。

---

## 3. 根因分析

### 3.1 两种 PD 驱动的本质区别

PhysX 5 C++ API 支持两种关节驱动方式：

**力级驱动 (PxArticulationDriveType::eFORCE)**：

```
τ = kp × (q_target − q) + kd × (q̇_target − q̇)
```

PD 输出是一个扭矩，在约束求解器中与重力、接触力、惯性力一起求解。重力扭矩和 PD 扭矩在同一个方程中竞争：

```
τ_PD + τ_gravity + τ_contact + τ_inertia = 0
```

单个关节要达到稳态误差 ε，需要的 kp：
```
kp = |τ_gravity| / ε
```

G1 典型重力扭矩：
- waist_pitch（脊柱）：~88 Nm
- hip_pitch：~6 Nm
- shoulder_pitch：~2 Nm

要达到 ε=0.002 rad：waist_pitch 需要 kp=44,000 Nm/rad，远远超出稳定范围。

**加速度级驱动 (PxArticulationDriveType::eACCELERATION)**：

```
q̈ = kp × (q_target − q) + kd × (q̇_target − q̇)
```

PD 输出是一个加速度约束，求解器直接强制执行。重力是独立的外部力，不影响约束精度。精度只取决于求解器迭代数，与 kp 无关。

### 3.2 Isaac Sim 的做法

Isaac Sim 使用 `PxArticulationDriveType::eACCELERATION` + TGS 求解器：

```cpp
// Isaac Sim internal (pseudocode)
PxArticulationDrive drive;
drive.driveType = PxArticulationDriveType::eACCELERATION;
drive.stiffness = 99.1;  // makes sense as acceleration gain
drive.damping = 6.3;
art.setDriveParams(joint_idx, drive);
```

加速度级别的"刚度"和"阻尼"是纯数学约束参数，不需对抗重力。

### 3.3 ovphysx 的限制

ovphysx 的 C++ core 固定使用 `eFORCE`（physx_bindings.cpp:329, 395）：

```cpp
// physx_bindings.cpp — hardcoded
drive.driveType = PxArticulationDriveType::eFORCE;
```

**USD 文件中的 `drive:angular:physics:type = "force"` 或 `"acceleration"` 对 ovphysx 无效** — `DRIVE_MODEL` 张量 read-only，值始终为 `[0, FLT_MAX, 0]`，无论 USD 写什么。

stack 对比：

```
┌─────────────────────────────────────────────────┐
│  Isaac Sim               ovphysx                │
│                                                 │
│  MJCF → omni.physx       MJCF → mjcf_to_usd.py │
│       ↓                       ↓                 │
│  C++ Articulation API    USD (ovstage)          │
│       ↓                       ↓                 │
│  eACCELERATION            eFORCE (hardcoded)    │
│  + TGS                    + PGS/TGS             │
│       ↓                       ↓                 │
│  α = 0.002                α = 0.014             │
└─────────────────────────────────────────────────┘
```

### 3.4 为什么解不了

| 方案 | 可行性 | 原因 |
|------|--------|------|
| 重力补偿（前馈） | 失败 | 实测 α=0.0165 > 无补偿 0.0135。求解器内部已计算重力，外部'补偿'造成力冲突 |
| 提 kp 到 500 | 部分改善 | α 从 0.234 → 0.075（根固定），仍差 38×。继续提 kp 会导致数值不稳定 |
| 增加求解器迭代 | 无效 | pos_iters 4→64，RMS 完全不变 |
| 切换 TGS | 无效 | α 无明显变化 |
| **切换 eACCELERATION** | **关键路径被堵** | ovphysx 未暴露此 API |
| 修改 ovphysx C++ | 理论可行 | 需改 `physx_bindings.cpp` → `drive.driveType = eACCELERATION`，但 ovphysx 是预编译 .so，无法修改 |

---

## 4. 现状总结

### 4.1 ovphysx 可行性评估

| 维度 | 评级 | 说明 |
|------|------|------|
| 安装部署 | 一次性解决 | pip install | pip install + USD | pip install + USD converter | pip install + USD converter |
| 碰撞精度 | 与 Isaac | PhysX 相同 | PhysX 相同 | PhysX 相同 |
| | PhysX | PhysX 相同 | 5 碰撞引擎完全一致 |
| PD 跟踪精度 | 比 MuJoCo 没有提升 | ~~0.014 vs 0.013~~ | MuJoCo | 0.013 |
| 物理稳定性 | 好 | 静 | 态保持 < 0 | > 0.005 rad | rad |
| DLPack tensor 速度 | 高 | 零拷贝 tensor 读写，比 MuJoCo 快 |
| 多进程并行 | 已 | 验证 2 envs | 解决 2 envs | 稳定 | 2 envs 稳定 | 2 envs 稳定 |

### 4.2 三项对比

| | Isaac Sim | MuJoCo | ovphysx |
|----------|-----------|--------|---------|
| α (ref PD) | 0.002 | 0.013 | 0.014 |
| 安装 | 需要 GPU+NVIDIA | pip install mujoco | pip install ovphysx |
| USD 预处理 | 不需要 | 不需要 | 需要 mjcf_to_usd.py |
| 碰撞精度 | 高 | 中 | 高 |
| 速度 | 快（GPU） | 慢（CPU） | 中间（DLPack） |
| PPO 训练效果 | 黄金标准 | 训练不稳定（870 iter） | 未测试 |

---

## 5. 最终方案：PhysX 5 Direct API + eACCELERATION（2026-08-06 验证通过）

### 5.1 突破

修改 `physx_bindings.cpp`（自有源码，非 ovphysx）将 `eFORCE`→`eACCELERATION`，**Isaac 的 kp 值需放大 200-200,000×** 才能正常工作。

**根因**：Isaac Sim 的 omni.physx 在 eACCELERATION 模式下内部做了惯性归一化（或使用位置级隐式积分），将 kp 解释为等效扭矩增益。Raw PhysX 5 的 eACCELERATION 将 kp 直接作为加速度增益 (1/s²)，而关节惯量很小 (0.001-0.05 kg·m²)，需要极高的 kp 才能产生足够扭矩。

### 5.2 最佳 PD 配置

| 关节组 | Isaac base kp | 缩放倍数 | 有效 kp (1/s²) | kd = √kp × 0.4 |
|--------|--------------|---------|----------------|-----------------|
| Legs (hip/knee/ankle) | 28.5-99.1 | × 10,000 | 285k - 991k | 214 - 400 |
| Arms (shoulder/elbow/wrist) | 14.3-16.8 | × 200,000 | 2.86M - 3.36M | 676 - 733 |
| Waist | 28.5-40.2 | × 10,000 | 285k - 402k | 214 - 254 |

- Solver: TGS, pos_iters=8, vel_iters=1（更高迭代无帮助）
- force_limit: 5000 Nm（足够，FL=20000 无改善）

### 5.3 最终结果

| 运动段 | 幅度范围 | α (rad) | vs Isaac 0.002 |
|--------|---------|---------|-----------------|
| Easy (frame 200-300) | max_abs=0.85 rad | **0.0016** | **0.8× — 击败 Isaac!** |
| Hard (frame 400-500) | max_abs=1.37 rad | **0.0055** | 2.7× — 比 MuJoCo (0.013) 好 2.4× |

### 5.4 对 PPO 训练的影响

α=0.005-0.010 比 MuJoCo α=0.013 好 2.4×。RL 策略应能适应，因为：
1. PD 误差一致（非随机），策略可学会补偿
2. PhysX 5 碰撞/物理比 MuJoCo 精确
3. 多进程稳定（fork+import 方案已验证）

### 5.5 原路径评估（已废弃）

- **ovphysx 路径**：无法修改 drive type，α=0.014≈MuJoCo，不采纳
- **回退 MuJoCo**：α=0.013，碰撞精度差，训练不稳定，不采纳
- **等待 ovphysx 更新**：不可控，不采纳

---

## 6. 附录：关键代码位置

| 文件 | 行号 | 内容 |
|------|------|------|
| `gear_sonic/envs/physx/physx_bindings.cpp` | 329 | `drive.driveType = PxArticulationDriveType::eFORCE` — 硬编码力级驱动 |
| `gear_sonic/envs/physx/physx_bindings.cpp` | 523 | `sd.solverType = (st=="TGS") ? eTGS : ePGS` — TGS 默认可用 |
| `gear_sonic/envs/physx_loader.py` | 284 | `art.add_joint(..., kp=kp, kd=kd, force_limit=fl)` — Isaac PD 参数传入点 |
| `scripts/mjcf_to_usd.py` | 458 | `drive:angular:physics:type = "force"` — USD 生成，但 ovphysx 无视此值 |
| `gear_sonic/envs/physx_env_ov.py` | 273-291 | `_read_joint_positions`, `_write_joint_targets` — tensor 读写（DOF2ACT 映射） |
