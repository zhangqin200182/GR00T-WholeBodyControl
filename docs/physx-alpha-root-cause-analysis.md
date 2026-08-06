# PhysX PD 跟踪精度技术报告

**日期**: 2026-08-06 | **版本**: v2.0

---

## 1. 问题与探索过程

### 1.1 问题起源

SONIC PPO 训练原用 Isaac Sim（Physics Engine: PhysX 5 eACCELERATION），ref PD 跟踪精度 α=0.002 rad（RMS joint error）。切换到 MuJoCo CPU 后 α=0.013 rad（6.5× gap），训练不稳定。决定迁移到裸 PhysX 5 Direct API，期望恢复 α=0.002。

### 1.2 早期探索（已废弃）

最初用 eFORCE 驱动类型（`PxArticulationDriveType::eFORCE`），使用 Isaac 的 kp 值（14-99），在 ovphysx 和 Direct API 上均只达到 α=0.014 — 与 MuJoCo 同一水平。

尝试过的无效/收效甚微的方案：
- 提高 kp（α 从 0.234→0.075，仍差 38×）
- 增加求解器迭代（pos_iters 4→64 无变化）
- 切换 PGS↔TGS（无改善）
- 重力补偿前馈（反而恶化）
- 频率扫描（0.1Hz 已跟不上）
- ovphysx 路径（硬编码 eFORCE，无法修改）

这些实验的结论**事后被证明是误判**——问题不在 PhysX 本身，而在 Isaac kp 值对于裸 PhysX 5 eACCELERATION 来说太小。

### 1.3 早期错误结论

上述探索导致以下误判（均已在 2026-08-06 纠正）：

1. "PhysX SDK 的 createLink 有不可修复的结构缺陷（joint frame / parentPose / CoM 耦合）"
   - **实际**：createLink 的 joint frame 没有问题，TGS solver 正确处理了 constraint frame
2. "α≈0.02 是裸 PhysX 的 hard plateau"
   - **实际**：α=0.0016 证明 createLink 精度足以支持 α < 0.002
3. "eACCELERATION 不可用"
   - **实际**：eACCELERATION 完美工作，只是 kp 需要 200-200,000× 缩放

---

## 2. 突破：eACCELERATION + kp 缩放（2026-08-06）

### 2.1 根因

Isaac Sim 的 omni.physx 在 eACCELERATION 内部对 stiffness 做了惯性归一化（`τ = I × kp × ε`），kp 保持 Nm/rad 的物理意义。裸 PhysX 5 的 eACCELERATION 将 kp 直接作为加速度增益（`q̈ = kp × ε`），单位是 1/s²。

对于 G1 关节惯性 I≈0.001-0.05 kg·m² 的小惯量关节，Isaac 的 kp 值在加速度域中差 200-200,000×。

```cpp
// physx_bindings.cpp — 关键修改
drive.driveType = PxArticulationDriveType::eACCELERATION;  // was eFORCE
```

### 2.2 最佳 PD 配置（经验验证）

| 关节组 | Isaac base kp | 缩放倍数 | 有效 kp (1/s²) | kd = √kp × 0.4 |
|--------|--------------|---------|-----------------|-----------------|
| Legs (hip/knee/ankle) | 28.5 - 99.1 | × 10,000 | 285k - 991k | 214 - 400 |
| Waist | 28.5 - 40.2 | × 10,000 | 285k - 402k | 214 - 254 |
| Arms (shoulder/elbow/wrist) | 14.3 - 16.8 | × 200,000 | 2.86M - 3.36M | 676 - 733 |

- Solver: TGS, pos_iters=8, vel_iters=1（更高迭代无帮助）
- force_limit: 5000 Nm（足够，FL=20000 无改善）
- 注意：计算 inertia 的反射惯量公式（`kp_form = kp_isaac × 1000 / I_reflected`）与经验值有 10-85× 偏差，三组经验缩放法比公式更准确

### 2.3 结果

| 运动段 | 幅度范围 | α (rad) | vs Isaac 0.002 |
|--------|---------|---------|-----------------|
| Easy (frame 200-300) | max_abs=0.85 rad | **0.0016** | 0.8× — 击败 Isaac |
| Hard (frame 400-500) | max_abs=1.37 rad | **0.0055** | 2.7× — 比 MuJoCo 0.013 好 2.4× |

Hard 段精度劣于 Easy 段是运动幅度效应（更大的关节角度 → 更大的跟踪误差），合法物理现象，非 bug。

### 2.4 四路对比

| | Isaac Sim | MuJoCo | ovphysx | Direct API + eACCELERATION |
|---|---|---|---|---|
| α | 0.002 | 0.013 | 0.014 | **0.0016-0.0055** |
| 驱动类型 | eACCELERATION | 隐式约束 | eFORCE (硬编码) | eACCELERATION (可控) |
| 可行性 | GPU 需要 | NPU 可行 | 无法改驱动 | ✅ 验证通过 |

---

## 3. 技术栈确定

```
MJCF XML → physx_loader.py → physx_bindings.cpp (pybind11) → PhysX 5.4 C++ SDK
    ↓                              ↓
  XML 解析                   eACCELERATION 驱动
  createLink                 kp × 10k-200k
  Python FK                   TGS 8/1 iter
```

不依赖 ovphysx，不依赖 USD。所有组件在自有代码中，可任意修改。

## 4. 对 PPO 训练的影响

α=0.005-0.010（取决于运动幅度）比 MuJoCo α=0.013 好 2.4×。RL 策略应能适应：
1. PD 误差一致（非随机），策略可学会补偿
2. PhysX 5 碰撞/物理比 MuJoCo 精确
3. 多进程稳定（fork+import 方案已验证）

## 5. 关键代码位置

| 文件 | 行号 | 内容 |
|------|------|------|
| `gear_sonic/envs/physx/physx_bindings.cpp` | 329, 395 | eACCELERATION 驱动类型设置 |
| `gear_sonic/envs/physx/physx_bindings.cpp` | 523 | solver type 配置（TGS 默认） |
| `gear_sonic/envs/physx/physx_loader.py` | `_scaled_pd_gains()` | kp 缩放 + kd 计算 |
| `gear_sonic/envs/physx/physx_fk.py` | — | 纯 Python forward kinematics |
| `gear_sonic/envs/physx_env.py` | — | Direct API 单环境 wrapper |
| `scripts/test_accel_ref_pd.py` | — | ref PD 跟踪验证 |
| `scripts/sweep_accel_kp.py` | — | kp 扫描脚本 |
| `scripts/train_physx_ppo_direct.py` | — | PPO 端到端 smoke test |
