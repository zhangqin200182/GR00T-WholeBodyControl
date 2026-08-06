# ovphysx 精度问题 ⚠️ 已废弃

> 2026-08-06 废弃：ovphysx 路径已弃用。eFORCE 硬编码无法修改，α=0.014 无法突破。
> 最终方案：Direct API (physx_bindings.cpp) + eACCELERATION，α=0.0016-0.0055。
> 当前文档：`docs/physx-status-and-plan.md`

**日期**: 2026-08-05

---

## 1. 现有数据回顾

| 实验 | α (rad) | 关键信息 |
|---|---|---|
| 零重力 ref PD | **0.0013** | PD + joint frame + 积分全部正确 |
| 重力静态 hold | **0.005** | PD 能对抗重力，稳态误差极小 |
| 重力 free-root ref tracking | **0.064** | 接触参数修复后，最好的浮基跟踪 |
| 零重力 1Hz sine | 0.047 | 单关节正弦扫描 vs 多关节 ref PD 差距大 |

**核心矛盾**：零重力 0.0013 证明引擎没问题，重力静态 hold 0.005 证明 PD 能对抗重力。问题只出在**重力 + 同时跟踪所有关节**的 ref tracking 场景。

## 2. `docs/physx-alpha-root-cause-analysis.md` 中的问题

### 2.1 静态 hold 数据与 eFORCE 数学矛盾

文档 §3.1 声称 eFORCE 稳态误差 ε = |τ_gravity| / kp，waist_pitch 需要 kp=44,000 才能达 ε=0.002。但 §2.3 报告重力静态 hold max error = 0.005 rad。

如果稳态误差公式成立，kp=28.5 的 waist_pitch 在 ~88 Nm 负载下应该是 3 rad 的静差，不是 0.005。这两个结果不可能同时成立。

**可能原因**：PhysX 约束求解器不只做简单的 τ = kp×Δq，接触约束 + 摩擦让系统有额外刚度。公式 τ_gravity/kp 是 DC motor 模型，不是有接触的多体系统模型。

### 2.2 waist_pitch 88 Nm 站姿不成立

G1 站立时躯干竖直，COM 在 waist 上方，重力矩 ≈ 0 Nm。88 Nm 是躯干水平的配置（上半身 15kg × 0.6m），站姿 ref 不会出现。

### 2.3 Isaac 用 eACCELERATION 没有证据

§3.2 给出的 pseudocode 是纯推测，没有引用 Isaac Sim 源码或文档。找不到任何公开资料确认 Isaac Sim 用 eACCELERATION 做 ref tracking。

### 2.4 零重力 1Hz vs ref PD 矛盾的归因

零重力 1Hz 正弦 α=0.047 远差于 ref PD α=0.0013。如果 eFORCE 是根因，零重力也应该差（没有重力需要补偿）。但 ref PD 实测 0.0013 完美。1Hz 正弦差的可能原因：
- 正弦波是单关节运动（其他关节不动），ref PD 是多关节协同（互相抵消一部分误差）
- 测试脚本差异

### 2.5 DRIVE_MODEL 张量 [0, FLT_MAX, 0] 推断太强

ovphysx 的 DRIVE_MODEL 张量可能是 read-only placeholder，不对应内部驱动类型。不为 eACCELERATION 不等于就是 eFORCE。

## 3. 已知事实 vs 未验证假设

### 已确认

| 事实 | 证据 |
|---|---|
| ovphysx 零重力精度达标 | α=0.0013，200 frame 稳定 |
| PD 能对抗重力稳态 | 静态 hold max error=0.005 |
| 碰撞几何重建完成 | 39 个 geom 全部转 USD primitive |
| bare SDK 有 createLink bug | α plateau @ 0.02，12 假设已排除 |
| ovphysx 精度 > bare SDK | 0.0013 vs 0.02，joint frame 解耦有效 |

### 未验证假设

| 假设 | 风险 | 验证成本 |
|---|---|---|
| "Isaac 用 eACCELERATION" | 没有证据，方向可能全错 | 高 -- 需要 Isaac 源码 |
| "eFORCE 是 ovphysx α=0.064 的根因" | 零重力 0.0013 与此矛盾 | 低 -- 可交叉验证 |
| "Isaac 浮基重力 ref PD α 也是 ~0.05" | 无法确认，没 Isaac 环境 | 高 |
| "RL 能在 α=0.064 物理上正常训练" | 未验证 | 低 -- smoke test |

## 4. 下一步验证计划（按优先级）

### V1: ovphysx vs bare SDK 同场景对比（0.5d）

**要回答的问题**：ovphysx 的精度是否显著优于 bare SDK？

bare SDK 有 createLink bug（α plateau @ 0.02），ovphysx 有 correct joint frame。同场景（固定 base + 重力 + ref PD）对比：
- bare SDK α 应当 ≥ 0.02（已知结果）
- ovphysx α 如果 < 0.01 → ovphysx 确实有改进，eFORCE 不是根因
- ovphysx α 如果也 ≈ 0.02 → joint frame 不是瓶颈，瓶颈在驱动方式

**最低成本**：用已有脚本，只换后端。

### V2: 重力 ref tracking 误差分解（0.5d）

**要回答的问题**：重力 ref tracking α=0.064 里，多少是 PD 静差、多少是 root 漂移、多少是运动耦合？

方法：逐关节分析稳态阶段的 RMS error，与 kp 归一化：
- 高 kp 关节（hip kp=99.1）error 应该小
- 低 kp 关节（shoulder kp=14.3）error 应该大
- 如果高 kp 关节 error 也大 → 问题不是 kp 不够，是其他机制
- 如果 error ∝ 1/kp → 就是 eFORCE 的静差 → 提 kp 或换驱动

**已有数据**：之前上半身 ratio 5-20× 已经在 bare SDK 上见过。需要在 ovphysx 上复现。

### V3: 全局 kp×5 验证（0.5d）

**要回答的问题**：如果 eFORCE 静差是根因，提 kp 应该改善？

在 `mjcf_to_usd.py` 里对上半身 kp 统一 ×5（shoulder 14.3→71.5, elbow 14.3→71.5, waist 28.5→142.5），跑重力 ref tracking。

- 如果 α 从 0.064 → 0.02 以下 → eFORCE 静差确实是问题，临时方案可行
- 如果 α 不变 → 有其他根因

### V4: PPO smoke test（1d）

**要回答的问题**：α=0.064 的物理环境 RL 能不能学？

只要 V1-V3 排除了 ovphysx 有隐藏的物理 bug，就应该直接跑 PPO smoke test。之前 MuJoCo α=0.013 都能做训练，ovphysx α=0.064 如果 V3 提 kp 能到 0.02 以下就更应该能训。

## 5. 执行顺序

```
V1 (ovphysx vs bare SDK 同场景)
 └─ ovphysx < bare SDK → joint frame 有效，继续
    ovphysx ≈ bare SDK → 有新问题，重新排查

V2 (误差分解)
 └─ error ∝ 1/kp → eFORCE 静差，走 V3
    error 不跟 kp 相关 → 接触/动力学问题，重新排查

V3 (kp×5)
 └─ α < 0.02 → 参数方案可行，走 V4
    α 不变 → 有其他根因，重新排查

V4 (PPO smoke test)
```

## 6. 结论

`docs/physx-alpha-root-cause-analysis.md` 的 "eACCELERATION 是根因" 结论**没有证据支撑**，且与零重力 α=0.0013 和静态 hold α=0.005 这两个实测结果矛盾。ovphysx 零重力表现完美 + 静态能 hold 证明引擎本身正确。

真正的瓶颈更可能是：
1. **eFORCE 静差在运动耦合时放大**（静态 hold 肌肉收缩没有大幅 δq，但 ref tracking 持续改 target，每个子步都在追新目标，误差机会多）
2. **KP 绝对值不足**（上半身 kp=14.3 对 30kg 人形来说太低）

建议按 V1→V2→V3 顺序快速验证，每一步都有明确的 go/no-go 条件。如果 V3 能把 α 拉到可接受范围，直接 V4 训练。
