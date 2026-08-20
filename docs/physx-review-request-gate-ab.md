# 审核请求：joint-order gate A/B 结果与 E8 前置清单

> 2026-08-20，请队友审核。你们的 obs/动作序补丁已推容器并完成零样本 A/B。

## A/B 结果（FORCE + Isaac 接触栈，24 eps @0.35/0.35，motion_seed 0，无裁剪）

| 格 | gate | 模式 | mean_len | 死亡模式 |
|----|------|------|----------|----------|
| A | ON | release | **12.17** | ank_pos 13, ori 11, height 2 |
| B | ON | PD | 27.88 | ank_pos 14, body_h 10, height 4, ori 1 |
| C | OFF | release | 5.25 | ank_pos 22, ori 5 |
| D | OFF | PD | 27.88（与 B 逐位相同）| 同 B |

**内部一致性**：C = 5.25 逐位复现 2×2 基线（legacy 路径无损）；B/D 逐位相同（gate 对 PD 路径零影响，sanity 通过）。

**判读**：obs/动作序修复生效（release 5.25 → 12.17，+132%），且死亡模式迁移（ori 5→11）显示策略拿回了正确关节身份、在真正尝试控制。**但距 Isaac 侧 87.33 仍有 7×**——修复是必要项，非充分项。

## 请审核的三点

**1. 剩余 7× 差距的归因与优先级**

我方当前候选排序：
- ① **clip 训练语义**：release 权重是 clip 时代产物，无裁剪下 target 满量程（E7 已证灾难性）。gate-ON 后 12 步的死亡仍是踝位置漂移主导（ank 13）——与踝 replay corr 0.45 的植物闭环残留互相印证。
- ② **obs 层遗留**（你们标注"低影响"的项）：mujoco actor 块序 [gdh, avh, ...] 独立项、SMPL 块全零、obs 噪声注入缺失、gdh/jvh 的 settle-捕获语义差。关节序修复后这些是否需要重评影响？
- ③ **植物闭环残留**：踝/根动力学（replay corr 层，绕过策略）。
你们认为哪项优先？有没有我们漏掉的候选？

**2. E8 前置清单审核**

我方计划：gate-ON + FORCE + Isaac 接触栈环境上 BC warmup → PPO（无裁剪，训练与评测语义一致）。release 12.17 只是旧权重的底数，不是 E8 目标。E8 前是否还有必须做的检查？

**3. 门控默认值确认**

`SONIC_PHYSX_ISAAC_JOINT_ORDER` 默认 ON（OFF = legacy XML 接口保 A/B）——生产/E8 默认 ON，是否同意？

## 附：可复现性

- 命令：`bash /tmp/gate_ab.sh`（容器 /tmp，4 格）与 `bash /tmp/gate_bd.sh`（B/D 补跑）
- 环境变量：SONIC_PHYSX_ISAAC_FEET=1、SONIC_PHYSX_GROUND_FRICTION=1.0,1.0
- 驱动：FORCE + vel_iters 4 + dt 0.005×4（Isaac 栈）
