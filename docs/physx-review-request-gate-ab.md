# 审核请求：joint-order gate A/B 结果与 E8 前置清单

> 2026-08-20，请队友审核。你们的 obs/动作序补丁已推容器并完成零样本 A/B。

## A/B 结果（FORCE + Isaac 接触栈，24 eps @0.35/0.35，motion_seed 0，无裁剪）

| 格 | gate | 模式 | mean_len | 死亡模式 |
|----|------|------|----------|----------|
| A | ON | release | **12.17** | ank_pos 13, ori 11, height 2 |
| B | ON | PD | 27.88 | ank_pos 14, body_h 10, height 4, ori 1 |
| C | OFF | release | 5.25 | ank_pos 22, ori 5 |
| D | OFF | PD | 27.88（与 B 逐位相同）| 同 B |

> 注：死亡模式计数按原因字符串多模式匹配（如 "height(0.6) ori(0.4)" 同时计入两类），各行计数可超过 24。

**内部一致性**：C = 5.25 逐位复现 2×2 基线（legacy 路径无损）；B/D 逐位相同（PD 路径无策略消费 obs，植物轨迹逐位相同——gate 只换接口不碰物理，sanity 通过）。

**判读**：obs/动作序修复生效（release 5.25 → 12.17，+132%），且死亡模式迁移（ori 5→11）显示策略拿回了正确关节身份、在真正尝试控制。**但距 Isaac 侧 87.33 仍有 7×**——修复是必要项，非充分项。

## 请审核的三点

**1. 剩余 7× 差距的归因与优先级**

我方当前候选排序：
- ① **植物闭环残留**（主候选）：replay corr 0.45-0.52（无策略的纯环境量）经策略闭环放大。gate-ON 后死亡仍是踝位置漂移主导（ank 13-15）——与 replay corr 的踝残留同源。
- ⚠️ **更正（2026-08-20）**："clip 训练语义"是 E7 时代的错误归因，撤回：release 权重是 NVIDIA 在 Isaac 无裁剪训练的；裁剪是我们评测侧遗留物，其真实作用曾是安全阀掩盖 obs 关节序错位（把策略推到 ±8 vs Isaac ±5.2）。gate-ON 修复后 clip 不参与当前差距解释。
- ② **obs 层遗留**（你们标注"低影响"的项）：mujoco actor 块序 [gdh, avh, ...] 独立项（仅影响 mujoco 管线一致性，生产是 physx）、SMPL 块全零、obs 噪声注入缺失、gdh/jvh 的 settle-捕获语义差。关节序修复后这些是否需要重评影响？其中 **critic/tokenizer 的逐块对齐可以用数据直接判决**：他们 npz 录制了全 clip 的 ref_qpos（50Hz），future 窗口（0.1s × 10 帧）可从后续帧重建——用我们重建的 critic/tokenizer obs 与他们录制的逐块 diff，比"低影响"标注更硬。SMPL 块同理：若他们录制行全零则该项关闭，非零则升级。
- ③ **植物闭环残留**：踝/根动力学（replay corr 层，绕过策略）。
- ④ **（补）训练分布对齐**：settle-捕获语义（他们 obs 捕获 qvel 0.5-4.0 rad/s，我们 reset 后 qvel=0）与 obs 噪声注入缺失——影响 E8 训练分布，不解释 release 零样本差距（eval 更干净只会更有利）。
- ⑤ **（补）语境修正**：7× 是"环境差距"的上限而非点估计——release 权重本身带 clip 语义，真实环境差距只有 E8 重训后（我们的策略 vs 87.33）才有定义。另注 FORCE PD 27.88（我们）> Isaac 自侧 PD 18.58：植物层在 PD 档不比 Isaac 严苛。
你们认为哪项优先？有没有我们漏掉的候选？

**2. E8 前置清单审核**

我方计划：gate-ON + FORCE + Isaac 接触栈环境上 BC warmup → PPO（无裁剪，训练与评测语义一致）。release 12.17 只是旧权重的底数，不是 E8 目标。E8 前是否还有必须做的检查？

我方自检清单（请补漏）：
1. **配置审计闭合**：alive_bonus=0、ignore_terminations=True（skip 会断 GAE）、isaac_action_space=True、reward 权重/终止阈值与 Isaac CFG 逐项核对——三个历史大坑显式钉进 E8 训练 config。
2. **门控钉进启动脚本**：SONIC_PHYSX_ISAAC_JOINT_ORDER=1 显式写入 launch script，训练产物自文档化。
3. **BC smoke**：固定接口上先跑 10-20 iter，确认 BC loss 收敛 + PD 等价 rollout 正常，再放全量。
4. **settle-on-reset 决策**：他们捕获是 settle 后（qvel 0.5-4.0），我们 reset 即零速——E8 训练分布对齐与否先拍板（自洽即可，但要显式决策）。
5. **评测集宽度**：12 条录制 clip 确定性进 E8 评测（唯一有 Isaac ground truth 的对照集）。

**3. 门控默认值确认**

`SONIC_PHYSX_ISAAC_JOINT_ORDER` 默认 ON（OFF = legacy XML 接口保 A/B）——生产/E8 默认 ON，是否同意？

## 附：可复现性

- **v1 核实结论（已修正）**：v1 runner（/tmp/gate_ab.sh）实际只设了 2 个 env 变量（ISAAC_FEET、GROUND_FRICTION）——原"全 Isaac 栈"措辞不准确，实为 2×2 矩阵栈。A/B **对比**不受影响（两侧配置逐位相同），但绝对值基于 2×2 栈。
- **v2（已入库）**：`scripts/gate_ab_eval.sh`（commit 见 git log），补全全栈 env 变量并重跑 A/C：
  - SONIC_PHYSX_ISAAC_JOINT_ORDER=1（A）或 0（C）
  - SONIC_PHYSX_ISAAC_FEET=1、SONIC_PHYSX_GROUND_FRICTION=1.0,1.0
  - SONIC_PHYSX_BOUNCE_THRESHOLD=0.5、SONIC_PHYSX_FRICTION_OFFSET=0.04、SONIC_PHYSX_FRICTION_CORR_DIST=0.025
  - SONIC_PHYSX_ROOT_Z_OFFSET=0.02
  - cross_eval 参数：--drive-type FORCE --vel-iters 4 --native-dt 0.005 --decimation 4 --isaac-space
- B/D（PD 两格）已在 v1 栈上确认逐位相同（27.88）——PD 无策略路径，gate 不敏感结论与接触栈无关，无需重跑。
- **v2 全栈 A/C 结果**：gate ON **10.62**（ank 15, ori 15）/ gate OFF 5.42（ank 23, ori 5）。对比结论不变（+96% vs v1 +132%，接触参数在 gate-OFF 侧噪声级 5.25→5.42；gate-ON 侧 −13% 主因 rz 0.02 起步低 2cm）。**序修复必要非充分的判决在两种栈上均成立。**
