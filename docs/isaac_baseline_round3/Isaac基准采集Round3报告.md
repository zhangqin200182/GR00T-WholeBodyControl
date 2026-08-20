# Isaac 基准采集 Round-3 报告（free-root + 植定脚承重 stance 响应）

> 日期：2026-08-20
> 环境：Isaac Sim 5.0.0.0 / IsaacLab v2.3.2 @ 37ddf6268
> 策略：SONIC 37M 参数 G1 humanoid（仅用其 env 配置，不跑策略）
> GPU：ssh.1617k.com:50555

---

## 1. 协议概述

R3 是植物分歧的最后一个判决：机器人站立、双脚踩地（5mm 预穿透）、髋/膝 27 组阶跃/正弦扰动，
录制承重 stance 响应（qpos/qvel/target/applied_torque + root 全序列 + 接触力）。

- **判读 fork**：响应一致 → 近端承重无分歧 → E8 放行；响应分歧+载荷同 → 传递侧；响应分歧+载荷差 → 接触侧
- 与 P1 的三处改动：① free-root（不锁根）② 非扰动关节 PD 持姿（不锁死，踝含在内）③ 初始姿态显式（默认站立、双脚并立、足底地面下 5mm）

## 2. 执行结果（27/27 完成）

| 类别 | 组数 | 结果 |
|------|------|------|
| hip_pitch step ±0.05/0.1/0.2 | 6 | 全部完整 100 步（2.0s）|
| hip_roll step ±0.05/0.1/0.2 | 6 | 全部完整 100 步 |
| knee step ±0.05/0.1/0.2 | 6 | 全部完整 100 步 |
| hip_pitch sine 0.5/2/5Hz | 3 | ~2.3s 截断（root pitch>0.35 rad，物理发散）|
| hip_roll sine 0.5/2/5Hz | 3 | ~2.3s 截断 |
| knee sine 0.5/2/5Hz | 3 | ~2.3s 截断 |

**文件**：27 主 npz + 9 截断保留 npz（`r3_*_truncated.npz`，触地前数据可用）
**命名**：`r3_<joint>_<kind>_<param>.npz`（与 physx_r3_replica.py 同款）

## 3. 关键执行决策（详见 protocol_notes.txt）

1. **记录频率 50Hz（实际）vs 200Hz（名义）**：Isaac physics_context 固定 substeps=4，
   一次 sim.step() 推进 0.02s，无法逐 0.005 子步取数。实际记录 = 控制 = 50Hz，与 P1 真实行为一致。
   npz 元数据盖章：record_hz_nominal=200, record_hz_actual=50。
2. **q0 = URDF 默认站立姿态**（非 walk 起始帧）：walk 起始帧踝预弯 -0.35~-0.47 rad 重心偏前，
   纯 PD 持姿必然前倾倒下（冒烟测试实证）。README 主协议要求"默认站立 ref 姿态、双脚并立"，
   故采用 default_joint_pos + root 正立。npz 内嵌 q0 字段供对齐。
3. **每试次前重置初始姿态**：free-root + 软 PD 无法从漂移状态自恢复（冒烟测试实证），
   每试次从标准站立重新开始，保证独立同分布初始条件。
4. **失衡规则**：root roll/pitch > 0.35 rad（replica ori_thresh）或双脚离地 200ms → 截断保留 + 重试一次。

## 4. 数据内容

每 npz 24 字段：t, qpos(29), qvel(29), target(29), applied_torque(29),
root_pos(3), root_quat(4,wxyz), contact_force_left(3), contact_force_right(3), n_contacts,
+ 元数据盖章（quat_format, control_dt, record_dt, record_hz_nominal/actual, drive_cfg, q0,
joint_name, kind, param, sign, truncated, trunc_reason, retries）

**响应质量抽查**（hip_pitch sine 2Hz）：
- qpos 幅度 0.050 vs target 幅度 0.100 —— 软增益 PD 下响应衰减（物理真实）
- 相位滞后存在（59 采样点 @50Hz）
- pitch 单调增长 0→0.28 rad —— free-root 不稳定发散（正是要对比的响应）

**step 响应**：rise 到稳态、踝/膝/髋承重 ~170N/脚，PD 持姿自然载荷分布。

## 5. 与 PhysX 侧对比注意事项

- 两侧 q0 必须一致（我们内嵌 q0 字段；若 PhysX 侧用 walk 起始帧请改默认站立）
- 记录频率两侧可能不同（我们 50Hz；replica 名义 200Hz）——建议 PhysX 侧按 50Hz 重采样对比
- applied_torque = 驱动原始力矩（act-then-record 一样本滞后），与 P1 语义相同

## 6. 交付清单

```
isaac_r3_20260820/
├── r3_*.npz (27 主文件 + 9 truncated)
├── r3_manifest.txt
├── joint_names.txt
├── protocol_notes.txt        ← 本包核心说明（运行命令/驱动版本/差异/截断统计）
├── isaac_r3_drive.py         ← 采集脚本（可复现）
├── Isaac基准采集Round3报告.md
└── isaac_request_round3_20260820/  ← 原始需求（README + 检查单 + replica）
```

## 7. 结论

- step 18 组全部完整：free-root 承重 stance 的阶跃响应可直接对比（rise90/幅值/相位/力矩波形）
- sine 9 组截断于 ~2.3s：free-root + 软 PD 持姿在持续正弦激励下前倾发散（0.5Hz 仅 ~1 周期，
  2/5Hz 有 4-11 周期可对比幅值/相位；发散时间点本身也是响应特征）
- 数据、脚本、协议说明齐全，可支撑"响应一致→E8 放行 / 分歧→定位驱动传递或接触侧"的判读
