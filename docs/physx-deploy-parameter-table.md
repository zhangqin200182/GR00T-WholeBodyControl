# 部署栈参数表提取与交叉核对（前置任务 #13）

> 来源：`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp`
> 日期：2026-08-17

## 一、提取的逐关节参数（29 关节，部署栈权威表）

| 关节 | 电机 | kp（N·m/rad）| kd | action_scale | default_angle |
|------|------|------------|-----|-------------|---------------|
| hip_pitch L/R | 7520_22 | 99.10 | 6.31 | 0.3506 | -0.312 |
| hip_roll L/R | 7520_22 | 99.10 | 6.31 | 0.3506 | 0.0 |
| hip_yaw L/R | 7520_14 | 40.18 | 2.56 | 0.5475 | 0.0 |
| knee L/R | 7520_22 | 99.10 | 6.31 | 0.3506 | 0.669 |
| ankle_pitch/roll L/R | 5020×2 | 28.50 | 1.81 | 0.4386 | -0.363 / 0.0 |
| waist_yaw | 7520_14 | 40.18 | 2.56 | 0.5475 | 0.0 |
| waist_roll/pitch | 5020×2 | 28.50 | 1.81 | 0.4386 | 0.0 |
| shoulder_pitch L/R | 5020 | 14.25 | 0.91 | 0.4386 | 0.2 |
| shoulder_roll L/R | 5020 | 14.25 | 0.91 | 0.4386 | +0.2 / -0.2 |
| shoulder_yaw L/R | 5020 | 14.25 | 0.91 | 0.4386 | 0.0 |
| elbow L/R | 5020 | 14.25 | 0.91 | 0.4386 | 0.6 |
| wrist_roll L/R | 5020 | 14.25 | 0.91 | 0.4386 | 0.0 |
| wrist_pitch/yaw L/R | 4010 | 16.78 | 1.07 | 0.0745 | 0.0 |

公式：stiffness = armature×ω²（ω=10Hz×2π，ζ=2）；action_scale = 0.25×effort/stiffness；target = action×scale + default。

## 二、交叉核对结果

| 对照源 | kp/kd | action_scale | default_angles | 力矩限幅 |
|--------|-------|--------------|----------------|---------|
| loader `_ISAAC_PD` | ✅ 一致（99.1/40.2/28.5/14.3/16.8；6.3/2.6/1.8/0.9/1.1）| — | — | — |
| 同事 E3 表 | ✅ 一致 | ✅ 一致（0.3506/0.5475/0.4386/0.0745）| ✅ 一致 | 见 Finding 2 |
| Isaac CFG | ✅ | ✅ | ✅ | 见 Finding 2 |

## 三、Findings（按 review 语义优先级：训练环境对齐目标 = Isaac CFG）

**Finding 1（已确认，任务 #17 修）**：XML hip_pitch/roll 限幅 ±88 是**旧电机**（7520_14）参数；部署栈注释明确"old is 7520_14 new is 7520_22"（effort 139）。修 XML 4 关节 → ±139。

**Finding 2（Phase D 呈现）**：踝关节 effort 三方表述：
- 部署栈 `EFFORT_LIMIT_5020 = 25`（仅用于 action_scale 计算）
- Isaac CFG / XML 力矩限幅：50
- 两者算出的 action_scale 一致（0.4386：deploy 用未乘刚度 14.25÷25，CFG 用乘后刚度 28.5÷50）
- 处置：env 对齐跟随 CFG（50 为力矩限幅、scale=0.4386 不变）；25 与 50 的限幅语义差异在 Phase D 部署兼容验证时呈现。

**Finding 3（无需行动）**：部署栈默认姿态与 E3 一致（shoulder_roll 左右 ±0.2），无冲突。

## 四、实验一实现的参数来源决议

实验一的 act_scale/act_offset 表以**本表为准**（部署栈提取值，已与 CFG/E3 三方核对一致）；力矩限幅用 CFG 值（139/88/50/25/5），与 Finding 2 的处置一致。
