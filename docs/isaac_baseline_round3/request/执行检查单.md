# R3 执行检查单（给录制侧）

> 配合 `physx-isaac-request-r3.md` 主协议使用。执行前后各过一遍。

## 执行前

- [ ] 读主协议 §采集协议（三处改动：free-root / PD 持姿 / 初始姿态）
- [ ] 确认驱动配置 = CFG 原始增益（kp=armature·ω², ω=2π·10Hz, ζ=2.0，**无任何 stiff 变体**）
- [ ] 确认控制 dt = 0.02（P1 教训：标注值与实际值必须一致）
- [ ] 确认记录通道：applied_torque / qpos / qvel / target / **root_pos 全序列** / **root_quat 全序列** / contact_force（可选）
- [ ] 确认四元数导出 **wxyz**
- [ ] 确认初始姿态：站立 ref、双脚并立、足底 z = 地面下 5mm（预穿透）

## 执行中

- [ ] 27 组试次：3 关节（hip_pitch / hip_roll / knee 单侧）× 阶跃 ±0.05/0.1/0.2 × 正弦 0.5/2/5Hz
- [ ] 失衡规则：root roll/pitch 超阈值或双脚离地 → 截断、数据保留、reset 重试一次、记录重试次数
- [ ] 200Hz 记录（0.005 子步）

## 执行后（回传）

- [ ] 打包：27 个 npz + npz 元数据盖章（quat=wxyz / control_dt=0.02 / drive=CFG 版本）
- [ ] 附一份 `protocol_notes.txt`：实际运行命令、驱动配置版本、任何与协议的偏差
- [ ] 文件名与 P1 同款：`p1_<joint>_<kind>_<param>.npz` 改为 `r3_<joint>_<kind>_<param>.npz`

## 对照方（我们侧）

`scripts/physx_r3_replica.py` = 协议的**可执行定义**（free-root、PD 持姿、200Hz、同扰动序列）。你们的数据到达后我们跑同协议，逐试次对比 q 响应（rise90/幅值/相位）与 applied_torque 波形。
