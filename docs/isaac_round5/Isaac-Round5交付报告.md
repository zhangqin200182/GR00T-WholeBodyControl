# Isaac Round-5 交付报告（obs 修复重采 + 接触探针 + R4 收口）

> 日期：2026-08-23 凌晨
> 环境：GPU 519 等价复原环境（isaacsim 5.0.0.0 / IsaacLab v2.3.2@37ddf6268 + urdf importer 2.4.19 + hasattr patch）

---

## A. obs 修复重采（P0，关键路径）—— ✅ 完成，验收 PASS

### 重要澄清：数据源问题

同事诊断的"12 npz 只有 3 条唯一轨迹"**对象是 replay kit 里附带的 npz**
（ep 号 00-11 全局编号那套，经 verify 实测确认 3 组内 action/qpos 逐位相同）。
**round-2 交付的重采数据（isaac_baseline_round2_20260819）当时已修复该 bug**
（B1 hijack `_resample_command` 的 eval 分支），verify 对其跑出 PASS（12 条两两不同）。
即 obs 侧注入修复在 round-2 已生效；本轮重采主要是补齐新字段。

### 本轮重采结果

- **release 12 npz + PD 12 npz**（同 round-2 协议：500 步 × 16 字段 × 12 clips）
- **验收自检 PASS**：`verify_recollection.py` → `RESULT: PASS -- safe to send`
  （12 文件 action_raw 两两 maxdiff > 0，qpos 两两不同）
- **新增 4 字段**（每 npz）：
  - `consumed_motion_id`（int，策略 obs 实际消费的 motion_lib index）
  - `consumed_motion_name`（str）
  - `ref_clip_name`（str，记录侧 ref clip）
  - `friction_sample_all`（45×3，episode 起始时 root_physx_view 读出的
    material properties 全表 [static, dynamic, restitution]）
- sidecar 保留：consumed_motions_{policy}.txt、obs_step0 逐集 npy、env_origins

**friction 说明**：本采集脚本中立化了全部随机化 event（round-2 起如此），
故 friction_sample_all 为默认材质表（row0 = 0.7873/0.5308/0.1625，
与 round-2 specs.txt A2 一致），非随机采样值。若同事需要随机化开启的版本再议。

---

## B. 接触合规 settle 探针 —— 两版数据 + 重要物理发现

### 协议字面版（自由 root，同事原脚本）→ 7 档全部倾倒

`settle_probe/`：offset -5~+20mm 各档 100 控制步，**全部在 ~2s 内倾倒**
（root_z → 0.086-0.090，尾段力为倒地后大腿/踝压地，非站立态）。
根因（R3 已证）：**官方软踝增益（feet kp=28.5 N·m/rad）下，纯 PD 持姿的
自由 root 站立在 2s 内失稳**——这是官方资产的固有物理，不是 bug。

### 修正版（pin root，`settle_pinned/`）→ 干净的穿透-高度对应 + 新发现

root 每控制步钉在"默认出生高度 + offset"（速度清零），穿透由高度唯一决定：

| 档 | root_z | ankle_roll_link z | 双足力 |
|---|---|---|---|
| -5mm | 0.7527 | 0.0370 | **0 N**（腿被压弯让位，q-q0≈0.09，tau 9.4 N·m）|
| +0mm | 0.7575 | 0.0358 | 0 N |
| +2~+20mm | 0.7595→0.7757 | 0.0378→0.0578 | 0 N |

**关键发现：官方默认出生高度（root z=0.76）下，G1 站姿 capsule 脚底距地面
约 5-6mm 悬空**（ankle_roll 中心 0.0358，脚底≈0.0058）——即 offset=0 并非
"贴地"，offset 需到约 **-6mm 以下才开始接触**。同事侧"支撑帧穿透 ~0mm
（-0.1~-3.1mm）"的对比请注意此参照系差异。
pin 版数据（含每档双足 4 body 位姿全序列 + 力全序列）在 settle_pinned/*.npz，
同事可用真胶囊表面公式本地算最低点重构穿透-力曲线的接触段。

### actuator 运行时表（组均值，同事要的收口数据）

| 组 | joints | stiffness | damping | armature | effort_limit |
|---|---|---|---|---|---|
| legs | 8 | 84.369 | 5.371 | 0.021371 | 126.25 |
| feet | 4 | 28.501 | 1.814 | 0.007219 | 50.0 |
| waist | 2 | 28.501 | 1.814 | 0.007219 | 50.0 |
| waist_yaw | 1 | 40.179 | 2.558 | 0.010178 | 88.0 |
| arms | 14 | 14.973 | 0.953 | 0.003793 | 19.29 |

注：armature 为组内关节均值（组内混合档位：legs=hip_pitch/roll 0.02510 与
hip_yaw 0.01018 的平均；arms=5020/4010 混合）。单关节精确值仍以 g1.py 表为准
（0.00361/0.00425/0.01018/0.02510）。运行时 stiffness/damping 与公式
kp=armature·ω²、kd=2ζ·armature·ω（ω=2π·10，ζ=2）在均值口径下吻合。

---

## C. R4 遗留收口 —— 3 答 1 跳过

1. **self-collision 力（C1）**：现有 P0 npz 无 per-body 通道（仅脚聚合）。
   补做了 pinned-squat 探针（root 钉住 + 深屈姿态：hip -0.75 / knee 1.0 /
   ankle -0.5，root -100mm）：**膝/踝 body 净接触力 = 0**（无自碰撞），
   但 **躯干-大腿（waist_roll_link）182 N、臂-腿（肩/肘/腕 ~10 N）有真实
   接触力**——深屈域 self-collision 主要在躯干-大腿与臂-腿，不在腿间。
2. **每脚 manifold 点数（C2）**：此 IsaacLab 版本的 RigidContactView 无
   get_contact_count API（实测 AttributeError），USD 层接触不可见（已知）。
   按文档许可跳过；你们侧 56 点基线仅作参考。
3. **contact_force_left/right 传感器语义（C3）**：**no**——源码注释明确：
   `net_forces_w` = "sum of the **normal** contact forces"（纯法向接触力，
   不含关节反力投影，也不含切向摩擦分量）。P0 npz 的 contact_force_*
   即此语义。
4. **运行时 getRestOffset/getContactOffset 回读（C4）**：跳过（USD 遍历
   不可见已证；config 事实同事已有）。

---

## 兼容性 patch 清单（本轮 settle 探针相关，均已验证）

1. `robot.num_dof` → `len(robot.joint_names)`（API 不存在）
2. per-body 接触力：ContactSensor 直连构造（MySceneCfg 不认 sensors 字段），
   prim 挂 `/World/envs/env_0/Robot/.*`，`_initialize_impl()` 手动初始化
   （错过 play 事件订阅）
3. **每控制步必须 `set_joint_position_target` + `write_data_to_sim`**
   （只写一次 → drive 不持续出力，tau=0、机器人倒地——已实证）
4. 每档不 `sim.reset()`（stop/play 丢 drive 状态）+ 收尾 simulation_app
   NameError 修复

## 交付物

```
isaac_r5_delivery/
├── data/release/         12 npz + sidecar（verify PASS，4 新字段）
├── data/pd/              12 npz + sidecar
├── settle_probe/         协议字面版 7 档（含倾倒过程，注明）
├── settle_pinned/        pin 版 9 npz + pinned_summary.txt（actuator 表）
├── scripts/              collect（R5 字段版）+ settle 两版 + c_probe + verify
└── logs/                 settle/release/pd 全部终端输出
```
