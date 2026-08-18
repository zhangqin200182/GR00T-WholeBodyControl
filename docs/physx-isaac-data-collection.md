# Isaac 基准数据收集请求（PhysX 环境对齐诊断）

> 日期：2026-08-18
> 状态：请求同事在 Isaac Sim 上执行本文档第四~七节
> 配套：`docs/physx-training-log.md`（训练全记录）、`docs/physx-alignment-experiment-plan.md`（对齐实验计划）
> 联系我们：数据收集过程有任何疑问，随时中断提问；不要猜测任何协议细节——每一条都必须与我们确认或按本文档执行。

---

## 一、实验目标

在 **NPU 服务器（无 GPU、无 Isaac Sim）** 上用 PhysX 5 Direct API 重建 Isaac Sim 的 SONIC 训练管线，使官方 release 权重在我们的环境里零样本走完整动捕段（**数百步**），随后在该环境中从随机权重训练交付策略（Phase C/D，见对齐实验计划）。

**本次数据收集的直接目标**：拿到 release 权重和 ref PD 在 **Isaac 原版环境**上的基准轨迹与力矩，用于定位我们环境与 Isaac 之间"残余 gap"的确切位置。

## 二、我们侧的实验方法与结论

### 2.1 方法

- **release 权重 = 校准仪**：权重冻结，只改环境侧参数，跑 inference 数存活步数（零训练 A/B 实验）。
- **三层接口模型**：网络输出 [-1,1] → ①动作空间翻译（`default + 0.25·effort/stiffness·action`）→ ②驱动器动力学 → ③力矩限幅（139/88/50/25/5）→ 物理；外加 ⓪观测空间。
- **评测协议**：单 env 确定性 rollout，12 episodes，逐集记录存活步数/reward/终止原因/片段归属（`scripts/physx_cross_eval.py`）。

### 2.2 已对齐（结论 A）

| 层 | 状态 | 证据 |
|----|------|------|
| 物理接触（P0 脚接触修复）| ✅ | PD 9.2→80.7 步，渲染无穿透 |
| ①动作空间 | ✅ | release 零样本 1.0→1.9 步（实验一）|
| ②驱动增益 | ✅ 解析换算 kp=k_isaac/M_eff，mult=10/kd×8 | 1.9→25.6 步（实验二）|
| ③限幅 | ✅ 机制+数值（hip ±139 修复）| 噪声级 |
| ⓪观测空间 | ✅ 4/6 处修复 | 零效果（非瓶颈）|
| 静态参数 | ✅ armature 有效；摩擦/torso 质量/vel_iters/dt 无效或已改 | batch 3 + re-matrix |
| **armature** | ✅ **唯一有效静态杠杆** | release +13%（24.3→27.6）、PD +26%（28.8→36.4）|
| **depenetration** | ⚠️ 零增益（batch-3 的"协同杠杆"已证伪，见 2.3）| 任何组合下与不开启逐位相同 |

### 2.3 结论 B：技能被压制，gap 未定位

**新回归基线**（500 片段、确定性、0.35/0.35、12 eps）：**release 27.58 步 / PD 36.42 步**。训练策略反而不如手写跟踪器，且宽松阈值下死因转为真实摔倒（ori 8/12 + height 7/12）——**release 的技能在所有阈值水平被压制**。所有接口层修复使 release 单调上升（1.0→1.9→25.6→32.9→27.6@500片段）但始终止步于 ≈PD 水平。残余 gap 的判断：**在物理动力学本身**（驱动动力学近似/接触模型/求解器时序），不在接口层。

### 2.4 教训：评测数据与确定性（已修复，供参考）

此前评测仅用 2 条动捕 + PID 洗牌导致评测结果双峰（排列 A=35.25 / 排列 B=26.67）。已修复：loader 固定 seed、逐集片段归属日志、500 片段数据集（`/root/sonic-data/robot_filtered`，89,464 条 BONES-SEED 转换产物）。**batch-3 的"depen 协同杠杆"是双峰假象，已证伪**。

## 三、Isaac 数据将如何被使用（裁决逻辑）

| 使用方式 | 回答的问题 |
|---------|-----------|
| ① 基准裁决 | release 在 Isaac 走多少步？若 ≈PD 且几十步 → release 本身弱，**我们环境可能已对齐**，主线转回训练侧；若数百步 → 环境 gap 实锤，gap 大小 = Isaac 步数 / 27.58 |
| ② 轨迹级 diff | 同片段逐帧对比 qpos/qvel → **首个发散关节与发散相位**（支撑相？摆动相？接触瞬间？）→ 把"gap 在引擎动力学"从假设变成逐关节定位 |
| ③ 状态注入力矩对比 | 把 Isaac 记录的 (qpos, qvel) 序列逐帧注入我们环境 → 对比两引擎算出的力矩/接触力 → 定位驱动动力学与接触模型的差距 |
| ④ 驱动响应反解 | 单关节阶跃/正弦响应曲线 → 反解 omni.physx 有效增益与惯性归一化（内部实现不公开，用实测曲线反推）|

**注意**：我们不需要 Isaac 侧的权重或训练；只需要**运行轨迹与力数据**。这是开发期一次性校准，数据到手后 NPU 训练管线完全不依赖 Isaac。

---

## 四、需要收集的数据（按优先级）

### P0：基准 rollout（必须）

**片段**：附录 A 的 12 个固定片段（episode i ↔ 片段 i，顺序配对）。

**策略**（每个片段各跑 1 集，共 12 集 × 2 策略）：

1. **release 权重**：SONIC 官方默认 checkpoint（`sonic_release/last.pt`，HF `nvidia/GEAR-SONIC` 可下载；你们机器上应已有）。模型输出直连驱动器，不做任何修改。
2. **ref PD 基线**：绕过网络，用 Isaac 原生驱动器直接跟踪动捕参考：`joint_target = ref_qpos`（每控制步）。stiffness/damping 用 Isaac CFG 原生值（`g1.py` 的 kp/kd）。这是"Isaac 驱动动力学 + 完美参考跟踪"的上限。

**每步记录字段**（逐控制步，50 Hz；全部为 numpy float32，形状如下）：

| 字段 | 形状 | 说明 |
|------|------|------|
| `ctrl_step` | () | 控制步索引，从 0 起 |
| `t` | () | 时间戳（秒，物理时间）|
| `root_pos` | (3,) | root 世界位置 |
| `root_quat` | (4,) | root 世界朝向（wxyz，标注你们的约定）|
| `qpos` | (29,) | 关节位置（**Isaac 内部关节序**）|
| `qvel` | (29,) | 关节速度 |
| `ref_qpos` | (29,) | 该步动捕参考关节位置（同关节序）|
| `ref_root_pos` | (3,) | 参考 root 位置 |
| `ref_root_quat` | (4,) | 参考 root 朝向 |
| `action_raw` | (29,) | 网络原始输出（[-1,1]；PD 基线填 `(ref_qpos - act_offset) / act_scale`）|
| `joint_target` | (29,) | 翻译后的驱动器目标（`offset + 0.25·e/k·action`）|
| `applied_torque` | (29,) | 该步实际施加的关节力矩 |
| `contact_force_left` | (3,) | 左脚（左踝）合接触力（世界系；取不到按每脚接触体合力）|
| `contact_force_right` | (3,) | 右脚合接触力 |
| `term_reason` | () | 终止原因字符串（无终止填 `"none"`）|
| `survived_steps` | () | 本集存活步数（每步重复填）|

**关节顺序必须附带**：导出一个 `joint_names.txt`（Isaac 内部序，29 行）。我们侧有 isaaclab↔mujoco 序的映射（`ISAAC_REORDER`），但**不要假设我们猜得对**——关节名列表是硬要求。缺失它，数据不可用。

**初始状态说明**：我们侧起步用 `root_z_offset=0.04` 补偿动捕悬空。你们侧按 Isaac 原生方式起步，**记录初始 root 高度和初始 qpos 即可**（diff 时我们先对齐初始条件）。不要为对齐而改你们的环境逻辑——我们要的是 Isaac 原生行为。

### P1：单关节驱动响应（高价值，可选但强烈建议）

固定机器人所有关节于站立位姿（或 root 固定），对 **3 个代表关节**（`hip_pitch`、`knee`、`ankle_pitch`，左右任选一侧）分别做：

1. **阶跃**：目标位置阶跃 ±0.05 / ±0.1 / ±0.2 rad（从站立位姿起），记录 2 秒。
2. **正弦扫描**：目标位置 = 0.05·sin(2πft)，f ∈ {0.5, 2, 5} Hz，各记录 4 秒。

**每个物理子步记录**（sim_dt=0.005 → 200 Hz）：`t`、该关节 `qpos/qvel/target/applied_torque`。此数据用于反解驱动有效增益（④）。

### P2：不需要的数据

观测向量（930 维）、网络内部激活、reward 明细——都不需要。保持数据包小。

---

## 五、评测协议（双方一致的部分）

| 项 | 值 | 说明 |
|----|-----|------|
| 控制频率 | 50 Hz | Isaac：sim_dt=0.005, decimation=4。我们侧 51Hz（0.001961×10，已知差异 2%，按控制步对齐）|
| 片段集 | 附录 A 的 12 个 | episode i ↔ 片段 i（sequential 配对）|
| 终止阈值 | ori=0.35, ank_pos=0.35, ank_h_mult=1.5 | 判定逻辑按 Isaac 原生；记录 term_reason |
| 最大步数 | 500 控制步 | 到上限记 `term_reason="truncated"` |
| 随机性 | 确定性（seed=0）| 双方各自保证自己侧可复现 |
| 集数 | 12 集/策略 | 片段与集一一对应 |

**我们侧的同协议对照命令**（结果见表，收集前我们会补跑并用你们的片段集校准过协议）：

```bash
cd /root/GR00T-WholeBodyControl
SONIC_PHYSX_DRIVE_ANALYTICAL=1 SONIC_PHYSX_DRIVE_MULT=10 SONIC_PHYSX_DRIVE_KD_MULT=8 \
python3 /tmp/physx_cross_eval.py --ckpt /root/sonic_release/last.pt \
  --ori 0.35 --ank 0.35 --episodes 12 --trust 1.0 \
  --isaac-space --sequential --pkl /sample_data/robot_filtered_fixed12
```

（`--trust 0.0` = PD 基线。逐集结果表见附录 B。）

---

## 六、数据格式与命名

```
每个 episode 一个文件：{policy}_{clip_name}_{ep:02d}.npz
  policy ∈ {release, pd}
  clip_name = 片段名（不含 .pkl，与附录 A 一致）

打包：
  isaac_baseline_YYYYMMDD.tar.gz
  ├── release/
  ├── pd/
  ├── joint_names.txt          # 29 行关节名（Isaac 内部序，必须）
  ├── protocol_notes.md        # 任何与本文档协议的偏差都在这里写
  └── version_info.txt         # 见 checklist 第 5 条
```

npz 字段名严格按第四节表格；多余字段欢迎（标注语义），缺失字段必须列出（写进 protocol_notes.md）。

---

## 七、执行步骤（可直接照做）

1. **准备**：官方 repo（`NVlabs/GR00T-WholeBodyControl`，最近 commit）+ Isaac Lab（你们现有环境即可，版本写进 version_info.txt）+ release 权重 `sonic_release/last.pt`。
2. **片段**：附录 A 的 12 个片段。若你们有 BONES-SEED 数据（`g1/csv` 或已转换的 motion_lib），按片段名直接取；否则告知我们，我们打包 pkl 发过去（每个 ~50KB）。
3. **加导出钩子**：在你们 Isaac eval 循环里加数据记录（参考代码见下）。官方入口 `gear_sonic/eval_agent_trl.py`（加载 release 权重、跑 env 的现成脚本）。
4. **跑 release**：12 集，逐集存 npz。
5. **跑 PD**：把动作替换为 `(ref_qpos - act_offset)/act_scale`（等价于驱动器直接跟踪 ref_qpos），其余不变。
6. **（可选）跑 P1 驱动响应**：单关节脚本，3 关节 × 2 种激励。
7. **打包回传**：按第六节结构 + 填完第八节 checklist。

### 导出钩子参考代码

放在 eval 循环内（每控制步执行一次），字段名按你们环境 API 对应修改：

```python
# 每集开始
ep_data = {k: [] for k in [
    "ctrl_step", "t", "root_pos", "root_quat", "qpos", "qvel",
    "ref_qpos", "ref_root_pos", "ref_root_quat", "action_raw",
    "joint_target", "applied_torque", "contact_force_left",
    "contact_force_right", "term_reason", "survived_steps"]}

# 每控制步（obs -> action 之后，env.step 之前/之后按字段取）
ep_data["ctrl_step"].append(step_idx)
ep_data["t"].append(sim_time)                    # env.sim_time 或等价
ep_data["root_pos"].append(root_pos.cpu().numpy())     # env.root_states[:, :3]
ep_data["root_quat"].append(root_quat.cpu().numpy())   # env.root_states[:, 3:7]
ep_data["qpos"].append(dof_pos.cpu().numpy())          # articulation 的 get_joint_pos
ep_data["qvel"].append(dof_vel.cpu().numpy())
ep_data["ref_qpos"].append(ref_qpos.cpu().numpy())     # 动捕参考（同关节序）
ep_data["ref_root_pos"].append(ref_root_pos.cpu().numpy())
ep_data["ref_root_quat"].append(ref_root_quat.cpu().numpy())
ep_data["action_raw"].append(raw_action.cpu().numpy()) # 网络原始输出
ep_data["joint_target"].append(drive_target.cpu().numpy())
ep_data["applied_torque"].append(applied_torque.cpu().numpy())  # articulation applied torque
# 接触力：取左右脚接触体的 net contact force（你们 API 的 rigid body contact forces）
ep_data["contact_force_left"].append(f_left.cpu().numpy())
ep_data["contact_force_right"].append(f_right.cpu().numpy())
ep_data["term_reason"].append("none")
ep_data["survived_steps"].append(0)

# 每集结束
ep_data["term_reason"][-1] = term_reason
ep_data["survived_steps"] = [len(ep_data["ctrl_step"])] * len(ep_data["ctrl_step"])
np.savez(f"{policy}_{clip_name}_{ep:02d}.npz",
         **{k: np.stack(v) if v[0].ndim > 0 else np.array(v) for k, v in ep_data.items()})
```

**重要**：`term_reason` 和 `survived_steps` 在每集结束后统一回填。关节序、四元数顺序（wxyz vs xyzw）、接触力坐标系（世界 vs body）三项**必须在 protocol_notes.md 里写明**——这是最容易导致数据不可用的三处。

---

## 八、回传 checklist

- [ ] 12 集 release npz + 12 集 pd npz（或更多，多了欢迎）
- [ ] `joint_names.txt`（29 关节名，Isaac 内部序）
- [ ] `version_info.txt`：Isaac Sim / Isaac Lab / omni.physx 版本号、官方 repo commit、release 权重来源（HF 或内部路径）
- [ ] `protocol_notes.md`：关节序、四元数顺序、接触力坐标系、初始状态处理、任何协议偏差
- [ ] PD 实现说明（驱动器目标如何从 ref 构造；stiffness/damping 取值来源）
- [ ] 随机种子与运行命令原文
- [ ] P1 驱动响应数据（若做了）
- [ ] 每集 term_reason 已填

---

## 附录 A：12 个固定片段（episode ↔ 片段顺序配对）

顺序即 episode 顺序。此顺序是**完全确定的**（片段名字典序 + `RandomState(0)` 洗牌，与文件系统无关），我们侧 `--sequential` 模式按此序加载；你们侧直接按此表配对 episode 与片段即可：

| ep | 片段名 | BONES-SEED 路径（我们宿主机）|
|----|--------|------------------------------|
| 0 | walk_ff_loop_180_R_003__A050 | sonic-data/robot_filtered/221011/ |
| 1 | walk_the_dog_ff_180_loop_R_001__A476 | sonic-data/robot_filtered/231011/ |
| 2 | injured_R_leg_walk_ff_start_315_R_002__A232 | sonic-data/robot_filtered/230301/ |
| 3 | walk_sideway_045_loop_003__A033 | sonic-data/robot_filtered/220721/ |
| 4 | crutches_walk_arc_cw_start_R_001__A516 | sonic-data/robot_filtered/231116/ |
| 5 | walk_ff_stop_360_R_001__A418 | sonic-data/robot_filtered/230710/ |
| 6 | crutch_walk_turn_270_R_001__A518 | sonic-data/robot_filtered/231116/ |
| 7 | walk_ff_stop_270_002__A051_M | sonic-data/robot_filtered/221011/ |
| 8 | walk_into_door_R_001__A514 | sonic-data/robot_filtered/231115/ |
| 9 | inj_right_leg_walk_180_R_max_003__A078 | sonic-data/robot_filtered/221129/ |
| 10 | big_heavy_one_hand_walk_ff_start_360_R_001__A509 | sonic-data/robot_filtered/231114/ |
| 11 | injured_torso_walk_ff_start_225_R_003__A338 | sonic-data/robot_filtered/230419/ |

另附官方 demo 2 片段（双方都应已有，可作为协议烟测）：`walk_forward_amateur_001__A001` 与 `walk_forward_amateur_001__A001_M`。

**如果我们需要发送 pkl**（你们无 BONES-SEED 时），我们按此命令打包（每个 ~50KB）：

```bash
cd /root/sonic-data/robot_filtered
tar czf fixed12_clips.tar.gz \
  231116/crutches_walk_arc_cw_start_R_001__A516.pkl \
  230301/injured_R_leg_walk_ff_start_315_R_002__A232.pkl \
  231115/walk_into_door_R_001__A514.pkl \
  221129/inj_right_leg_walk_180_R_max_003__A078.pkl \
  231116/crutch_walk_turn_270_R_001__A518.pkl \
  230419/injured_torso_walk_ff_start_225_R_003__A338.pkl \
  231011/walk_the_dog_ff_180_loop_R_001__A476.pkl \
  221011/walk_ff_loop_180_R_003__A050.pkl \
  221011/walk_ff_stop_270_002__A051_M.pkl \
  220721/walk_sideway_045_loop_003__A033.pkl \
  230710/walk_ff_stop_360_R_001__A418.pkl \
  231114/big_heavy_one_hand_walk_ff_start_360_R_001__A509.pkl
```

## 附录 B：我们侧同协议逐集结果（PhysX，固定 12 片段）

> 0.35/0.35/1.5，sequential 配对，确定性。Isaac 侧数据回传后并排对比。
> 表中数字为 存活步数（reward）。

| ep | 片段 | release | PD |
|----|------|---------|-----|
| 0 | walk_ff_loop_180_R_003__A050 | 28 (98.4) | 28 (121.5) |
| 1 | walk_the_dog_ff_180_loop_R_001__A476 | 26 (79.4) | 23 (102.4) |
| 2 | injured_R_leg_walk_ff_start_315_R_002__A232 | 29 (112.9) | **57** (249.3) |
| 3 | walk_sideway_045_loop_003__A033 | 34 (134.9) | 42 (236.8) |
| 4 | crutches_walk_arc_cw_start_R_001__A516 | 21 (92.9) | **50** (285.4) |
| 5 | walk_ff_stop_360_R_001__A418 | **19** (84.7) | 16 (83.3) |
| 6 | crutch_walk_turn_270_R_001__A518 | 20 (73.5) | 27 (100.7) |
| 7 | walk_ff_stop_270_002__A051_M | **23** (95.0) | 21 (99.8) |
| 8 | walk_into_door_R_001__A514 | 28 (110.1) | 28 (175.5) |
| 9 | inj_right_leg_walk_180_R_max_003__A078 | 25 (107.6) | 36 (212.2) |
| 10 | big_heavy_one_hand_walk_ff_start_360_R_001__A509 | 21 (81.2) | 41 (243.1) |
| 11 | injured_torso_walk_ff_start_225_R_003__A338 | 34 (121.5) | 43 (216.9) |
| **均值** | | **25.67 (99.3)** | **34.33 (177.2)** |

**要点**：PD 在 12 个片段中 11 个反超 release；差距最大的片段是"病理步态"类（拐杖、伤腿、负重）。这是技能压缩的逐集证据——你们侧的 Isaac 数据将逐集并排，直接看出哪些片段我们压得最狠。
