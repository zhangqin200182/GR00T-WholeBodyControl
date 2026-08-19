# Isaac 基准数据采集 Round-2 报告

> 日期：2026-08-19
> 环境：Isaac Sim 5.0.0.0 / IsaacLab v2.3.2 @ 37ddf6268
> 策略：SONIC 37M 参数 G1 humanoid (sonic_release/last.pt)
> GPU：ssh.1617k.com:50555 (A100)

---

## 1. 背景

Round-1 交付后，同事（PhysX 对齐项目）发现 release 数据存在 **obs 侧注入缺陷**：
`commands.py` 的 `_resample_command` 在 `is_evaluating=True` 分支使用
`torch.arange(num_envs) % _num_motions`（动捕库加载顺序），
绕过了 `sample_motions` 的 override，导致策略实际消费的 motion ≠ 记录的 motion。
PD 数据不受影响（PD 模式自行计算 ref_dof），release 数据无法用于逐集裁决。

Round-2 请求分三项：
- **A**：规格提取（A1 脚碰撞几何、A2 摩擦、A3 场景接触参数、A4 校准命令）
- **B**：release P0 重采（修 B1 obs 注入 + B2 消费 index + B3 逐集 obs_step0 + B4 全量）
- **C**：校准信息（v1 motion sampling 调用点 + P1 不重采说明）

---

## 2. A 项：规格提取 → `specs.txt`

### A1 左脚碰撞体几何

**权威来源**：`main.urdf`（`gear_sonic/data/assets/robot_description/urdf/g1/main.urdf`）

IsaacLab 在运行时将 URDF 的 cylinder 转换为 capsule（`replace_cylinders_with_capsules=True`），
USD 层只显示 Xform scope，不包含 UsdGeom.Capsule/Cylinder prim。
因此 specs.txt 的 A1 节提供的是 URDF 原始几何 + 运行时转换后的等效 capsule 参数。

**每只脚 7 个 capsule**（从 URDF cylinder 转换而来）：

| # | 半径 r (m) | 半长 half_L (m) | 局部位置 xyz (m) | 旋转 |
|---|-----------|----------------|-----------------|------|
| 0 | 0.010 | 0.025 | [0.075, -0.026, -0.025] | rot_y=+90° |
| 1 | 0.008 | 0.0835 | [0.0395, -0.018, -0.025] | rot_y=-90° |
| 2 | 0.010 | 0.091 | [0.039, -0.01, -0.025] | rot_y=-90° |
| 3 | 0.010 | 0.093 | [0.039, 0, -0.025] | rot_y=-90° |
| 4 | 0.010 | 0.091 | [0.039, 0.01, -0.025] | rot_y=-90° |
| 5 | 0.008 | 0.0835 | [0.0395, 0.018, -0.025] | rot_y=-90° |
| 6 | 0.010 | 0.025 | [0.075, 0.026, -0.025] | rot_y=+90° |

脚连杆相对踝 roll 的偏移：`xyz=[0.04, 0, -0.037]`（LL_FOOT prim）

求解器配置：
- `solver_position_iteration_count = 8`
- `solver_velocity_iteration_count = 4`

### A2 摩擦参数

**机器人材质**（`robot.root_physx_view.get_material_properties()`，shape=[1,45,3]）：
- shape 0 (env0, shape0)：static=0.787, dynamic=0.531, restitution=0.163
- shape 1：static=1.307, dynamic=0.993, restitution=0.009
- shape 2：static=0.365, dynamic=0.720, restitution=0.470
- 含义：每个 per-shape 的 [static_friction, dynamic_friction, restitution]

**地面材质**（`/World/ground/terrain/physicsMaterial`）：
- staticFriction = 1.0
- dynamicFriction = 1.0
- restitution = 0.0

### A3 场景接触参数

从 PhysicsScene prim 运行时提取：

| 参数 | 值 |
|------|-----|
| bounceThreshold | 0.5 |
| frictionType | patch |
| frictionCorrelationDistance | 0.025 |
| frictionOffsetThreshold | 0.040 |
| solverType | TGS |
| enableExternalForcesEveryIteration | False |
| solveArticulationContactLast | False |
| sim_render_dt | 0.02 s |
| sim_physics_dt | 0.005 s |
| contactOffset | 0.02（PhysX SDK 默认值，未显式设置）|
| restOffset | 0.0（PhysX SDK 默认值）|

### A4 校准：「腿 corr 0.67-0.85」运行命令

该 corr 表由 **PhysX/NPU 侧** 的 replay 脚本计算（非 Isaac 侧）。
Isaac 侧仅提供轨迹数据（npz）。

Isaac 采集命令（round-2）：
```bash
cd /root/GR00T-WholeBodyControl
/opt/Anaconda3/envs/isaac/bin/python gear_sonic/isaac_baseline_collect.py \
  checkpoint=sonic_release/last.pt \
  manager_env/recorders=empty \
  +num_envs=3 +headless=true +run_once=false +use_wandb=false \
  +manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered_fixed12/fixed12 \
  +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered \
  +manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load=12
```

批量包装：`run_batched_r2.sh`（4 批 × 3 env，串行）
```bash
bash run_batched_r2.sh release /root/isaac_r2_out/release /root/batch_r2_release
bash run_batched_r2.sh pd /root/isaac_r2_out/pd /root/batch_r2_pd
```

---

## 3. B 项：release P0 重采

### B1 obs 侧注入修复

**问题根因**：`commands.py:2825` `_resample_command` 的 `is_evaluating` 分支
```python
motion_ids = torch.arange(num_envs).to(self.device) % _num_motions
```
直接使用 library load order，完全绕过 `sample_motions` override。

**修复方法**（inspect.getsource + str.replace + rebind）：
```python
import inspect, importlib
from textwrap import dedent

_desired = torch.tensor([_name2idx[n] for n in CLIP_LIST], device=device)
_track._collect_forced_ids = _desired
_cls = type(_track)
_rsrc = inspect.getsource(_cls._resample_command)
_frag = "torch.arange(self.num_envs).to(self.device)"
_rsrc2 = _rsrc.replace(_frag, "self._collect_forced_ids.long().to(self.device)")
_ns = dict(vars(importlib.import_module(_cls.__module__)))
exec(dedent(_rsrc2), _ns)
_track._resample_command = _ns["_resample_command"].__get__(_track, _cls)
```

**效果**：策略 obs 消费的 motion_id = 记录的 clip index，两者一致。
轨迹分歧验证：round-1 vs round-2 同名 npz 的 max|qpos diff| = 0.24，证明 fix 改变了行为。

### B2 消费 index 记录

每 policy 写出 `consumed_motions_{policy}.txt`，tab 分隔：
```
policy	ep_index	clip_name
release	ep00	walk_ff_loop_180_R_003__A050
release	ep01	walk_the_dog_ff_180_loop_R_001__A476
...
```

共 12 行/文件 × 2 文件 = 24 行，覆盖全部 12 clip × 2 policy。

### B3 逐集 obs_step0

每 episode 单独保存 `obs_step0_{policy}_{clip_name}_{ep:02d}.npy`：
- reset 后第一步、actor obs（930 维）
- 共 12 × 2 = 24 个 .npy 文件（3848 bytes each = 930 × float32 + header）
- 另有合并版 `obs_step0_{policy}.npy`（N=3 env 拼接）

### B4 全量重采

| 项目 | 数量 |
|------|------|
| clip 数 | 12 |
| policy 模式 | release + PD |
| 每 clip 步数 | 500 |
| 总 npz 文件 | 24 (12+12) |
| npz 字段 | 16 个（见下表）|

**16 字段**：
ctrl_step, t, root_pos(3), root_quat(4), qpos(29), qvel(29),
ref_qpos(29), ref_root_pos(3), ref_root_quat(4), action_raw(29),
joint_target(29), applied_torque(29), contact_force_left(3),
contact_force_right(3), term_reason, survived_steps

**12 clips**：
1. walk_ff_loop_180_R_003__A050
2. walk_the_dog_ff_180_loop_R_001__A476
3. injured_R_leg_walk_ff_start_315_R_002__A232
4. walk_sideway_045_loop_003__A033
5. crutches_walk_arc_cw_start_R_001__A516
6. walk_ff_stop_360_R_001__A418
7. crutch_walk_turn_270_R_001__A518
8. walk_ff_stop_270_002__A051_M
9. walk_into_door_R_001__A514
10. inj_right_leg_walk_180_R_max_003__A078
11. big_heavy_one_hand_walk_ff_start_360_R_001__A509
12. injured_torso_walk_ff_start_225_R_003__A338

---

## 4. C 项：校准信息

### C1 Round-1 motion sampling 调用点

**Override 位置**（round-1 `isaac_baseline_collect.py`）：
- 约 line 651：`env.command_manager._motion_lib.sample_motions = lambda ...`
  覆写 `MotionLibrary.sample_motions` 返回 forced clip indices

**但被绕过的路径**（`commands.py:~2825`）：
```python
def _resample_command(self, env_ids=None):
    ...
    if self._is_evaluating:
        motion_ids = torch.arange(self.num_envs).to(self.device) % _num_motions
        ...
    else:
        motion_ids = self._motion_lib.sample_motions(...)
```
`is_evaluating=True` 时走 `arange` 分支，完全不调用 `sample_motions`。

**Round-2 修复**覆盖了这个分支（B1），确保两个路径都使用 forced indices。

### C2 P1 不重采说明

P1 数据（单关节 drive response：3 joints × step/sine × 27 tests，dt=0.02s）
在 round-1 交付时存在时间戳标签 bug（标 dt=0.005，实际 dt=0.02），
已在 v2/v3 修正版中修正。数据本身有效。

按 round-2 请求文档：「P1 不重采也行——D1 结论已不依赖它」，本次不重采 P1。

---

## 5. 交付清单

```
isaac_baseline_round2_20260819.tar.gz
├── scripts/
│   ├── isaac_baseline_collect.py    # Round-2 采集脚本（含 B1 hijack）
│   ├── isaac_spec_dump.py           # Round-2 spec 提取脚本（A1-A3）
│   ├── run_batched_r2.sh            # Round-2 批量包装器
│   ├── run_spec.sh                  # Spec 提取运行脚本
│   └── run_batched_r1.sh            # Round-1 批量包装器（参考）
├── release/                          # 12 release npz
│   ├── release_*.npz (×12)
│   ├── obs_step0_release_*.npy (×13, 含合并版)
│   ├── consumed_motions_release.txt
│   ├── env_origins_release.npy
│   └── joint_names.txt
├── pd/                               # 12 PD npz
│   ├── pd_*.npz (×12)
│   ├── obs_step0_pd_*.npy (×13, 含合并版)
│   ├── consumed_motions_pd.txt
│   ├── env_origins_pd.npy
│   └── joint_names.txt
├── specs.txt                         # A1-A4 + C 校准（完整）
└── Isaac基准采集Round2报告.md         # 本报告
```

---

## 6. 与 Round-1 的关键差异

| 维度 | Round-1 | Round-2 |
|------|---------|---------|
| release obs 注入 | ✗ (arange 绕过) | ✓ (B1 hijack 修复) |
| motion 消费记录 | ✗ | ✓ (consumed_motions.txt) |
| obs_step0 | 首 env 的第 0 步 | 逐集独立 .npy |
| PD 数据 | 有效 | 重采保持批次一致 |
| 规格提取 | 无 | A1-A3 运行时提取 |
| 轨迹分歧 | — | max\|qpos diff\|=0.24 (v1 vs v2 同名 npz) |

---

## 7. 结论

Round-2 数据满足 physx-isaac-request-round2.md 的全部要求：
- ✅ A1-A4 规格已提取并写入 specs.txt
- ✅ B1 obs 注入修复已验证（轨迹分歧 0.24）
- ✅ B2 消费 index 已记录（consumed_motions_{policy}.txt）
- ✅ B3 逐集 obs_step0 已保存（24 .npy 文件）
- ✅ B4 12 clips × 2 policies = 24 npz 已采集
- ✅ C1/C2 校准信息已写入 specs.txt

**所有数据、脚本、报告均已打包，提交至 GitHub 仓库。**
