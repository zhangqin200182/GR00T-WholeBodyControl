# PhysX 跟踪质量提升 — Task 拆解

> 基于 `docs/physx-next-steps.md` | 2026-08-03

## 依赖关系

```
T1 (per-joint kp) 0.25d ────────────────────┐
                                              │
T2 (reward FK 统一) 0.25d ───────────────────┤
                                              ├─ T4 (ref PD 验证) 0.5d
T3 (joint frame 修复) 1d ────────────────────┘
                                              │
T5 (dt/decimation 对齐) 0.25d ───────────────┘

T6 (手动 PD fallback) 1d — 仅在 T1-5 不够时启用
```

**T1-T3 互相独立，可并行执行。** T4 是它们的验证节点。

---

## T1：Per-joint PD gains 对齐 Isaac Sim

**目标**：将 PD gains 从全 100/5 改为 Isaac Sim 的 per-joint 配置，减少上肢过驱引起的耦合振动。

**前置**：无（纯 Python 改动）

**步骤**：

1. 在 `physx_loader.py` 的 `_parse_joint` 中，根据 joint name 查表设置 kp/kd：

```python
ISAAC_PD = {
    "hip_pitch": (99.1, 6.3), "hip_roll": (99.1, 6.3), "knee": (99.1, 6.3),
    "hip_yaw": (40.2, 2.6),
    "ankle_pitch": (28.5, 1.8), "ankle_roll": (28.5, 1.8),
    "waist_roll": (28.5, 1.8), "waist_pitch": (28.5, 1.8), "waist_yaw": (40.2, 2.6),
    "shoulder_pitch": (14.3, 0.9), "shoulder_roll": (14.3, 0.9), "shoulder_yaw": (14.3, 0.9),
    "elbow": (14.3, 0.9),
    "wrist_roll": (14.3, 0.9),
    "wrist_pitch": (16.8, 1.1), "wrist_yaw": (16.8, 1.1),
}
for pattern, (kp, kd) in ISAAC_PD.items():
    if pattern in jname:
        break
else:
    kp, kd = 100.0, 5.0  # default
```

2. 上传 loader.py，重建 articulation 后跑 20-episode ref PD

**验收**：α 从 0.053 下降到 < 0.04

**估时**：0.25 天

---

## T2：全局替换 get_link_world_pose → Python FK

**目标**：将所有 `get_link_world_pose` 的 body 位置查询统一为 Python FK，消除 PhysX FK 和 MJCF FK 不一致导致的所有计算偏差。

**前置**：无（纯 Python 改动）

**影响范围**（5 处）：

| 位置 | 当前 FK 源 | 修改 |
|---|---|---|
| `_compute_reward()` r3/r4/r5/r6/r11 | PhysX get_link_world_pose | Python FK |
| `reset()` `_prev_body_pos` / `_prev_body_quat` | PhysX get_link_world_pose | Python FK |
| `_anti_shake()` `curr_quat` | PhysX get_link_world_pose | Python FK |
| `_compute_critic_obs()` body_pos_w / body_quat_w | PhysX get_link_world_pose | Python FK |
| `_check_termination()` | 已改为 Python FK ✅ | — |

**步骤**：

1. 在类里加缓存方法（避免每步多次调 FK）：

```python
def _get_body_state_fk(self):
    """Compute body pos/quat via Python FK (cached per step)."""
    if getattr(self, '_fk_cache_step', -1) == self.ep:
        return self._fk_body_pos, self._fk_body_quat
    root_pos, root_quat = self.art.get_root_world_pose()
    actual_qpos = self.art.get_joint_positions()
    tracked = self._fk.get_tracked_poses(root_pos, root_quat, actual_qpos)
    self._fk_body_pos = np.array([t[0] for t in tracked], dtype=np.float64)
    self._fk_body_quat = np.array([t[1] for t in tracked], dtype=np.float64)
    self._fk_cache_step = self.ep
    return self._fk_body_pos, self._fk_body_quat
```

2. 替换 4 处（termination 已改）：
   - `reset()`：`self.art.get_link_world_pose(i)[0]` → `self._get_body_state_fk()[0][i]`
   - `_compute_reward()`：同上
   - `_compute_critic_obs()`：同上
   - `_anti_shake()`：`self.art.get_link_world_pose(idx)[1]` → `self._get_body_state_fk()[1][list(BODY_NAMES).index(name)]`

**产出**：`physx_env.py` 中不再有 `get_link_world_pose` 调用（除 root pose 外）

**验收**：body pos/quat 在所有函数中使用同一 FK 源；20-ep 中 reward 无 NaN

**估时**：0.25 天

---

## T3：Joint frame 修复 — setParentPose / setChildPose

**目标**：在 PhysX articulation 中显式设置每个 joint 的 parent frame 和 child frame，使 joint anchor 和 MJCF 定义一致。**这是最可能解决 FK 不一致和 α 过大的修复。**

**前置**：无（C++ + Python 改动，但和 T1/T2 独立）

**步骤**：

### 3.1 理解 MJCF joint frame

MJCF 中的 body 结构：
- `<body pos="..." quat="...">` 定义了 child body 在 parent body frame 中的位置
- `<joint pos="..." axis="...">` 定义 joint 在 child body frame 中的位置（默认 `pos="0 0 0"`）
- G1 所有 joint 的 pos 都是 `0 0 0`，即 joint 在 child body 的 origin 处

在 PhysX 中：
- `parentPose`：joint anchor 在 parent body frame 中的位置
- `childPose`：joint anchor 在 child body frame 中的位置

对应关系（G1 所有 joint pos=0 的情况）：
- `parentPose = PxTransform(child_body_local_pos, child_body_local_quat)` — child body origin 在 parent frame 中
- `childPose = PxTransform(PxIdentity)` — joint 在 child body origin 处

### 3.2 C++ 改动

**结构修正 — 变量作用域**：当前 `local` 变量（pos/quat）在 createLink 循环（loop 1）中计算，joint 配置循环（loop 2）中访问不到。两个方案：

- **方案 A（推荐）**：在 loop 1 中把 local transforms 存到 `std::vector<PxTransform> link_locals`，loop 2 中取 `link_locals[child]`
- **方案 B**：合并两个循环——loop 内先 createLink 再配 joint，用 `(i>0)` 判断非 root

推荐方案 A，改动最小。

**parentFrame quat 修正**：

```cpp
// ❌ 错误：丢了 local_quat（大部分 body 是 identity 所以不明显，但少数有旋转）
PxTransform parentFrame(local_pos, PxQuat(PxIdentity));

// ✅ 正确：包含 body 的完整 6-DOF 变换
PxTransform parentFrame(local_pos, local_quat);
```

G1 大部分 body 的 MJCF quat 是 identity，但 `left_hip_roll_link`、`left_knee_link` 等有非零 quat。用 identity 会导致这些 joint 的旋转轴方向错误。

**完整代码**（插入到 loop 2 中，`setJointType` 之后、`setMotion` 之前）：

```cpp
// Set joint frames explicitly — matches MJCF body transform
PxTransform local_joint = link_locals[child];
PxTransform parentFrame(local_joint.p, local_joint.q);
PxTransform childFrame(PxTransform(PxIdentity));
joint->setParentPose(parentFrame);
joint->setChildPose(childFrame);
```

### 3.3 API 前置验证 — setParentPose 坐标系

**问题**：PhysX `setParentPose` 是相对于 actor frame（body origin）还是 CoM frame？`createLink` 内部用 `getCMassLocalPose().transformInv()` 是 CoM frame，但 `setParentPose` 的文档需要确认。

**验证步骤**（T3 开工前必须完成）：

1. 建一个简单 2-link chain：parent link 带 CoM 偏移 (0,0,0.1)，child link 的 local_pos = (0,0,-0.5)
2. 调 `setParentPose(PxTransform(0,0,-0.5))` 和 `setChildPose(PxTransform(PxIdentity))`
3. 跑 `get_link_world_pose(1)` 看 child 的位置是否在 parent.z - 0.5 处
4. 如果在 parent.z - 0.4（差了 CoM 偏移）→ setParentPose 用 CoM frame → 需要用 CoM 偏移调整
5. 如果在 parent.z - 0.5（和 local pos 一致）→ setParentPose 用 actor frame → 直接传 local_quat 即可

**预期结果**：`setParentPose` 文档（PxArticulationJointReducedCoordinate.h:70-73）："Sets the joint pose in the parent link actor frame." — 明确是 actor frame，不是 CoM frame。所以直接传 `PxTransform(local_pos, local_quat)` 即可，不需要 CoM 调整。

---

## T4：Ref PD 验证 — 集成测试

**目标**：在 T1/T2/T3 的修改基础上，跑 20-episode ref PD，评估 α 和 survival。

**前置**：T1, T2, T3 中至少完成一个

**步骤**：

1. 确保所有改动已 build + deploy
2. 跑 `scripts/test_physx_ref_pd.py` (20 episodes)
3. 记录 α、survival、reward 分布
4. 和基线 (α=0.053, survival=2.5) 对比

**验收**（分阶段目标）：

| 阶段 | α 目标 | survival 目标 | 说明 |
|---|---|---|---|
| T4 初验 | < 0.01 | > 10 步 | 任一 task 有实质改善 |
| T4 中验 | < 0.005 | > 40 步 | 所有 P0/P1 task 完成 |
| T4 终验 | < 0.002 | > 80 步 | Isaac Sim 同级 |

**估时**：0.5 天（每次改动后跑 20-ep 需要 ~150s）

---

## T5：dt/decimation 对齐 Isaac Sim

**目标**：将仿真参数从 dt=0.002×10 改为 dt=0.005×4（和 Isaac Sim 一致），让 drive 在每个 substep 内有更多时间收敛。

**前置**：T1-T3 完成后，如果 α 仍 > 0.005

**步骤**：

1. 修改 `physx_env.py` 的 `native_dt` 和 `decimation`
2. 可能需要同时修改 PhysX scene 的 `simulate(dt)` 调用（当前每个 substep 调用一次 simulate）
3. 跑 20-episode ref PD 对比

**验收**：α 改善 20%+

**估时**：0.25 天

---

## T6：手动 PD fallback（最后手段）

**目标**：如果 PhysX 内置 articulation drive 无法达到所需跟踪带宽，改为 per-substep 手动 torque 计算（MuJoCo 方案）。

**前置**：T1-T5 完成后 α 仍 > 0.005

**步骤**：

1. 在 bindings 中暴露 `setJointForce(axis, torque)` 或每 substep 读 qpos/qvel + 写 drive target 的能力
2. 在 `physx_env.py` 的 `_physics_step` 循环中，每 substep 重新计算 torque 并施加
3. 跑 20-episode ref PD 对比

**估时**：1 天（涉及 C++ 改动 + 物理调试）

---

## Task 汇总

| Task | 名称 | 估时 | 前置 | 独立 | 改动文件 |
|---|---|---|---|---|---|
| T1 | Per-joint PD gains 对齐 | 0.25d | — | ✅ | loader.py |
| T2 | Reward FK 统一 | 0.25d | — | ✅ | physx_env.py |
| T3 | Joint frame 修复 | 1d | — | ✅ | bindings.cpp |
| T4 | Ref PD 验证 | 0.5d | T1/T2/T3 | — | 测试 |
| T5 | dt/decimation 对齐 | 0.25d | T1-T3 | — | physx_env.py |
| T6 | 手动 PD fallback | 1d | T1-T5 | — | bindings.cpp + physx_env.py |

**总估时**：~3.5 天（不含 T6）

**并行度**：T1/T2/T3 完全独立，可同时开发

**关键路径**：T3 → T4（joint frame 是最可能的主因，其他是辅助优化）

## 启动建议

按投入产出比排序：
1. **T1 first**（10 分钟，纯 Python，最快见效）
2. **T2 second**（10 分钟，纯 Python，消除 reward NaN）
3. **T3 third**（核心修复，需要 C++ 改动）
4. T4 每次改动后跑一次（150s/次）
5. T5/T6 看 T4 结果决定
