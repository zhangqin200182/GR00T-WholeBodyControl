# PhysX 跟踪质量提升方案

> 2026-08-03 | T6 完成后的系统分析与实施路线

## 1. 当前状态

| 指标 | 当前值 | 目标 |
|---|---|---|
| α (per-step drift) | 0.053 | < 0.002 |
| Survival (mean) | 2.5 步 | > 80 步 |
| Physics 稳定性 | ✅ | — |
| createLink pose | ✅ world-space | — |

## 2. 关键发现：问题不在 PD 增益

### 2.1 Isaac Sim 的真实 PD 参数

查阅 `gear_sonic/envs/manager_env/robots/g1.py`，Isaac Sim 使用 `ImplicitActuatorCfg`
（即 PhysX 内置 drive，和我们的方式完全相同）。关键参数：

| 关节组 | Isaac kp | Isaac kd | 我们的 kp | 我们的 kd |
|---|---|---|---|---|
| hip_pitch/roll, knee | 99.1 | 6.3 | 100 | 5 |
| hip_yaw | 40.2 | 2.6 | 100 | 5 |
| ankle | 28.5 | 1.8 | 100 | 5 |
| waist | 28.5~40.2 | 1.8~2.6 | 100 | 5 |
| shoulder/elbow | 14.3 | 0.9 | 100 | 5 |
| wrist | 16.8 | 1.1 | 100 | 5 |

**Isaac Sim 的腿部 kp≈99，和我们的 kp=100 几乎一样。**
**上肢 kp 甚至更低（14~17）。**

这意味着："kp 不够高" 不是 α=0.053 的根因。Isaac Sim 用同样的 PD gains、同样的 PhysX 引擎，能做到 α<0.002。

### 2.2 那差距在哪？

对比 Isaac Sim vs 我们的 PhysX pipeline，有以下不同：

| 差异 | Isaac Sim | 我们的 PhysX |
|---|---|---|
| **PD 增益分布** | 每关节不同 (14~99) | 全部 100 |
| **dt × decimation** | 0.005 × 4 = 20ms | 0.002 × 10 = 20ms |
| **单步物理精度** | dt=0.005, 4 substep | dt=0.002, 10 substep |
| **Solver type** | TGS (Isaac 默认) | TGS (我们设了) |
| **Solver iters** | Isaac 默认 (4 pos / 1 vel) | 8 pos / 1 vel |
| **FK 来源** | PhysX getGlobalPose | Python FK (workaround) |
| **reward 的 FK** | PhysX getGlobalPose | PhysX getGlobalPose ❌ |
| **termination FK** | PhysX getGlobalPose | Python FK |
| **action scale** | 0.25 * effort/kp | jh (half-range) |
| **Joint frame 设置** | URDF→PhysX (USD) | MJCF→world pose createLink |
| **CoM frame** | URDF 内嵌 | setCMassLocalPose |

## 3. 根因分析

### 3.1 最可能的根因：joint parent/child frame 不正确

**这是 §5 "body position FK 不一致" 背后的真正问题。**

PhysX articulation 的 joint 约束精度取决于 `createArticulationJoint(parent, parentFrame, child, childFrame)` 中 parent frame 和 child frame 的正确性。我们的代码：

```cpp
// physx_bindings.cpp:279-280
links[i] = ptr->createLink(parent, world);
```

PhysX 在 `NpFactory.cpp:265-268` 内部自动计算 joint frame：
```cpp
PxTransform parentPose = parent->getCMassLocalPose().transformInv(pose);
PxTransform childPose = PxTransform(PxIdentity);
```

这里 **childPose 被强制设为 Identity**。在 Isaac Sim (URDF→USD) 流程中，joint frame 是 URDF 中精确指定的；而我们通过 MJCF → world pose → createLink 的路径，joint frame 被 PhysX 自动推导，可能和 MJCF 的 joint frame 语义不一致。

具体问题：
- MJCF 中 joint 的 anchor 是 body 的 local origin
- PhysX 自动推导的 joint parent frame = `parent_CoM⁻¹ * child_world_pose`
- 如果 parent 的 CoM 不在 body origin（即 `setCMassLocalPose` 有偏移），joint anchor 位置就会偏

这解释了为什么 `get_link_world_pose()` 返回的位置和 MJCF FK 不一致——joint frame 错了，PhysX 的正运动学就和 MJCF 对不上。

### 3.2 验证方法

```
对每个 link，计算：
  MJCF_joint_anchor = parent_body_origin + joint_pos_in_parent
  PhysX_joint_anchor = parent_CoM⁻¹ * child_world_pose

如果两者不一致（误差 > 1e-4），就是 joint frame 问题。
```

### 3.3 次要因素

1. **PD 增益不分关节**：我们的上肢用 kp=100（Isaac 用 14~17），上肢 PD 过强可能导致高频振动，通过关节耦合影响下肢稳定性
2. **reward 和 termination 的 FK 不一致**：reward 用 `get_link_world_pose`（PhysX FK，数值偏差大），termination 用 Python FK。两个系统看到不同的 robot 状态
3. **dt 不同**：Isaac 用 dt=0.005 × 4，我们用 dt=0.002 × 10。虽然控制周期都是 20ms，但每步 0.005s 比 0.002s 给 drive 更多时间在一个 substep 内稳定

## 4. 实施路线

### Phase 0：诊断确认（~1天）

**目标**：确认 joint frame 是否是主因。

**E0-1：Joint frame 审计脚本**

写一个脚本，对比每个 joint 的：
- MJCF 定义的 joint anchor（body local origin，因为 MJCF joint pos 默认在 body frame 原点）
- PhysX `createLink` 后 joint 的 parent frame（通过 `getParentArticulationJointTransform()` 或类似 API 查询）
- 如果 PhysX 没有直接查询 API，用 `get_link_world_pose` 和 Python FK 的差值间接推断

**E0-2：Per-joint kp 测试**

将 PD gains 改为 Isaac Sim 的 per-joint 值（见 §2.1），看 α 是否改善。这是最快的一步，只改 `physx_loader.py`。

### Phase 1：修复 joint frame（~2-3天）

**如果 E0-1 确认 joint frame 有问题：**

**方案 A：手动设置 joint frame（推荐）**

在 `finalize()` 中，`createLink` 之后、添加到 scene 之前，用 `setParentPose()` / `setChildPose()` 手动设置每个 joint 的 parent frame 和 child frame：

```cpp
// After createLink:
auto *joint = links[child]->getInboundJoint();

// MJCF joint anchor is at the child body's local origin relative to parent body origin
// parent frame: child body origin in parent body frame (= local_pos in MJCF)
PxTransform parentFrame(local_pos_in_parent_body, PxQuat(PxIdentity));
// child frame: joint is at child body origin
PxTransform childFrame(PxTransform(PxIdentity));
joint->setParentPose(parentFrame);
joint->setChildPose(childFrame);
```

需要验证 PhysX 5 的 `setParentPose` / `setChildPose` 是否在 finalize 前可用。

**方案 B：调整 world pose 使 PhysX 自动推导正确**

在计算 `world_poses[i]` 时，考虑 parent 的 CoM 偏移，使得 `parent_CoM⁻¹ * world` 产生正确的 joint anchor。这比较 trick，不推荐。

### Phase 2：统一 FK + 精细参数调优（~2天）

**E2-1：统一 reward FK**

`_compute_reward()` 中的 body pos/quat 也改用 Python FK：

```python
# 在 _compute_reward 开头：
actual_qpos = self.art.get_joint_positions()
tracked = self._fk.get_tracked_poses(root_pos, root_quat, actual_qpos)
body_pos_w = np.array([t[0] for t in tracked], dtype=np.float64)
body_quat_w = np.array([t[1] for t in tracked], dtype=np.float64)
```

**E2-2：Per-joint PD gains**

按 Isaac Sim 的 per-joint 配置设置 PD gains：

```python
# physx_loader.py
ISAAC_KP = {
    "hip_pitch": 99.1, "hip_roll": 99.1, "knee": 99.1,
    "hip_yaw": 40.2,
    "ankle_pitch": 28.5, "ankle_roll": 28.5,
    "waist_roll": 28.5, "waist_pitch": 28.5, "waist_yaw": 40.2,
    "shoulder_pitch": 14.3, "shoulder_roll": 14.3, "shoulder_yaw": 14.3,
    "elbow": 14.3, "wrist_roll": 14.3,
    "wrist_pitch": 16.8, "wrist_yaw": 16.8,
}
```

**E2-3：dt/decimation 组合**

测试 dt=0.005 × 4（和 Isaac Sim 一样），替代当前的 dt=0.002 × 10。

### Phase 3：验证（~1天）

20 episode ref PD 测试：
- 目标：α < 0.005, survival > 40 步
- 如果达标 → 继续到 100 episode 全量测试（α < 0.002, survival > 80 步）
- 如果不达标 → 分析哪个 phase 的修复贡献最大，针对性继续

### Phase 4：手动 PD fallback（仅在 Phase 1-3 不够时）

如果 joint frame 修正 + per-joint gains + dt 调优后 α 仍 > 0.005：

在 bindings 里加 per-substep 回调，改为手动 PD（和 MuJoCo env 一样）。
这需要较大改动（约 100 行 C++），作为最后手段。

## 5. 预期结果

| Phase | 预期 α 改善 | 原因 |
|---|---|---|
| Phase 0 (per-joint kp) | 0.053 → 0.03~0.04 | 上肢 kp 从 100 降到 14~17，减少耦合振动 |
| Phase 1 (joint frame) | → 0.005~0.01 | 正确的 joint constraint → 正确的 FK → 物理精度提升 |
| Phase 2 (统一 FK + dt) | → 0.002~0.005 | 消除 reward/termination 不一致 + 优化 substep 配置 |
| Phase 4 (手动 PD) | → < 0.002 | 和 Isaac Sim 完全一致的控制策略 |

## 6. 优先级排序

```
P0 (必做): Phase 1 — joint frame 修复
     → 这很可能是 α=0.053 的主因（PhysX FK 和 MJCF FK 差几米级别）

P1 (必做): Phase 2 — 统一 FK + per-joint PD + dt 调优
     → 消除观测不一致，匹配 Isaac Sim 配置

P2 (观察): Phase 0 — 诊断确认
     → 如果你有信心直接改，可以跳过诊断直接做 Phase 1

P3 (备选): Phase 4 — 手动 PD
     → 仅在前面不够时启用
```

## 7. 参考

- Isaac Sim G1 配置：`gear_sonic/envs/manager_env/robots/g1.py:210-250`
- Isaac Sim action config：`gear_sonic/config/manager_env/actions/terms/joint_pos.yaml`
- Sim2sim PD gains：`gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml`
- PhysX bindings：`gear_sonic/envs/physx/physx_bindings.cpp`
- PhysX loader：`gear_sonic/envs/physx/physx_loader.py`
- PhysX env：`gear_sonic/envs/physx_env.py`
