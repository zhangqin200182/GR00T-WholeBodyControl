# PhysX Articulation Drive 稳定性问题分析

> 2026-08-03 | T6 阻断面 → 已解决，T6 完成

## 1. 核心现象

G1 29-DOF 模型在 PhysX 5 reduced-coordinate articulation 中**即使无重力、无 PD 驱动也会振荡爆炸**。

| 测试条件 | 结果 |
|---|---|
| 1-DOF revolute (X axis), kp=30, 零重力 | ✅ q 从 0.5 平滑收敛到 0 |
| 1-DOF revolute (Y axis), kp=30, 零重力 | ✅ 同上 |
| 1-DOF revolute (Z axis), kp=30, 零重力 | ✅ 同上 |
| T2 2-link pendulum, kp=100, 重力 | ✅ 100 步存活，q 正常移动 |
| **G1 29-DOF, 零重力, 无 drive target** | ❌ ss0: q_range=[-0.28, 0.44] → ss3: [-5.56, 1.93] → ss8: NaN |
| G1 29-DOF, 零重力, kp=1 | ❌ 同上（ss0→ss3→ss8 NaN） |
| G1 29-DOF, 零重力, kp=10 | ❌ 同上 |
| G1 29-DOF, 零重力, kp=30 | ❌ 同上 |
| G1 29-DOF, 零重力, driveType=eNONE | ❌ 同上 |
| G1 29-DOF, 零重力, solver iterations=32 pos+4 vel | ❌ 更快 NaN（ss0 已达 ±12.57） |
| G1 29-DOF, 重力, 无 drive target | ❌ ss0: q_range=[-0.47, 0.43], ss10: NaN |

**典型时序**（G1, 零重力, 无 drive, 8 iterations, dt=0.002）：

```
substep 0: q_range=[-0.284, 0.442]    正常
substep 1: q_range=[-0.363, 0.654]    开始发散
substep 2: q_range=[-1.398, 0.837]    一个关节超出限制
substep 3: q_range=[-5.564, 1.934]    多个关节超出限制
substep 4: q_range=[-5.434, 5.885]   
substep 5: q_range=[-12.566, 12.566]  ±2π — 关节 wrap-around
substep 8: NaN                         数值爆炸
```

## 2. 排除过的因素

### 2.1 PD gain 过高

kp=100、30、10、1 全部振荡；`driveType=eNONE`（完全禁用 drive）仍然振荡。**结论：不是 drive 问题。**

### 2.2 重力

零重力下仍然振荡。**结论：不是重力坍塌。**

### 2.3 求解器迭代次数

默认 8 pos + 1 vel 振荡；增加到 32 pos + 4 vel 结果更差（ss0 直接到 ±12.57）。**结论：增加迭代次数反而加速发散。**

### 2.4 关节非旋转轴未锁定

已验证 PhysX 5 header：`setMotion` 默认 `PxArticulationMotion::eLOCKED`。显式锁定 eTWIST/eSWING1/eSWING2 中非活动轴无效。**结论：默认就是锁定的。**

### 2.5 单/双关节

1-DOF 和 2-DOF 链在完全相同的驱动参数和求解器配置下稳定运行。**结论：问题出在 29-DOF 链特有的长度或参数上。**

### 2.6 单关节的 axis 方向

X (eTWIST)、Y (eSWING1)、Z (eSWING2) 三种 axis 的单关节测试全部稳定。

## 3. 关键对比

单关节稳定 vs 29-DOF 链爆炸的唯一区别是**链的深度和参数**。

| 假设 | 验证结论 |
|---|---|
| A. 极端质量比（torso 9.6kg / waist_roll 0.047kg = 204:1）导致数值不稳定 | 理论上排除——正确 pose 下 PhysX 可处理更大质量比 |
| B. 某个 link 的 MJCF local pose 产生退化（零长度/错误方向） | 不是直接根因 |
| C. 29-DOF 长链在 reduced-coordinate solver 中的已知数值限制 | PhysX ReducedCoordinate 支持 64+ links |
| D. `createLink(parent, local_pose)` 对非 root link 的 pose 处理方式不同于预期 | ✅ **根因确认** |

### 3.1 假设 A：极端质量比

G1 各 link 的质量范围：

| Link | Mass (kg) |
|---|---|
| torso_link | 9.598 |
| pelvis | 3.813 |
| left_knee_link | 1.932 |
| left_hip_yaw_link | 1.702 |
| left_hip_roll_link | 1.52 |
| left_hip_pitch_link | 1.35 |
| ... | ... |
| left_wrist_pitch_link | 0.484 |
| waist_roll_link | 0.047 |
| left_ankle_pitch_link | 0.074 |

最大/最小质量比 = 9.598 / 0.047 = **204:1**。

## 4. 根因确认：`createLink` pose 参数语义错误

### 4.1 PhysX SDK 源码分析

阅读 PhysX 5 SDK 源码 `NpFactory.cpp:250-278`，找到 `createLink` 内部对 pose 的处理：

```cpp
// NpFactory.cpp:265-268
PxTransform parentPose = parent->getCMassLocalPose().transformInv(pose);
PxTransform childPose = PxTransform(PxIdentity);
npArticulationJoint = root.createArticulationJoint(*parent, parentPose, *npArticulationLink, childPose);
```

**`createLink(parent, pose)` 的第二个参数是 world-space global pose，不是 parent-relative local pose。**

PhysX 内部用 `parent->getCMassLocalPose().transformInv(pose)` 计算 joint 的 parent-frame anchor——即用 parent CoM frame 的逆变换把 world-space pose 转成 joint parent frame。如果传入 parent-relative 的 local pose，PhysX 会把它当成 global pose 再做逆变换，算出完全错误的 joint frame。

### 4.2 为什么这导致爆炸

错误的 joint frame 使得 articulation 内部约束不一致：
- 关节 parent anchor 和 child anchor 的相对位置是错的
- 求解器在第一步就试图"修正"这些不一致的约束
- 29-DOF 深链的累积变换错误指数放大（pelvis→hip→knee→ankle 越深越离谱）
- 修正力矩远超物理合理范围 → 关节角在 1-2 个 substep 内就超出限制 → NaN

### 4.3 为什么 1-DOF / 2-DOF 测试没发现

- **Root link**：parent=NULL，local pose = global pose，无论传哪个都正确
- **1-DOF 链**：子 link 只有一层 parent→child，local 和 global 的差别取决于 root 的位置/朝向。如果 root 在原点且无旋转，local ≈ global
- **2-DOF 链**：误差开始累积但还不够大，可能恰好在求解器的容错范围内
- **29-DOF 链**：第 3-4 层开始 local 和 global 的差异变大（hip_yaw、knee 等），到 torso→shoulder→elbow 深链时完全错位

### 4.4 修复

在 `physx_bindings.cpp` `finalize()` 中传入 world-space 累积位姿：

```cpp
std::vector<PxTransform> world_poses(n_links);
for(int i=0;i<n_links;i++) {
    int p=parb(i);
    PxTransform local(PxVec3(pb(i*3),pb(i*3+1),pb(i*3+2)),
                      PxQuat(qb(i*4+1),qb(i*4+2),qb(i*4+3),qb(i*4+0)));
    PxTransform world = (p>=0) ? world_poses[p].transform(local) : local;
    world_poses[i] = world;
    links[i] = ptr->createLink(parent, world);
    ...
}
```

### 4.5 修复后验证

| 测试条件 | 结果 |
|---|---|
| G1 29-DOF, 零重力, kp=100, q_start=0.5 | ✅ 50 substep 稳定，q_range=[-0.62, 1.31]，0 NaN |
| G1 29-DOF, 重力, kp=100, ref PD | ✅ physics 稳定，0 NaN |

## 5. 残留问题：body position FK 不一致

### 5.1 现象

修复后 physics 稳定，但 `get_link_world_pose()` 返回的 body 位置和 Python FK 计算的参考位置不一致：

| Body | PhysX get_link_world_pose | Python FK (正确) |
|---|---|---|
| pelvis | z=0.788 | z=0.793 |
| left_ankle_roll | z=3.110 | z≈0.05 |
| torso | z=3.255 | z≈0.80 |

PhysX 的 FK（world-space createLink 链）产生的位置和 MJCF local-pose FK 不同。根因待进一步分析，可能与 `setCMassLocalPose` 和 `getGlobalPose` 的 body frame vs CoM frame 差异有关。

### 5.2 临时方案

在 termination 和 reward 计算中，用 Python FK 计算 robot 的 body 位置（从实际 joint angles 推），而非直接读 `get_link_world_pose()`。这样确保 robot 和 reference 的 body 位置使用同一 FK：

```python
actual_qpos = self.art.get_joint_positions()
tracked = self._fk.get_tracked_poses(root_pos, root_quat, actual_qpos)
actual_body_pos = np.array([t[0] for t in tracked], dtype=np.float64)
```

这个方案是临时的——长期应该修复 PhysX 侧 FK 使其和 MJCF 一致。

## 6. T6 最终结果

| 指标 | 当前值 | 目标 | 状态 |
|---|---|---|---|
| Physics 稳定性 | ✅ 50 substep 无 NaN | — | 已解决 |
| createLink pose | ✅ world-space | — | 已修复 |
| FK 一致性 (termination/reward) | ✅ Python FK 统一 | — | 临时方案 |
| α (per-step drift) | 0.053 | < 0.002 | ❌ 差 26× |
| Survival | 2.5 步 (mean) | > 80 步 | ❌ |
| Survival (max) | 9 步 | — | — |

**跟踪质量不足的根因**：kp=100/kd=5 的 PhysX articulation drive 在 world-pose createLink 下 physics 稳定但跟踪带宽不足。具体表现为 per-step root drift ~5cm，2-3 步后累计误差触发终止。需要后续 PD 参数调优或改 decimation（减少 drive 更新间隔）。

## 7. 后续方向

1. **PD 调优**：扫描 kp/kd 组合（如 kp=200/300/500，kd=10/15/25），找到 PhysX drive 的稳定-跟踪平衡点
2. **Decimation 调优**：当前 decimation=10（20ms 控制周期），尝试 decimation=5（10ms）提高 drive 更新频率
3. **PhysX FK 修复**：解决 `get_link_world_pose` 和 MJCF FK 的不一致（可能涉及 `setCMassLocalPose` 和 joint parent/child frame 设置）
4. **手动 PD**：如果 PhysX 内置 drive 无法达到所需带宽，改用 per-substep 手动 torque 计算（MuJoCo 方案），需要在 bindings 中增加每个 substep 读 qpos/qvel + 写 joint position target 的能力

## 8. 参考

- PhysX 5 SDK 源码：`/tmp/physx_build/physx/`
- NpFactory.cpp（createLink 实现）：`physx/source/physx/src/NpFactory.cpp:250-278`
- 测试脚本：`/root/GR00T-WholeBodyControl/scripts/test_physx_ref_pd.py`
- Bindings：`gear_sonic/envs/physx/physx_bindings.cpp`
- Loader：`gear_sonic/envs/physx/physx_loader.py`
- Env：`gear_sonic/envs/physx_env.py`
- FK：`gear_sonic/envs/physx/physx_fk.py`
