# PhysX 训练管线：现状、调查、分析与计划

> 2026-08-04 | 从 MuJoCo 迁移到 PhysX 5 的完整技术总结

## 1. 为什么从 MuJoCo 换到 PhysX

MuJoCo 经过 15+ 轮实验，最佳 α=0.013（Isaac 目标 <0.002），差距 6.5×。

| | MuJoCo | PhysX |
|---|---|---|
| 瓶颈类型 | 引擎级（硬约束求解器 vs 软约束） | 参数级（PD gain、solver iters） |
| 调优空间 | 已穷尽 | 刚开始 |
| 与 Isaac 引擎 | 不同 | **相同**（都是 PhysX 5） |

核心判断：MuJoCo 是天花板问题，PhysX 是配置问题。

## 2. 已完成的工作

### T1：Per-joint PD gains 对齐 Isaac Sim ✅

`physx_loader.py` 中 flat kp=100/kd=5 → Isaac Sim 的 per-joint 配置。

数据来源：`gear_sonic/envs/manager_env/robots/g1.py`，Isaac Sim 用 `ImplicitActuatorCfg`（内置 PhysX joint drive，和我们相同）。增益按 10Hz 自然频率 + 阻尼比 2.0 从 motor armature 计算：

| 关节组 | Isaac kp | Isaac kd | force_limit |
|---|---|---|---|
| hip_pitch/roll, knee | 99.1 | 6.3 | 139 Nm |
| hip_yaw | 40.2 | 2.6 | 88 Nm |
| ankle | 28.5 | 1.8 | 50 Nm |
| waist_roll/pitch | 28.5 | 1.8 | 50 Nm |
| waist_yaw | 40.2 | 2.6 | 88 Nm |
| shoulder/elbow/wrist_roll | 14.3 | 0.9 | 25 Nm |
| wrist_pitch/yaw | 16.8 | 1.1 | 5 Nm |

效果：手臂 kp 从 100 降到 14~17，减少了上肢过驱对下肢的耦合振动。

### T2：全局 FK 统一为 Python FK ✅

`physx_env.py` 中所有 `get_link_world_pose` 调用替换为 `_get_body_state_fk()`：

| 位置 | 修改前 | 修改后 |
|---|---|---|
| `_compute_reward()` r3/r4/r5/r6/r11 | PhysX `getGlobalPose` | Python FK |
| `reset()` `_prev_body_pos/quat` | PhysX `getGlobalPose` | Python FK |
| `_anti_shake()` | PhysX `getGlobalPose` | Python FK |
| `_compute_critic_obs()` | PhysX `getGlobalPose` | Python FK |
| `_check_termination()` | 此前已改为 Python FK | 不变 |

效果：reward、termination、critic obs 使用同一个 FK 源，消除了观测不一致。

### T3：Joint frame 修复（setParentPose）❌ 已终止

**尝试了什么**：在 `createLink` 之后调 `setParentPose` / `setChildPose` 修正关节 frame。

**实验过程**：

- 在 G1 全模型上测试：`createLink(world)` → `setCMassLocalPose` → `setParentPose(correct)` → `addArticulation`
- FK 精度大幅改善：PhysX `getGlobalPose` 误差从 3.77m 降到 0.06m
- 但物理稳定性被破坏：第 11 步 NaN，而不修复可以稳定跑 100+ 步

- 在 2-link 简化模型上测试：2-link 上物理稳定，但 addArticulation 后子链接在错误位置（传播的符号或方向有问题）

**为什么失败**：`createLink(world_poses)` 同时设置 body 位置和 joint frame，它们是同一个 `pose` 参数推导出的自洽对。修改 joint frame 后 body 位置没有同步更新——solver 看到 joint 约束锚点和 body 当前位置的矛盾，把机器人撕碎。

**为什么 Isaac Sim 可以**：见 §3。

此路线彻底终止。

### T4：Ref PD 验证 ✅

```text
物理稳定性: 100+ 步（之前 2.5 步）  ✅
静态 PD 保持: jerr=0.0000（完美）   ✅
动态轨迹跟踪 α: ~0.10 rad          ❌ 目标 <0.002（差 50 倍）
FK 误差: ~3.15m（不影响训练）       —  Python FK 绕过
```

## 3. Isaac Sim 关节约束调研

> 本节来自子 agent 对 PhysX USD 桥接层、PhysX 5 API 文档和开源代码的调研。

### 3.1 Isaac Sim 的关节 frame 数据路径

Isaac Sim 走三条路径将机器人模型喂给 PhysX：

```
URDF →（Isaac Lab convert_urdf.py / Omniverse URDF importer）→ USD → PhysX articulation
```

关键模块：**PhysX-USD 桥接层（omni.physx / PhysxSchema）**。构建 articulation 时不依赖 `createLink` 的自动推导。

USD 里每个 joint prim 上有**四个独立属性**，关节 frame 和 link 创建位姿是完全解耦的两份数据：

```python
# Omniverse 文档中的示例代码
revoluteJoint.CreateLocalPos0Attr().Set(revoluteJointlocalPositions[0])  # 在 body0/parent 坐标系中
revoluteJoint.CreateLocalRot0Attr().Set(revoluteJointLocalRotations[0])
revoluteJoint.CreateLocalPos1Attr().Set(revoluteJointlocalPositions[1])  # 在 body1/child 坐标系中
revoluteJoint.CreateLocalRot1Attr().Set(revoluteJointLocalRotations[1])
```

| USD 属性 | 含义 | 对应我们的 MJCF 数据 |
|---|---|---|
| `localPos0` / `localRot0` | 关节在 parent body 坐标系中的位姿 | `<body pos="..." quat="...">` |
| `localPos1` / `localRot1` | 关节在 child body 坐标系中的位姿 | `<joint pos="0 0 0">` |

### 3.2 Isaac Sim 的 articulation 构建步骤

1. URDF `<origin xyz rpy>` →（converter）→ USD joint 的 `localPos0/localRot0`
2. USD 场景加载 → PhysX-USD bridge 读取 localPos0/1、localRot0/1
3. 构建时直接调 `PxArticulationJointReducedCoordinate::setParentPose(localPos0)` + `setChildPose(localPos1)`
4. 然后 `addArticulation`

关键约束：`setParentPose` 必须在 simulation 开始前调用（PhysX 5.4.1 API 文档明确规定 "not allowed while simulation is running"），但 Isaac 在 `addArticulation` 之前调，完全合法。

来源：
- [Omniverse Articulations 文档](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.1/dev_guide/rigid_bodies_articulations/articulations.html)
- [PhysX API setParentPose](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxArticulationJointReducedCoordinate.html)
- [IsaacLab 资产导入](https://deepwiki.com/fan-ziqi/IsaacLab/4.3-asset-import)

### 3.3 我们和 Isaac 的差距

```
Isaac 的路径：
  createLink(parent, world_pose)  → 只决定 link 的初始位置
  setParentPose(local)            → 独立设置关节 frame
  两份数据解耦

我们的路径（裸 PhysX SDK）：
  createLink(parent, pose) → 一个参数同时决定 link 初始位置 + joint frame
  两份数据耦合
```

**我们不缺数据**——MJCF `<body pos quat>` 就是 parentPose 的值，`<joint pos="0 0 0">` 就是 childPose 的值。但裸 `createLink` API 把这两个概念绑死了。

### 3.4 为什么 T3 的 setParentPose 在 Issac Sim 有效在我们无效

Isaac 在 `createLink` 之后立即 `setParentPose` → 此时 articulation 还没加入场景 → PhysX 允许修改 joint frame → `addArticulation` 时 solver 用修正后的正确 joint frame 初始化约束。**Isaac 不需要 updateKinematic 传播**——它的 body 初始位置来自 USD，和 joint frame 从一开始就是独立设置的。

而我们：`createLink(world)` 设置的 body 初始位置是基于错误 parentPose 推导的，`setParentPose` 改 joint frame 后 body 位置没更新。solver 看到矛盾——joint 锚点位置 vs body 当前位置不一致。

### 3.5 一个未验证的方案：修改 createLink 的 pose 参数

子 agent 提出了一个数学方案，不在 `createLink` 之后修 joint frame，而是**改 createLink 传入的 pose 值**来补偿 CoM 偏移：

```
当前传入：  pose = parent_world × child_local
建议传入：  pose' = parent_world × child_local × child_CoM
```

理论效果：
- body 初始位置 → 更接近正确值（消掉 childCoM⁻¹ 偏移）
- FK 误差 → 可能从 3.15m 降到 ~5cm
- 物理约束 → child CoM 偏移仅引入 cm 级 joint anchor 偏差，kp=99 可压住

**未验证**，优先级低于 P0/P1。

## 4. 关键技术认知

### 4.1 createLink pose 语义

```cpp
// PhysX 5 NpFactory.cpp:265-268
PxTransform parentPose = parent->getCMassLocalPose().transformInv(pose);
PxTransform childPose = PxTransform(PxIdentity);
npArticulationJoint = root.createArticulationJoint(*parent, parentPose, *npArticulationLink, childPose);
```

`pose` 是 world-space global pose。PhysX 计算 `parentPose = parent_CoM⁻¹ × pose`，childPose 恒为 Identity（关节在 child CoM 原点）。

### 4.2 两种 createLink 方式对比

| 方式 | parentPose | FK | 物理 |
|---|---|---|---|
| Local pose（`createLink(parent, local)`）| 正确值 | ✅ | ❌ NaN @ step 2 |
| World-accumulated（`createLink(parent, world)`）| 基于 identity CoM | ❌ 3m+ 误差 | ✅ 稳定 |

两种方式不可兼得。我们选择 world-accumulated + Python FK。

### 4.3 setCMassLocalPose 的顺序问题

`finalize()` 中 `createLink` 在先（line 280），`setCMassLocalPose` 在后（line 285-287）：

```cpp
links[i] = ptr->createLink(parent, world);          // 此时 CoM = identity
links[i]->setMass(mb(i));
links[i]->setMassSpaceInertiaTensor(...);
links[i]->setCMassLocalPose(PxTransform(com, ...));   // 改 CoM，joint frame 已固化
```

PhysX 在 `createLink` 时用**当时的 CMassLocalPose（identity）**推导并存储 joint frame。此后改 CoM 不会重算 frame。这是一个真实 bug，但裸 PhysX SDK 无 API 可修复——`createLink` 之前 link 还不存在，无法预设 CoM。

## 5. 当前状态

```text
┌────────────────┬────────────────────────┬─────────────────────┐
│      指标      │          结果          │        目标         │
├────────────────┼────────────────────────┼─────────────────────┤
│ 物理稳定性     │ 100+ 步                │ ✅ 解决             │
│ 静态 PD 保持   │ jerr=0.0000            │ ✅ 解决             │
│ FK 观测统一    │ Python FK 全局替代     │ ✅ 解决             │
│ 动态跟踪 α     │ ~0.10 rad              │ <0.002（差 50 倍）  │
│ 存活步数       │ 100+ 步                │ >80 步 ✅           │
└────────────────┴────────────────────────┴─────────────────────┘
```

物理层已稳定，核心剩余问题：**动态轨迹跟踪精度差 50 倍**。

## 6. 动态跟踪差的待排查方向

### P0：关节顺序审计

**假设**：PKL 数据的 29 维 dof 顺序与 MJCF actuator/motor 顺序不一致。

**为什么可能性高**：静态保持 jerr=0 不能排除——所有关节 drive target = 实际 qpos 时，无论顺序如何都不会有 PD 误差。但动态跟踪时若数据错配，跟踪必然失败。

**验证方法**：打印单 step 的 ref_qpos 和 jm 的差值向量，人工确认偏离中性的关节是否和当前动作语义一致。

**估时**：30 分钟。

### P1：单关节 PD 带宽测试

**假设**：kp=99.1 的跟踪带宽不足以跟上 30FPS 参考轨迹的变化率（每帧 ~0.1 rad 跳变）。

**验证方法**：1-DOF revolute + 正弦波 target（频率扫 1~20Hz），测 -3dB 跟踪带宽。

**估时**：1 小时。

### P2：PD 参数扫描

**假设**：更高的 kp/kd 可提升带宽。

**验证方法**：在 G1 上扫 kp=200/500/1000，20-episode ref PD 对比 α。

**估时**：2 小时。

### P3：惯性参数审计

**假设**：MJCF body inertia 值偏大导致响应迟钝。

**可能性最低**：inertial 从 CAD 导出，数量级不会错。

**估时**：1 小时。

## 7. 建议执行顺序

```text
P0（关节顺序）──→ 如果命中 → 修了直接解决
    │
    └──→ 如果排除

P1（单关节带宽）──→ 如果 kp=99 带宽 < 5Hz → 必须提 kp
    │
    └──→ 如果带宽 > 10Hz → 问题不在 PD

P2（PD 扫描）──→ 在完整 G1 上找最优 kp/kd
    │
    └──→ 如果 α < 0.01 → 继续精调

P3（惯性审计）──→ 最后手段
```

## 8. 不可行的路线（已排除）

| 路线 | 排除原因 |
|---|---|
| MuJoCo 继续调参 | α=0.013 是引擎天花板 |
| `createLink(local)` + `updateKinematic` | 深链位置不一致，solver 爆炸 |
| `createLink(world)` + `setParentPose` 修正 | body 位置和 joint frame 矛盾，物理 NaN |
| 换 URDF 格式 | 还是走同一个 `createLink` API |
| 换 ovphysx pip 包 | 封装层级更高但底层还是 `createLink` |

## 9. 参考文件

| 文件 | 用途 |
|---|---|
| `gear_sonic/envs/physx/physx_bindings.cpp` | PhysX 5 pybind11 wrapper |
| `gear_sonic/envs/physx/physx_loader.py` | MJCF→PhysX articulation 转换器 |
| `gear_sonic/envs/physx/physx_fk.py` | 纯 Python 前向运动学 |
| `gear_sonic/envs/physx_env.py` | PhysX 单环境（SONIC 兼容） |
| `gear_sonic/envs/manager_env/robots/g1.py` | Isaac Sim G1 PD 配置（参考） |
| `scripts/test_physx_ref_pd.py` | ref PD 精度验证脚本 |
| `docs/physx-stability-analysis.md` | T6 createLink 根因分析 |
| `docs/physx-next-steps.md` | 跟踪质量提升方案（旧，已被本文档替代） |
| `docs/physx-tracking-tasks.md` | Task 拆解 T1-T6 |
