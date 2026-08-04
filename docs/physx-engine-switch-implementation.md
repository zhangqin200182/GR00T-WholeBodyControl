# PhysX 5 引擎切换 — 详细实施方案 v2

> 状态：待审查 | 2026-08-02 | 修订：代码审查后修正 7 项问题

## 0. 背景摘要

经过 15+ 轮 MuJoCo 实验 (v6→v18, E0→E2)，已确证：

```
BC warmup = 20.6 步 ≈ 纯 ref PD = 21 步  → 策略已达物理天花板
参数推顶后 ~28 步                        → 所有可调参数已最优
Isaac Sim (PhysX) = 100+ 步, α < 0.002   → 6.5× 精度差距
```

**根因**：MuJoCo 硬约束接触模型 + 力-接触解耦，属于 C++ 引擎级差异，参数调不动。

**决策**：切换到裸 PhysX 5 C++ SDK（与 Isaac Sim 同一物理引擎），通过 pybind11 封装 + MJCF→PhysX 转换器，保留全部训练架构。

PhysX 5 已在 NPU 服务器 (aarch64) 上编译通过（Phase 0 ✅）。

---

## 1. 目标架构

```
ppo_trainer (复用,不改) → PhysXEnvManager (独立文件, 复用 SHM+Barrier 模式)
                             → PhysXEnv (~500行, 新写)
                               → physx_core.pyd (pybind11, ~300行 C++)
                               → physx_loader.py (MJCF→PhysX, ~400行 Python)
```

**改动范围**：只换物理引擎层。ppo_trainer、模型架构、训练流程、SHM 布局——全部不变。

**核心思路**：现有 `MuJoCoEnv` (520行) 就是模板。`PhysXEnv` 只替换物理调用：

| MuJoCo 调用 | PhysX 替代 |
|---|---|
| `mujoco.mj_step` | `scene.simulate(dt)` → `fetchResults()` |
| `self.data.qpos[7:]` | `art.getJointPositions()` |
| `self.data.qvel[6:]` | `art.getJointVelocities()` |
| `self.data.xpos[idx]` | `art.getLinkWorldPose(idx)[0]` |
| `self.data.xquat[idx]` | `art.getLinkWorldPose(idx)[1]` |
| `mujoco.mj_kinematics` | Python FK（基于关节树 + 刚体变换，见 §4.4） |
| `self.data.ctrl[:] = torque` | `art.setJointDriveTargets(target_qpos)` (PhysX 内置 PD) |
| `self.data.contact` + `mj_contactForce` | `scene.getContacts()` |

所有 MDP 逻辑（obs 拼接、reward 公式、termination 条件）都是纯数学——直接移植。

---

## 2. 四元数约定对照表（⚠️ 最容易出 bug）

三种数据源使用不同四元数顺序。当前 MuJoCo 代码中有多处手动转换，切换到 PhysX 后需要重新对齐：

| 数据源 | 顺序 | 示例 `[w,x,y,z]` 或 `[x,y,z,w]` |
|---|---|---|
| MuJoCo (`data.xquat`) | **`[w,x,y,z]`** | `[1, 0, 0, 0]` = identity |
| PhysX `PxQuat` | **`[x,y,z,w]`** | `[0, 0, 0, 1]` = identity |
| Motion PKL (`root_rot`) | **`[x,y,z,w]`** | `[0, 0, 0, 1]` = identity |
| `mujoco_math.py` 全部函数 | **`[w,x,y,z]`** | 输入输出均为此约定 |

**关键推论**：

- PhysX 输出和 PKL 数据约定一致（都是 `[x,y,z,w]`），互相直接兼容
- `mujoco_math.py`（`[w,x,y,z]`）用于 PhysX 时需要转换输入输出
- 当前 MuJoCo 代码中 `[x,y,z,w] → [w,x,y,z]` 的转换（`mujoco_env.py:164,196,480`）在 PhysX 下**全部要去掉**

**推荐方案**：在 PhysX 绑定的 C++ 侧统一为 `[w,x,y,z]`（`py::array` 返回前转换），这样 `mujoco_math.py` 全部函数零改动复用，且 MuJoCoEnv→PhysXEnv 移植时不需要改任何 quat 相关逻辑。

具体做法——`physx_bindings.cpp` 中每个返回 `PxQuat` 的函数，在 pybind11 lambda 里做：

```cpp
// PxQuat(x,y,z,w) → numpy [w,x,y,z]
auto q = link->getGlobalPose().q;
return py::array_t<float>({q.w, q.x, q.y, q.z});
```

**验证方法**：`assert quat_mul(q, quat_inv(q)) ≈ [1,0,0,0]` 在 PhysX 输出的四元数上通过。

---

## 3. Phase 1：pybind11 C++ 封装（第 1-2 周）

### 3.1 最小 API 集（~25 个函数）

```python
# 目标：import physx_core as px

# ── 全局生命周期 ──
px.init_foundation()           # PxFoundation + PxPhysics (进程级, 仅一次)
px.release_foundation()

# ── 场景 ──
px.create_scene(gravity=(0,0,-9.81), num_threads=1,
                solver_type="TGS",          # Temporal Gauss-Seidel (与 Isaac 一致)
                position_iters=8,           # Isaac Sim 默认
                velocity_iters=1,           # Isaac Sim 默认
                contact_distance=0.02,       # 接触检测余量
                ) → PxScene
scene.add_articulation(art)
scene.add_actor(rigid_static)
scene.simulate(dt=0.002)
scene.fetch_results(wait=True, block=True)
scene.get_contacts() → [(link_a_idx, link_b_idx, point[3], normal[3], force_mag), ...]

# ── 材质 ──
px.create_material(static_friction=0.6, dynamic_friction=0.5, restitution=0.0) → PxMaterial

# ── 浮基关节体 (Reduced Coordinate) ──
px.create_articulation() → PxArticulation
art.add_link(parent_idx, name, mass, diag_inertia[3], local_pos[3], local_quat[4])
art.add_joint(parent_idx, child_idx,
              type=eREVOLUTE,          # 29 个全是 revolute
              axis[3],                 # 如 (0, 1, 0)
              limits=(lower, upper),   # 如 (-2.53, 2.88)
              damping=2.0,             # 关节阻尼 (来自 XML joint damping 属性)
              friction=3.0)            # 关节摩擦 (来自 XML joint frictionloss 属性)
art.finalize()
art.get_joint_positions() → np[29]      # 标量角度 (rad)
art.get_joint_velocities() → np[29]     # 标量角速度 (rad/s)
art.get_root_world_pose() → (pos[3], quat[4])
art.get_root_world_velocity() → (lin[3], ang[3])
art.get_link_world_pose(idx) → (pos[3], quat[4])
art.get_link_world_velocity(idx) → (lin[3], ang[3])
art.set_root_world_pose(pos[3], quat[4])
art.set_joint_positions(qpos[29])
art.set_joint_velocities(qvel[29])

# ── PD 驱动 (PhysX 内置——和 Isaac Sim 一致!) ──
art.set_joint_drive_targets(target_qpos[29])   # 设 PD 目标
art.set_joint_drive_params(idx, kp, kd, force_limit)  # 每关节配置

# ── 碰撞几何 ──
px.create_sphere_shape(radius) → PxShape
px.create_box_shape(half_x, half_y, half_z) → PxShape
px.create_capsule_shape(radius, half_height) → PxShape
px.create_plane_shape() → PxRigidStatic
link.attach_shape(shape, local_pos[3], local_quat[4])
```

### 3.2 文件结构

```
gear_sonic/envs/physx/
  CMakeLists.txt          # cmake 构建 (aarch64)
  physx_bindings.cpp      # pybind11 模块 (~400行)
  Makefile                # 便捷: cmake && make -j64
```

### 3.3 CMakeLists.txt（⚠️ 修正：PhysX 5 是多库架构）

```cmake
cmake_minimum_required(VERSION 3.18)
project(physx_core)

set(PhysX_DIR /opt/physx/physx CACHE PATH "PhysX SDK root")
set(PhysX_LIB_DIR ${PhysX_DIR}/bin/linux.aarch64/release)

# PhysX 5 SDK 架构（所有库都需要链接）：
#   PhysXFoundation   — 基础类型、内存分配、数学库
#   PhysXCommon       — 几何、材质、序列化
#   PhysXExtensions   — PxArticulationReducedCoordinate、PxArticulationDrive
#   PhysX             — PxScene、PxRigidStatic、PxShape（核心物理）
#   PhysXCooking      — mesh/capsule 碰撞几何构建（如果要用 mesh collision）
find_library(PX_FOUNDATION NAMES PhysXFoundation_static_64 PhysXFoundation_64
             PATHS ${PhysX_LIB_DIR} REQUIRED)
find_library(PX_COMMON NAMES PhysXCommon_static_64 PhysXCommon_64
             PATHS ${PhysX_LIB_DIR} REQUIRED)
find_library(PX_EXTENSIONS NAMES PhysXExtensions_static_64 PhysXExtensions_64
             PATHS ${PhysX_LIB_DIR} REQUIRED)
find_library(PX_PHYSICS NAMES PhysX_static_64 PhysX_64
             PATHS ${PhysX_LIB_DIR} REQUIRED)
find_library(PX_COOKING NAMES PhysXCooking_static_64 PhysXCooking_64
             PATHS ${PhysX_LIB_DIR} REQUIRED)

find_package(pybind11 REQUIRED)

pybind11_add_module(physx_core physx_bindings.cpp)
target_link_libraries(physx_core PRIVATE
    ${PX_FOUNDATION} ${PX_COMMON} ${PX_EXTENSIONS} ${PX_PHYSICS} ${PX_COOKING})
target_include_directories(physx_core PRIVATE
    ${PhysX_DIR}/include
    ${PhysX_DIR}/include/physx
    ${PhysX_DIR}/include/physx/extensions)
```

### 3.4 Phase 1 验收

```python
import physx_core as px
import numpy as np

px.init_foundation()
scene = px.create_scene(gravity=(0, 0, -9.81), solver_type="TGS",
                         position_iters=8, velocity_iters=1)
art = load_g1_from_mjcf("g1_29dof.xml")  # Phase 2
scene.add_articulation(art)

art.set_root_world_pose((0,0,0.8), (1,0,0,0))
art.set_joint_positions(np.zeros(29))
scene.simulate(0.002); scene.fetch_results()

qpos = art.get_joint_positions()
assert qpos.shape == (29,)
assert not np.any(np.isnan(qpos))
px.release_foundation()
print("Phase 1: PASS")
```

---

## 4. Phase 2：MJCF→PhysX 转换器（第 1-2 周，可与 Phase 1 并行）

### 4.1 技术路线

不走 ovphysx + USD（OpenUSD 200万行代码 70% 是渲染）。G1 的 MJCF XML 已有全部物理参数。直接解析 XML 调 PhysX API 建模型。

PhysX 5 的 `PxArticulationReducedCoordinate` 和 MuJoCo 运动学模型一一对应。

### 4.2 映射表

| MJCF XML | PhysX API |
|---|---|
| `<body name="pelvis">` (root) | `art = px.create_articulation()` |
| `<body name="left_knee_link">` | `art.add_link(parent_idx, name, mass, diag_inertia, pos, quat)` |
| `<inertial mass="1.5" diaginertia="..."/>` | 质量+惯性直接传给 `add_link` |
| `<inertial pos="..." quat="..."/>` | CoM 局部位姿直接传给 `add_link` |
| `<joint name="left_knee" axis="0 1 0" range="..." damping="2.0"/>` | `art.add_joint(parent, child, type=eREVOLUTE, axis, limits, damping, friction)` |
| `frictionloss="3.0"` (joint 属性) | `friction` 参数传给 `add_joint` |
| `actuatorfrcrange="-88 88"` (joint 属性) | `art.set_joint_drive_params(idx, ..., force_limit=88.0)` |

### 4.3 G1 运动学树

从 `g1_29dof.xml` 解析得到 29 个 revolute 关节 + 1 个浮基：

```
pelvis (浮基 7DOF)
├─ left_hip_pitch → left_hip_roll → left_hip_yaw
│   → left_knee → left_ankle_pitch → left_ankle_roll        (左腿 6)
├─ right_hip_pitch → right_hip_roll → right_hip_yaw
│   → right_knee → right_ankle_pitch → right_ankle_roll      (右腿 6)
├─ waist_yaw → waist_roll → waist_pitch → torso               (腰部 3)
│   ├─ left_shoulder_pitch → left_shoulder_roll → left_shoulder_yaw
│   │   → left_elbow → left_wrist_roll → left_wrist_pitch
│   │   → left_wrist_yaw                                      (左臂 7)
│   └─ right_shoulder_pitch → right_shoulder_roll → right_shoulder_yaw
│       → right_elbow → right_wrist_roll → right_wrist_pitch
│       → right_wrist_yaw                                     (右臂 7)
```

### 4.4 碰撞几何 — 逐 body 完整清单（⚠️ 修正：基于实际 XML 逐行核对）

`g1_29dof.xml` 使用 **"双 geom"模式**——大多数 body 有两个 mesh geom：第一个 `contype="0" conaffinity="0"` 为视觉专用，第二个无 contype/conaffinity 属性（MuJoCo 默认 `contype=1, conaffinity=1`）为碰撞。

以下是对 PhysX 转换器真正需要创建的碰撞几何：

#### 4.4.1 启用碰撞的 mesh geom（无 contype/conaffinity = 默认开启）

##### 独立 body 的碰撞 mesh

| Body | XML 行 | Mesh | 说明 |
|---|---|---|---|
| pelvis | 49 | `pelvis_contour_link` | 骨盆碰撞 |
| left_hip_pitch | 54 | `left_hip_pitch_link` | 髋部碰撞 |
| left_hip_roll | 59 | `left_hip_roll_link` | |
| left_hip_yaw | 64 | `left_hip_yaw_link` | |
| left_knee | 69 | `left_knee_link` | |
| left_ankle_pitch | 74 | `left_ankle_pitch_link` | |
| right_hip_pitch | 93 | `right_hip_pitch_link` | |
| right_hip_roll | 98 | `right_hip_roll_link` | |
| right_hip_yaw | 103 | `right_hip_yaw_link` | |
| right_knee | 108 | `right_knee_link` | |
| right_ankle_pitch | 113 | `right_ankle_pitch_link` | |
| torso | 140 | `torso_link` | 躯干碰撞（line 139 是视觉，**line 140 存在且无 contype**） |
| left_shoulder_yaw | 162 | `left_shoulder_yaw_link` | |
| left_elbow | 167 | `left_elbow_link` | |
| left_wrist_roll | 172 | `left_wrist_roll_link` | |
| left_wrist_pitch | 177 | `left_wrist_pitch_link` | |
| left_wrist_yaw | 182 | `left_wrist_yaw_link` | |
| right_shoulder_yaw | 206 | `right_shoulder_yaw_link` | |
| right_elbow | 211 | `right_elbow_link` | |
| right_wrist_roll | 216 | `right_wrist_roll_link` | |
| right_wrist_pitch | 221 | `right_wrist_pitch_link` | |
| right_wrist_yaw | 226 | `right_wrist_yaw_link` | |

##### torso_link body 下的额外子 geom（非独立 body，但仍有碰撞）

这些 geom 没有自己的 `<body>` 元素，而是直接挂在 `torso_link` body 内部。在 PhysX 中作为 torso link 的附加 shape（同 link、不同 local pose）：

| Geom | XML 行 | Mesh | 说明 |
|---|---|---|---|
| logo_link | 142 | `logo_link` | 躯干上的 logo（line 141 视觉，line 142 碰撞） |
| head_link | 144 | `head_link` | 头部（line 143 视觉，line 144 碰撞） |
| waist_support_link | 146 | `waist_support_link` | 腰部支撑（line 145 视觉，line 146 碰撞） |

##### wrist 末端的手部碰撞

| Body | XML 行 | Mesh | 说明 |
|---|---|---|---|
| left_wrist_yaw | 184 | `left_rubber_hand` | 左手橡胶手（line 183 视觉，line 184 碰撞） |
| right_wrist_yaw | 228 | `right_rubber_hand` | 右手橡胶手（line 227 视觉，line 228 碰撞） |

#### 4.4.2 圆柱碰撞 geom（⚠️ 近似映射——PhysX 无原生 cylinder）

MuJoCo `type="cylinder"` 有平面端盖，PhysX 没有等价原生 shape。两个选项：

| 方案 | PhysX Shape | 精度 | 性能 |
|---|---|---|---|
| A: capsule 近似 | `px.create_capsule_shape(radius, half_height)` | 端盖圆滑（碰撞比 cylinder 更早触发） | 原生，快 |
| B: convex mesh | 用 `PxConvexMeshCooking` 从圆柱顶点构建凸包 | 精确匹配 cylinder 几何 | 需 Cooking，稍慢 |

**推荐先用方案 A（capsule）**。如果肩部接触行为异常（过早碰撞或穿透），再切换到方案 B。

| Body | XML 行 | MuJoCo 尺寸 `size="r h"` | 方案 A PhysX Shape |
|---|---|---|---|
| left_shoulder_pitch | 152 | `size="0.03 0.025"` | `px.create_capsule_shape(radius=0.03, half_height=0.025)` |
| left_shoulder_roll | 157 | `size="0.03 0.015"` | `px.create_capsule_shape(radius=0.03, half_height=0.015)` |
| right_shoulder_pitch | 196 | `size="0.03 0.025"` | `px.create_capsule_shape(radius=0.03, half_height=0.025)` |
| right_shoulder_roll | 201 | `size="0.03 0.015"` | `px.create_capsule_shape(radius=0.03, half_height=0.015)` |

> **MuJoCo cylinder 语义说明**：`size="r h"` 表示半径 `r`、半高 `h`。总长度 = `2h`，端盖是垂直于轴线的平面圆。Capsule 的半高含义相同，但端盖是半球。两者的碰撞检测在端盖区域有差异（capsule 的半球在边缘处比 cylinder 的平面端盖"更早"触发接触）。

**方案 B 实现参考**（如果 capsule 近似不够）：

```python
# Cylinder convex hull via PhysX cooking
def create_cylinder_convex(radius, half_height):
    """Generate convex hull vertices for a cylinder."""
    n_segments = 16
    verts = []
    for i in range(n_segments):
        angle = 2 * np.pi * i / n_segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        verts.append((x, y, -half_height))  # bottom cap
        verts.append((x, y, +half_height))  # top cap
    return px.create_convex_mesh(verts)
```

#### 4.4.3 脚部球体碰撞 geom（8 个，4 个/脚）

| Body | XML 行 | 位置 | PhysX Shape |
|---|---|---|---|
| left_ankle_roll_link | 79 | `pos="-0.05 0.025 -0.03"` | `px.create_sphere_shape(0.005)` |
| left_ankle_roll_link | 80 | `pos="-0.05 -0.025 -0.03"` | `px.create_sphere_shape(0.005)` |
| left_ankle_roll_link | 81 | `pos="0.12 0.03 -0.03"` | `px.create_sphere_shape(0.005)` |
| left_ankle_roll_link | 82 | `pos="0.12 -0.03 -0.03"` | `px.create_sphere_shape(0.005)` |
| right_ankle_roll_link | 118 | `pos="-0.05 0.025 -0.03"` | `px.create_sphere_shape(0.005)` |
| right_ankle_roll_link | 119 | `pos="-0.05 -0.025 -0.03"` | `px.create_sphere_shape(0.005)` |
| right_ankle_roll_link | 120 | `pos="0.12 0.03 -0.03"` | `px.create_sphere_shape(0.005)` |
| right_ankle_roll_link | 121 | `pos="0.12 -0.03 -0.03"` | `px.create_sphere_shape(0.005)` |

#### 4.4.4 无碰撞几何的 body（所有 geom 均为 contype=0）

| Body | XML 行 | 说明 |
|---|---|---|
| `waist_yaw_link` | 131 | 只有 1 个 visual mesh，无碰撞对 |
| `waist_roll_link` | 135 | 只有 1 个 visual mesh，无碰撞对 |

> **确认依据**：遍历了整个 XML，仅 waist_yaw_link 和 waist_roll_link 缺少"双 geom"模式的第二个 collision geom。所有其他 body 要么有双 geom（视觉+碰撞），要么有圆柱/球体碰撞 geom。

#### 4.4.5 地面

```xml
<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
```
→ `px.create_plane_shape()`

#### 4.4.6 ⚠️ 源码模型注意事项

- 训练入口 `train_agent_trl.py:371` 引用的是 `g1_29dof_v17.xml`（服务器上），不是本地仓库的 `g1_29dof.xml`。**Phase 2 开工前必须从 NPU 服务器拉取 `g1_29dof_v17.xml` 作为转换器的实际输入**。
- v17 的 P0 修复（box 脚 16cm×7cm）可能已替换了上述球体脚。转换器需要同时支持两种（球体脚和 box 脚），通过 XML 中 geom 类型自动判断。

#### 4.4.7 Mesh 碰撞的性能权衡

上述 ~22 个 mesh collision geom 是 MuJoCo 当前的碰撞配置。但如果 PhysX 的 mesh-mesh 碰撞（通过 PhysXCooking 构建 triangle mesh shape）性能太差，可以退化为：
- 下肢 → capsule（大腿+小腿）
- 上肢 → capsule（上臂+前臂）
- 躯干 → box 或 capsule

转换器支持 `--collision-mode={mesh|primitive|hybrid}` 命令行参数来切换。

### 4.5 验收

```python
art = load_g1_from_mjcf("gear_sonic_deploy/g1/g1_29dof_v17.xml")
assert art.num_joints == 29

# 1000 步物理仿真无 crash
art.set_root_world_pose((0,0,0.8), (1,0,0,0))
art.set_joint_positions(np.zeros(29))
for _ in range(1000):
    scene.simulate(0.002); scene.fetch_results()
print("Phase 2: PASS — 1000 physics steps")
```

---

## 5. Phase 3：PhysXEnv — MDP 层移植（第 2-3 周）

### 5.1 设计原则

`PhysXEnv` 是 `MuJoCoEnv` (520行) 的**直接替换**。相同 API，相同输出语义，不同物理引擎。

### 5.2 求解器参数（⚠️ 新增：匹配 Isaac Sim 行为的关键配置）

这些值直接影响接触精度，必须和 Isaac Sim 对齐：

| 参数 | Isaac Sim 默认值 | 正确 PhysX 5 API | 说明 |
|---|---|---|---|
| `solver_type` | TGS (Temporal Gauss-Seidel) | `PxSolverType::eTGS` | 比 PGS 收敛更快 |
| `position_iters` | 8 | `PxArticulationReducedCoordinate::setSolverIterationCounts(pos, vel)` | 关节体专用，不是 `PxSceneDesc::nbContactDataBlocks`（那是接触缓冲区大小） |
| `velocity_iters` | 1 | 同上 | |
| `contact_offset` | 0.02 (2cm) | `PxShape::setContactOffset(float)` | **per-shape** 设置，不是 scene 级别。每个碰撞 geom 需要调用 |
| `bounce_threshold` | 2.0 m/s | `PxSceneDesc::bounceThresholdVelocity` | 低于此速度不弹跳 |
| `friction_type` | PATCH (各向异性) | `PxFrictionType::ePATCH` | 比默认单值摩擦更精确 |
| `ccd_enabled` | False | 不启用 | 连续碰撞检测（性能代价高，训练不需要） |

```python
def _create_physx_scene(self):
    self.scene = px.create_scene(
        gravity=(0, 0, -9.81),
        solver_type="TGS",
        position_iters=8,
        velocity_iters=1,
        contact_distance=0.02,
        bounce_threshold=2.0,
        friction_type="PATCH",
    )
```

### 5.3 关键差异：PD 控制

| | MuJoCo | PhysX |
|---|---|---|
| 力矩计算 | 手动 `kp*(target-q) - kd*qvel` | **引擎内置** PxArticulationDrive |
| 每子步重算 | 是（我们的修复） | 是（自动） |
| 力-接触耦合 | **无**（先算力矩，再解接触） | **有**（同时求解） |

这是 Isaac Sim 精度优势的核心——PhysX 的关节驱动力和接触力在同一个 TGS 迭代中求解。

```python
class PhysXEnv:
    def __init__(self, model_xml, pkl_dir, config=None):
        # ── PhysX 全局初始化 (进程级单例) ──
        self._ensure_foundation()   # 调用 px.init_foundation()（仅第一次）

        # ── 场景 ──
        self.scene = self._create_physx_scene()
        self.art = load_g1_from_mjcf(model_xml)
        self.ground_mat = px.create_material(
            static_friction=0.6, dynamic_friction=0.5, restitution=0.0)
        self.ground = px.create_plane_shape()
        self.scene.add_articulation(self.art)
        self.scene.add_actor(self.ground, self.ground_mat)

        # ── 仿真参数 ──
        self.native_dt = 0.002; self.decimation = 10; self.ctrl_dt = 0.020

        # ── 动作空间 (同 MuJoCoEnv——硬编码 Isaac WBC config) ──
        self.nu = 29
        self.jm = np.array([...])   # 完全相同
        self.jh = np.array([...])

        # ── PD 驱动 ──
        for i in range(29):
            kp = 100.0; kd = 5.0
            self.art.set_joint_drive_params(i, kp, kd, force_limit=torque_limit[i])

        # ── Body 索引 ──
        self._body_idx = {n: self.art.get_link_index(n) for n in BODY_NAMES}
        ...

    @staticmethod
    def _ensure_foundation():
        """PxFoundation 进程内只能创建一次."""
        if not hasattr(PhysXEnv, '_foundation_initialized'):
            px.init_foundation()
            PhysXEnv._foundation_initialized = True

    def _pd_control(self, action):
        target = action * self.jh + self.jm
        self.art.set_joint_drive_targets(target)

    def _physics_step(self):
        for _ in range(self.decimation):
            self.scene.simulate(self.native_dt)
            self.scene.fetch_results()
```

### 5.4 不变的部分（直接移植）

| 函数 | 行数 | 改动 |
|---|---|---|
| `_load_motions()` | 20 | 无 |
| `_sample_motion()` | 10 | 无 |
| `_advance_motion_time()` | 1 | 无 |
| `_future_dof()` | 5 | 无 |
| `_future_dof_vel()` | 6 | 无 |
| `_compute_actor_obs()` | 8 | `data.qpos[7:]` → `art.get_joint_positions()` |
| `_compute_critic_obs()` | 25 | body 查询替换 |
| `_build_tokenizer()` | 40 | body 查询替换；⚠️ 去掉 PKL quat → MuJoCo quat 转换 |
| `_compute_reward()` (12+1项) | 55 | body 查询替换 |
| `_undesired_contact()` | 15 | `data.contact` → `scene.get_contacts()` |
| `_anti_shake()` | 10 | body 速度查询替换 |
| `_vr_local_error()` | 18 | body 查询替换 |
| `_feet_acc()` | 12 | `qvel` 查询替换 |
| `_check_termination()` | 35 | body 查询替换；⚠️ PhyX 下收紧阈值到 Isaac 原始值 |
| `reset()` | 25 | `mj_forward` → set pose + Python FK |
| `step()` | 22 | 无（相同编排） |
| `mujoco_math.py` (7个 quat 函数) | 64 | **零改动**（C++ 侧统一为 `[w,x,y,z]` 后直接兼容） |

**约 280 行 MDP 逻辑直接移植，约 60 行物理 API 调用替换。**

### 5.5 FK 参考模型（⚠️ 修正：不能用 simulate(0)）

PhysX 没有 `mj_kinematics` 的等价物。`simulate(0)` 在 PhysX 中行为未定义，不能依赖。

**推荐方案：Python 前向运动学**

G1 只有 29 个 revolute 关节 + 1 个浮基，运动学链是固定的。用 Python 实现 FK——给定 `(root_pos, root_quat, 29 joint_angles)` → 所有 link 的世界位姿。

```python
# gear_sonic/envs/physx_fk.py (~150 行)

def build_fk_chain(xml_path: str):
    """从 MJCF XML 构建 FK 链——每个 link 的相对位姿和父索引."""
    # 解析 XML → 提取每个 body 的 local pos/quat 和其 parent
    # 返回: List[{"name": str, "parent": int, "pos": [3], "quat": [4]}]
    ...

class G1ForwardKinematics:
    """G1 前向运动学（纯 Python，不依赖任何物理引擎）."""

    def __init__(self, xml_path: str):
        self.chain = build_fk_chain(xml_path)       # ~30 links
        self._link_name_to_idx = {l["name"]: i for i, l in enumerate(self.chain)}

    def compute(self, root_pos, root_quat, joint_angles):
        """返回所有 link 的世界位姿 [(pos[3], quat[4]), ...].

        Args:
            root_pos:   浮基位置 [x, y, z]     — 来自 PKL root_trans_offset
            root_quat:  浮基朝向 [w, x, y, z]  — 来自 PKL root_rot（已转约定）
            joint_angles: 29D 关节角            — 来自 PKL dof
        """
        # 从 pelvis 开始，沿父链逐个计算 world transform
        # T_world_i = T_world_parent * T_local_i * R_joint_i
        poses = [(root_pos, root_quat)]  # pelvis
        for link in self.chain[1:]:  # skip root
            parent_pos, parent_quat = poses[link["parent"]]
            # 累积变换
            ...
        return poses

    def get_link_pose(self, poses, name):
        return poses[self._link_name_to_idx[name]]
```

**使用方式**（`PhysXEnv._compute_ref_body_state()` 的替代）：

```python
# 初始化
self._fk = G1ForwardKinematics(model_xml)

def _compute_ref_body_state(self):
    """用 Python FK 计算参考 motion 的 body 位姿."""
    idx = min(int(self._ref_time * self._ref_fps), len(self._ref_dof) - 1)
    ref_q = self._ref_dof[idx]                        # (29,) joint angles
    ref_root_pos = self._ref_root_trans[idx]           # (3,) from PKL
    pk_quat = self._ref_root_rot[idx]                  # [x,y,z,w] from PKL
    ref_root_quat = np.array([pk_quat[3], pk_quat[0], pk_quat[1], pk_quat[2]])  # → [w,x,y,z]

    self._ref_body_poses = self._fk.compute(ref_root_pos, ref_root_quat, ref_q)
    # 结果: [(pos3, quat4), ...] 按 BODY_NAMES 顺序
```

**优势**：
- 不依赖任何物理引擎，纯 Python，无副作用
- 和 MuJoCo 的 `mj_kinematics` 数学上等价（都是刚体 FK）
- 性能和 `mj_kinematics` 同级（~0.001ms）
- 可单元测试（给定已知关节角，手工验算特定 body 位姿）

**验收**：用已知的 qpos 在 MuJoCo 和 Python FK 上跑 100 帧，逐 body 对比位姿误差 < 1e-6。

### 5.6 接触查询

```python
def _get_contacts(self):
    """获取当前帧所有接触."""
    contacts = self.scene.get_contacts()
    # contacts: [(link_a_idx, link_b_idx, point[3], normal[3], force_mag), ...]
    return contacts
```

---

## 6. Phase 4：PhysXEnvManager — 多进程并行（第 3 周）

### 6.1 设计：独立文件，复用 SHM 模式（⚠️ 修正：不是"改一行"）

`MuJoCoEnvManager` (402行) 中 SHM 布局、Barrier 同步、`_to_numpy` 等都是物理无关的，但直接改 `_worker_loop` 里的 class import 是脆弱的。更好的做法：

**新文件 `gear_sonic/envs/physx_env_manager.py`**（~200 行），结构和 `MuJoCoEnvManager` 相同但引用 `PhysXEnv`：

```python
# gear_sonic/envs/physx_env_manager.py
"""PhysX parallel environment manager — same SHM+Barrier pattern as MuJoCoEnvManager."""

from gear_sonic.envs.mujoco_env_manager import (
    EnvSharedMemory, _to_numpy, OBS_DIM, ACT_DIM,
    OBS_BYTES, ACT_BYTES, REW_BYTES, DONE_BYTES, TIMEOUT_BYTES, ORIG_DONE_BYTES,
)
from gear_sonic.envs.physx_env import PhysXEnv

def _physx_worker_loop(worker_id, start_env, num_envs, shm_names, barrier,
                        model_xml, pkl_dir, env_config=None):
    """与 _worker_loop 结构相同，envs 列表用 PhysXEnv 而非 MuJoCoEnv."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    shm = EnvSharedMemory.attach(shm_names)
    # ... (SHM buffer 重建代码完全相同)
    envs = [PhysXEnv(model_xml, pkl_dir, config=env_config) for _ in range(num_envs)]
    # ... (循环逻辑完全相同)


class PhysXEnvManager:
    """与 MuJoCoEnvManager 相同接口，内部用 PhysXEnv + PhysX worker loop."""

    def __init__(self, num_envs, num_workers, model_xml, pkl_dir, env_config=None):
        # 结构完全复用 MuJoCoEnvManager.__init__:
        #   计算 env 分配 → 创建 SHM → 创建 Barrier → spawn _physx_worker_loop
        #   初始 barrier 同步 → 创建 _EnvStub
        ...

    # step(), reset(), reset_all(), close(), _handle_worker_crash()
    # 全部和 MuJoCoEnvManager 相同
```

**复用关系**：
- `EnvSharedMemory`、`_to_numpy`、全部常量 → 从 `mujoco_env_manager.py` import
- `_physx_worker_loop` → 和 `_worker_loop` 结构相同，只改 env 类名
- `PhysXEnvManager` 类 → 和 `MuJoCoEnvManager` 结构相同，改 worker target 函数

### 6.2 全局生命周期

PhysX 需要进程级初始化（`PxFoundation` 单例）。每个 worker 进程启动时调用一次 `PhysXEnv._ensure_foundation()`，由 `PhysXEnv.__init__` 自动处理。

`PxFoundation` 在 worker 进程退出时自动释放（进程结束时 OS 回收资源）。

### 6.3 训练入口

```python
# train_agent_trl.py
if os.environ.get("SONIC_PHYSX_ENV"):
    from gear_sonic.envs.physx_env_manager import PhysXEnvManager
    env = PhysXEnvManager(
        num_envs=local_envs,
        num_workers=getattr(config, "mujoco_workers", 160) // accelerator.num_processes,
        model_xml="/gear_sonic_deploy/g1/g1_29dof_v17.xml",  # ⚠️ 和 MuJoCo 训练同一个 XML!
        pkl_dir="/sample_data/robot_filtered",
        env_config=OmegaConf.create({"alive_bonus": 0.0, "ignore_terminations": True}),
    )
```

---

## 7. Phase 5：验证策略

### 7.1 Phase 1 验证：pybind11 smoke test

```bash
cd gear_sonic/envs/physx && make
python -c "import physx_core as px; px.init_foundation(); s = px.create_scene(); print('OK')"
```

### 7.2 Phase 2 验证：G1 模型加载 + 物理稳定性

```python
# scripts/verify_physx_model.py
art = load_g1_from_mjcf("g1_29dof_v17.xml")
assert art.num_joints == 29
for _ in range(1000):
    scene.simulate(0.002); scene.fetch_results()
print("PASS: 1000 physics steps")
```

### 7.3 Phase 2b 验证：Python FK 精度

```python
# scripts/verify_physx_fk.py
fk = G1ForwardKinematics("g1_29dof_v17.xml")
# 100 帧随机 qpos，对比 MuJoCo mj_kinematics 的输出
# 每帧 14 个 body 的 pos 误差 < 1e-6
```

### 7.4 Phase 3 验证：ref PD 精度（⚠️ 修正）

```python
# scripts/test_physx_ref_pd.py
env = PhysXEnv(model_xml, pkl_dir)
survivals = []
for ep in range(100):
    obs = env.reset()
    for step in range(500):
        # ⚠️ 修正：ref PD = 把参考 motion 关节角转为 action 空间
        ref_qpos = env.get_current_ref_qpos()     # (29,) 当前帧参考关节角
        action = (ref_qpos - env.jm) / env.jh     # 归一化到 [-1,1]
        _, _, done, _ = env.step(action)
        if done:
            break
    survivals.append(step + 1)

print(f"ref PD survival: {np.mean(survivals):.1f} ± {np.std(survivals):.1f}")

# 同时测 α (per-step drift)
# α = mean(|root_pos - ref_root_pos|) over 100 episodes, first 20 steps
```

**验收标准**：

| 指标 | MuJoCo 当前 | PhysX 目标 |
|---|---|---|
| ref PD α | 0.013 | **< 0.002** |
| ref PD 存活 (ANK=0.2) | 21 | **> 80** |
| ref PD 存活 (ANK=0.2, ORI=0.2) | — | **> 50**（Isaac 原始严格阈值） |

### 7.5 Phase 4 验证：训练 smoke test

```bash
SONIC_PHYSX_ENV=1 accelerate launch train_agent_trl.py +exp=stub_train \
  num_envs=64 headless=True algo.config.num_learning_iterations=100
```

**验收**：无 crash、reward 趋势可见、0 NaN。

---

## 8. 文件清单

| 文件 | 状态 | 估计行数 | 说明 |
|---|---|---|---|
| `gear_sonic/envs/physx/CMakeLists.txt` | **新建** | ~40 | aarch64 构建（5 个 PhysX 库） |
| `gear_sonic/envs/physx/physx_bindings.cpp` | **新建** | ~400 | pybind11 C++ 封装（含 quat 约定转换） |
| `gear_sonic/envs/physx/physx_loader.py` | **新建** | ~400 | MJCF→PhysX 转换器 |
| `gear_sonic/envs/physx/physx_fk.py` | **新建** | ~150 | Python FK（替代 mj_kinematics） |
| `gear_sonic/envs/physx_env.py` | **新建** | ~500 | 单 PhysX 环境 |
| `gear_sonic/envs/physx_env_manager.py` | **新建** | ~200 | 独立多进程管理器（复用 SHM 模式） |
| `gear_sonic/train_agent_trl.py` | **修改** | +10 | 加 `SONIC_PHYSX_ENV` 分支 |
| `scripts/verify_physx_model.py` | **新建** | ~60 | Phase 2 验证 |
| `scripts/verify_physx_fk.py` | **新建** | ~60 | Phase 2b FK 验证 |
| `scripts/test_physx_ref_pd.py` | **新建** | ~80 | Phase 3 验证 |

**总计**：~1900 行新代码 + 10 行修改。

---

## 9. 风险矩阵

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| aarch64 上 PhysX 5 库无法链接（缺依赖） | 中 | 高 | Phase 0 已编译通过；第一天验证链接 |
| Python FK 和 MuJoCo/Isaac FK 有数值偏差 | 低 | 中 | 100 帧对比验证；偏差 > 1e-6 则排查坐标系 |
| PhysX `PxArticulationDrive` 行为和 Isaac Sim 不一致 | 低 | 高 | 同 motion 逐关节对比力矩轨迹 |
| aarch64 PhysX 性能 < MuJoCo | 中 | 中 | Phase 3 benchmark，太慢减 env |
| 碰撞几何转换遗漏（mesh 碰撞 vs capsule 简化） | 中 | 中 | 提供 `--collision-mode` 切换 |
| 接触查询 API 和 MuJoCo 语义不同 | 中 | 低 | `r9` 是小项，可暂时禁用 |
| 容器缺少编译工具链 | 中 | 高 | 第一天确认 gcc/cmake/pybind11 可用性 |

---

## 10. 成功标准

**Phase 3 完成时**：
- [ ] G1 模型通过 MJCF 转换器在 PhysX 中加载
- [ ] Python FK 和 MuJoCo FK 逐帧位姿误差 < 1e-6
- [ ] 1000 步物理仿真：0 crash，0 NaN
- [ ] ref PD α < 0.002（Isaac Sim 同级）
- [ ] ref PD 存活 > 80 步

**Phase 4 完成时**：
- [ ] 64 env × 2 worker 训练：100 iter，0 crash
- [ ] Reward 趋势正向且上升
- [ ] 4096 env × 160 worker：不 OOM

**最终验收**：
- [ ] PhysX 上训练的 BC+PPO 策略，entropy 收敛，length > 60

---

## 附录 A：现有项目结构速查

| 文件 | 行数 | 角色 |
|---|---|---|
| `gear_sonic/envs/mujoco_env.py` | 520 | MuJoCo 单环境（模板） |
| `gear_sonic/envs/mujoco_env_manager.py` | 402 | 多进程管理（SHM 模式复用） |
| `gear_sonic/envs/mujoco_math.py` | 64 | quat 工具（`[w,x,y,z]` 约定） |
| `gear_sonic/train_agent_trl.py` | 622 | 训练入口 |
| `gear_sonic_deploy/g1/g1_29dof_v17.xml` | — | ⚠️ 训练实际用的 XML（服务器上） |
| `gear_sonic_deploy/g1/g1_29dof.xml` | 295 | 本地仓库版本（和 v17 可能不同） |
| `gear_sonic/config/exp/stub_train.yaml` | — | 训练配置 |

## 附录 B：容器环境

NPU 服务器 `113.46.41.54`，容器 `sonic-train` (aura-a3:latest)，openEuler aarch64，Kunpeng 920 320核。Python 3.11.15，PyTorch 2.7.1 + torch_npu。

Phase 1 前需确认/安装：`gcc`, `cmake`, `pybind11`（`pip install pybind11`），以及验证 PhysX 5 `.a`/`.so` 文件位置。

## 附录 C：本地 PhysX 源码

本地 `/Users/kevin/code/PhysX` 有 PhysX SDK 5.10.0 源码（今日 clone），含 `physx/`（C++ SDK）、`ovphysx/`（Python 绑定 v0.5.9）和示例代码。ovphysx pip wheel 在 aarch64 上缺 `libovstage.so`，但 `ovphysx/tests/python_samples/` 中的示例（`hello_world.py`、`contact_binding.py` 等）可作为 Phase 1 pybind11 封装的 API 参考。

## 附录 D：现有 pybind11 先例

`external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/` — 含 aarch64 分支的 CMakeLists.txt 和 `bindings/py_bindings.cpp`。PhysX 封装可参考其 CMake 结构（特别是 aarch64 的 include/lib 路径处理）。
