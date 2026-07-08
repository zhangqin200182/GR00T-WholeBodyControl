# MuJoCoEnv 完整设计

## 1. 定位与边界

### 1.1 MuJoCoEnv 在系统中的位置

```
┌─ NPU 服务器 ─────────────────────────────────────────────────────┐
│                                                                  │
│  MuJoCoEnvManager (trainer 进程)                                  │
│  ├─ SharedMemory × 6 (obs / terminal / actions / rewards / ...)  │
│  ├─ Barrier(N_workers + 1)                                       │
│  │                                                               │
│  └─ Worker 0..159 (每个 worker 一个进程, ~26 envs)                │
│       └─ MuJoCoEnv × 26  ← 本文档的设计对象                        │
│                                                                  │
│  PPO Trainer (16 × Ascend 910, DDP)                              │
│       └─ 从 SHM 读 obs/reward/done → GAE → PPO update             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**MuJoCoEnv 只负责一件事**：给定 action，执行物理仿真，产出训练信号（obs / reward / termination）。

它不关心：
- 多进程同步（由 MuJoCoEnvManager 的 Barrier 负责）
- 共享内存布局（EnvSharedMemory 负责序列化/反序列化）
- PPO / GAE / DDP（trainer 侧负责）

### 1.2 核心原理

```
每一步:
  action (29D float, [-1, 1])
    │
    ▼
  ┌─────────────────────────────────────┐
  │  MuJoCoEnv                          │
  │                                     │
  │  1. PD control: action → torque     │
  │  2. mj_step × N (physics)           │
  │  3. advance motion reference        │
  │  4. compute observations            │
  │  5. compute reward                  │
  │     (仿真状态 vs 参考轨迹的差异)       │
  │  6. check termination               │
  │     (差异超阈值则终止)                │
  │  7. auto-reset if done              │
  │                                     │
  │  → (obs, reward, done, info)        │
  └─────────────────────────────────────┘
```

评判逻辑的核心公式：
- **Reward**：`r_i = w_i · exp(-error²/σ_i²)` — Gaussian kernel，完美跟踪时 r → w_i
- **Termination**：`‖sim - ref‖ > threshold` — 任一条件触发则 terminated
- **Observation**：混合仿真状态（过去 10 帧）和参考轨迹（未来 10 帧）

---

## 2. 类结构与初始化

### 2.1 完整 API

```python
class MuJoCoEnv:
    # ── 生命周期 ──
    def __init__(self, model_xml: str, pkl_dir: str, config=None)
    def reset(self) -> dict[str, np.ndarray]
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]
    def close(self)  # 可选, 释放 MuJoCo 资源

    # ── 物理仿真 ──
    def _pd_control(self, action: np.ndarray) -> None
    def _physics_step(self) -> None

    # ── Motion Reference ──
    def _load_motions(self, pkl_dir: str) -> list[dict]
    def _sample_motion(self) -> None
    def _advance_motion_time(self) -> None
    def _compute_ref_body_state(self) -> None  # FK

    # ── Observation (4336D total) ──
    def _obs(self) -> dict[str, np.ndarray]
    def _compute_actor_obs(self) -> np.ndarray      # 930D
    def _compute_critic_obs(self) -> np.ndarray     # 1645D
    def _build_tokenizer(self) -> np.ndarray        # 1761D

    # ── Reward (12 项) ──
    def _compute_reward(self, action: np.ndarray) -> float

    # ── Termination (5 条件) ──
    def _check_termination(self) -> tuple[bool, bool]
```

### 2.2 `__init__` — 初始化

```python
def __init__(self, model_xml: str, pkl_dir: str, config=None):
    # ═══ MuJoCo 模型与数据 ═══
    self.model = mujoco.MjModel.from_xml_path(model_xml)
    self.data = mujoco.MjData(self.model)

    # 仿真用的模型(PD控制 + mj_step)
    # self.data 的 xpos/xquat/qpos/qvel 是机器人的"真实"物理状态

    # 运动学参考模型(纯 FK, 用于算参考 body 位姿)
    self._ref_model = mujoco.MjModel.from_xml_path(model_xml)
    self._ref_data = mujoco.MjData(self._ref_model)

    # ═══ 仿真参数 ═══
    self.native_dt = self.model.opt.timestep   # 0.002 (XML 中定义)
    self.decimation = 10                        # 每 ctrl_step 物理步数
    self.ctrl_dt = self.native_dt * self.decimation  # 0.020 = 50Hz

    # ═══ 动作空间 ═══
    self.nu = self.model.nu  # 29
    # 关节限位 (排除第一行 free joint)
    jr = self.model.jnt_range[1:]  # (29, 2)
    self.jm = (jr[:, 1] + jr[:, 0]) / 2  # 关节中点
    self.jh = (jr[:, 1] - jr[:, 0]) / 2  # 关节半范围

    # ═══ PD 控制器增益 ═══
    self.kp = np.ones(self.nu) * 30.0
    self.kd = np.ones(self.nu) * 3.0

    # ═══ Body 索引 (14 个 tracked body) ═══
    self._body_names = BODY_NAMES
    self._body_idx = {
        name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in BODY_NAMES
    }
    self._body_indices = np.array([self._body_idx[n] for n in BODY_NAMES])

    # ═══ Episode 管理 ═══
    self.ep = 0
    self.max_ep = getattr(config, "max_episode_length", 500) if config else 500

    # ═══ Motion Reference 数据 ═══
    self._load_motions(pkl_dir)

    # ═══ History Buffers (10 帧) ═══
    self._init_history_buffers()

    # ═══ 上一帧缓存 (用于速度/加速度计算) ═══
    self._prev_action = np.zeros(self.nu, dtype=np.float32)
    self._prev_root_pos = np.zeros(3, dtype=np.float64)
    self._prev_body_pos = np.zeros((14, 3), dtype=np.float64)
    self._prev_body_quat = np.zeros((14, 4), dtype=np.float64)
    self._prev_joint_vel = np.zeros(self.nu, dtype=np.float64)
```

### 2.3 History Buffer 布局

所有 history buffer 统一为 `(10, dim)` 形状，dim 0 是时间（0=最旧, 9=最新）。

```python
def _init_history_buffers(self):
    """Actor obs 用 (5 个 buffer)."""
    H = 10  # history_length, 来自 sonic_release.yaml
    self._gdh = np.zeros((H, 3), dtype=np.float32)    # gravity_dir_history
    self._avh = np.zeros((H, 3), dtype=np.float32)    # ang_vel_history
    self._jph = np.zeros((H, self.nu), dtype=np.float32)  # joint_pos_history
    self._jvh = np.zeros((H, self.nu), dtype=np.float32)  # joint_vel_history
    self._ah  = np.zeros((H, self.nu), dtype=np.float32)  # action_history

    """Critic obs 额外用 (1 个 buffer, 其余复用 actor 的)."""
    self._lvh = np.zeros((H, 3), dtype=np.float32)    # lin_vel_history
    # _avh, _jph, _jvh, _ah 与 actor 共享
```

### 2.4 Motion 数据加载

```python
def _load_motions(self, pkl_dir: str):
    """从 PKL 文件加载 motion 参考数据.

    每个 motion 是一个 dict:
      {"dof": np.ndarray (T, 29)}  — 当前可用
      // 未来可能扩展: body_pos, body_quat, root_pos, root_quat
    缺少的 body 级数据通过 FK 实时计算。
    """
    self.motions = []
    for p in sorted(glob.glob(os.path.join(pkl_dir, "**/*.pkl"), recursive=True)):
        if os.path.basename(p).startswith("._"):
            continue
        data = joblib.load(p)
        for v in data.values():
            if isinstance(v, dict) and "dof" in v:
                self.motions.append(v)

    if len(self.motions) == 0:
        raise RuntimeError(f"No valid motion PKL files found in {pkl_dir}")
```

**当前 PKL 只有 dof 字段。** Body 位姿（body_pos, body_quat, root_pos, root_quat）通过 `_compute_ref_body_state()` 用 MuJoCo FK 从 dof 实时计算（见第 3.3 节）。

---

## 3. 物理仿真

### 3.1 PD 控制

SONIC 的 action 空间是 `[-1, 1]` 的 normalized joint offset。MuJoCo 需要 torque 控制。使用 PD 控制器桥接：

```python
def _pd_control(self, action: np.ndarray):
    """将 normalized action → target joint position → PD torque."""
    # action ∈ [-1, 1] → target joint position
    target = action * self.jh + self.jm  # shape (29,)

    # PD: torque = kp * (target - current) - kd * current_velocity
    torque = (self.kp * (target - self.data.qpos[7:])
              - self.kd * self.data.qvel[6:])
    self.data.ctrl[:] = np.clip(torque, -50, 50)
```

关键参数：
- `kp=30, kd=3`：经验值，使跟踪带宽约 5-10 Hz
- `ctrl` clamp ±50：防止数值爆炸（与 Isaac Sim 的 `actuatorfrcrange` 一致）
- `qpos[7:]` / `qvel[6:]`：排除 free joint 的 7 维 pos / 6 维 vel

### 3.2 物理步进

```python
def _physics_step(self):
    """执行 decimation 次 mj_step."""
    for _ in range(self.decimation):  # 10 次
        mujoco.mj_step(self.model, self.data)
```

- `native_dt = 0.002` (XML `timestep`)
- `decimation = 10` → `ctrl_dt = 0.020s = 50Hz`
- Isaac Sim 是 `dt=0.005 × 4 = 0.020s`，两种方式在控制频率上等效

### 3.3 Motion Reference 管理

```python
def _sample_motion(self):
    """随机采样一个 motion clip 和起始帧."""
    m = self.motions[np.random.randint(len(self.motions))]
    self._ref_dof = m["dof"]  # (T, 29)
    self._ref_start = np.random.randint(0, max(1, len(self._ref_dof) - self.max_ep))
    self._ref_idx = self._ref_start

def _advance_motion_time(self):
    """推进 reference 帧索引."""
    self._ref_idx += 1

def _get_current_ref_frame(self):
    """获取当前帧的 reference joint position."""
    idx = min(self._ref_idx, len(self._ref_dof) - 1)
    return self._ref_dof[idx]

def _future_dof_pos(self, n_future=10):
    """获取未来 n_future 帧的 reference joint position."""
    idx = self._ref_idx
    n = len(self._ref_dof)
    indices = np.clip(np.arange(idx, idx + n_future), 0, n - 1)
    return self._ref_dof[indices].astype(np.float32)  # (n_future, 29)

def _future_dof_vel(self, n_future=10):
    """获取未来 n_future 帧的 reference joint velocity (差分)."""
    dof = self._ref_dof
    idx = self._ref_idx
    n = len(dof)
    # 用当前帧和后一帧的差分算速度
    t0 = np.clip(np.arange(idx, idx + n_future), 0, n - 2)
    t1 = np.clip(np.arange(idx + 1, idx + n_future + 1), 1, n - 1)
    vel = (dof[t1] - dof[t0]) / self.ctrl_dt
    return vel.astype(np.float32)
```

### 3.4 Reference Body 位姿 (FK)

这是整个设计的关键——从 dof reference 算出 body pos/quat。

```python
def _compute_ref_body_state(self):
    """用 MuJoCo FK 从 reference joint pos 计算参考 body 位姿.

    将 self._ref_data 的关节角设为 reference 值，做 mj_kinematics，
    然后从 self._ref_data.xpos/xquat 读取 body 位姿。
    """
    ref_joint_pos = self._get_current_ref_frame()  # (29,)
    self._ref_data.qpos[7:] = ref_joint_pos
    # Free joint: 从仿真 data 复制 root pos/quat (保持一致的全局坐标系)
    self._ref_data.qpos[:7] = self.data.qpos[:7]
    self._ref_data.qvel[:] = 0
    mujoco.mj_kinematics(self._ref_model, self._ref_data)
    # 现在 self._ref_data.xpos[body_idx] 和 self._ref_data.xquat[body_idx]
    # 包含参考姿态下各 body 的世界位姿

def _get_ref_root_pos(self):
    self._compute_ref_body_state()
    return self._ref_data.xpos[self._body_idx["pelvis"]].copy()

def _get_ref_root_quat(self):
    self._compute_ref_body_state()
    return self._ref_data.xquat[self._body_idx["pelvis"]].copy()

def _get_ref_body_pos(self):
    """返回 14 个 body 的参考位置 (14, 3)."""
    self._compute_ref_body_state()
    return self._ref_data.xpos[self._body_indices].copy()

def _get_ref_body_quat(self):
    """返回 14 个 body 的参考朝向 (14, 4)."""
    self._compute_ref_body_state()
    return self._ref_data.xquat[self._body_indices].copy()
```

**为什么需要两个 model/data？**
- `self.data`：PD 控制 → `mj_step` → 机器人"真实"物理状态
- `self._ref_data`：dof reference → `mj_kinematics` → 参考姿态的 body 位姿

两者物理状态不同（前者有动力学误差，后者是运动学上的完美姿态），需要独立实例。

**性能注意**：`_compute_ref_body_state()` 每 step 调用一次（被 reward、termination、critic_obs、tokenizer 共用），耗时约 0.05ms。

### 3.5 Future Reference Body 位姿

Critic_obs 和 tokenizer 需要未来 10 帧的 body 参考数据。实现方式：

```python
def _future_ref_body_pos(self, n_future=10):
    """未来 n_future 帧的参考 body 位置 (n_future, 14, 3)."""
    idx = self._ref_idx
    n = len(self._ref_dof)
    results = []
    for i in range(n_future):
        t = min(idx + i, n - 1)
        self._ref_data.qpos[7:] = self._ref_dof[t]
        self._ref_data.qpos[:7] = self.data.qpos[:7]
        self._ref_data.qvel[:] = 0
        mujoco.mj_kinematics(self._ref_model, self._ref_data)
        results.append(self._ref_data.xpos[self._body_indices].copy())
    return np.stack(results, axis=0)

def _future_ref_root_pos(self, n_future=10):
    """未来 n_future 帧的参考根位置 (n_future, 3)."""
    body_pos = self._future_ref_body_pos(n_future)  # (n_future, 14, 3)
    pelvis_idx = list(self._body_names).index("pelvis")
    return body_pos[:, pelvis_idx, :]

def _future_ref_root_quat(self, n_future=10):
    """未来 n_future 帧的参考根朝向 (n_future, 4)."""
    idx = self._ref_idx
    n = len(self._ref_dof)
    results = []
    for i in range(n_future):
        t = min(idx + i, n - 1)
        self._ref_data.qpos[7:] = self._ref_dof[t]
        self._ref_data.qpos[:7] = self.data.qpos[:7]
        self._ref_data.qvel[:] = 0
        mujoco.mj_kinematics(self._ref_model, self._ref_data)
        results.append(self._ref_data.xquat[self._body_idx["pelvis"]].copy())
    return np.stack(results, axis=0)
```

**性能注意**：`_future_ref_body_pos` 做 10 次 FK loop（每次 0.005ms），总共约 0.05ms。可以和当前帧的 FK 合并优化（一次计算 11 帧），但作为 v1 实现可以接受。

---

## 4. Observation 计算

### 4.1 整体结构

```python
def _obs(self) -> dict[str, np.ndarray]:
    return {
        "actor_obs": self._compute_actor_obs(),    # 930D
        "critic_obs": self._compute_critic_obs(),   # 1645D
        "tokenizer": self._build_tokenizer(),        # 1761D
    }
```

总计 4336D。三种 observation 的构成逻辑：
- **Actor obs (930D)**：全来自仿真状态（policy 在真机上能观测到的 proprioception）
- **Critic obs (1645D)**：仿真状态 + 参考轨迹 + 参考 body 位姿（训练时的特权信息）
- **Tokenizer obs (1761D)**：参考轨迹 + body 位姿 + 少量仿真状态（给 encoder 的输入）

### 4.2 Actor Obs (930D)

```
gravity_dir_history:  10 × 3   = 30
ang_vel_history:      10 × 3   = 30
joint_pos_history:    10 × 29  = 290
joint_vel_history:    10 × 29  = 290
action_history:       10 × 29  = 290
──────────────────────────────────
Total: 930
```

```python
def _compute_actor_obs(self) -> np.ndarray:
    # 1. Gravity direction in robot body frame
    root_quat = self.data.xquat[self._body_idx["pelvis"]]
    g_body = quat_apply(quat_inv(root_quat), np.array([0, 0, -1]))

    # 2. Angular velocity (directly from MuJoCo qvel)
    ang_vel = self.data.qvel[3:6].copy()

    # 3. Joint state (from MuJoCo physics)
    joint_pos = self.data.qpos[7:].copy()
    joint_vel = self.data.qvel[6:].copy()

    # 4. Shift histories (left-shift, insert new at end)
    self._gdh[:-1] = self._gdh[1:]; self._gdh[-1] = g_body
    self._avh[:-1] = self._avh[1:]; self._avh[-1] = ang_vel
    self._jph[:-1] = self._jph[1:]; self._jph[-1] = joint_pos
    self._jvh[:-1] = self._jvh[1:]; self._jvh[-1] = joint_vel
    # _ah is shifted in step() when action is known

    return np.concatenate([
        self._gdh.flatten(),
        self._avh.flatten(),
        self._jph.flatten(),
        self._jvh.flatten(),
        self._ah.flatten(),
    ]).astype(np.float32)
```

### 4.3 Critic Obs (1645D)

```
command_multi_future:    10×(29+29) = 580
motion_anchor_pos_b:     1×3        = 3
motion_anchor_ori_b:     1×6        = 6
body_pos_b:              14×3       = 42
body_ori_b:              14×6       = 84
lin_vel_history:         10×3       = 30
ang_vel_history:         10×3       = 30
joint_pos_history:       10×29      = 290
joint_vel_history:       10×29      = 290
action_history:          10×29      = 290
────────────────────────────────────────
Total: 1645
```

```python
def _compute_critic_obs(self) -> np.ndarray:
    root_pos = self.data.xpos[self._body_idx["pelvis"]]
    root_quat = self.data.xquat[self._body_idx["pelvis"]]
    ref_root_pos = self._get_ref_root_pos()
    ref_root_quat = self._get_ref_root_quat()

    # 1. command_multi_future (580D)
    future_pos = self._future_dof_pos()           # (10, 29)
    future_vel = self._future_dof_vel()            # (10, 29)
    command_mf = np.concatenate([future_pos.flatten(), future_vel.flatten()])

    # 2. motion_anchor_pos_b (3D): ref root pos in robot root frame
    pos_b, _ = subtract_frame_transforms(root_pos, root_quat, ref_root_pos, ref_root_quat)

    # 3. motion_anchor_ori_b (6D): ref root ori in robot root frame (6D rotation)
    _, ori_b = subtract_frame_transforms(root_pos, root_quat, ref_root_pos, ref_root_quat)
    mat = quat_to_matrix(ori_b)
    ori_6d = mat[..., :2].flatten()

    # 4. body_pos_b (42D): 14 body positions in robot root frame
    body_pos_w = self.data.xpos[self._body_indices]
    body_pos_b = quat_apply(quat_inv(root_quat), body_pos_w - root_pos)

    # 5. body_ori_b (84D): 14 body orientations in robot root frame (6D)
    body_quat_w = self.data.xquat[self._body_indices]
    body_quat_b = quat_mul(quat_inv(root_quat[None, :].repeat(14, axis=0)), body_quat_w)
    body_mat = np.stack([quat_to_matrix(q) for q in body_quat_b])
    body_ori_flat = body_mat[..., :2].reshape(-1)

    # 6. lin_vel_history (30D)
    lin_vel = (root_pos - self._prev_root_pos) / self.ctrl_dt
    self._lvh[:-1] = self._lvh[1:]; self._lvh[-1] = lin_vel
    self._prev_root_pos = root_pos.copy()

    # 7-10. reuse actor history buffers
    return np.concatenate([
        command_mf,            # 580
        pos_b.flatten(),       # 3
        ori_6d,                # 6
        body_pos_b.flatten(),  # 42
        body_ori_flat,         # 84
        self._lvh.flatten(),   # 30
        self._avh.flatten(),   # 30
        self._jph.flatten(),   # 290
        self._jvh.flatten(),   # 290
        self._ah.flatten(),    # 290
    ]).astype(np.float32)
```

### 4.4 Tokenizer Obs (1761D)

12 个子项。大部分数据来自 motion reference（可从 PKL dof + FK 计算），少量需要 robot root state（来自 MuJoCo physics）。

```python
# 常量
LOWER_JOINT_INDICES = list(range(12))  # 下半身 12 个关节
VR_3POINT_BODY = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
VR_3POINT_BODY_OFFSET = np.array([
    [0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]
], dtype=np.float32)

def _build_tokenizer(self) -> np.ndarray:
    N_FUTURE = 10
    N_SMPL_FUTURE = 10

    root_quat = self.data.xquat[self._body_idx["pelvis"]]

    # 1. encoder_index (3D) — one-hot, 当前训练数据无 SMPL 所以固定 g1
    encoder_index = np.zeros(3, dtype=np.float32)
    encoder_index[0] = 1.0  # g1 encoder

    # 2. command_multi_future_nonflat (580D): 未来 10 帧 pos+vel
    future_pos = self._future_dof_pos(N_FUTURE)     # (10, 29)
    future_vel = self._future_dof_vel(N_FUTURE)      # (10, 29)
    cmd_nonflat = np.concatenate([future_pos, future_vel], axis=-1).flatten()

    # 3. command_z_multi_future_nonflat (10D): 未来 10 帧根高度
    future_root_pos = self._future_ref_root_pos(N_FUTURE)  # (10, 3)
    cmd_z_mf = future_root_pos[:, 2:3].flatten()

    # 4. motion_anchor_ori_b_mf_nonflat (60D): 未来 10 帧根朝向 6D
    future_root_quat = self._future_ref_root_quat(N_FUTURE)  # (10, 4)
    root_q_expanded = np.tile(root_quat, (N_FUTURE, 1))       # (10, 4)
    rot_diff = quat_mul(quat_inv(root_q_expanded), future_root_quat)
    mats = np.stack([quat_to_matrix(q) for q in rot_diff])     # (10, 3, 3)
    ori_mf = mats[..., :2].reshape(-1)                         # 60

    # 5. command_multi_future_lower_body (240D): 未来 10 帧下半身 pos+vel
    lower_pos = future_pos[:, LOWER_JOINT_INDICES]   # (10, 12)
    lower_vel = future_vel[:, LOWER_JOINT_INDICES]   # (10, 12)
    cmd_lower = np.concatenate([lower_pos.flatten(), lower_vel.flatten()])

    # 6. vr_3point_local_target (9D)
    vr_indices = [self._body_idx[n] for n in VR_3POINT_BODY]
    ref_body_pos = self._get_ref_body_pos()                  # (14, 3)
    ref_body_quat = self._get_ref_body_quat()                # (14, 4)
    ref_root_pos = ref_body_pos[self._body_idx["pelvis"]]
    ref_root_quat_full = ref_body_quat[self._body_idx["pelvis"]]
    # 3 VR bodies
    vr_body_pos = ref_body_pos[[
        list(self._body_names).index(n) for n in VR_3POINT_BODY
    ]]  # (3, 3)
    vr_body_quat = ref_body_quat[[
        list(self._body_names).index(n) for n in VR_3POINT_BODY
    ]]  # (3, 4)
    # Apply body offsets
    vr_pos_w = vr_body_pos + quat_apply(vr_body_quat, VR_3POINT_BODY_OFFSET)
    ref_root_q_3p = np.tile(ref_root_quat_full, (3, 1))
    vr_diff = vr_pos_w - ref_root_pos
    vr_3point_local = quat_apply(quat_inv(ref_root_q_3p), vr_diff).flatten()

    # 7. vr_3point_local_orn_target (12D)
    ref_3point_quat_local = quat_mul(quat_inv(ref_root_q_3p), vr_body_quat)
    vr_3point_orn = ref_3point_quat_local.flatten()

    # 8. motion_anchor_ori_b (6D): 当前帧参考根在 robot 坐标系中的朝向
    _, ori_b = subtract_frame_transforms(
        self.data.xpos[self._body_idx["pelvis"]], root_quat,
        ref_root_pos, ref_root_quat_full,
    )
    mat_single = quat_to_matrix(ori_b)
    ori_b_6d = mat_single[..., :2].flatten()

    # 9. command_z (1D): 当前帧参考根高度
    cmd_z = np.array([ref_root_pos[2]], dtype=np.float32)

    # 10-12. SMPL (全零 — 训练数据无 SMPL)
    smpl_joints = np.zeros(N_SMPL_FUTURE * 24 * 3, dtype=np.float32)  # 720
    smpl_root_ori = np.zeros(N_SMPL_FUTURE * 6, dtype=np.float32)     # 60
    smpl_wrist = np.zeros(N_SMPL_FUTURE * 6, dtype=np.float32)        # 60

    return np.concatenate([
        encoder_index,       # 3
        cmd_nonflat,         # 580
        cmd_z_mf,            # 10
        ori_mf,              # 60
        cmd_lower,           # 240
        vr_3point_local,     # 9
        vr_3point_orn,       # 12
        ori_b_6d,            # 6
        cmd_z,               # 1
        smpl_joints,         # 720
        smpl_root_ori,       # 60
        smpl_wrist,          # 60
    ]).astype(np.float32)  # Total: 1761
```

---

## 5. Reward 计算

### 5.1 总体公式

$$r = \sum_{i=1}^{12} w_i \cdot r_i$$

Tracking reward 使用 Gaussian kernel：$r_i = \exp(-error^2 / \sigma^2) \in [0, 1]$

Penalty 项直接计算 L2 或 relu 后的值。

### 5.2 配置 (来自 sonic_release.yaml + 各 term YAML)

| # | 项 | 权重 | σ | 范围 |
|---|-----|------|---|------|
| 1 | tracking_anchor_pos | 0.5 | 0.3 | [0, 0.5] |
| 2 | tracking_anchor_ori | 0.5 | 0.4 | [0, 0.5] |
| 3 | tracking_relative_body_pos | 1.0 | 0.3 | [0, 1.0] |
| 4 | tracking_relative_body_ori | 1.0 | 0.4 | [0, 1.0] |
| 5 | tracking_body_linvel | 1.0 | 1.0 | [0, 1.0] |
| 6 | tracking_body_angvel | 1.0 | 3.14 | [0, 1.0] |
| 7 | action_rate_l2 | -0.1 | — | (-∞, 0] |
| 8 | joint_limit | -10.0 | — | (-∞, 0] |
| 9 | undesired_contacts | -0.1 | — | (-∞, 0] |
| 10 | anti_shake_ang_vel | -0.005 | — | (-∞, 0] |
| 11 | tracking_vr_5point_local | 2.0 | 0.1 | [0, 2.0] |
| 12 | feet_acc | -2.5e-6 | — | (-∞, 0] |

> 注：sonic_release.yaml 的 `reward_point_body` 实际只有 3 个点 (torso, left_wrist, right_wrist)，与 reward 函数名 `tracking_local_vr_5point_error` 不一致。按 YAML override 的实际值用 3 点实现。

### 5.3 完整实现

```python
def _compute_reward(self, action: np.ndarray) -> float:
    # 确保 body 参考数据已计算
    # (如果前面 _obs() 已经调过 FK, 这里可以读取缓存)

    root_pos = self.data.xpos[self._body_idx["pelvis"]]
    root_quat = self.data.xquat[self._body_idx["pelvis"]]
    ref_root_pos = self._get_ref_root_pos()
    ref_root_quat = self._get_ref_root_quat()

    # ── 项 1: tracking_anchor_pos (w=0.5, σ=0.3) ──
    err = np.linalg.norm(root_pos - ref_root_pos)
    r1 = 0.5 * np.exp(-err**2 / 0.09)  # 0.09 = 0.3²

    # ── 项 2: tracking_anchor_ori (w=0.5, σ=0.4) ──
    angular_err = quat_error_magnitude(ref_root_quat, root_quat)
    r2 = 0.5 * np.exp(-angular_err**2 / 0.16)

    # ── 项 3: tracking_relative_body_pos (w=1.0, σ=0.3) ──
    ref_body_pos = self._get_ref_body_pos()
    # anchor-relative: 将参考 body 位置平移到与机器人共享 root 位置
    ref_body_pos_aligned = ref_body_pos - ref_root_pos + root_pos
    body_pos_w = self.data.xpos[self._body_indices]
    per_body_err = np.sum((body_pos_w - ref_body_pos_aligned)**2, axis=-1)
    r3 = 1.0 * np.exp(-per_body_err.mean() / 0.09)

    # ── 项 4: tracking_relative_body_ori (w=1.0, σ=0.4) ──
    ref_body_quat = self._get_ref_body_quat()
    body_quat_w = self.data.xquat[self._body_indices]
    angular_errs = np.array([
        quat_error_magnitude(ref_body_quat[i], body_quat_w[i])
        for i in range(14)
    ])
    r4 = 1.0 * np.exp(-(angular_errs**2).mean() / 0.16)

    # ── 项 5: tracking_body_linvel (w=1.0, σ=1.0) ──
    body_lin_vel = (body_pos_w - self._prev_body_pos) / self.ctrl_dt
    ref_body_lin_vel = (ref_body_pos - self._prev_ref_body_pos) / self.ctrl_dt \
        if hasattr(self, '_prev_ref_body_pos') else np.zeros_like(body_lin_vel)
    vel_diff = body_lin_vel - ref_body_lin_vel
    per_body_err = np.sum(vel_diff**2, axis=-1)
    r5 = 1.0 * np.exp(-per_body_err.mean() / 1.0)
    self._prev_body_pos = body_pos_w.copy()
    self._prev_ref_body_pos = ref_body_pos.copy()

    # ── 项 6: tracking_body_angvel (w=1.0, σ=3.14) ──
    # 从朝向差分反算角速度
    body_ang_vel = _quat_diff_to_angvel(self._prev_body_quat, body_quat_w, self.ctrl_dt)
    ref_body_ang_vel = _quat_diff_to_angvel(
        self._prev_ref_body_quat, ref_body_quat, self.ctrl_dt
    ) if hasattr(self, '_prev_ref_body_quat') else np.zeros_like(body_ang_vel)
    vel_diff = body_ang_vel - ref_body_ang_vel
    per_body_err = np.sum(vel_diff**2, axis=-1)
    r6 = 1.0 * np.exp(-per_body_err.mean() / 9.86)  # 3.14²
    self._prev_body_quat = body_quat_w.copy()
    self._prev_ref_body_quat = ref_body_quat.copy()

    # ── 项 7: action_rate_l2 (w=-0.1) ──
    r7 = -0.1 * np.sum((action - self._prev_action)**2)
    self._prev_action = action.copy()

    # ── 项 8: joint_limit (w=-10.0) ──
    q = self.data.qpos[7:]
    excess = np.maximum(np.abs(q - self.jm) - self.jh, 0)
    r8 = -10.0 * np.sum(excess)

    # ── 项 9: undesired_contacts (w=-0.1, threshold=1.0) ──
    r9 = -0.1 * self._compute_undesired_contact_force()

    # ── 项 10: anti_shake_ang_vel (w=-0.005, threshold=1.5) ──
    r10 = -0.005 * self._compute_anti_shake()

    # ── 项 11: tracking_vr_5point_local (w=2.0, σ=0.1) ──
    r11 = 2.0 * self._compute_vr_local_error()

    # ── 项 12: feet_acc (w=-2.5e-6) ──
    r12 = -2.5e-6 * self._compute_feet_acc()

    return float(r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12)
```

### 5.4 Penalty 辅助方法

```python
def _compute_undesired_contact_force(self) -> float:
    """计算非预期接触力 (排除 ankle, wrist, elbow)."""
    excluded = {"left_ankle_roll_link", "right_ankle_roll_link",
                "left_wrist_yaw_link", "right_wrist_yaw_link",
                "left_elbow_link", "right_elbow_link"}
    total_force = 0.0
    for i in range(self.data.ncon):
        contact = self.data.contact[i]
        b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                               self.model.geom_bodyid[contact.geom1])
        b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                               self.model.geom_bodyid[contact.geom2])
        if b1 not in excluded and b2 not in excluded:
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            total_force += np.linalg.norm(force[:3])
    return max(total_force - 1.0, 0.0)

def _compute_anti_shake(self) -> float:
    """计算腕+头角速度超 threshold 的 penalty."""
    target = ["left_wrist_yaw_link", "right_wrist_yaw_link", "head_link"]
    excesses = []
    for name in target:
        idx = self._body_idx.get(name)
        if idx is None:
            continue
        dof_adr = self.model.body_dofadr[idx]
        # cvel 每个 body 占 6 个元素: [linvel_x, linvel_y, linvel_z, angvel_x, angvel_y, angvel_z]
        w = self.data.cvel[dof_adr + 3 : dof_adr + 6]
        speed = np.linalg.norm(w)
        excesses.append(max(speed - 1.5, 0))
    return np.mean(np.array(excesses)**2) if excesses else 0.0

def _compute_vr_local_error(self) -> float:
    """计算 VR 3 点局部坐标跟踪 error (用于 Gaussian kernel)."""
    # reward_point_body: ["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    # reward_point_body_offset: [[0, 0, 0.5], [0, 0, 0], [0, 0, 0]]
    point_bodies = ["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    point_offsets = np.array([[0, 0, 0.5], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
    n_pts = len(point_bodies)

    ref_root_pos = self._get_ref_root_pos()
    ref_root_quat = self._get_ref_root_quat()
    ref_body_pos = self._get_ref_body_pos()
    ref_body_quat = self._get_ref_body_quat()

    # Reference VR points in ref root local frame
    ref_pt_indices = [list(self._body_names).index(n) for n in point_bodies]
    ref_pt_pos_w = ref_body_pos[ref_pt_indices]
    ref_pt_quat_w = ref_body_quat[ref_pt_indices]
    ref_pt_w = ref_pt_pos_w + quat_apply(ref_pt_quat_w, point_offsets)
    ref_root_q_tiled = np.tile(ref_root_quat, (n_pts, 1))
    ref_pt_local = quat_apply(quat_inv(ref_root_q_tiled), ref_pt_w - ref_root_pos)

    # Robot VR points in robot root local frame
    root_pos = self.data.xpos[self._body_idx["pelvis"]]
    root_quat = self.data.xquat[self._body_idx["pelvis"]]
    robot_pt_pos = self.data.xpos[[self._body_idx[n] for n in point_bodies]]
    robot_pt_quat = self.data.xquat[[self._body_idx[n] for n in point_bodies]]
    robot_pt_w = robot_pt_pos + quat_apply(robot_pt_quat, point_offsets)
    robot_root_q_tiled = np.tile(root_quat, (n_pts, 1))
    robot_pt_local = quat_apply(quat_inv(robot_root_q_tiled), robot_pt_w - root_pos)

    error = np.sum((robot_pt_local - ref_pt_local)**2)
    # Gaussian: r = exp(-error.mean() / σ²)
    return float(np.exp(-error / (n_pts * 0.01)))  # σ=0.1, σ²=0.01

def _compute_feet_acc(self) -> float:
    """计算 ankle 关节加速度 L2."""
    ankle_dof_indices = []
    for i, name in enumerate(
        ["left_ankle_pitch_joint", "left_ankle_roll_joint",
         "right_ankle_pitch_joint", "right_ankle_roll_joint"]
    ):
        try:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            # qvel 中第一个非 free joint 的 dof 地址是 6
            dof_idx = self.model.jnt_dofadr[jid]
            if dof_idx >= 0:
                ankle_dof_indices.append(dof_idx - 6)  # 减去 free joint 的 6 维
        except Exception:
            pass
    if not ankle_dof_indices:
        return 0.0
    joint_vel = self.data.qvel[6:]
    ankle_vel = joint_vel[ankle_dof_indices]
    prev_ankle_vel = self._prev_joint_vel[ankle_dof_indices]
    acc = (ankle_vel - prev_ankle_vel) / self.ctrl_dt
    self._prev_joint_vel = joint_vel.copy()
    return float(np.sum(acc**2))
```

### 5.5 Reward 分布预期

| 场景 | 预期 r |
|------|--------|
| 完美跟踪 (qpos == ref) | 5.0 - 7.0 |
| 正常运动跟踪 (有轻微误差) | 2.0 - 5.0 |
| 随机动作 | -2.0 - 0 |
| 摔倒/崩溃 | < -5.0 |

---

## 6. Termination 计算

### 6.1 配置 (来自 base_adaptive_strict_ori_foot_xyz.yaml, 被 sonic_release.yaml override)

| # | 条件 | 阈值 | 类型 |
|---|------|------|------|
| 1 | exceeded_anchor_height | 0.15m / 0.75m (自适应) | terminated |
| 2 | exceeded_anchor_ori | 0.2 rad² | terminated |
| 3 | exceeded_body_height | 0.15m / 0.75m (自适应) | terminated |
| 4 | exceeded_body_pos | 0.2m | terminated |
| 5 | tracking_time_out | — | truncated |

### 6.2 完整实现

```python
def _check_termination(self) -> tuple[bool, bool]:
    """返回 (terminated, truncated)."""
    ref_root_pos = self._get_ref_root_pos()
    ref_root_quat = self._get_ref_root_quat()
    root_pos = self.data.xpos[self._body_idx["pelvis"]]
    root_quat = self.data.xquat[self._body_idx["pelvis"]]
    ref_root_height = ref_root_pos[2]
    root_height = root_pos[2]

    terminated = False

    # ── 条件 1: exceeded_anchor_height ──
    height_err = abs(ref_root_height - root_height)
    thresh = 0.75 if ref_root_height < 0.5 else 0.15
    if height_err > thresh:
        terminated = True

    # ── 条件 2: exceeded_anchor_ori ──
    angular_err = quat_error_magnitude(ref_root_quat, root_quat)
    if angular_err**2 > 0.2:
        terminated = True

    # ── 条件 3: exceeded_body_height ──
    ref_body_pos = self._get_ref_body_pos()
    check_bodies = ["left_ankle_roll_link", "right_ankle_roll_link",
                    "left_wrist_yaw_link", "right_wrist_yaw_link"]
    for name in check_bodies:
        idx = list(self._body_names).index(name)
        ref_h = ref_body_pos[idx, 2]
        sim_h = self.data.xpos[self._body_idx[name]][2]
        err = abs(ref_h - sim_h)
        thresh = 0.75 if ref_root_height < 0.5 else 0.15
        if err > thresh:
            terminated = True
            break

    # ── 条件 4: exceeded_body_pos ──
    foot_bodies = ["left_ankle_roll_link", "right_ankle_roll_link"]
    for name in foot_bodies:
        idx = list(self._body_names).index(name)
        ref_pos_aligned = ref_body_pos[idx] - ref_root_pos + root_pos
        sim_pos = self.data.xpos[self._body_idx[name]]
        if np.linalg.norm(ref_pos_aligned - sim_pos) > 0.2:
            terminated = True
            break

    # ── 条件 5: tracking_time_out ──
    truncated = self._ref_idx >= len(self._ref_dof) - 1

    return terminated, truncated
```

### 6.3 GAE 中的区分

```python
terminated, truncated = self._check_termination()
done = terminated or truncated

if done:
    terminal_obs = obs.copy()
    obs = self.reset()

info = {
    "time_outs": truncated,    # GAE: truncated → bootstrap with V(terminal_obs)
    "terminal_obs": terminal_obs,  # terminated → bootstrap to 0
}
```

- `time_outs=True`（truncated，motion 自然播完）：GAE 用 `V(terminal_obs)` 作为后续价值
- `time_outs=False`（terminated，摔倒/超限）：GAE 后续价值归零

---

## 7. step() 完整流程

```python
def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
    """执行一步仿真，返回 (obs, reward, done, info).

    Args:
        action: (29,) np.ndarray, float32, values in [-1, 1]

    Returns:
        obs: dict with keys "actor_obs"(930), "critic_obs"(1645), "tokenizer"(1761)
        reward: float
        done: bool
        info: dict with keys "time_outs", "terminal_obs"
    """
    if action.ndim == 2:
        action = action[0]  # 兼容 (1, 29) 格式

    action = np.clip(action, -1, 1).astype(np.float64)

    # ═══ 1. 更新 action history ═══
    self._ah[:-1] = self._ah[1:]
    self._ah[-1] = action

    # ═══ 2. PD 控制 ═══
    self._pd_control(action)

    # ═══ 3. 物理步进 ═══
    self._physics_step()  # mj_step × decimation

    # ═══ 4. 推进 motion reference ═══
    self._advance_motion_time()

    # ═══ 5. Computing reference body state (FK, 5-8 共用) ═══
    self._compute_ref_body_state()

    # ═══ 6. Observation ═══
    obs = self._obs()

    # ═══ 7. Reward ═══
    reward = self._compute_reward(action)

    # ═══ 8. Termination ═══
    terminated, truncated = self._check_termination()
    done = terminated or truncated

    # ═══ 9. Episode counter ═══
    self.ep += 1

    # ═══ 10. Auto-reset ═══
    terminal_obs = None
    if done:
        terminal_obs = {k: v.copy() for k, v in obs.items()}
        obs = self.reset()

    return obs, reward, done, {
        "time_outs": truncated,
        "terminal_obs": terminal_obs,
    }
```

**时序依赖**：
1. action_history 更新必须在 PD 控制之前（这样 actor obs 中 action_history 是最新的）
2. obs 计算在 reward/termination 之前（termination 触发 reset 后 obs 被覆盖）
3. `_compute_ref_body_state()` 只调一次，被 obs/reward/termination 共享读取

### 7.1 reset() 流程

```python
def reset(self) -> dict[str, np.ndarray]:
    """重置环境到新的 motion clip，返回初始 obs."""
    # 1. 采样新 motion
    self._sample_motion()

    # 2. 设置初始 joint position
    ref_q0 = self._ref_dof[self._ref_start]
    self.data.qpos[7:] = ref_q0.astype(np.float64)
    self.data.qvel[:] = 0

    # 3. 初始 forward
    mujoco.mj_forward(self.model, self.data)

    # 4. 清理 history buffer
    for buf in [self._gdh, self._avh, self._jph, self._jvh, self._ah, self._lvh]:
        buf.fill(0)

    # 5. 初始化缓存
    self._prev_action.fill(0)
    self._prev_root_pos = self.data.xpos[self._body_idx["pelvis"]].copy()
    self._prev_body_pos = self.data.xpos[self._body_indices].copy()
    self._prev_body_quat = self.data.xquat[self._body_indices].copy()
    self._prev_joint_vel = self.data.qvel[6:].copy()
    self._prev_ref_body_pos = self._prev_body_pos.copy()
    self._prev_ref_body_quat = self._prev_body_quat.copy()

    self.ep = 0

    # 6. 计算 FK 参考 + 初始 obs
    self._compute_ref_body_state()
    return self._obs()
```

---

## 8. 数学工具函数

所有函数为纯 numpy 实现，不依赖 isaaclab/torch。

```python
# gear_sonic/envs/mujoco_math.py  (或在 mujoco_env.py 内定义)

import numpy as np

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法 (w,x,y,z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)

def quat_inv(q: np.ndarray) -> np.ndarray:
    """四元数共轭."""
    inv = q.copy()
    inv[..., 1:] *= -1
    return inv

def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数旋转向量."""
    # v shape: (..., 3), q shape: (..., 4) or (4,)
    qv = np.zeros(v.shape[:-1] + (4,), dtype=v.dtype)
    qv[..., 1:] = v
    qv_rot = quat_mul(quat_mul(q, qv), quat_inv(q))
    return qv_rot[..., 1:]

def quat_error_magnitude(q1: np.ndarray, q2: np.ndarray) -> float:
    """四元数误差角度 (rad), 返回标量."""
    dot = abs(np.dot(q1, q2))
    dot = min(dot, 1.0)
    return 2.0 * np.arccos(dot)

def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """四元数 → 3×3 旋转矩阵."""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y],
    ])

def subtract_frame_transforms(
    t01: np.ndarray, q01: np.ndarray,
    t02: np.ndarray, q02: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """将 (t02, q02) 变换到 frame 1 坐标系."""
    q01_inv = quat_inv(q01)
    q_rel = quat_mul(q01_inv, q02)
    t_rel = quat_apply(q01_inv, t02 - t01)
    return t_rel, q_rel

def _quat_diff_to_angvel(
    q_prev: np.ndarray, q_curr: np.ndarray, dt: float
) -> np.ndarray:
    """从两个连续的四元数近似角速度 (n, 4) → (n, 3)."""
    dq = quat_mul(q_curr, quat_inv(q_prev))
    # 小角度近似: angvel ≈ 2 * imag(dq) / dt
    ang_vel = 2.0 * dq[..., 1:] / dt
    return np.clip(ang_vel, -100, 100)
```

---

## 9. 常量与配置

```python
# gear_sonic/envs/mujoco_env.py

import numpy as np
import mujoco
import joblib
import os, glob

NUM_DOF = 29
NUM_FUTURE = 10
NUM_SMPL_FUTURE = 10
NUM_BODIES = 14
NUM_SMPL_JOINTS = 24

BODY_NAMES = [
    "pelvis",                    # 0
    "left_hip_roll_link",        # 1
    "left_knee_link",            # 2
    "left_ankle_roll_link",      # 3
    "right_hip_roll_link",       # 4
    "right_knee_link",           # 5
    "right_ankle_roll_link",     # 6
    "torso_link",                # 7
    "left_shoulder_roll_link",   # 8
    "left_elbow_link",           # 9
    "left_wrist_yaw_link",       # 10
    "right_shoulder_roll_link",  # 11
    "right_elbow_link",          # 12
    "right_wrist_yaw_link",      # 13
]

VR_3POINT_BODY = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
VR_3POINT_BODY_OFFSET = np.array([
    [0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]
], dtype=np.float32)

LOWER_JOINT_INDICES = list(range(12))  # 0-11: hip pitch/roll/yaw, knee, ankle pitch/roll ×2

# Observation dimensions
ACTOR_OBS_DIM = 930
CRITIC_OBS_DIM = 1645
TOKENIZER_DIM = 1761
TOTAL_OBS_DIM = ACTOR_OBS_DIM + CRITIC_OBS_DIM + TOKENIZER_DIM  # 4336
```

---

## 10. 与现有代码的对照

### 10.1 当前 muJoCoEnv.py (120 行) vs 完整设计

| 组件 | 当前状态 | 完整设计 |
|------|----------|----------|
| PD 控制 | ✅ kp=30, kd=3 | 不变 |
| 物理步进 | ✅ decimation=10, mj_step | 不变 |
| Motion 加载 | ✅ joblib PKL | 不变 |
| reset() | ✅ 采样 + set qpos | 增加历史缓存初始化 |
| step() | 骨架 | 完整流程（10 步） |
| Actor obs | 历史只有 1 帧 | 改为 10 帧 history buffer |
| Critic obs | 700+ 维为零 | 完整 1645D |
| Tokenizer | 只有 future dof + 随机 encoder_index | 12 子项完整 1761D |
| Reward | 单 L2 误差 × 0.01 + 0.1 | 12 项 Gaussian kernel |
| Termination | `xpos[1][2] < 0.3` | 5 条件 + 自适应阈值 |
| Body 参考 | 无 | FK 计算参考 body 位姿 |
| Quat 工具 | 无 | 6 个工具函数 |

### 10.2 MuJoCoEnvManager 改动

- SHM 布局不变（总维度 4336 不变）
- `_worker_loop` 不变
- EnvSharedMemory 不变

### 10.3 train_mujoco_ppo.py 改动

- `TinyPolicy` 输入维度不变（actor=930, critic=1645）
- 训练循环不变

---

## 11. 性能估算

| 组件 | 耗时 (ms) | 备注 |
|------|-----------|------|
| PD 控制 | 0.01 | 简单乘加 |
| mj_step × 10 | 0.20 | 实测值 |
| 推进 motion ref | 0.01 | 索引更新 |
| FK 参考 body (当前帧) | 0.05 | mj_kinematics |
| Actor obs (930D) | 0.05 | history shift + concat |
| Critic obs (1645D) | 0.10 | FK future + quat ops |
| Tokenizer (1761D) | 0.15 | FK future + 12 项计算 |
| Reward (12 项) | 0.15 | quat ops + contact |
| Termination (5 条件) | 0.05 | 简单比较 |
| **每 env 总计** | **~0.77 ms** | 与设计文档 ~0.8ms 一致 |
| 4096 envs / 160 workers | ~21 ms | 每个 worker ~26 envs |

---

## 12. 分阶段验证策略

### 12.1 核心问题

MuJoCoEnv 的 obs/reward/termination 计算涉及 12 项 reward、5 项 termination、4336D obs，
逻辑复杂且需要与 Isaac Sim 保持语义一致。一次写完再调试的方式风险很高——
某个 reward 项的 quaternion 方向搞反了、某个 FK body 索引偏移了、
某个 Gaussian kernel 的 σ² 用错了，都需要在全量训练中才能发现，定位成本极高。

### 12.2 分层验证原则

```
Layer 0: 静态单元验证 — 给定已知输入，检查输出是否等于手工计算值
Layer 1: 动态一致性验证 — 仿真运行 N 步，检查统计分布是否在合理范围
Layer 2: 训练信号验证 — PPO training loop，检查 reward 是否呈上升趋势
```

**每一层通过后才进入下一层**。Layer 0 和 Layer 1 是自动化脚本，Layer 2 是 POC 训练。

### 12.3 Layer 0：静态单元验证

目标：**每个函数在隔离状态下输出正确的数值。**

验证方法：构造一个确定性的仿真快照（固定 `qpos`、`qvel`、`action`、`ref_dof`），
对每个函数手工计算期望输出，然后 assert 误差小于 1e-5。

```
验证脚本: scripts/verify_mujoco_obs_reward.py

  1. quat 工具函数 (6 个)
     └─ 已知四元数 → 手工计算 → assert 一致

  2. _compute_ref_body_state (FK)
     └─ 已知 qpos → mj_kinematics → 抽样 3 个 body 的 xpos/xquat → assert 非零且有限

  3. _compute_actor_obs (930D)
     └─ assert shape=(930,), no NaN, no Inf
     └─ history buffer: 填满 10 帧后 assert 最新帧与当前 qpos 一致

  4. _compute_critic_obs (1645D)
     └─ assert shape=(1645,), no NaN, no Inf
     └─ command_multi_future 段与 _future_dof 输出一致

  5. _build_tokenizer (1761D)
     └─ assert shape=(1761,), no NaN, no Inf
     └─ encoder_index 是合法 one-hot (3 维, 恰好一项为 1)

  6. _compute_reward (12 项, 逐项)
     └─ 完美跟踪状态 (qpos == ref): r ≥ 4.5
     └─ 随机扰动状态: r 比完美状态下降
     └─ 每项的符号正确 (6 个正 reward 均 > 0, 4 个 penalty 均 ≤ 0)

  7. _check_termination (5 条件)
     └─ 完美跟踪状态: terminated=False, truncated=False
     └─ 制造 4 种失败: 根超限/朝向超限/身体高度超限/脚位置超限 → terminated=True
     └─ ref_idx 接近 motion 末尾 → truncated=True

  8. step() 集成
     └─ 单步: obs 被正确写入, reward 是标量 float
     └─ 多步无 crash: 1000 steps 不报错
     └─ done 后 auto-reset: terminal_obs 非 None, obs 来自新 episode
```

### 12.4 Layer 1：动态一致性验证

目标：**仿真运行的数据分布合理，没有异常值。**

验证方法：创建 64 个 MuJoCoEnv，跑 200 steps，对输出做统计检查。

```
验证脚本: scripts/verify_mujoco_distribution.py

  1. reward 分布
     └─ 输出 reward 的 mean/std/min/max histogram
     └─ 预期: mean ∈ [2, 7], 80% 的 reward > 0, 无 NaN

  2. termination 比率
     └─ 输出每类 termination 的触发频率
     └─ 预期: terminated rate < 5%, truncated rate < 2%

  3. obs 协方差
     └─ actor_obs 的 930 维不应全零（至少 50% 的维度方差 > 0）
     └─ critic_obs 同理

  4. reward 逐项分解
     └─ 12 项分别输出 mean/std/min/max
     └─ 找出异常项（均值异常高/低、方差为零）

  5. 性能基线
     └─ 记录每步耗时，建立性能基线供后续对比
```

### 12.5 Layer 2：训练信号验证

目标：**reward 曲线能上升，说明信号方向正确。**

使用当前 `train_mujoco_ppo.py`，`TinyPolicy` 不变，跑 30 iteration：

```
预期:
  - reward 从初始 ~r_mean 逐步上升，至少提升 20%
  - loss 下降
  - 无 NaN gradient
  - 无 worker crash

失败判定:
  - reward 持续下降 → reward 符号可能反了
  - reward 不变化 → gradient 没有正确传播
  - loss → NaN → 数值问题（如 reward scale 过大）
```

### 12.6 完整验证流程

```
Phase A 完成 → 跑 Layer 0 第 1-2 项
Phase B 完成 → 跑 Layer 0 第 3-5 项
Phase C 完成 → 跑 Layer 0 第 6 项
Phase D 完成 → 跑 Layer 0 第 7 项
Phase E 完成 → 跑 Layer 0 第 8 项 → Layer 1 (全量) → Layer 2 (训练)
```

每一步发现有问题的项，立即在对应 phase 修复，不带到下一层。

---

## 13. 实现计划与验证检查点

### Phase A: 基础设施 + quat 工具

| 步骤 | 内容 | 验证 |
|------|------|------|
| A1 | 新建 `gear_sonic/envs/mujoco_math.py`，实现 6 个 quat 函数 | — |
| A2 | 实现 `_compute_ref_body_state()`（FK 参考 body 位姿） | — |
| A3 | 实现 `_future_dof_pos/vel()`、`_future_ref_body_pos/root_pos/root_quat()` | — |
| A4 | 写 Layer 0 验证脚本 `scripts/verify_unit_quat_fk.py` | ✅ |

**A4 检查清单：**
- [ ] `quat_mul(q, quat_inv(q))` ≈ identity quat
- [ ] `quat_apply(q, [1,0,0])` 手工验算
- [ ] `quat_error_magnitude(q, q)` = 0
- [ ] `subtract_frame_transforms` 输出在合理范围
- [ ] `_compute_ref_body_state` 后 pelvis 的 xpos 非零
- [ ] `_future_ref_body_pos` 返回 (10, 14, 3)，无 NaN

### Phase B: Observation

| 步骤 | 内容 | 验证 |
|------|------|------|
| B1 | history buffer 改为 (10, dim) 形状 | — |
| B2 | 实现完整 `_compute_actor_obs()` (930D) | — |
| B3 | 实现完整 `_compute_critic_obs()` (1645D) | — |
| B4 | 实现完整 `_build_tokenizer()` (1761D) | — |
| B5 | 写 Layer 0 验证脚本 `scripts/verify_unit_obs.py` | ✅ |

**B5 检查清单：**
- [ ] actor_obs.shape = (930,)，critic_obs.shape = (1645,)，tokenizer.shape = (1761,)
- [ ] 三个 obs 均无 NaN / Inf
- [ ] buffer 左移逻辑正确：`_jph[-1]` 等于当前 `qpos[7:]`
- [ ] encoder_index 是合法 one-hot
- [ ] tokenizer 的 SMPL 段（720+60+60=840D）全为零

### Phase C: Reward

| 步骤 | 内容 | 验证 |
|------|------|------|
| C1 | 实现 12 项 reward（5 个 penalty helper 方法） | — |
| C2 | 写 Layer 0 验证脚本 `scripts/verify_unit_reward.py` | ✅ |

**C2 检查清单（在完美跟踪状态 qpos==ref 下）：**
- [ ] r1 (anchor_pos): ≈ 0.5（误差 ≈ 0 时 Gaussian → 1.0 × w=0.5）
- [ ] r2 (anchor_ori): ≈ 0.5
- [ ] r3 (relative_body_pos): ≈ 1.0
- [ ] r4 (relative_body_ori): ≈ 1.0
- [ ] r5 (body_linvel): 物理初态速度为零，ref 速度也接近零 → ≈ 1.0
- [ ] r6 (body_angvel): ≈ 1.0
- [ ] r7 (action_rate_l2): 初始 action=0 → 0
- [ ] r8 (joint_limit): qpos == ref 在关节限位内 → 0
- [ ] r9 (undesired_contacts): 无接触 → 0
- [ ] r10 (anti_shake): 低于 threshold → 0
- [ ] r11 (vr_local): ≈ 2.0
- [ ] r12 (feet_acc): 加速度 ≈ 0 → ≈ 0
- [ ] total reward ≥ 4.5

**"堕落检查"（在随机扰动后）：**
- [ ] total reward 显著下降（证明 reward 对误差敏感，方向正确）

### Phase D: Termination

| 步骤 | 内容 | 验证 |
|------|------|------|
| D1 | 实现 `_check_termination()` (5 条件) | — |
| D2 | 写 Layer 0 验证脚本 `scripts/verify_unit_termination.py` | ✅ |

**D2 检查清单：**
- [ ] 完美状态 → terminated=False, truncated=False
- [ ] 手动设 root_pos 偏移 1.0m → terminated=True (条件 1)
- [ ] 手动设 root_quat 旋转 90° → terminated=True (条件 2)
- [ ] 手动设 ankle z 偏移 1.0m → terminated=True (条件 3)
- [ ] 手动设 foot 位置偏移 0.5m → terminated=True (条件 4)
- [ ] ref_idx 强制设到 dof.length - 1 → truncated=True (条件 5)
- [ ] 自适应阈值：ref_root_height < 0.5 时用 0.75，否则用 0.15

### Phase E: 集成 + 系统级验证

| 步骤 | 内容 | 验证 |
|------|------|------|
| E1 | 更新 `mujoco_env.py` step() 流程，收口所有组件 | — |
| E2 | `Layer 0 第 8 项`: step/reset 集成测试 | ✅ |
| E3 | `Layer 1`: 64 envs × 200 steps 分布检查 | ✅ |
| E4 | `Layer 2`: 30 iter POC 训练 | ✅ |
| E5 | 训练曲线 review → 确认可以进入 Task 7 | ✅ |

### 文件清单

| 文件 | 改动 | 用途 |
|------|------|------|
| `gear_sonic/envs/mujoco_env.py` | 120 → ~500 行 | 主实现 |
| `gear_sonic/envs/mujoco_math.py` | 新建, ~80 行 | 6 个 quat 工具函数 |
| `gear_sonic/envs/mujoco_env_manager.py` | 不变 | — |
| `scripts/train_mujoco_ppo.py` | 不变 | Layer 2 POC 训练 |
| `scripts/verify_unit_quat_fk.py` | 新建, ~100 行 | Phase A 验证 (Layer 0 第 1-2 项) |
| `scripts/verify_unit_obs.py` | 新建, ~80 行 | Phase B 验证 (Layer 0 第 3-5 项) |
| `scripts/verify_unit_reward.py` | 新建, ~120 行 | Phase C 验证 (Layer 0 第 6 项) |
| `scripts/verify_unit_termination.py` | 新建, ~100 行 | Phase D 验证 (Layer 0 第 7 项) |
| `scripts/verify_integration.py` | 新建, ~150 行 | Phase E 验证 (Layer 0 第 8 项 + Layer 1) |

---

## 14. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| FK 计算的参考 body 位姿与 Isaac Sim MotionLib 有系统偏差 | reward 基准偏移，训练出的 policy 行为异常 | 生成 golden 数据：用同一 PKL 的 10 帧 dof，分别在 MuJoCo FK 和 Isaac Sim 算 body 位姿，逐帧对比。如果偏差 >1cm 或 >2°，需要对齐坐标系或 FK 参数 |
| `tracking_relative_body_pos` 的 anchor-relative 变换方向搞反 | reward 惩罚正确跟踪，policy 学到错误行为 | 此为 Layer 0 C2 完美跟踪检查的核心目的。在 qpos==ref 条件下 r3 必须 ≈1.0，否则立即暴露 |
| contact force 在 MuJoCo (mj_contactForce) 和 PhysX 之间物理含义不同 | r9 (undesired_contacts) 基线偏移 | Layer 1 分布检查中看 r9 的均值。如果 MuJoCo 下 r9 mean 与预期偏差 > 10×，暂时从 reward 中移除 r9 |
| quat_error_magnitude 符号约定与 Isaac Sim 不一致 | r2、r4 计算错误 | 直接对比——Isaac Sim 的 `quat_error_magnitude` 也是 `2*arccos(|dot|)`，与我们的实现相同 |
| body 名称在 XML 和 IK 计算结果中不完全一致 | 某些 body 的 FK 结果为零向量 | A4 验证中显式检查所有 14 个 body 的 xpos 非零 |
| history buffer 左移时 copy 方向错误 | obs 显示"过去"的数据来自"未来" | B5 验证：填满 buffer 后 assert `_jph[-1]` == 当前 `qpos[7:]` |
