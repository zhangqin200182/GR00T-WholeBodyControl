# MuJoCo CPU + NPU 单机训练开发方案

## 1. 目标

在 NPU 服务器 (Kunpeng 920 320c + 16×Ascend 910) 上，用 MuJoCo CPU 物理仿真替代 Isaac Sim，实现 **零 GPU 依赖** 的 SONIC 完整训练。

## 2. 基准数据

```
MuJoCo 3.10.0, Kunpeng 920 aarch64, 单核实测:
  humanoid (14b/17DOF):  0.118 ms/step (纯 mj_step)
  G1 29-DOF 估算:        0.20  ms/step (纯 mj_step)

实际每 env 完整开销 (含 obs/reward/termination):
  mj_step:                0.20 ms
  compute_obs (930+1645+1761):  ~0.3 ms (tensor 拼接+history buffer)
  compute_reward (9项):         ~0.2 ms
  check_termination:            ~0.05 ms
  ctrl 更新 + 其他:              ~0.05 ms
  ─────────────────────────────────────
  每 env 总计:              ~0.8 ms

  单核 @20ms:  ~25 envs (保守)
  4096 envs:   ~160 workers (160 核, 50% CPU)

NPU 推理 (16卡 StubEnv 实测):
  batch=256, FP32 eager: ~26ms/step ← 瓶颈
  + compile + FP16:     ~6ms/step  ← 优化后
```

## 3. 架构设计

### 3.1 整体架构

```
┌─ NPU 服务器 (单机) ────────────────────────────────────────────────┐
│                                                                     │
│  ┌─ CPU Worker Pool (320 cores) ───────────────────────────────┐   │
│  │                                                              │   │
│  │  Worker 0:   MuJoCo × 26 envs  (进程, 顺序执行, 1 核)         │   │
│  │  Worker 1:   MuJoCo × 26 envs                                 │   │
│  │  ...                                                         │   │
│  │  Worker 159: MuJoCo × 25 envs  (共 160 workers, 160 核)       │   │
│  │                                                              │   │
│  │  每个 worker 在单个 CPU 核上顺序跑 ~26 个 env                    │   │
│  │  NUMA 亲和: Workers 按 env id 范围绑定到对应 NUMA node         │   │
│  │                                                              │   │
│  │  每 worker:                                                   │   │
│  │    ┌──────────────────────────────────────────────┐          │   │
│  │    │ for env in 0..25:                            │          │   │
│  │    │   mj_step(model, data)  → physics            │          │   │
│  │    │   compute_obs()          → dict              │          │   │
│  │    │   compute_reward()       → float             │          │   │
│  │    │   check_termination()    → bool              │          │   │
│  │    │   if done: reset_env()                       │          │   │
│  │    └──────────────────────────────────────────────┘          │   │
│  │                                                              │   │
│  │  SharedMemory: obs_dict [4096, ...]                          │   │
│  │               rewards [4096]                                 │   │
│  │               dones [4096]                                   │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             │ SHM (零拷贝)                          │
│  ┌──────────────────────────┴───────────────────────────────────┐   │
│  │  NPU Training Process (16 × Ascend 910, DDP)                 │   │
│  │                                                              │   │
│  │  for step in range(24):                                      │   │
│  │    obs = shm.read_obs()           # 从共享内存读               │   │
│  │    actions = policy_step(obs)      # NPU 推理                 │   │
│  │    shm.write_actions(actions)      # 写回共享内存              │   │
│  │    # CPU workers 并发执行 env.step(actions)                   │   │
│  │    barrier()                       # 等待所有 worker 完成     │   │
│  │    rewards, dones = shm.read()                                │   │
│  │    storage.store(...)                                         │   │
│  │                                                              │   │
│  │  GAE → advantages → PPO update × 20                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 同步模型 (使用 Barrier)

```python
# 关键: 使用 multiprocessing.Barrier(N + 1), 不是 Event
_barrier = multiprocessing.Barrier(num_workers + 1)
# N workers + 1 trainer = N+1 方, 全部到达才放行
# 同一个 Barrier 对象复用多次, 每次自动 reset

每步流程 (同一个 Barrier 的 3 次 wait):
  Workers (N 进程, 并行)              Trainer (1 进程, 主控)
  ─────────────────────               ───────────────────
  _barrier.wait() ← 全部到这          policy_step(obs[4096])
    (阻塞, 不占 CPU)                     ↓ 26ms (NPU 推理)
                                       actions → SHM
  ← _barrier.wait() 放行              _barrier.wait()
  mj_step + compute × N envs
  obs/reward/done → SHM
  _barrier.wait() ← 全部写入完成      ← _barrier.wait() 放行
    (阻塞)                              read obs/reward/done
                                       storage.store()
  ← _barrier.wait() 放行              _barrier.wait() → 进入下一步

每步时间: max(worker 最慢时间, NPU 推理 26ms) ≈ 26ms (NPU 瓶颈)
```

```
Barrier 计数: num_workers + 1, 同一对象复用 3 次/step

  时刻 T0:  trainer + all workers arrive → 第 1 次放行
  时刻 T1:  trainer 推理, workers idle
  时刻 T2:  trainer 写入 actions → arrive 第 2 次
  时刻 T3:  all workers arrive → 第 2 次放行
  时刻 T4:  workers 执行, trainer idle
  时刻 T5:  all workers arrive + trainer arrive → 第 3 次放行 (= 下一步的第 1 次)
```

### 3.3 进程模型与启动顺序

```
⚠️ 关键: 必须先 spawn workers, 再初始化 DDP/HCCL

  1. spawn workers (multiprocessing.Process)
     每个 worker 独立进程, 不继承 HCCL 句柄
  
  2. 初始化 HCCL + accelerator.prepare()
     DDP 在 workers 之后初始化, 避免通信资源泄漏到子进程
  
  3. 开始训练循环

  # train_agent_trl.py 伪代码:
  if is_main_process():
      # Step 1: 先启动 worker pool
      env_manager = MuJoCoEnvManager(num_envs=4096, num_workers=160)
      # 内部 spawn 160 个子进程
  
      # Step 2: 再初始化 DDP
      accelerator = Accelerator()
      model = accelerator.prepare(model)
  
      # Step 3: 训练
      train(model, env_manager, accelerator)
```

### 3.4 内存估算

```
每 env MuJoCo data:
  qpos (36) + qvel (35) + act + ctrl + xpos + xquat + ...
  ~50 KB / env

4096 envs:
  MuJoCo data:   4096 × 50 KB   = 200 MB
  模型 (共享):   1 × MjModel      = 5 MB
  MotionLib:     160 workers × 共享 MotionLib per worker
                 每 worker ~40 MB (AMASS 数据, mmap) = 6.4 GB
  obs_dict:      4096 × 17 KB   = 70 MB  (FP32)
  actions:       4096 × 116 B   = 0.5 MB
  rewards:       4096 × 4 B     = 16 KB
  ─────────────────────────────────────
  CPU 侧总计:    ~6.7 GB (MotionLib 占大头)
  MotionLib 优化: 所有 worker 共享同一份 mmap, 降至 ~40 MB

NPU 侧:
  模型 + optimizer: ~2 GB (37M params, Adam)
  训练 batch:       ~5 GB
  ─────────────────────────────────────
  总计 (单机 24GB+): 绰绰有余
  服务器内存:        远大于 24GB ✅
```

## 4. 核心模块开发

### 4.1 `MuJoCoEnv` — 单环境类

```python
# gear_sonic/envs/mujoco_env.py

class MuJoCoEnv:
    """单个 MuJoCo 环境, 替代 Isaac Sim env."""
    
    def __init__(self, model_xml: str, config: dict):
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.data = mujoco.MjData(self.model)
        self.motion_lib = MotionLibRobot(...)  # 复用现有, 同一 worker 内所有 env 共享
        self.motion_id = None
        self.episode_length = 0
        self.max_episode_length = config.get("max_episode_length", 500)
        self._init_history_buffers()
    
    def reset(self):
        self.motion_id = self.motion_lib.sample_motion()
        self.start_time = self.motion_lib.sample_time(self.motion_id)
        self.data.qpos[:] = self._get_init_qpos()
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)  # 同步运动学
        self._clear_history_buffers()
        self.episode_length = 0
        return self._compute_obs()
    
    def step(self, action: np.ndarray) -> tuple:
        # 1. 应用 PD 控制
        self.data.ctrl[:] = action_to_torque(action, self.data)
        
        # 2. 物理步进 (decimation=10, native_dt=0.002 → ctrl_dt=0.02=50Hz)
        #    注: MuJoCo G1 模型原生 timestep=0.002, 非 Isaac Sim 的 0.005
        #    10×0.002=20ms 与 Isaac Sim 的 4×0.005=20ms 等效
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        
        # 3. 推进 motion reference
        self._advance_motion_time()
        
        # 4. 计算 observation (termination 前!)
        obs = self._compute_obs()
        
        # 5. 计算 reward
        reward = self._compute_reward()
        
        # 6. 检查 termination
        terminated, truncated = self._check_termination()
        done = terminated or truncated
        
        self.episode_length += 1
        
        # ⚠️ Auto-reset: 保存 terminal obs 再 reset
        terminal_obs = obs if done else None
        if done:
            obs = self.reset()
        
        return obs, reward, done, {
            "time_outs": truncated,
            "terminal_obs": terminal_obs,  # ← GAE 需要!
        }
```

### 4.2 `MuJoCoEnvManager` — 并行环境管理器

```python
# gear_sonic/envs/mujoco_env_manager.py

class MuJoCoEnvManager:
    """管理 N 个 MuJoCo 环境, 替代 ManagerEnvWrapper.
    
    支持两种模式:
      - Serial: 单进程顺序执行 (开发/调试, num_envs ≤ 100)
      - Parallel: 多进程 pool (训练, num_envs ≤ 4096)
    
    同步: 使用 multiprocessing.Barrier(num_workers + 1)
          同一 Barrier 对象每步复用 3 次, 自动 reset
    """
    
    def __init__(self, num_envs: int, num_workers: int = 160):
        self.num_envs = num_envs
        self.num_workers = num_workers
        # 不均匀分配: 前 N-1 个 worker 各跑 math.ceil 个 env, 最后 1 个跑剩余
        import math
        self.envs_per_worker = math.ceil(num_envs / num_workers)
        
        # 共享内存 (布局见 4.2.1)
        self._obs_shm = SharedMemory(create=True, size=obs_total_bytes)
        self._term_obs_shm = SharedMemory(create=True, size=term_obs_total_bytes)
        self._act_shm = SharedMemory(create=True, size=act_total_bytes)
        self._rew_shm = SharedMemory(create=True, size=rew_total_bytes)
        self._done_shm = SharedMemory(create=True, size=done_total_bytes)
        
        # 同步: 单一 Barrier, N workers + 1 trainer
        self._barrier = multiprocessing.Barrier(num_workers + 1)
        
        # 启动 worker 进程 (必须在 DDP/HCCL 初始化之前)
        self._workers = []
        for i in range(num_workers):
            start_env = i * self.envs_per_worker
            end_env = min(start_env + self.envs_per_worker, num_envs)
            num_envs_for_worker = end_env - start_env
            p = multiprocessing.Process(
                target=self._worker_loop,
                args=(i, start_env, num_envs_for_worker)
            )
            p.start()
            self._workers.append(p)
    
    def _worker_loop(self, worker_id: int, start_env: int, num_envs: int):
        """每个 worker 进程的主循环."""
        # MotionLib 在 worker 内共享一份 (mmap 方式, 避免 4096 份拷贝)
        motion_lib = MotionLibRobot(...)
        envs = [MuJoCoEnv(MODEL_XML, config, motion_lib) for _ in range(num_envs)]
        
        # 初始 reset, 写入 SHM (按 env_id 全局偏移)
        for i, env in enumerate(envs):
            obs = env.reset()
            self._write_obs_to_shm(start_env + i, obs)
        
        while True:
            # --- 第 1 次 barrier: 等待 trainer 写入 actions ---
            self._barrier.wait()
            
            # 读取本 worker 负责的 actions
            actions = self._read_actions_from_shm(start_env, num_envs)
            
            # 执行所有 env
            for i, env in enumerate(envs):
                env_id = start_env + i
                obs, reward, done, info = env.step(actions[i])
                self._write_obs_to_shm(env_id, obs)
                self._write_reward_to_shm(env_id, reward)
                self._write_done_to_shm(env_id, done)
                # 保存 terminal obs (GAE 需要)
                if info.get("terminal_obs") is not None:
                    self._write_terminal_obs_to_shm(env_id, info["terminal_obs"])
            
            # --- 第 2 次 barrier: 通知 trainer 写入完成 ---
            self._barrier.wait()
    
    def step(self, actions: torch.Tensor) -> tuple:
        """Trainer 调用: 分发 actions, 等待 workers, 返回结果."""
        # 写入 actions 到共享内存
        self._write_actions(actions)
        
        # --- 第 1 次 barrier: 触发 workers 开始执行 ---
        self._barrier.wait()
        
        # --- 第 2 次 barrier: 等待所有 workers 完成 ---
        self._barrier.wait()
        
        # 读取结果, 转为 torch tensors on NPU
        obs_dict = self._read_obs()
        rewards = self._read_rewards()
        dones = self._read_dones()
        terminal_obs = self._read_terminal_obs()  # GAE 用的 terminal value
        
        return obs_dict, rewards, dones, {"terminal_obs": terminal_obs}
```

### 4.2.1 共享内存布局

```
SharedMemory 布局 (单块连续内存):

┌────────────── obs_dict ──────────────────────────────────────┐
│  Layout: [env_0, env_1, ..., env_4095]                       │
│  每 env stride = actor_obs(930) + critic_obs(1645)           │
│                 + tokenizer_obs(1761) = 4336 floats           │
│  每 env 大小: 4336 × 4B = 17,344 bytes                       │
│  总大小:      4096 × 17,344 = 71 MB                          │
├────────────── terminal_obs ──────────────────────────────────┤
│  Layout: [env_0, ..., env_4095]                              │
│  每 env: 4336 floats (同 obs 格式)                             │
│  非 terminal 时填充 0                                         │
│  总大小: 71 MB                                               │
├────────────── actions ───────────────────────────────────────┤
│  Layout: [env_0, ..., env_4095]                              │
│  每 env stride: 29 floats                                     │
│  每 env 大小: 29 × 4B = 116 bytes                            │
│  总大小:      4096 × 116 = 475 KB                            │
├────────────── rewards ───────────────────────────────────────┤
│  Layout: [env_0, ..., env_4095]                              │
│  每 env: 1 float = 4 bytes                                   │
│  总大小: 4096 × 4 = 16 KB                                    │
├────────────── dones ─────────────────────────────────────────┤
│  Layout: [env_0, ..., env_4095]                              │
│  每 env: 1 uint8                                              │
│  总大小: 4096 × 1 = 4 KB                                     │
├────────────── time_outs ─────────────────────────────────────┤
│  Layout: [env_0, ..., env_4095]                              │
│  每 env: 1 uint8                                              │
│  总大小: 4 KB                                                │
└──────────────────────────────────────────────────────────────┘
总计: ~143 MB 共享内存

Worker 写入: 使用全局 env_id 计算偏移, 无需 worker_id
  write_offset = env_id × obs_stride

Trainer 读取: 直接 memcpy 到 NPU tensor (torch.from_numpy 从 SHM buffer)
```

### 4.3 Observation 适配

```python
# 需要实现的 observation 计算函数

def _compute_actor_obs(self):
    """930 维: 10 帧 history × (joint_pos + joint_vel + gravity + ang_vel + action)"""
    # MuJoCo 直接提供:
    #   joint_pos: self.data.qpos[7:]   (除去 free joint 的 7 维)
    #   joint_vel: self.data.qvel[6:]   (除去 free joint 的 6 维)
    #   gravity:   self._compute_gravity_dir()
    #   ang_vel:   self.data.qvel[3:6]  (base angular velocity)
    return concat_histories(...)

def _compute_critic_obs(self):
    """1645 维: actor_obs + 运动目标 + 地形/接触"""
    # 地形: MuJoCo 射线传感器 或 geom 高度查询
    # 接触: self.data.contact (body pair + force)
    return concat(...)

def _compute_tokenizer_obs(self):
    """1761 维: 运动参考数据"""
    # motion_lib.get_motion_state() → encoder input
    # 完全复用 StubEnv 的计算逻辑
    return concat(...)
```

### 4.4 Reward 适配

```python
# 9 个 reward 项, MuJoCo 均可实现

def _compute_rewards(self):
    rewards = []
    
    # 运动学跟踪 (6 项):
    rewards.append(tracking_anchor_pos())    # self.data.xpos[0] vs reference
    rewards.append(tracking_anchor_ori())    # self.data.xquat[0] vs reference
    rewards.append(tracking_body_pos())      # self.data.xpos[1:] vs reference
    rewards.append(tracking_body_ori())      # self.data.xquat[1:] vs reference
    rewards.append(tracking_body_linvel())   # from self.data.qvel
    rewards.append(tracking_body_angvel())   # from self.data.qvel
    
    # 惩罚 (3 项):
    rewards.append(action_rate_l2())         # action diff
    rewards.append(joint_limit())            # model.jnt_range
    rewards.append(undesired_contacts())     # self.data.contact
    
    return weighted_sum(rewards)
```

### 4.5 Termination 适配

```python
def _check_termination(self):
    # Terminated (失败)
    terminated = (
        exceeded_anchor_pos(threshold=0.15) or
        exceeded_anchor_ori(threshold=0.2) or
        exceeded_foot_pos(threshold=0.2) or
        exceeded_ee_body_pos(threshold=0.15) or
        cumulative_error()
    )
    
    # Truncated (运动播完)
    truncated = (self.episode_length >= self.max_episode_length)
    
    return terminated, truncated
```

### 4.6 OSMesa 渲染 (Step 2 数据生成用)

```python
# gear_sonic/envs/mujoco_render.py

import mujoco
from mujoco import renderer as mjr

class MuJoCoRenderer:
    """CPU 软件渲染, 替代 Isaac Sim camera."""
    
    def __init__(self, model, width=640, height=480):
        # OSMesa: CPU-based OpenGL
        self.renderer = mjr.Renderer(
            model, height, width,
            backend='osmesa'  # ← 关键: CPU 渲染
        )
    
    def render(self, data, camera_name='ego_view'):
        """渲染一帧 RGB 图像 → numpy [H, W, 3]."""
        self.renderer.update_scene(data, camera=camera_name)
        pixels = self.renderer.render()
        return pixels  # [480, 640, 3] uint8
    
    def close(self):
        self.renderer.close()
```

```python
# 渲染性能 (480×640, OSMesa, Kunpeng 920)
# 单帧: ~15-20ms (CPU 软件渲染)
# 4096 envs 全部渲染: 4096 × 15ms = 61s → 不能用
# 只渲染 10% envs (采样): 400 × 15ms = 6s → 可以接受
# 训练时不需要每步都渲染, 每 N 步采样一次即可
```

## 5. 与现有代码的接口对齐

### 5.1 PPO Trainer 改动（最小）

```python
# gear_sonic/train_agent_trl.py

# 原版 (Isaac Sim):
# from gear_sonic.envs.manager_env_wrapper import ManagerEnvWrapper
# env = ManagerEnvWrapper(...)

# MuJoCo 版:
from gear_sonic.envs.mujoco_env_manager import MuJoCoEnvManager
env = MuJoCoEnvManager(
    num_envs=4096,
    num_workers=160,
    model_xml="/path/to/g1_29dof.xml",
    motion_lib_cfg=motion_lib_cfg,  # 复用现有 MotionLib
)

# 接口完全一致:
obs_dict, rewards, dones, infos = env.step(actions)
```

### 5.2 复用现有模块

```
✅ 复用 (不改):
  MotionLib (motion_lib_base.py + robot + smpl)
  Encoder routing (universal_token_modules.py)
  PPO Trainer (ppo_trainer.py) — 只需 adapter
  StubEnv observation 计算 (大部分逻辑)
  G1 关节映射 + 运动学

🆕 新开发:
  MuJoCoEnv / MuJoCoEnvManager
  OSMesa 渲染管线
  MuJoCo PD 控制器 (action → torque)

🔧 需适配:
  ManagerEnvWrapper → MuJoCoEnvManager 接口
  Isaac Sim specific 的 sensor 调用 → MuJoCo 等价
```

## 6. 开发阶段

### Phase 1: 单 env 验证 (1-2 天)

```
□ MuJoCo G1 XML 模型就绪 (已有, 在 gear_sonic_deploy/g1/)
□ MuJoCoEnv 实现:
  □ step() — PD 控制 + mj_step × 4
  □ _compute_actor_obs() — 930 维
  □ _compute_critic_obs() — 1645 维
  □ _compute_tokenizer_obs() — 1761 维
  □ _compute_reward() — 9 项
  □ _check_termination() — 5 个条件
□ 单 env stand-alone 测试: step 1000 次, 不掉帧
□ 与 Isaac Sim 单 env 对比: obs 分布/reward 范围
```

### Phase 2: 并行环境管理器 (2-3 天)

```
□ MuJoCoEnvManager 实现:
  □ SharedMemory layout
  □ Worker 进程池 (multiprocessing)
  □ Barrier 同步
  □ Auto-reset 逻辑
□ 小规模测试:
  □ 64 envs, 2 workers → StubEnv baseline 对比
  □ 256 envs, 10 workers → 性能测试
  □ 4096 envs, 160 workers → 全量压力测试
```

### Phase 3: PPO Trainer 集成 (1-2 天)

```
□ train_agent_trl.py 适配 MuJoCoEnvManager
□ rollout 循环验证: 24 steps × 4096 envs
□ GAE 计算 (不变)
□ PPO update (不变)
□ 100 iter 小训练: 指标对比 StubEnv
```

### Phase 4: OSMesa 渲染 (1-2 天)

```
□ MuJoCoRenderer 实现
□ 渲染性能测试
□ 数据采集流程: (image, token) 对
□ LeRobot 格式导出
```

### Phase 5: 完整训练 (2-3 天)

```
□ 100K iter 完整训练
□ TensorBoard 监控
□ Checkpoint 保存/恢复
□ 与 StubEnv/Isaac Sim 结果对比分析
```

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| MuJoCo 碰撞几何与 PhysX 不同 | reward 分布偏移 | 调 reward 权重; 先用 StubEnv 的 reward 分布对照 |
| 多进程 SHM 竞争 | 性能下降 > 预期 | 双缓冲; 分 NUMA node |
| SMPL 人体模型在 MuJoCo 中加载 | 运动参考格式不兼容 | 直接用 MotionLib 数据, 不渲染 SMPL 人体 |
| OSMesa 在 aarch64 上不可用 | 无法渲染 camera | 改 EGL headless; 或仅用于训练(不需 camera) |
| MuJoCo G1 模型关节限制不匹配 | action 夹持不正确 | 验证 jnt_range 与 Isaac Sim 一致 |

## 8. 总结

```
一台 NPU 服务器 (Kunpeng 920 320c + 16 × Ascend 910)
  CPU:   MuJoCo 物理 × 4096 envs (~160 workers, 50% CPU)
  NPU:   PPO 训练 × 16 DDP
  GPU:   0
  内存:  ~143 MB 共享内存 + ~275 MB MuJoCo data + ~7 GB NPU 模型

开发周期: ~10 天 (5 phases)
核心工作: MuJoCoEnvManager (并行环境管理 + SHM 布局 + Barrier 同步)
最大风险: 物理真实性 (reward 分布偏移)、obs 计算开销超预期
最高回报: 完全消除 GPU 依赖

关键设计注意事项:
  1. 使用 Barrier(N+1) 而非 Event 同步 (同一 Barrier 复用 3 次)
  2. 先 spawn workers, 再初始化 HCCL/DDP
  3. 单 worker 顺序执行, 1 核够用 (~26 envs per worker)
  4. terminal obs 单独保存 (GAE 需要)
  5. SHM 每 env 固定偏移, 避免动态分配
  6. MotionLib 跨 worker 共享 (mmap), 避免 O(envs) 内存膨胀
```
