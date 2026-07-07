# MuJoCo CPU + NPU 训练 — 任务分解与详细设计

## 任务总览

> ⚠️ **所有任务的开发和验证都在 NPU 服务器上进行**
>
> 服务器: `113.46.41.54`, 容器: `rlinf-train`
> 代码路径: `/root/GR00T-WholeBodyControl/`
> 模型路径: `/root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/`
> MuJoCo: 3.10.0 已安装, Kunpeng 920 320c + 16 × Ascend 910

```
Task 1: MuJoCo G1 模型验证          (独立)  ──┐
Task 2: MotionLib 集成              (独立)  ──┤
Task 3: MuJoCoEnv 单环境            ←────────┘  依赖 1,2
Task 4: 观测一致性验证               ←── Task 3 ┤
Task 5: MuJoCoEnvManager 并行管理器   ←── Task 3 ┤
Task 6: PPO Trainer 集成             ←── Task 5 ┤
Task 7: 全量训练 + 性能分析           ←── Task 6 ┘
Task 8: OSMesa 渲染 (可选)           (独立)

可并行开发: Task 1 ∥ Task 2 ∥ Task 8
串行依赖:   Task 1+2 → Task 3 → Task 4
                            → Task 5 → Task 6 → Task 7
Task 4 依赖 Task 3, 但可在 Task 3 开发期间先写好对比脚本框架
```

---

## Task 1: MuJoCo G1 模型验证

### 目标

在 NPU 服务器上运行 MuJoCo G1 29-DOF 模型，验证物理仿真的正确性和性能。

### 输入

- `gear_sonic_deploy/g1/g1_29dof.xml` — G1 MuJoCo 模型 (服务器路径: `/root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/`)
- `gear_sonic_deploy/g1/meshes/` — STL 网格 (服务器已含原始 ASCII 文件, 需转 binary)
- NPU 服务器 (rlinf-train 容器, MuJoCo 3.10.0 已安装)

### 验证服务器

```
服务器:  113.46.41.54
容器:    rlinf-train
MuJoCo:  3.10.0
XML:     /root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml
Meshes:  /root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/meshes/

### 详细设计

```
Step 1.1: 准备 G1 模型文件
  - 将 ASCII STL 转为 binary STL (MuJoCo 只支持 binary)
    使用 Python trimesh 或 meshio 批量转换
  - 或将 <mesh> 替换为 <geom type="capsule/box"> 的简化碰撞模型
  - 验证: mjModel 加载不报错

Step 1.2: 基本物理验证
  - 从站立姿态跌落测试: qpos 初始化为 T-pose, 无控制, 自由落体
  - PD 控制站立测试: 固定目标关节角, 验证机器人保持站立
  - 随机动作测试: random ctrl, 运行 1000 步, 检查是否 NaN/爆炸

Step 1.3: 性能 benchmark
  - 单 env 顺序 10000 步, 测量 mj_step 耗时
  - 多 env 并行 (同一进程内 N 个 MjData, 顺序执行)
  - 记录: bodies, joints, contacts 数量; 每步耗时; 内存占用
  - 与 humanoid 基准比较 (0.118ms → G1 预期 0.20ms)
```

### 输出

- `benchmarks/mujoco_g1_perf.json`: 性能数据
- `notes/g1_model_issues.md`: 模型问题记录 (关节限制、碰撞几何等)
- 可运行的脚本: `scripts/verify_mujoco_g1.py`

### 验证标准

- [x] G1 模型成功加载, 无 XML 解析错误
- [x] mj_step 无崩溃, 10000 步无 NaN
- [x] PD 控制下机器人能保持站立 ≥ 500 步
- [x] 单 env 物理耗时 ≤ 0.3ms (实测: 0.939ms/ctrl_step, 0.235ms/substep)
- [x] 内存占用 ≤ 100KB/env (实测: ~7 KB/env)

### 实测结果 (2026-07-07, Kunpeng 920)

```
模型:     31 bodies, 29 hinge joints, 29 actuators, 35 DOF
Drop:     z=0.793 → 0.120, 正常倒地
PD stand: 500/500 步不倒
Random:   1000 步无 NaN
Bench:    0.939 ms/step (含 decimation×4), 21 envs/core@20ms
          320 cores → 6720 envs (远超 4096)
Memory:   ~7 KB/env
```

### 预估工时: 0.5 天 ✅ 已完成

---

## Task 2: MotionLib 集成

### 目标

在 NPU 服务器容器内加载 MotionLib 运动数据，验证数据格式兼容性和采样正确性。此 Task 独立于 MuJoCo，可并行开发。

### 输入

- `sample_data/robot_filtered/*.pkl` — Robot 运动 PKL (服务器路径: `/root/GR00T-WholeBodyControl/sample_data/robot_filtered/`)
- `sample_data/smpl_filtered/*.pkl` — SMPL 运动 PKL (服务器路径: `/root/GR00T-WholeBodyControl/sample_data/smpl_filtered/`)
- `gear_sonic/utils/motion_lib/` — 现有 MotionLib 代码

### 详细设计

```
Step 2.1: 上传运动数据
  - smpl_filtered 数据已在服务器 (? 确认)
  - 或上传 sample_data 用于开发测试

Step 2.2: 独立加载测试
  from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot
  lib = MotionLibRobot(motion_file="sample_data/robot_filtered")
  motion_ids, times = lib.sample_motions(n=64)
  state = lib.get_motion_state(motion_ids[0], times[0])
  # 验证: state 包含 dof, root_rot, root_pos 等字段, shape 正确

Step 2.3: SMPL MotionLib 加载测试
  from gear_sonic.utils.motion_lib.motion_lib_smpl import MotionLibSMPL
  # 同上验证

Step 2.4: 编码器路由逻辑验证
  # 复用 StubEnv 的 encoder_masks 逻辑
  # 验证: motion_has_smpl, encoder_sample_probs 按预期工作

Step 2.5: 性能测试
  - 10000 次 sample_motions() + get_motion_state(), 测量耗时
  - 单 worker 内多 env 共享同一 MotionLib 实例的内存评估
```

### 输出

- `scripts/verify_motion_lib.py`: 独立验证脚本
- `notes/motion_lib_compatibility.md`: 兼容性问题和数据路径记录

### 验证标准

- [ ] MotionLibRobot 加载无报错
- [ ] sample_motions 返回合法的 motion_id 和 time_step
- [ ] get_motion_state 返回完整数据, shape 与 StubEnv 一致
- [ ] SMPL MotionLib 同样验证通过
- [ ] 采样耗时 ≤ 0.1ms (对训练影响可忽略)

### 预估工时: 0.5 天

---

## Task 3: MuJoCoEnv 单环境实现

### 目标

实现单个 MuJoCo 环境类，接口与 Isaac Sim `ManagerEnvWrapper` 兼容，输入 action 返回 (obs_dict, reward, done, info)。

### 依赖

- Task 1 (G1 模型可用)
- Task 2 (MotionLib 可用)

### 详细设计

```
Step 3.1: 核心类骨架
  class MuJoCoEnv:
      def __init__(self, model, motion_lib, config):
          self.data = mujoco.MjData(model)
          self.motion_lib = motion_lib
          self.motion_id = None
          self.episode_length = 0
          self._init_history_buffers()  # 复用 StubEnv 代码

Step 3.2: reset()
  - sample_motions() → motion_id, start_time
  - 从 motion_lib 获取初始关节角 → model.qpos0
  - mj_forward() 初始化运动学
  - 清零 history buffer
  - 返回 _compute_obs()

Step 3.3: step(action)
  1. action 反归一化: target_qpos = action * joint_range + default_pose
  2. PD torque: self.data.ctrl[:] = kp*(target_qpos - qpos) - kd*qvel
  3. mj_step() × 4 (decimation=4)
  4. _advance_motion_time()
  5. obs = _compute_obs()
  6. reward = _compute_reward()
  7. done, truncated = _check_termination()
  8. if done: terminal_obs = obs; obs = self.reset()
  9. return obs, reward, done, {"time_outs": truncated, "terminal_obs": terminal_obs}

Step 3.4: _compute_obs()
  返回 dict:
    "actor_obs":   930D  (10帧 history × 5项, 复用 StubEnv 计算)
    "critic_obs":  1645D (actor_obs + motion targets + contact/terrain)
    "tokenizer":   1761D (motion refs for encoder, 复用 StubEnv 计算)

Step 3.5: _compute_reward()
  9 项奖励, 权重从 config 读取 (不硬编码):
    # 从 sonic_release.yaml manager_env.rewards 读取 weight/std
    # 与 Isaac Sim 权重完全一致，确保训练信号可比
    公式: reward_i = exp(-error² / std²) × weight

Step 3.6: _check_termination()
  terminated: 位置/朝向/脚部/末端偏差超过阈值
  truncated: episode_length >= max_length

Step 3.7: PD 控制器
  # 与 Step 3.3 一致, 这里给出完整实现
  def _compute_pd_torque(self, action):
      # action: [29] normalized, range [-1, 1]
      joint_range = self.model.jnt_range[1:]  # (29, 2) — [min, max] per joint
      default_pose = self.model.qpos0[7:]      # (29,) — 默认关节角
      # 反归一化
      target_qpos = action * (joint_range[:,1] - joint_range[:,0]) / 2 + default_pose
      # PD: torque = kp * (target - current) - kd * velocity
      kp, kd = self._pd_gains  # 从 config 读取
      qpos = self.data.qpos[7:]   # 当前关节角 (跳过 free joint 的 7 维)
      qvel = self.data.qvel[6:]   # 当前关节速度 (跳过 free joint 的 6 维)
      return kp * (target_qpos - qpos) - kd * qvel
```

### 输出

- `gear_sonic/envs/mujoco_env.py`
- `tests/test_mujoco_env.py`: 单 env 测试
  - test_reset: reset 后 obs shape 正确
  - test_step: step 100 次无崩溃, reward 在合理范围
  - test_termination: 极端 action 触发 terminated
  - test_history: history buffer 正确滑动

### 验证标准

- [ ] 与 StubEnv 的 obs_dict 输出维度完全一致 (930, 1645, 1761)
- [ ] 1000 步连续执行, 无 crash, 无 NaN
- [ ] reward 范围 [-5, 10] 内 (参考 Isaac Sim)
- [ ] done=True 时正确 auto-reset, terminal_obs 非 None
- [ ] 内存无泄漏 (10000 步后内存稳定)

### 预估工时: 2 天

---

## Task 4: 观测一致性验证

### 目标

对比 MuJoCo 产出的 observation 与 StubEnv 产出的 observation，确认数值分布一致。

### 依赖

- Task 3 (MuJoCoEnv 可用)
- 与 Task 1/2/8 可并行开发

### 详细设计

```
Step 4.1: 对照实验设置
  - 使用相同的 motion_id 和 start_time
  - StubEnv: 纯运动学计算 obs
  - MuJoCoEnv: step(action=0) → obs (action=0 让机器人保持初始姿态)
  - 比较 1000 个随机 motion 的 obs 差异

Step 4.2: 对比指标
  - actor_obs: 每维 mean/std 差异, 相关性 > 0.95
  - critic_obs: 同上
  - tokenizer_obs: 应完全一致 (同 motion_id/time)
  - reward: 同 action 下差异 < 20%

Step 4.3: 差异来源分析
  - MuJoCo 有物理 (重力、接触) → critic_obs 中接触力和地形会不同
  - StubEnv 无物理 → critic_obs 缺少接触信息
  - 预期: tokenizer_obs 完全相同, actor_obs 接近, critic_obs 有差异
  - 记录差异量, 判断是否需要调整 reward 权重

Step 4.4: 长时间序列对比
  - 同一 motion, 500 步 rollout
  - 对比 StubEnv (开环) vs MuJoCo (闭环) 的 trajectory 差异
  - 验证 MuJoCo 物理对行为的影响在合理范围内
```

### 输出

- `reports/obs_comparison.md`: 对比报告
- `reports/obs_comparison_plots/`: 分布对比图

### 验证标准

- [ ] tokenizer_obs 完全相同 (同输入 → 同输出)
- [ ] actor_obs 各维度相关性 > 0.90
- [ ] reward 均值差异 < 30%
- [ ] 差异来源有合理解释 (物理 vs 无物理)

### 预估工时: 1 天

---

## Task 5: MuJoCoEnvManager 并行管理器

### 目标

实现多进程并行环境管理器，通过共享内存 + Barrier 同步，管理 4096 个 MuJoCo 环境。

### 前置要求 (Task 5 开始前必须完成)

- [ ] `_obs()` critic_obs 不再用零占位 — 需要真实 motion reference + contact 信息
- [ ] `_obs()` tokenizer 不再用零占位 — 需要 encoder 输入格式的 motion 数据
- [ ] `_compute_reward()` 接入 config 权重 — 9 项奖励按 sonic_release.yaml 计算
- [ ] MuJoCoEnv 拆分为独立模块 `gear_sonic/envs/mujoco_env.py`

### 依赖

- Task 3 (MuJoCoEnv 稳定，上述 4 项已完成)

### 详细设计

```
Step 5.1: SHM 内存管理模块
  ⚠️ 跨进程传递: SharedMemory 对象不可 pickle!
     正确做法: 父进程创建 → 拿到 shm.name (字符串)
              子进程 SharedMemory(name=shm.name) attach

  class EnvSharedMemory:
      def __init__(self, num_envs):
          创建 5 个 SharedMemory 区域, 保留 name 字符串:
            obs_shm:        num_envs × 4336 × 4B  = 71 MB
            terminal_shm:   num_envs × 4336 × 4B  = 71 MB
            actions_shm:    num_envs × 29 × 4B    = 475 KB
            rewards_shm:    num_envs × 4B          = 16 KB
            dones_shm:      num_envs × 1B          = 4 KB
          self._names = {k: shm.name for k, shm in ...}  # 传给子进程
          self._shms = {...}  # 父进程直接引用
          提供 read/write 方法, 使用全局 env_id 计算偏移

      def attach(self):
          """子进程调用: 通过 name 重新 attach."""
          self._shms = {k: SharedMemory(name=n) for k, n in self._names.items()}

Step 5.2: Worker 进程实现
  # ⚠️ ceil(4096/160)=26 时, 实际只用 158 workers (26×157 + 14)
  #    最后几个 worker 的 num_envs==0, 不 spawn
  #    Barrier 计数用实际 worker 数 actual_workers + 1

  def _worker_loop(worker_id, start_env, num_envs, shm_names, barrier):
      # ⚠️ shm_names 是 dict[str, str], 可 pickle
      shm = EnvSharedMemory.attach_from_names(shm_names)
      motion_lib = MotionLibRobot(...)  # mmap, 所有 worker 共享
      envs = [MuJoCoEnv(model, motion_lib) for _ in range(num_envs)]
      
      # 初始 reset
      for i, env in enumerate(envs):
          shm.write_obs(start_env + i, env.reset())
      
      while True:
          barrier.wait()  # 等 trainer 写好 actions
          actions = shm.read_actions(start_env, num_envs)
          for i, env in enumerate(envs):
              obs, reward, done, info = env.step(actions[i])
              shm.write_obs(start_env + i, obs)
              shm.write_reward(start_env + i, reward)
              shm.write_done(start_env + i, done)
              if info.get("terminal_obs"):
                  shm.write_terminal(start_env + i, info["terminal_obs"])
          barrier.wait()  # 通知 trainer 完成

Step 5.3: Trainer-side step() 与 Worker 启动
  class MuJoCoEnvManager:
      def __init__(self, num_envs, num_workers):
          # 按需分配, 跳过 num_envs==0 的 worker
          envs_per_worker = ceil(num_envs / num_workers)
          self._actual_workers = 0
          for i in range(num_workers):
              start = i * envs_per_worker
              num = min(envs_per_worker, num_envs - start)
              if num > 0:
                  self._actual_workers += 1
                  spawn_worker(i, start, num)
          # Barrier 用实际 worker 数, 不是 num_workers
          self._barrier = Barrier(self._actual_workers + 1)

      def step(self, actions):
          self._shm.write_actions(actions)
          self._barrier.wait()
          self._barrier.wait()
          return (
              self._shm.read_obs(),
              self._shm.read_rewards(),
              self._shm.read_dones(),
              {"terminal_obs": self._shm.read_terminal()},
          )

Step 5.4: 小规模正确性测试
  - 64 envs, 4 workers
  - 与单进程串行结果逐位对比 (确定性: 同 seed 同 action 序列应同输出)
  - 不要求 bit-exact (浮点顺序可能不同), 但要求误差 < 1e-6

Step 5.5: 性能测试
  - 256 envs, 10 workers → 测量每步 worker 耗时
  - 4096 envs, 160 workers → 测量每步 worker 耗时
  - 预期: 最慢 worker ≤ 0.8ms × 26envs ≈ 21ms (4096/160=25.6→ceil=26; NPU 26ms 内)
  - 记录: 各 worker 的 step 耗时分布 (mean/P99/max)

Step 5.6: Worker 崩溃处理
  # 160 workers × 长时间训练, OOM/segfault 是大概率事件
  # Barrier 模型下任一 worker 崩溃 → trainer 永久阻塞

  class MuJoCoEnvManager:
      _BARRIER_TIMEOUT = 60  # 秒, worker 崩溃检测超时

      def step(self, actions):
          ...
          try:
              self._barrier.wait(timeout=self._BARRIER_TIMEOUT)
          except multiprocessing.BrokenBarrierError:
              # 有 worker 挂了 → 日志 + 重建
              self._handle_worker_crash()

      def _handle_worker_crash(self):
          # 1. kill 所有剩余 worker
          # 2. 重新 spawn, 所有 env 重新 reset
          # 3. ⚠️ PPO rollout 是 24-step 连续序列, 中间断一处则整段 GAE 不可靠
          #    必须丢弃当前 rollout 的全部 storage 数据, 从 reset 后的 obs
          #    重新开始新一轮 rollout, 不能只跳过当前 step
          # 4. 记录 crash 统计 (worker_id, iter, 时间)

      def _health_check(self):
          # 每 N iter 主动检查: 所有 worker.is_alive()
          # 如有死亡, 提前发现, 避免卡在 barrier
```

### 输出

- `gear_sonic/envs/mujoco_env_manager.py`
- `tests/test_env_manager.py`:
  - test_shm_read_write: SHM 读写正确性
  - test_small_parallel: 64 envs 并行 vs 串行结果一致
  - test_barrier_sync: Barrier 正确同步, 无死锁
  - test_auto_reset: done env 正确 reset
  - test_stress_4096: 4096 envs 100 步无 crash
  - test_worker_crash: 模拟 worker crash, trainer 检测并恢复

### 验证标准

- [ ] 64 envs 并行 vs 串行, obs/reward/done 差异 < 1e-6
- [ ] 4096 envs, 100 步, 无死锁, 无 crash
- [ ] SHM 读写无越界, 各 env 数据独立
- [ ] SHM 跨进程通过 name attach, 无 pickle 错误
- [ ] 最慢 worker 耗时 ≤ 25ms (安全余量: NPU 26ms/步)
- [ ] Worker 崩溃检测 + 自动恢复 (BrokenBarrierError 处理)
- [ ] 内存占用符合设计 (143MB SHM + 200MB MuJoCo data)

### 预估工时: 2.5 天

---

## Task 6: PPO Trainer 集成

### 目标

将 MuJoCoEnvManager 接入现有 PPO 训练管线，完成小规模端到端训练。

### 依赖

- Task 5 (MuJoCoEnvManager 可用)
- Task 4 (obs 一致性已验证)

### 详细设计

```
Step 6.1: 训练脚本适配
  # train_agent_trl.py 改动:
  
  # 原版:
  # from gear_sonic.envs.manager_env_wrapper import ManagerEnvWrapper
  # env = ManagerEnvWrapper(...)
  
  # MuJoCo 版:
  from gear_sonic.envs.mujoco_env_manager import MuJoCoEnvManager
  env = MuJoCoEnvManager(
      num_envs=cfg.num_envs,
      num_workers=cfg.mujoco_workers,
      model_xml=cfg.mujoco_model,
      motion_lib_cfg=cfg.motion_lib_cfg,
  )

Step 6.2: rollout 循环适配
  # ppo_trainer.py 不需要改动!
  # MuJoCoEnvManager.step() 接口与 ManagerEnvWrapper.step() 一致
  for step in range(num_steps_per_env):
      actions = policy_step(obs_dict)
      obs_dict, rewards, dones, infos = env.step(actions)
      storage.store(...)

Step 6.3: GAE 适配
  # GAE 反向递推时需要 next_values:
  #   delta = r + gamma * next_values * (1-done) - values
  #
  # done=True 时, env 已 auto-reset, storage 里存的是 reset 后的 obs
  # 但 GAE 需要的是 done 前那一刻的 value: V(s_terminal)
  # 用 infos["terminal_obs"] 算这个 bootstrap value:
  
  terminal_obs_batch = infos["terminal_obs"]  # shape [num_envs, 4336] (完整 obs)
  done_mask = dones.bool()
  if done_mask.any():
      # SHM 存的是完整 4336 维, Critic 只用其中 critic_obs(1645) 部分
      critic_obs = terminal_obs_batch[done_mask][:, actor_dim:actor_dim+critic_dim]
      next_values = values[step+1].clone()  # 当前用 reset 后的 obs 算的 value
      terminal_values = critic(critic_obs)
      next_values[done_mask] = terminal_values  # 替换为正确的 terminal value

Step 6.4: 小规模端到端测试
  - 配置: 64 envs, 1 worker, 100 iter, 1 NPU
  - 期望: 训练正常开始, loss 曲线合理 (与 StubEnv 类似)
  - 对比: 与 StubEnv 64 envs 的 loss/entropy/reward 趋势

Step 6.5: 中等规模测试
  - 配置: 256 envs, 4 workers, 100 iter, 4 NPU
  - 验证 DDP 梯度同步正常
  - 验证多 worker 下 rollout 数据一致性

Step 6.6: 全量配置测试
  - 配置: 4096 envs, 160 workers, 100 iter, 16 NPU
  - 测量每 iter 时间
  - 与设计值比较 (~2.2s/iter)

Step 6.7: DDP world_size 变化处理
  # 1 NPU (world_size=1): 无 DDP, 直接训练
  # 4 NPU (world_size=4): DDP 4-way
  # 16 NPU (world_size=16): DDP 16-way
  # 
  # ⚠️ 每次切换需要重启进程 (HCCL init 不支持运行时 resize)
  # 启动脚本: accelerate launch --num_processes=$NPU_COUNT
  # 配置对齐: cfg.algo.config.world_size = accelerator.num_processes
```

### 输出

- `gear_sonic/config/exp/mujoco_train.yaml`: MuJoCo 训练配置
- `scripts/train_mujoco.sh`: 训练启动脚本
- `reports/trainer_integration.md`: 集成记录和问题修复

### 验证标准

- [ ] 64 envs, 100 iter: 训练正常完成, 无 crash
- [ ] loss 曲线与 StubEnv 趋势一致 (entropy 上升, weighted_ppo_loss 下降)
- [ ] 256 envs, 100 iter: DDP 梯度同步正常
- [ ] 4096 envs, 100 iter: rollout 数据无 corruption
- [ ] GAE 在 terminal obs 边界正确处理 (无 value 跳变)

### 预估工时: 1.5 天

---

## Task 7: 全量训练 + 性能分析

### 目标

运行完整 100K iteration 训练，产出可部署的模型 checkpoint，对比分析性能。

### 依赖

- Task 6 (PPO 集成完成)

### 详细设计

```
Step 7.1: 完整训练运行
  - 配置: 4096 envs, 160 workers, 16 NPU, 100K iter
  - 监控: TensorBoard (loss, entropy, rewards, FPS)
  - 预计时间: ~2.0s/iter × 100K = ~56 小时 ≈ 2.3 天

Step 7.2: 性能分析
  - 记录每 iter 各阶段耗时: rollout, GAE, PPO update
  - 记录最慢 worker 的每步耗时分布
  - 与设计值对比, 识别瓶颈

Step 7.3: 与 StubEnv 结果对比
  - 训练指标: entropy, rewards, ep_length, latent losses
  - 模型质量: token → decoder → action 的一致性
  - Checkpoint: 模型参数分布

Step 7.4: 与 Isaac Sim 已知结果对比 (如有)
  - reward 分布, ep_length 收敛速度
  - sim-to-real 能力分析 (如果有真机测试条件)
```

### 输出

- `logs_rl/mujoco_train/` — 训练日志和 checkpoint
- `reports/mujoco_training_analysis.md` — 训练分析报告

### 验证标准

- [ ] 100K iter 无 crash 完成
- [ ] 产出可用 checkpoint (428MB, 包含 policy+value+optimizer)
- [ ] 吞吐量 ≥ 15K steps/s (保守, 与设计值 49K 有差距可接受)
- [ ] 训练指标合理: entropy 上升, latent_loss 收敛, ep_length 增长

### 预估工时: 2.5 天 (含训练等待时间)

---

## Task 8: OSMesa 渲染 (可选)

### 目标

实现 MuJoCo OSMesa 渲染，产出 `(camera_image, token)` 对用于 Step 2 VLA 数据集。

### 独立任务，无代码依赖

### 详细设计

```
Step 8.1: OSMesa 可用性验证
  renderer = mujoco.Renderer(model, 480, 640, backend='osmesa')
  # 如果 aarch64 不支持 OSMesa, 尝试 EGL headless
  # 如果都不支持, 标记为 "需要 GPU 渲染" 或 "用 Isaac Sim 离线渲染"

Step 8.2: 渲染管线
  class MuJoCoRenderer:
      def __init__(self, model, width, height):
          self.renderer = mujoco.Renderer(model, height, width)
      def render(self, data, camera='ego_view'):
          self.renderer.update_scene(data, camera=camera)
          return self.renderer.render()  # [H, W, 3]

Step 8.3: 数据采集脚本
  # 离线运行: 加载 MuJoCoEnv + encoder ONNX
  # 每个 env step: render camera + encoder(motion_ref, state) → token
  # 保存 (image, token) 为 LeRobot 格式

Step 8.4: 渲染性能测试
  - 单帧渲染耗时 (480×640, OSMesa, Kunpeng 920)
  - 估计: 15-20ms/帧 (CPU 软件渲染)
  - 批量渲染 150K 帧 (100 demos) 预估时间
```

### 输出

- `gear_sonic/envs/mujoco_render.py`
- `scripts/render_vla_data.py` — 数据采集脚本

### 验证标准

- [ ] OSMesa 或 EGL 在 aarch64 上可用 (或确认不可用)
- [ ] 渲染图像分辨率正确 480×640, RGB
- [ ] 图像内容合理 (能看到机器人和地面, 不是黑屏)
- [ ] 单帧耗时记录

### 预估工时: 1 天

---

## 任务依赖图

```
Day 1-2:   Task 1 ─┐          Task 2 ─┐
                    │                   │
Day 3-4:            ├──→ Task 3 ←──────┘
                    │       │
Day 4-5:            │       ├──→ Task 4 (对比验证, 1天)
                    │       │
Day 5-7:            │       └──→ Task 5
                    │               │
Day 8-9:            │               └──→ Task 6
                    │                       │
Day 10-12:          │                       └──→ Task 7 (训练中)
                    │
Day 3 (并行):      Task 8 (OSMesa, 独立)

总计: 10-12 天 (含训练等待时间)
并行开发可压缩到 ~8 天
```

## 里程碑

| 里程碑 | 完成标志 | 预计 |
|--------|---------|------|
| M1: 模型就绪 | G1 MuJoCo 模型跑通, PD 控制站立 | Day 1 |
| M2: 单环境可用 | MuJoCoEnv 通过全部单元测试 | Day 4 |
| M3: 并行就绪 | 4096 envs × 100 步无 crash | Day 7 |
| M4: 训练集成 | 64 envs × 100 iter 端到端 | Day 9 |
| M5: 全量训练 | 100K iter 完成, checkpoint 可加载 | Day 12 |
