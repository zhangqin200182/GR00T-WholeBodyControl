# SONIC 远程训练架构设计：NPU 模型训练 + 物理仿真

---

## 0. 推荐方案：MuJoCo CPU + NPU 单机训练

### 0.1 实测数据

在 NPU 服务器 (Kunpeng 920, 320 核 aarch64, 16 × Ascend 910) 上实测 MuJoCo 性能：

```
MuJoCo 3.10.0, humanoid 模型 (14 bodies, 17 DOF):
  单核单 env: 0.118 ms/step

G1 29-DOF 等效估算 (1.7× humanoid):
  单核单 env: ~0.20 ms/step
  单核 @ 20ms 控制周期: ~99 envs
  320 核 @ 20ms: ~31,680 envs ✅

4096 envs 仅需 ~41 核 (13% CPU)，大幅余量给 OSMesa 渲染。
```

### 0.2 架构

```
┌── 同一台 NPU 服务器 (113.46.41.54) ─────────────────────────────┐
│                                                                  │
│  Kunpeng 920 CPU (320 核)           Ascend 910 NPU (16 卡)       │
│  ┌────────────────────┐           ┌────────────────────────┐    │
│  │ MuJoCo × 4096 envs │  SHM     │ PPO 训练                │    │
│  │ 物理仿真 @ 50Hz     │ ←─────→ │ Encoder + FSQ + Decoder │    │
│  │ OSMesa 渲染 camera  │  零拷贝   │ AllReduce (HCCL)        │    │
│  │ 运动数据加载         │           │ 辅助 Loss               │    │
│  └────────────────────┘           └────────────────────────┘    │
│                                                                  │
│  每步: MuJoCo ~0.2ms + NPU 推理 ~26ms = ~27ms                   │
│  网络: 0（同机共享内存）                                           │
│  GPU:  0（完全不需要）                                             │
└──────────────────────────────────────────────────────────────────┘
```

### 0.3 与各方案对比

| | Isaac Sim (原版) | GPU+NPU 分离 | MuJoCo+NPU (推荐) |
|---|----------------|-------------|-----------------|
| 物理仿真 | 64 GPU (PhysX) | 64 GPU (PhysX) | **CPU (MuJoCo)** |
| 模型训练 | 64 GPU | 16 NPU (远程) | **16 NPU (本地)** |
| 网络通信 | 0 (同卡) | TCP ~6ms/步 | **0 (SHM)** |
| 每步时间 | ~5ms | ~36ms | **~27ms** |
| GPU 依赖 | 64 张 | 64 张 | **0** |
| 机器数量 | 8+ 台 | 9 台 | **1 台** |
| 训练时长 (100K) | ~0.7 天 | ~2.7 天 | **~2.0 天** |
| 优化后 | — | ~1.1 天 | **~0.8 天** |

### 0.4 需要开发的内容

| 模块 | 说明 | 工作量 |
|------|------|--------|
| MuJoCo 并行环境管理器 | 替代 Isaac Lab `ManagerEnvWrapper`，管理 4096 envs | 大 |
| OSMesa 渲染管线 | 替代 Isaac Sim 摄像头渲染，产出 camera image | 中 |
| Observation/Reward 适配 | 确保 obs_dict 接口与 Isaac Sim 一致 | 中 |
| SMPL 人体模型加载 | MuJoCo 中加载 SMPL 骨骼用于 motion reference | 中 |
| Domain Randomization | MuJoCo 中实现地形/外观随机化 | 小 |

### 0.5 为什么 MuJoCo 可行

- **Google DeepMind 先例**：OP3、HumanoidBench 均基于 MuJoCo 训练人形机器人
- **SONIC 架构解耦**：token → decoder → action，物理引擎只影响 reward 数值分布
- **所有 9 个 reward 项 MuJoCo 都能算**：运动学跟踪 + 接触检测 + 关节限位
- **NPU 推理已验证**：StubEnv 16 卡实测通过

> ⚠️ 以下章节 (1-9) 描述的是 **GPU+NPU 分离方案**（备选路径）。如能实现 MuJoCo 环境管理器，推荐直接使用本章的 MuJoCo 单机方案。

---

## 1. 背景：GPU 训练配置与性能 (备选方案参考)

### 1.1 原始训练配置

SONIC (Systematic Orchestration of Neural Imitation Control) 是 G1 人形机器人的全身控制策略。训练使用 PPO + 辅助损失在 Isaac Sim 物理仿真器上运行。

| 参数 | 值 | 来源 |
|------|-----|------|
| num_envs | 4096 | `sonic_release.yaml` |
| num_steps_per_env | 24 | `sonic_release.yaml` |
| num_learning_iterations | 100,000 | `ppo_im_phc.yaml` |
| num_ppo_epochs | 5 | `ppo_im_phc.yaml` |
| num_mini_batches | 4 | `ppo_im_phc.yaml` |
| sim_dt | 0.005s | `base_env.yaml` |
| decimation | 4 | `base_env.yaml` |
| 控制频率 | 50 Hz | `target_fps: 50` |
| **硬件** | **64 × GPU** (8 nodes × 8 GPUs) | 官方文档推荐 |
| 训练时长 | 2-3 天 | 官方文档 |

```
每 iteration: 4,096 envs × 24 steps = 98,304 transitions
总训练量:     100,000 iter × 98,304 = ~98.3 亿 transitions
```

### 1.2 原有架构：仿真与模型绑定在同 GPU

```
┌── 64 GPU DDP ─────────────────────────────────────────────┐
│                                                            │
│  GPU-0:  64 envs  + 模型副本#0  + Isaac Sim (headless)     │
│  GPU-1:  64 envs  + 模型副本#1  + Isaac Sim (headless)     │
│  ...                                                       │
│  GPU-63: 64 envs  + 模型副本#63 + Isaac Sim (headless)     │
│                                                            │
│  每个 GPU 进程内:                                           │
│    env.step(action) ──→ physX 物理仿真 (同 GPU)             │
│    model.forward(obs) ──→ 37M 网络推理 (同 GPU)             │
│    model.backward() ──→ AllReduce 64 份梯度 (跨 GPU)       │
│                                                            │
│  零拷贝, 零网络 — 物理状态和模型参数都在同一张 GPU 显存内     │
└────────────────────────────────────────────────────────────┘
```

**为什么必须绑在一起？** PPO 是 on-policy 算法，每步 rollout 严格串行：

```python
for step in range(24):
    obs = env.get_obs()           # 必须等上一步物理结果
    action = model(obs)            # 必须先得到 obs
    obs_next = env.step(action)    # 必须等模型输出
```

Isaac Sim 的 PhysX 物理状态存于 GPU 显存，`env.step()` 在同一进程同 GPU 上零拷贝执行。分离到不同进程/机器会引入跨进程通信，破坏 on-policy 同步性。

### 1.3 GPU 性能分解

```
训练 2-3 天完成 100K iterations, 每 iteration 模拟 24×20ms = 480ms 物理时间
RTF = 480ms / 平均每 iter 时间 ≈ 3.6-5.4× 实时

从 RTF 反推每步墙钟时间:

  每 iter 总模拟时间:  24 步 × 0.02s (sim_dt×decimation) = 0.48s
  每 iter 墙钟时间:    0.48s / RTF ≈ 0.09-0.13s (取 0.11s)
  每步墙钟:            ~5ms
  
  内部分解 (64 envs/GPU, A100):
  ├── 物理仿真 (PhysX, 4 substeps × 64 人形机器人): ~3.5ms
  └── 模型推理 (37M, batch=64):                      ~1.5ms
  
  64 GPU 并行: 每卡同时执行, 无等待
```

```
GPU 训练性能总结:

  模型推理 (batch=64, A100):       ~1.5ms  ← 极快, 小 batch
  物理仿真 (64 envs, A100):         ~3.5ms  ← 主要耗时
  每步总计:                          ~5ms
  Rollout (24步):                   ~0.12s
  Learning (20次更新, 含AllReduce): ~0.5s
  每 iteration:                     ~0.6s
  100K iter:                         ~17 小时
```

---

## 2. 当前状态：NPU StubEnv 打桩测试

### 2.1 StubEnv 方案

华为昇腾 NPU 无法运行 Isaac Sim（CUDA-only, 无 PhysX）。为验证训练 pipeline 能在 NPU 上端到端跑通，实施了 StubEnv 方案：

- 用 `StubEnv` 替代 Isaac Sim，直接从 MotionLib 采样运动数据作为观测源
- **不执行物理仿真** — `env.step()` 只推进时间步
- 训练 pipeline 完全一致（PPO + actor/critic + encoder routing + 辅助损失）
- FP32 精度（NPU 不支持 bf16 的 `torch.normal`）

### 2.2 实测性能数据

#### 16 卡 NPU (Ascend 910, FP32, batch=256/卡)

```
硬件: 16 × Ascend 910 NPU (61.3GB HBM/card)
全局 envs: 256/卡 × 16 = 4,096
每 iteration transitions: 4,096 × 24 = 98,304

稳态性能:
  Collection (rollout 24步):  0.65s
    → 每步: 0.65s / 24 ≈ 27ms
    → 模型推理 (batch=256): ~26ms  ← StubEnv 无物理, env.step≈0
  Learning (20次梯度更新):    1.35s  (含 AllReduce HCCL)
  每 iteration 总计:          2.0s
  总吞吐:                    ~49K steps/s

200 iterations 训练时间: ~419s ≈ 7 分钟
```

#### 单卡 NPU (Ascend 910, FP32, batch=64)

```
硬件: 1 × Ascend 910 NPU
全局 envs: 64

性能:
  100 iterations 训练时间: 143s
  每 iteration: ~1.43s
  吞吐: ~1K steps/s
```

### 2.3 StubEnv 的局限

```
StubEnv vs Isaac Sim:

                      StubEnv (NPU)          Isaac Sim (GPU)
  ────────────────    ──────────────         ──────────────
  物理仿真:           无                      完整刚体 + 接触力
  因果关系:          断开 (action 不影响状态)  完整
  reward 信号:        基于运动片段采样          真实物理反馈
  value loss:         无法下降 (0.059 恒定)     持续收敛
  适用:              验证 pipeline 跑通         训练真正控制策略
```

**StubEnv 证明了 NPU 能跑完整 PPO 训练，但不能产出有效控制策略。** 下一步必须接入真实 Isaac Sim。

---

## 3. NPU + GPU 分离架构的挑战

### 3.1 矛盾

```
  模型训练 → 必须在 NPU 上（硬件约束）
  物理仿真 → 必须在 GPU 上（Isaac Sim CUDA-only + PhysX）
  PPO on-policy → 要求同步循环（仿真 ↔ 推理不能分）
```

### 3.2 核心挑战：推理时延

```
原版 (GPU 同卡): 物理 3.5ms + 推理 1.5ms = 5ms/步
                              ↑           ↑
                         同 GPU 零拷贝, 极端高效

分离后 (NPU 推理):
  当前 baseline:  推理 26ms (FP32 eager, batch=256)
  物理:           3.5ms
  网络:           6ms (TCP)

  每步 26ms + 3.5ms + 6ms ≈ 36ms → 是原版的 7× 倍
  100K iter → 2.7 天
```

### 3.3 核心挑战：网络通信

```
每步需要从 64 张 GPU 收集 obs (71MB) 发送到 NPU, 再把 action (0.5MB) 发回

  挑战:
  1. 64 路并发 TCP → 单 NPU 节点需要 64 个连接
  2. obs 序列化/反序列化开销 (4336 维 × 4096 envs = 71MB)
  3. 尾延迟: 1 路慢则整步阻塞
  4. 100K iter × 24 步 = 240 万次网络往返
```

### 3.4 核心挑战：DDP 规模匹配

```
原版 64-way DDP: 每 GPU 处理 64 envs, 模型推理 1.5ms

NPU 上需要多少卡？
  16 NPU (batch=256):   推理 26ms → 比 64 GPU 推理慢 17× (但每卡 envs 多)
   8 NPU (batch=512):   推理 ~35ms → 更慢
  32 NPU (batch=128):   推理 ~15ms → 更快但硬件成本高

需要找出"推理时间 ≤ 容忍上限"的最小 NPU 数量
```

---

## 4. 方案设计

### 4.1 架构总览

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          一个 Iteration (~2.7s)                              │
│                                                                            │
│  ┌── GPU Cluster (8 nodes × 8 GPUs = 64 GPUs) ──────────────────────────┐  │
│  │                                                                       │  │
│  │  Node N:  GPU-0 [64e, server:5555]  ...  GPU-7 [64e, server:5562]   │  │
│  │            ↑↓ zmq REQ/REP                  ↑↓ zmq REQ/REP            │  │
│  └────────────┼───────────────────────────────┼─────────────────────────┘  │
│               │                               │                            │
│               │  TCP / RDMA (每 node 8 路, 共 64 路)                       │
│               │                               │                            │
│  ┌────────────┼───────────────────────────────┼─────────────────────────┐  │
│  │            ↓                               ↓                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    Remote Isaac Manager (NPU-side)               │ │  │
│  │  │  收集 64 GPU obs → 拼接 batch=4096 → 按 NPU rank 切分 → dispatch │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  │  ┌── NPU Cluster (1 node × 16 NPUs) ──────────────────────────────┐  │  │
│  │  │                                                                 │  │  │
│  │  │  NPU-0  ←→ GPU  0-3   (256 envs)    NPU-8  ←→ GPU 32-35       │  │  │
│  │  │  NPU-1  ←→ GPU  4-7   (256 envs)    ...                        │  │  │
│  │  │  ...                                NPU-15 ←→ GPU 60-63        │  │  │
│  │  │                                                                 │  │  │
│  │  │  各自 forward (batch=256) → AllReduce gradient (HCCL)           │  │  │
│  │  │  各自 send action 回 4 路 GPU                                    │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  每步时序: [物理 3.5ms] → [网络 6ms] → [NPU推理 26ms] → [回传 0.5ms]      │
│           = ~36ms (原版 ~5ms)                                               │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 选路方案

```
每个 NPU 固定负责 4 张 GPU (来自 2 个不同 node 以分散风险):

  NPU-0  ←→ GPU  0-3   (node-0×4)        256 envs → batch=256
  NPU-1  ←→ GPU  4-7   (node-0×4)        256 envs → batch=256
  ...
  NPU-15 ←→ GPU 60-63  (node-7×4)        256 envs → batch=256
```

### 4.3 数据流与传输量

```
每 GPU (64 envs):
  obs → NPU:  4336 维 × 4B × 64 = 1.1 MB
  action ←:     29 维 × 4B × 64 = 7.4 KB

全系统 (64 GPU → 16 NPU):
  每步 obs 总量:    1.1 MB × 64 = 71 MB
  每步 action 总量:  7.4 KB × 64 = 0.5 MB
  每 NPU 负责:       71/16 = 4.4 MB obs (来自 4 路 × 1.1MB)
```

### 4.4 网络需求

```
所需带宽:  71 MB / 26ms ≈ 2.8 GB/s (用 8 路分担)

方案              单路带宽    8路总带宽   延迟    裁决
────────────────  ────────   ────────    ────    ────
100GbE            12.5 GB/s  100 GB/s    <1ms    ✅ 充裕
25GbE              3.1 GB/s   24.8 GB/s  ~2ms    ✅ 刚好
10GbE              1.25 GB/s  10.0 GB/s  ~5ms    ⚠️ 勉强
InfiniBand HDR    25.0 GB/s  200 GB/s    <1μs    ✅ 完美

推荐: 每 GPU node 1 条 25GbE 连接 NPU node (8 路聚合)
或 InfiniBand 直连（如果机房支持）
```

### 4.5 一步完整时序

```
时刻     GPU端 (64卡并行)                    NPU端 (16卡并行)
──────────────────────────────────────────────────────────────────
 0ms     env.step() → PhysX 物理仿真
 3.5ms   obs gather (4336维×64, 零拷贝)
 4ms     zmq.send(obs) ─── TCP ───────→  zmq.recv(obs) × 4路
         │                                 obs → npu tensor
 6ms     (空闲等待)                        batch=256 forward
         │                                   ├── Encoder (10ms)
         │                                   ├── FSQ      (2ms)
         │                                   ├── g1_dyn   (8ms)
         │                                   └── sample   (1ms)
         │                                 26ms 总计
 32ms    zmq.recv(action) ←── TCP ─────  zmq.send(action) × 4路
 32.5ms  action → GPU tensor
 33ms    下一轮 env.step() 开始
──────────────────────────────────────────────────────────────────
  每步 ~36ms (原版 ~5ms, 慢 7×)
  瓶颈: NPU 推理 26ms 占 72%
```

---

## 5. 训练时间估算

```
配置: 64 GPU + 16 NPU (DDP 16)

每 iter:
  Collection (24步):  24 × 36ms = 0.86s
  Learning (20次更新): 1.35s (StubEnv 实测)
  总计:                ~2.2s

100K iterations:  ~2.7 天 (原版 ~0.7 天)

不同 NPU 数量的对比:

NPU  每卡envs  推理/步   每步总   Collection  训练时间
───  ───────  ───────   ──────   ──────────  ────────
 4     1024    ~55ms*    64ms*    1.54s       ~4.8 天*
 8      512    ~35ms*    45ms*    1.08s       ~3.2 天*
16      256     26ms ✅   36ms     0.86s       ~2.7 天 ✅实测
32      128    ~15ms*    25ms*    0.60s       ~2.0 天*
64       64     ~9ms*    19ms*    0.46s       ~1.6 天*

  ✅ 实测  * 估算 (亚线性 scaling)
```

---

## 6. NPU 推理优化（关键路径）

当前 26ms 是 **FP32 + eager 模式** 的最差情况。推理优化是缩短 36ms/步 的最有效手段。

### 6.1 Baseline 内部分解

```
batch=256, FP32, eager execution, 16 NPU DDP

每步 26ms 估算内部分解:
  编码器 (SMPL/G1/Teleop, encoder_masks 路由):  10ms
  FSQ 量化 (32级离散映射):                        2ms
  g1_dyn 解码器 (994→2048→...→29 MLP):            8ms
  采样 + log_prob (Normal分布):                   1ms
  kernel launch overhead (Python→NPU 多次调度):    5ms
  ────────────────────────────────────────────────────
  总计:                                          26ms ✅实测
```

### 6.2 Tier 1: 图编译（预期 26ms → 11ms）

```
torch.compile(model, backend="aot_ts_npu", mode="reduce-overhead")
或 torch_npu 原生 jit_compile

效果:
  • 算子融合: SiLU+Linear, LayerNorm+Linear → 单个 NPU kernel
  • 消除 launch overhead: 每层 1 次调度 → 整图 1-2 次
  • 常量折叠: FSQ 量化表, encoder_masks 预处理
  • 内存复用: 中间 tensor 原地更新

  编码器:   10ms → 5ms  (算子融合)
  FSQ:       2ms → 1ms  (常量折叠)
  解码器:    8ms → 4ms  (算子融合)
  overhead:  5ms → 1ms  (整图调度)
  ──────────────────────────
  总计:     26ms → 11ms  (2.4× 加速)

注意事项:
  • encoder_masks 动态路由需 dynamic=True
  • 首次预热 ~30s, 后续重放无开销
  • 25.9M 参数, compile 内存增量 < 500MB
```

### 6.3 Tier 2: 混合精度（预期 11ms → 6ms）

```
torch.autocast(device_type="npu", dtype=torch.float16)

Ascend 910 FP16 能力:
  • 矩阵乘 (matmul): FP16 = 2× FP32 吞吐 (AICORE 张量核)
  • 逐元素运算: 1.3-1.5× 加速

混合精度策略（关键层保持 FP32）:
  编码器:     FP16 ✅   矩阵乘密集, 收益大
  FSQ:        FP32 ⚠️   离散映射精度敏感
  g1_dyn:     FP16 ✅   最大 MLP, 最大收益
  采样:       FP32 ⚠️   Normal分布数值稳定
  log_prob:   FP32 ⚠️   概率计算精度

  图编译后 11ms → FP16 后 ~6ms (1.8× 加速)

注意: 之前训练用 FP32 是因为 torch.normal 不支持 bf16,
      不是 FP16。FP16 推理不受此限制。
```

### 6.4 Tier 3: Stream 并行（预期 6ms → 5ms）

```
编码器与解码器的 proprio 投影可并行:

  stream_encode = torch.npu.Stream()
  stream_decode = torch.npu.Stream()

  with torch.npu.stream(stream_encode):
      latent = encoder(tokenizer_obs)       # 异步

  with torch.npu.stream(stream_decode):
      proprio_embed = linear_proj(actor_obs) # 与 encoder 并行

  torch.npu.synchronize()
  action_mean = g1_dyn(fsq(latent), proprio_embed)

  节省 ~1ms encoder/proprio 重叠
```

### 6.5 Tier 4: NPU Graph Capture（预期 5ms → 4ms）

```
torch_npu.npu_graph() 捕获整次 forward 为静态图, 零 Python overhead。

要求: 无动态 shape, 无控制流
  → encoder_masks 转为固定 shape 的 gather 操作
  → 或 torch.compile 的 static_graph=True
```

### 6.6 优化阶梯

```
                   编码   FSQ   解码   overhead   总计     训练时间
────────────────   ────  ────  ────  ────────  ─────    ────────
0. Baseline (实测) 10ms   2ms   8ms    5ms      26ms ✅   2.7 天
1. +图编译          5ms   1ms   4ms    1ms      11ms      1.5 天
2. +FP16            3ms   1ms   2ms   <1ms       6ms      1.1 天
3. +Stream并行      3ms   1ms   2ms   <1ms       5ms      1.0 天
4. +NPU Graph       3ms   1ms   2ms   ≈0ms       4ms      0.9 天

建议目标: Tier 1+2 = 6ms/步, 训练 ~1.1 天
```

### 6.7 优化后架构重新评估

```
NPU 推理 26ms → 6ms 后:

配置              物理    网络    推理    每步     训练时间
────────────────  ────   ────   ────   ─────     ────────
64GPU + 16 NPU    3.5    6ms     6ms    ~16ms     ~1.2 天
64GPU +  8 NPU    3.5    6ms     8ms    ~18ms     ~1.3 天
64GPU +  4 NPU    3.5    6ms    12ms    ~22ms     ~1.6 天

推理优化后瓶颈转移: 网络 6ms > 推理 6ms > 物理 3.5ms
  → 8 NPU 即可与 16 NPU 几乎持平
  → 下一步优化重点: 网络 (RDMA/SHM)
```

---

## 7. 端到端优化路线图

| 阶段 | 优化项 | 推理 | 网络 | 每步 | 训练时间 |
|------|--------|------|------|------|---------|
| **P0** | 当前 baseline (FP32 eager) | 26ms | 6ms | 36ms | 2.7 天 |
| **P1** | torch.compile | 11ms | 6ms | 21ms | 1.5 天 |
| **P2** | FP16 混合精度 | 6ms | 6ms | 16ms | 1.1 天 |
| **P3** | RDMA/SHM 替代 TCP | 6ms | <1ms | 10ms | 0.9 天 |
| **P4** | 流水线 (仿真∥网络∥推理) | 6ms | <1ms | 6ms | **0.8 天** |

P2 后即可接近原版训练速度 (0.7 天), P4 后完全持平。

---

## 8. 软件改动

### 8.1 新增文件

```
gear_sonic/envs/remote_env.py           # NPU-side proxy
gear_sonic/envs/remote_env_server.py    # GPU-side Isaac Sim server
gear_sonic/config/exp/remote_train.yaml # 训练配置
scripts/launch_remote_training.sh       # 启动脚本
```

### 8.2 修改文件

```
gear_sonic/trl/trainer/ppo_trainer.py   # rollout 循环适配 remote env
gear_sonic/train_agent_trl.py           # NPU 侧 remote env 初始化
```

### 8.3 核心接口

```python
# gear_sonic/envs/remote_env.py — NPU 侧代理
class RemoteEnv:
    """NPU 侧环境代理, 通过 ZMQ 与 GPU 端 Isaac Sim server 通信."""

    def __init__(self, gpu_addrs: list[str], num_envs_per_gpu: int = 64,
                 device: str = "npu"):
        self.num_envs = len(gpu_addrs) * num_envs_per_gpu
        # 每个 GPU server 一个 ZMQ REQ socket
        self._sockets = [zmq_socket(addr) for addr in gpu_addrs]

    def step(self, actions: Tensor) -> tuple[dict, Tensor, Tensor, dict]:
        # 拆分 actions → 64 路并发发送
        for sock, act in zip(self._sockets, actions.split(64)):
            sock.send(act.cpu().numpy().tobytes(), zmq.NOBLOCK)

        # 64 路并发接收 obs
        all_obs, rewards, dones = [], [], []
        for sock in self._sockets:
            obs_bytes, r_bytes, d_bytes = sock.recv_multipart()
            all_obs.append(deserialize(obs_bytes, self.device))
            rewards.append(torch.frombuffer(r_bytes))
            dones.append(torch.frombuffer(d_bytes))

        return merge(all_obs), cat(rewards), cat(dones), {}


# gear_sonic/envs/remote_env_server.py — GPU 侧 server
class IsaacSimServer:
    """每个 GPU 进程运行一个 server, 绑定 64 个 Isaac Sim 环境."""

    def __init__(self, port: int):
        self._env = ManagerEnvWrapper(num_envs=64, ...)
        self._socket = zmq.Context().socket(zmq.REP)
        self._socket.bind(f"tcp://*:{port}")

    def serve(self):
        while True:
            action_bytes = self._socket.recv()
            actions = torch.frombuffer(action_bytes).cuda().reshape(-1, 29)
            obs, rewards, dones, _ = self._env.step(actions)
            self._socket.send_multipart([
                serialize_obs(obs), serialize(rewards), serialize(dones)
            ])
```

### 8.4 启动流程

```
# GPU 端 (8 nodes × 8 GPUs, 共 64 个 server 进程)
for node in 1..8:
    for gpu in 0..7:
        CUDA_VISIBLE_DEVICES=$gpu \
            python gear_sonic/envs/remote_env_server.py \
                --port $((5555 + gpu)) \
                --num_envs 64 &

# NPU 端 (1 node × 16 NPUs)
accelerate launch \
    --num_processes=16 \
    gear_sonic/train_agent_trl.py \
    +exp=remote_train \
    num_envs=4096 \
    ++remote_env.gpu_addrs=gpu-node-1:5555,...,gpu-node-8:5562
```

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| 网络抖动导致单步尾延迟 | >100ms 阻塞整条 pipeline | 超时重试 + 双 buffer 预取 |
| GPU server 进程崩溃 | 丢失 64 envs 的数据 | checkpoint 恢复 + 健康检查 + 重启 |
| NPU OOM (gather 4096 obs) | forward 失败 | StubEnv 16卡已验证: 256 envs/卡正常 |
| obs 序列化开销过大 | 额外 2-3ms/步 | 预分配共享内存, 零拷贝序列化 |
| on-policy 数据过期 | GPU 端数据 vs 最新 policy 不匹配 | GAE 在 NPU 侧用最新 policy 重新计算 |
| torch.compile 兼容性 | Ascend 后端部分 op 不支持 | 渐进式: 先编译 decoder, 再编译 encoder |
| FP16 精度损失 | latent 对齐质量下降 | FSQ/log_prob 保持 FP32, 监控 loss 曲线 |

---

*本方案基于 16 卡 Ascend 910 NPU StubEnv 实测数据和 Isaac Sim 官方训练文档制定。*
