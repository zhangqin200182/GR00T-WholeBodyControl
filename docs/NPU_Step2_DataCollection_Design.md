# Step 2 NPU 数据收集设计：遥操作 Token 生成

## 1. 背景：Step 2 需要什么计算

VLA 训练需要 `(camera_image, robot_state, latent_token)` 三元组数据集。其中 token 由 SONIC Encoder 实时产生：

```
VR 遥操作 → motion_reference → Encoder(motion_ref, robot_state) → token[64]
                                                                      │
                                                              顺手记录 + camera → 数据集
```

**Step 2 的计算完全是无物理仿真的神经网络推理。** 没有 Isaac Sim、没有 PhysX、没有渲染。只有两件事：

| 组件 | 输入 | 输出 | 频率 |
|------|------|------|------|
| Encoder | motion_ref + robot_state | token[64] | 50 Hz |
| Decoder (g1_dyn) | token[64] + robot_state | action[29] | 50 Hz |

## 2. 为什么 NPU 就够了

当前 C++ 部署栈使用 TensorRT（NVIDIA GPU），但实际计算量极小：

```
Encoder: 单 encoder MLP [in→2048→1024→512→512→out(64)]
         参数量 ~3-5M，batch=1
         FP32 推理: GPU ~0.3ms, NPU ~1-2ms (eager)

Decoder (g1_dyn): MLP [994→2048→2048→1024→1024→512→512→29]
                  参数量 ~8-10M，batch=1
                  FP32 推理: GPU ~0.5ms, NPU ~2-3ms (eager)
```

**NPU 推理延迟远低于 20ms 控制周期要求。** 不需要 GPU。

## 3. 两种 NPU 部署方案

### 方案 A：NPU 实时推理（在线）

```
┌── 机器人端 (CPU) ──────────────────────────────────────────┐
│                                                              │
│  VR 遥操作 → motion_reference + 摄像头 + IMU + 电机编码器      │
│       │                                                      │
│       │ robot_state (154D) + motion_ref                      │
│       │                                                      │
│       ▼ ZMQ/TCP                                              │
│  ┌──────────────┐                                            │
│  │ NPU Server   │  ← 局域网内一台 NPU 机器                    │
│  │              │                                            │
│  │ Encoder.om → token[64]                                    │
│  │ Decoder.om → action[29]                                   │
│  └──────┬───────┘                                            │
│         │ token + action                                     │
│         ▼                                                    │
│  action → 电机驱动                                           │
│  token + camera → 数据集记录                                  │
└──────────────────────────────────────────────────────────────┘
```

- **优点**：实时，encoder/decoder 都在回路里，token 就是当时机器人正在执行的
- **缺点**：网络延迟 1-5ms（同机房），需要 NPU 一直在线

### 方案 B：NPU 离线后处理（推荐起步）

```
┌── 机器人端 (CPU/Jetson) ────────────────────────────────────┐
│                                                              │
│  VR 遥操作 → motion_reference 直接发给简单轨迹跟踪器            │
│       │                                                      │
│       ├──→ MPC/Joint PD → 电机驱动                           │
│       │                                                      │
│       └──→ 记录: camera_image + robot_state + motion_ref     │
│                              ↑                               │
│                        不需要 encoder!                        │
└──────────────────────────────────────────────────────────────┘
           │
           │ 离线
           ▼
┌── NPU 离线处理 ─────────────────────────────────────────────┐
│                                                              │
│  遍历录制的每一帧:                                            │
│    Encoder.om(motion_ref, robot_state) → token[64]          │
│    → 写入数据集: (camera_image, robot_state, token)          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

- **优点**：零实时性压力，机器人端零改动（不需要跑 encoder），NPU 离线慢慢算
- **缺点**：录制的 token 和"如果当时用 SONIC decoder"的 token 可能有细微差异（因为 decoder 闭环 vs MPC 开环），但不影响 VLA 训练质量 — VLA 只是学 image→token 映射

## 4. 编码器模型详解

### 4.1 模型结构

ONNX 导出（`export_universal_token_encoders_as_onnx`）包含全部 3 个 encoder：

```
obs_dict [encoder_index(1) + tokenizer_obs(D)]
     │
     ├── encoder_index=0 → G1 Encoder
     ├── encoder_index=1 → Teleop Encoder
     └── encoder_index=2 → SMPL Encoder
     │
     ▼
encoded_tokens [64]  ← 对所有encoder统一输出
```

每个 encoder backbone：`Linear(in→2048) → SiLU → Linear(2048→1024) → SiLU → Linear(1024→512) → SiLU → Linear(512→512) → SiLU → Linear(512→64)`

### 4.2 部署时观测输入

从 C++ observation_config.yaml，部署时 encoder 需要：

| 观测 | 维度 | 说明 |
|------|------|------|
| encoder_index | 1 | 选择哪个 encoder（通常用 g1=0） |
| motion_joint_positions | 29 | 运动参考关节位置 |
| motion_joint_velocities | 29 | 运动参考关节速度 |
| motion_anchor_orientation | 6 | 根节点朝向 (6D) |
| base_angular_velocity | 3 | 机器人角速度 |
| body_joint_positions | 29 | 当前关节位置 |
| body_joint_velocities | 29 | 当前关节速度 |
| last_actions | 29 | 上一步动作 |
| **总计** | **~155** | |

> 注：实际部署可能使用多帧观测（如 `motion_joint_positions_5frame_step5`），维度会更大。单帧是最简配置。

### 4.3 模型大小与推理时间

```
Encoder (3-in-1 ONNX):
  参数量:      ~8-12M (3个encoder合计，共享 tokenizer_obs 解析)
  ONNX 大小:   ~50 MB
  输出:        [64] float32

Decoder (g1_dyn ONNX):
  参数量:      ~8-10M
  ONNX 大小:   ~40 MB
  输入:        token[64] + proprioception[930] ≈ 994
  输出:        action[29]
```

NPU 推理时间（Ascend 910, batch=1, FP32）：

| 模型 | eager | +compile | +FP16 |
|------|-------|----------|-------|
| Encoder | ~2ms | ~0.8ms | ~0.4ms |
| Decoder | ~3ms | ~1.2ms | ~0.6ms |
| **合计** | **~5ms** | **~2ms** | **~1ms** |

远低于 20ms 控制周期。

## 5. 数据收集流程

### 5.1 方案 B 详细流程（推荐）

```
Phase 1: 机器人端录制（不需要 NPU）
─────────────────────────────────
  操作员在机器人前做遥操作演示
  摄像头录制 @ 30-60 Hz
  每 20ms 记录一帧: camera_image, robot_state, motion_ref
  
  工具: C++ deploy 的 Data Exporter + 摄像头
  产出: LeRobot 原始数据集（缺 token 列）

Phase 2: NPU 离线批处理
────────────────────────
  加载 Encoder.om
  for each frame in dataset:
    encoder_input = concat([encoder_index=0, motion_ref, robot_state])
    token = encoder(encoder_input)  # [64]
    写入 frame["action.motion_token"]

  产出: 完整 LeRobot 数据集（带 token）
  
Phase 3: 质量检查
─────────────────
  随机抽 10 个 episode:
    token → Decoder.om(token, robot_state) → predicted_action
    vs recorded motion_ref 对比 → 动作一致性
```

### 5.2 处理时间估算

```
单个 demo:  30 秒 × 50 Hz = 1,500 帧
100 个 demo: 150,000 帧

NPU encoder batch=1:  ~2ms/帧 (eager)
  150,000 帧 × 2ms ≈ 300s ≈ 5 分钟

NPU encoder batch=8:  ~3ms/8帧 ≈ 0.4ms/帧
  150,000 × 0.4ms ≈ 60s ≈ 1 分钟

实际瓶颈: 数据 I/O，不是 NPU 推理
```

## 6. 资源需求

### 6.1 硬件

| 阶段 | 硬件 | 用途 |
|------|------|------|
| 数据录制 | 机器人 + 摄像头 + VR | 遥操作采集 |
| Token 生成 | **1 × NPU (Ascend 910)** | encoder 推理 |
| 数据存储 | ~500GB 磁盘 | 100 demo × ~5GB/demo (含视频) |

### 6.2 软件

| 组件 | 说明 |
|------|------|
| Encoder.om | SONIC checkpoint → ONNX → ATC 转换 |
| Python/NPU runtime | torch_npu + ONNX Runtime (Ascend backend) |
| 或 C++ ACL | Ascend Computing Library, 低延迟 |

### 6.3 不需要的

- ❌ Isaac Sim / PhysX
- ❌ NVIDIA GPU / TensorRT
- ❌ PPO 训练栈
- ❌ FSQ 量化器（encoder 输出连续 latent，量化在训练用，部署 encoder 可以直接输出 latent）

## 7. 与现有 C++ 部署栈的关系

```
当前 C++ 部署栈 (Jetson + TensorRT):
  EncoderEngine (TRT) → PolicyEngine (TRT) → 电机

NPU 方案:
  Option 1: 替换 InferenceEngine
    将 TensorRT 替换为 Ascend ACL
    C++ 代码改动: 推理接口适配，其余不变
    适用: 机器人端有 NPU

  Option 2: Python 离线处理 (推荐)
    机器人端录制原始数据 + 简单轨迹跟踪
    NPU 离线跑 Python encoder
    机器人端零改动
```

## 8. 总结

```
Step 2 NPU 数据收集:

  为什么 NPU 够:     只有 encoder/decoder 推理，没有物理仿真
  推理延迟:           ~2ms/帧 (eager), ~0.4ms (compile+FP16)
  控制周期:           20ms → 推理占比 <10%
  数据集生成:         100 demo × 1500 帧 ≈ 5 分钟 NPU 离线处理
  硬件:               1 × NPU
  不需要:             Isaac Sim, GPU, TensorRT

推荐方案 B（离线处理）:
  机器人正常录数据（不需要 SONIC encoder）
  → NPU 离线批量生成 token
  → 零实时性风险，机器人端零改动
```

> *本方案基于 `gear_sonic/utils/inference_helpers.py` 的 ONNX 导出逻辑和 `gear_sonic_deploy/` 的 C++ 部署架构分析。*
