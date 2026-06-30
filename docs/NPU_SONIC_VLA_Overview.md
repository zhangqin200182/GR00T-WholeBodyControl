# SONIC + VLA 全 NPU 训练方案

## 总览

当前 SONIC 训练全链路依赖 NVIDIA GPU（Isaac Sim 物理仿真 + TensorRT 推理 + GR00T N1.7 VLA 微调）。本方案将 **模型训练、数据生成、VLA 微调** 全部迁移到华为昇腾 NPU，GPU 仅保留 Isaac Sim 物理仿真。

```
                        ┌─ Step 1 ─┐      ┌── Step 2 ──┐      ┌── Step 3 ──┐
                        │          │      │            │      │            │
    BONES-SEED ────→  SONIC      ──→  Encoder  ──→  LeRobot   ──→  VLA 微调   ──→  部署
    运动数据         预训练         ONNX    离线    数据集        (NPU)        token
                    (NPU+GPU)      token生成  token             image→token  →SONIC
                        │          │      │            │      │            │  decoder
                        │          │      │            │      │            │
    GPU 资源:         64 GPU        GPU   0             0      0            0
    NPU 资源:          16 NPU        NPU  1 NPU          0     4 NPU         0
    产出:           checkpoint    token  (img,token)   VLA    控制策略
```

## 方案对比

| | 原版 (NVIDIA) | 本方案 (NPU) |
|---|-------------|------------|
| **Step 1 训练** | 64 GPU (仿真+模型同卡) | 64 GPU (仿真) + 16 NPU (模型) |
| **Step 2 数据** | 机器人遥操作采集 | NPU 离线 encoder → token |
| **Step 3 VLA** | Isaac-GR00T N1.7 (4 GPU) | Qwen2.5-VL-3B (4 NPU) |
| **部署推理** | TensorRT + GPU | OM + NPU / CPU decoder |

---

## Step 1: SONIC 预训练

### 目标

训练 SONIC 的 Universal Token Module（三编码器 → FSQ → 两解码器），产出：
- Encoder ONNX：`obs → token[64]`
- Decoder ONNX：`token[64] + state → action[29]`

### 架构

```
64 GPU (Isaac Sim 物理仿真) ←→ 16 NPU (PPO 训练)

  64 × GPU                             16 × NPU
  ┌──────────┐                        ┌──────────┐
  │ PhysX 3.5ms│ ── TCP ──────────→   │ Encoder   │
  │ 64 envs   │ ←── action ────────   │ FSQ       │
  │ obs 1.1MB │                        │ Decoder   │
  └──────────┘                        │ PPO Loss  │
      ...                             │ 辅助 Loss  │
  ┌──────────┐                        │ AllReduce  │
  │ GPU-63   │ ←──────────────→      └──────────┘
  └──────────┘                             ...
                                      ┌──────────┐
                                      │ NPU-15   │
                                      └──────────┘
```

### 关键数据

| 参数 | 值 |
|------|-----|
| 模型大小 | 37.4M (Actor 25.9M + Critic 11.5M) |
| 全局 envs | 4,096 |
| 训练量 | 100K iter × 98K transitions = ~98.3 亿 |
| 每步耗时 | ~36ms (优化后 ~6ms) |
| 训练时长 | ~2.7 天 (优化后 ~0.8 天) |
| 产出 | checkpoint (428MB) + ONNX encoder/decoder |

### 详细文档

→ [`docs/NPU_GPU_Remote_Training_Design.md`](NPU_GPU_Remote_Training_Design.md)

---

## Step 2: VLA 训练数据生成

### 目标

产出 `(camera_image, robot_state, latent_token)` 三元组，用于 VLA 微调。

### 方案：NPU 离线 token 生成

SONIC 预训练已经学到了 token 的语义。不需要真人遥操作：

```
方案 A: Isaac Sim 渲染 (有 GPU 时)
  BONES-SEED SMPL → Isaac Sim (headless, no physics)
    ├──→ 渲染 camera image @ 30 Hz
    └──→ SMPL Encoder → token
  → 完整数据集

方案 B: 真人遥操作 (有机器人时)
  VR 遥操作 → 记录 camera + robot_state + motion_ref
  NPU 离线: Encoder(motion_ref, robot_state) → token
  → 完整数据集

方案 C: 纯离线 (最低成本)
  已有运动数据 + NPU Encoder → token
  camera 来自仿真渲染或真实录制回放
```

### 关键数据

| 参数 | 值 |
|------|-----|
| Encoder 推理延迟 | ~2ms/帧 (FP32), ~0.4ms (compile+FP16) |
| 每 demo 帧数 | 30s × 50Hz = 1,500 帧 |
| 100 demo 处理时间 | ~5 分钟 (NPU batch=1) |
| 硬件需求 | 1 × Ascend 910 |
| 产出 | LeRobot v2.1 数据集 |

### 详细文档

→ [`docs/NPU_Step2_DataCollection_Design.md`](NPU_Step2_DataCollection_Design.md)

---

## Step 3: VLA 微调

### 目标

训练 VLA 学会 `image + text → token`，替代 SONIC Encoder 在部署时的角色。

### 模型架构

```
Camera [384×384] → Vision Encoder (SigLIP, frozen)
Text "pick up cup" → Language Model (Qwen2.5-2.5B, LoRA)
Robot State → MLP Proj
    │
    ▼
Cross-Attention → Action Head → token[64] + hands[14]
                                     │
                              MSE(pred, GT token)
```

### 训练策略

| 组件 | 方式 | 参数量 |
|------|------|--------|
| Vision Encoder | 冻结 | 400M (不训) |
| Language Model | LoRA rank=64 | 2.5B → ~100M |
| Action Head | 全量 | ~50M |
| **可训练总计** | | **~150M / 3B** |

### 关键数据

| 参数 | 值 |
|------|-----|
| Base model | Qwen2.5-VL-3B |
| 训练数据 | 50-100 demos, ~100K frames |
| 训练步数 | 20,000 |
| Global batch | 32 |
| NPU 需求 | 4 × Ascend 910 |
| 训练时长 | ~11 小时 / task |
| 产出 | VLA checkpoint (LoRA weights + action head) |

### 详细文档

→ [`docs/NPU_Step3_VLA_Finetune_Design.md`](NPU_Step3_VLA_Finetune_Design.md)

---

## 部署

三步完成后，部署链路：

```
┌──────────────────────────────────────────────────────────┐
│                       部署架构                            │
│                                                          │
│  Camera → VLA (NPU/GPU) → token[64] ──→ SONIC Decoder   │
│    ↑                       ↑                (C++/OM)     │
│  @ 30 Hz                @ 2.5 Hz              @ 50 Hz    │
│                                         token+state→29   │
│                                              ↓           │
│                                           电机驱动        │
└──────────────────────────────────────────────────────────┘

组件分布:
  VLA Model:       NPU (OM format, Ascend ACL runtime)
                   或 GPU (ONNX/TensorRT, 如果需要)
  SONIC Decoder:   机器人端 CPU/NPU (OM, C++ 控制循环)
  Camera Server:   机器人端 (systemd service)
  ZMQ Bridge:      Python (VLA client → C++ deploy)
```

---

## 资源总览

| 阶段 | GPU | NPU | 时长 | 产出 |
|------|-----|-----|------|------|
| Step 1 训练 | 64 × A100 | 16 × Ascend 910 | 1-3 天 | Encoder + Decoder ONNX |
| Step 2 数据 | 0 | 1 × Ascend 910 | ~1 小时 | LeRobot 数据集 |
| Step 3 VLA | 0 | 4 × Ascend 910 | ~11 小时 | VLA checkpoint |
| **总计 NPU** | **0** (GPU仅仿真) | **16 (训练) + 4 (VLA)** | **~3 天** | 完整 VLA + SONIC |

---

## 与原版路径对比

```
原版 (NVIDIA):
  Isaac-GR00T repo + GR00T N1.7 + 4 GPU
  → 需要 NVIDIA 专有模型和 TensorRT
  → 不可迁移到 NPU

本方案:
  开源 VLM (Qwen2.5-VL-3B) + torch_npu
  → 全链路 NPU 可控
  → SONIC decoder 复用原版 ONNX
  → VLA 只替代 Encoder，Decoder 不动
```

## 关键设计原则

1. **SONIC Decoder 永远不动** — 37M 小模型，PPO 训练好的，任何情况下都是 `token + state → action`
2. **Encoder 可被 VLA 替代** — 训练时需要（产 GT token），部署时 VLA 产出 token 注入
3. **token 是跨模态统一语言** — BONES-SEED / VR 遥操作 / camera → 都是同一个 64-dim 空间
4. **物理仿真是唯一的 GPU 依赖** — 其余全部可在 NPU 上完成

---

> *本方案综合了以下设计文档：*
> - *[NPU+GPU Remote Training Design](NPU_GPU_Remote_Training_Design.md)*
> - *[Step 2 Data Collection Design](NPU_Step2_DataCollection_Design.md)*
> - *[Step 3 VLA Fine-tuning Design](NPU_Step3_VLA_Finetune_Design.md)*
> - *[SONIC Training Report](../SONIC_Training_Report.md)*
