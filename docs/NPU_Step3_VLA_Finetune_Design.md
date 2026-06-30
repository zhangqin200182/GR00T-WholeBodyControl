# Step 3 NPU VLA 微调设计：Image → SONIC Token

## 1. 背景

### 1.1 VLA 在 SONIC 中的角色

```
部署时:
  Camera + Text → VLA → token[64] → SONIC Decoder (frozen) → action[29] → 机器人

训练时 (Step 3):
  LeRobot 数据集 → VLA 微调: (image, text, state) → token[64] + hands[14]
                   目标: 预测的 token 接近 GT token (Step 2 产出)
```

VLA 只需要学会一件事：**从图片预测 SONIC token**。SONIC decoder 完全不动。

### 1.2 当前方案（NVIDIA 技术栈）

```bash
# Isaac-GR00T fine-tuning (vla_workflow.md:114-129)
export NUM_GPUS=4
uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag UNITREE_G1_SONIC \
    --num-gpus 4 \
    --global-batch-size 32 \
    --max-steps 20000
```

| 参数 | 值 | 说明 |
|------|-----|------|
| Base model | GR00T N1.7-3B | NVIDIA 专有 VLA |
| 参数量 | ~3B | Vision + LLM + Action head |
| GPU 需求 | 4 × A100/H100 | — |
| Batch size | 32 | 全局 |
| Training steps | 20,000 | — |
| 输出维度 | 78 | token[64] + 双手[14] |
| 推理频率 | 2.5 Hz | 每次产出 40 帧 chunk |

### 1.3 目标

GR00T N1.7 是 NVIDIA 专有模型，依赖 CUDA。需要在 **Ascend NPU** 上用开源 VLM 替代，实现同等的 fine-tuning 能力。

## 2. 模型架构

### 2.1 基础选型

GR00T N1.7 不是选项（NVIDIA 专有 + CUDA-only）。开源替代：

| 模型 | 参数量 | 视觉编码器 | 语言模型 | NPU 适配难度 |
|------|--------|----------|---------|------------|
| Qwen2.5-VL-3B | 3B | SigLIP-400M | Qwen2.5-2.5B | 中（已有 Ascend 适配） |
| InternVL2-2B | 2B | InternViT-300M | InternLM2-1.8B | 中 |
| SmolVLM-2B | 2B | SigLIP-400M | SmolLM2-1.7B | 低 |

**推荐：Qwen2.5-VL-3B**。3B 参数与 GR00T N1.7 对齐，Ascend 社区已有适配经验，视觉+语言双模完整。

### 2.2 模型结构

```
┌────────────────────────────────────────────────────────────┐
│  VLA Model (~3B)                                           │
│                                                            │
│  ┌──────────────────┐   ┌──────────────────┐              │
│  │ Camera Image     │   │ Text Prompt      │              │
│  │ [H×W×3]          │   │ "pick up cup"    │              │
│  └────────┬─────────┘   └────────┬─────────┘              │
│           │                      │                         │
│           ▼                      ▼                         │
│  ┌────────────────┐   ┌──────────────────┐               │
│  │ Vision Encoder │   │ Language Model   │               │
│  │ (SigLIP, 冻结)  │   │ (Qwen2.5-2.5B)  │               │
│  │ → [N, 1152]    │   │ → [L, 2048]     │               │
│  └────────┬───────┘   └────────┬─────────┘               │
│           │                    │                           │
│           └────────┬───────────┘                           │
│                    │                                       │
│                    ▼                                       │
│  ┌─────────────────────────────────────┐                  │
│  │ Cross-Attention / Projector         │                  │
│  │ vision tokens ⊕ text tokens         │                  │
│  │         ⊕ state_proj                │                  │
│  └────────────────┬────────────────────┘                  │
│                   │                                        │
│                   ▼                                        │
│  ┌─────────────────────────────────────┐                  │
│  │ Action Head (MLP)                   │   ← 从头训练      │
│  │ [hidden] → [2048] → [1024] → [78]  │                  │
│  │ token[64] + left_hand[7] +         │                  │
│  │ right_hand[7]                       │                  │
│  └─────────────────────────────────────┘                  │
│                                                            │
│  Robot State [N] ──→ MLP Proj ──→ state tokens [D]       │
└────────────────────────────────────────────────────────────┘
```

### 2.3 训练策略：LoRA + 冻结视觉编码器

```
组件              冻结?    训练方式      原因
────────────────  ──────  ───────────  ──────────────────
Vision Encoder    冻结    不训练        预训练表示足够好，省显存
Language Model    LoRA    rank=64       NL 指令理解需要微调
Projector         全量    全量训练      视觉-语言-动作对齐
Action Head       全量    全量训练      新模块，从头训练
State Proj        全量    全量训练      新模块
───────────────────────────────────────────────────────
可训练参数:        ~150M (含 LoRA)，总 3B
```

## 3. 训练数据

### 3.1 数据集格式

LeRobot v2.1（由 Step 2 产出）：

```
outputs/2026-04-03-14-30-00-G1-robot01/
├── data/train-00000.parquet      ← 每帧一行
│   ├── observation.images.ego_view  ← camera frame (bytes/路径)
│   ├── observation.state.*          ← 机器人状态
│   ├── action.motion_token          ← GT token [64]
│   ├── action.left_hand_joints      ← [7]
│   ├── action.right_hand_joints     ← [7]
│   └── task_prompt                  ← 文本指令
├── videos/observation.images.ego_view/  ← MP4 视频
└── meta/
```

### 3.2 数据规模

```
每 task:  50-100 个 demonstration
每 demo:  30 秒 × 50 Hz = 1,500 帧
每 task:  75,000-150,000 帧
训练时:  每帧 1 个训练样本: (image, text, state) → (token, hands)
```

## 4. 训练配置

### 4.1 超参数（对齐 Isaac-GR00T）

| 参数 | 值 | 说明 |
|------|-----|------|
| base_model | Qwen2.5-VL-3B | 替代 GR00T N1.7 |
| LoRA rank | 64 | 语言模型 |
| LoRA alpha | 128 | — |
| max_steps | 20,000 | 对齐原版 |
| global_batch_size | 32 | 对齐原版 |
| learning_rate | 2e-4 | LoRA 适配 |
| lr_schedule | cosine | warmup=500 steps |
| optimizer | AdamW | weight_decay=0.01 |
| gradient_accumulation | 按 NPU 数量调整 | 保持 global batch=32 |
| precision | FP16 (mixed) | NPU 原生支持 |
| max_seq_length | 2048 tokens | 视觉+文本 |
| image_size | 384×384 | 对齐 SigLIP |

### 4.2 数据增强（对齐 Isaac-GR00T）

```bash
--color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

## 5. NPU 资源需求

### 5.1 显存估算

```
Qwen2.5-VL-3B, LoRA, batch=1, FP16:

  Vision Encoder (冻结):       ~0.8 GB
  Language Model (LoRA):       ~5.0 GB (base) + ~0.3 GB (LoRA)
  Projector + Action Head:     ~0.2 GB
  Optimizer (AdamW):           ~1.2 GB (仅可训练参数)
  激活值 (batch=1):             ~2.0 GB
  ─────────────────────────────────────
  每卡 batch=1 显存:           ~9.5 GB

  Ascend 910 单卡 HBM:        61.3 GB
  可容纳 batch size:           61.3 / 9.5 ≈ 6
  安全 batch (80%):            batch=4
```

### 5.2 NPU 数量与配置

| NPU 数量 | 每卡 batch | 梯度累积 | 等效 global batch | 每步时间 | 总训练时间 |
|---------|-----------|---------|------------------|---------|----------|
| 4 | 4 | 2 | 32 | ~2.0s | ~11 小时 |
| 8 | 4 | 1 | 32 | ~1.5s | ~8 小时 |
| 16 | 4 | — | 64 | ~1.2s | ~7 小时 |

> 注：VLA forward 包含视觉编码器（~200ms）和 LLM decode（~500ms），比纯 MLP 慢很多。估算基于 Qwen2.5-VL-3B 在 Ascend 910 上的性能数据。

### 5.3 硬件推荐

```
推荐配置:  4 × Ascend 910 NPU (61GB)
最小配置:  2 × Ascend 910 (batch=2, accumulate=4)
训练时长:  ~11 小时 / task (50-100 demos)
存储:      ~50GB (数据集) + ~20GB (checkpoint × 5)
```

## 6. 训练流程

### 6.1 数据预处理

```bash
# Step 1: 验证数据集格式
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/my_task \
    --output-path outputs/my_task_cleaned

# Step 2: 切分 train/val (90/10)
python scripts/split_vla_dataset.py \
    --dataset outputs/my_task_cleaned \
    --train-ratio 0.9

# Step 3: 生成 NPU 训练格式
python scripts/prepare_npu_training_data.py \
    --dataset outputs/my_task_cleaned \
    --output data/vla_train/ \
    --image-size 384 \
    --num-workers 8
```

### 6.2 训练启动

```bash
# 4 × NPU DDP
accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    train_vla_npu.py \
    --base-model Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset data/vla_train/ \
    --output-dir checkpoints/vla_my_task/ \
    --max-steps 20000 \
    --global-batch-size 32 \
    --gradient-accumulation 2 \
    --lora-rank 64 \
    --learning-rate 2e-4 \
    --precision fp16 \
    --save-steps 5000 \
    --save-total-limit 5 \
    --color-jitter brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --use-wandb
```

### 6.3 训练监控

```
关键指标:
  loss/train:          MSE(pred_token, gt_token)，应持续下降至 ~0.01
  token_accuracy:      预测 token 各维度 sigmoid 后的准确率
  action_recovery:     每 N steps: pred_token → Decoder → action vs GT action
  eval/loss:           验证集 loss，防止过拟合

收敛判断:
  loss 在 15K-20K steps 稳定 → 停止
```

### 6.4 Checkpoint 导出

```bash
# 转为部署格式
python scripts/export_vla_for_deploy.py \
    --checkpoint checkpoints/vla_my_task/checkpoint-20000 \
    --output deploy/vla_my_task/ \
    --format onnx \       # ONNX → ATC → OM
    --merge-lora          # 合并 LoRA 权重
```

## 7. 与 Isaac-GR00T 的差异

| | Isaac-GR00T | NPU 方案 |
|---|------------|---------|
| Base model | GR00T N1.7 (3B) | Qwen2.5-VL-3B |
| 框架 | NVIDIA TRT-LLM | torch_npu / Ascend |
| 视觉编码器 | 内部 SigLIP | SigLIP-400M (标准) |
| LoRA | 支持 | rank=64 (默认) |
| 推理部署 | TRT-LLM server | OM + ACL runtime |
| 训练时间 | ~8h (4×A100) | ~11h (4×Ascend 910) |

## 8. 风险评估

| 风险 | 影响 | 缓解 |
|------|------|------|
| Qwen2.5-VL NPU 适配不完全 | 算子不支持，训练失败 | 提前用 dummy batch 验证全流程 |
| LoRA rank 太小 | 欠拟合，token 预测不准 | 从 rank=64 起步，可调大到 128 |
| 视觉编码器 NPU 推理慢 | 训练时间翻倍 | torch.compile + FP16 + 降低 image_size |
| 数据集 token 质量差 | VLA 学不到有效行为 | Step 2 质量检查：token→decoder→action 一致性 |
| 多 NPU 通信瓶颈 | DDP 效率低 | HCCL 已验证（16 卡 StubEnv） |

## 9. 总结

```
Step 3 NPU VLA 微调:

  模型:      Qwen2.5-VL-3B + LoRA + Action Head
  训练:      image → token[64]+hands[14], MSE loss
  数据:      50-100 demos, ~100K frames (Step 2 产出)
  NPU:       4 × Ascend 910 (61GB)
  训练时长:   ~11 小时 / task
  不需要:    NVIDIA GPU, Isaac-GR00T, GR00T N1.7

核心变化:
  GR00T N1.7 → Qwen2.5-VL-3B (开源替代)
  TensorRT    → torch_npu + Ascend
  CUDA        → HCCL + CANN

SONIC decoder 全程冻结，不受影响。
```

> *本方案基于 `docs/source/tutorials/vla_workflow.md` 的训练参数和 `gear_sonic/scripts/run_vla_inference.py` 的模型接口分析。*
