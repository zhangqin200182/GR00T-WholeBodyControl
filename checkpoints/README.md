# 微调权重（纯策略，CPU 格式）

> 完整 checkpoint（含优化器状态，448MB）在 NPU 服务器：
> `/data/z00666713/GR00T-WholeBodyControl/logs_rl/TRL_G1_Stub/stub_train_mujoco_ft-20260815_214321/last.pt`

| 文件 | 来源 | global_step | 说明 |
|---|---|---|---|
| `policy_round1_5000.pt` | 一轮微调完成 | 5000 | 奖励+4088/长度1145；能站稳不推进 |
| `policy_round2_final_11000.pt` | 二轮训练完成 | 11000 | **最终版**：行进段 0.73 vs 0.72 m/s，训练均值奖励 +9207/长度1476 |

格式：`{"policy_state_dict": {name: tensor}, "global_step": int}`，已剥离 torch_npu 引用，任意机器可直接 `torch.load(..., map_location="cpu")`。

加载示例（Mac 诚实模式渲染）：
```bash
SONIC_NOCLIP=1 MUJOCO_GL=cgl SONIC_CKPT=$PWD/checkpoints/policy_round2_6550.pt \
  .venv_sim/bin/python scripts/record_walk.py 2000 out.mp4
```

完整二轮曲线见 docs/assets/tb_*.png（至 11000 步）。

## Round-3（多动作课程，17000 迭代）
- `policy_round3_final_17000.pt` — 33 条 BONES-SEED 动作课程（31 新 + 原 2 条步行），从 round-2 11000 续训
- MuJoCo 成绩：步行 1000 步满/速度 0.64 vs 0.63 m/s；舞蹈整段；跳 275 步；疾跑未解（0.82 vs 1.79）
- Isaac 回传 4 步（特化地板），详见 docs/EXPERIMENTS.md E13/E14
