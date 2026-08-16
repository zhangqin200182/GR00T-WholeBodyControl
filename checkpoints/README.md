# 微调权重（纯策略，CPU 格式）

> 完整 checkpoint（含优化器状态，448MB）在 NPU 服务器：
> `/data/z00666713/GR00T-WholeBodyControl/logs_rl/TRL_G1_Stub/stub_train_mujoco_ft-20260815_214321/last.pt`

| 文件 | 来源 | global_step | 说明 |
|---|---|---|---|
| `policy_round1_5000.pt` | 一轮微调完成 | 5000 | 奖励+4088/长度1145；能站稳不推进 |
| `policy_round2_6550.pt` | 二轮训练中段快照 | 6550 | **当前最佳**：速度贴合 0.58 vs 0.56 m/s |

格式：`{"policy_state_dict": {name: tensor}, "global_step": int}`，已剥离 torch_npu 引用，任意机器可直接 `torch.load(..., map_location="cpu")`。

加载示例（Mac 诚实模式渲染）：
```bash
SONIC_NOCLIP=1 MUJOCO_GL=cgl SONIC_CKPT=$PWD/checkpoints/policy_round2_6550.pt \
  .venv_sim/bin/python scripts/record_walk.py 2000 out.mp4
```

注意：二轮训练（目标 11000 步）完成后，最终权重将更新到本目录。
