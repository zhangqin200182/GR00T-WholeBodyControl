# NPU 微调 + Mac 渲染全流程指导（含全部踩坑记录）

> 2026-08-16 整理 | NPU：119.8.36.80（8× 昇腾 910B3，192 核 CPU）；Mac：本地渲染
> 任务：基于官方 SONIC 37M 权重，在 MuJoCo CPU 物理里微调（NPU 训练），并导出视频
> 状态：微调进行中 ✅ checkpoint 导出 Mac 渲染 ✅
> 配套文档：《GPU环境搭建指导-IsaacSim.md》（云 GPU / Isaac Sim 侧）

---

## 1. 架构总览（谁在哪干什么）

```
NPU 服务器（容器 sonic-train-zhouzhi，--runtime=ascend）
 ├─ 学习端：npu:0 单卡（37M 模型只占 4.6G/64G 显存、AICore ~16%）
 ├─ 物理端：64 个 CPU 进程跑 MuJoCo（192 核机器，采体 1.5s + 学习 1.3s/迭代）
 └─ 每 50 迭代自动存 checkpoint → logs_rl/TRL_G1_Stub/<run>/last.pt

Mac（本地）
 ├─ 从 NPU 抽取纯策略权重（绕开 torch_npu 序列化）
 └─ .venv_sim + MUJOCO_GL=cgl 离屏渲染视频
```

关键路径（容器与宿主机同路径挂载 `/data/z00666713`，宿主机可直接 scp）：
- 代码仓：`/data/z00666713/GR00T-WholeBodyControl`（容器内相同路径）
- 官方权重：`sonic_release/last.pt`（step 41550）
- 微调输出：`logs_rl/TRL_G1_Stub/stub_train_mujoco_ft-20260815_214321/`
- 训练日志：`/data/z00666713/mujoco_ft4.log`（当前）
- 过程记录：`/data/z00666713/debug_notes.md`

---

## 2. 微调启动（验证过的命令）

```bash
ssh root@119.8.36.80
docker exec sonic-train-zhouzhi bash -c "cd /data/z00666713/GR00T-WholeBodyControl && \
  SONIC_MUJOCO_ENV=1 WANDB_MODE=disabled nohup python3 gear_sonic/train_agent_trl.py \
  +exp=stub_train exp_var=mujoco_ft +resume=true \
  checkpoint=logs_rl/TRL_G1_Stub/stub_train_mujoco_ft-20260815_214321/last.pt \
  algo.config.num_learning_iterations=5000 \
  num_envs=256 +mujoco_workers=64 use_wandb=false \
  > /data/z00666713/mujoco_ft4.log 2>&1 &"
```

要素说明：
- `SONIC_MUJOCO_ENV=1`：切到 MuJoCoEnvManager（CPU 物理），不用 Isaac
- `+exp=stub_train`：提供 PPO/模型超参（继承官方 ppo_im_phc，全部未改）
- `checkpoint=`：起点权重；`+resume=true`：带优化器状态续训
- **首次从官方权重开始**时不加 resume：`checkpoint=sonic_release/last.pt`（不恢复官方优化器，全新 PPO 进程）
- NPU 自动检测：脚本强制 fp32（NPU 无 bf16 torch.normal）、device npu:0

监控：
```bash
docker exec sonic-train-zhouzhi bash -c 'ps aux | grep -c "[t]rain_agent_trl"'   # ~66=正常
grep -E "Mean rewards|Mean length" /data/z00666713/mujoco_ft4.log | tail -4
```

---

## 3. NPU 侧踩坑全记录（症状 → 根因 → 解法）

**N1. `Could not override 'xxx'`（hydra 启动即退出）**
stub_train 配置里没有的键**全部要 `+` 前缀**：`+resume=true`、`+mujoco_workers=64`、`+max_train_steps=N`。已有键（num_envs/checkpoint/use_wandb）不用加。

**N2. `max_train_steps=1000` 只跑了 100 迭代就存盘退出**
真正控制迭代数的是 **`algo.config.num_learning_iterations`**（meta 里的 max_train_steps 只是它存档时的镜像）。顶层 `+max_train_steps` 会被静默忽略，用默认 100。

**N3. resume 找错 checkpoint：`FileNotFoundError: ...20260815_221631/last.pt`**
resume 的 glob 取"最新时间戳目录"，但失败运行也会创建自己的日志目录（空的）→ 可能选中空目录。**永远显式传 `checkpoint=<绝对路径>`**，不要依赖自动查找。

**N4. NPU 克隆里 mujoco_env.py 是同事的旧版（动作空间错误）**
git 克隆不带我在 Mac 上的修复 → 官方权重进去动作缩放全错、立即摔。**重建环境时必须把 Mac 修复版 `gear_sonic/envs/mujoco_env.py` 覆盖过去**（特征自查：`grep -c act_scale mujoco_env.py` 应 >0；原件备份在 `/data/z00666713/mujoco_env.py.orig`）。

**N5. 存档崩溃：`Expected a Storage of type float ... got type Byte`（deepcopy）**
训练 ~150 迭代首次定期存档时 `copy.deepcopy(state)` 遇到 torch_npu 的 Byte storage 张量崩溃（官方起点跑 100 迭代不触发，续训加载优化器状态后才出现）。补丁位置 `gear_sonic/trl/callbacks/model_save_callback.py`：deepcopy 包 try/except，失败退 `copy.copy`（torch.save 反正序列化当前值）。

**N6. 训练用的入口是 `gear_sonic/train_agent_trl.py`，不是 scripts/ 下的**
scripts/ 里是同事/我的辅助脚本；训练入口在 gear_sonic 包根目录。

**N7. 容器挂载必须同路径**
`-v /data/z00666713:/data/z00666713`（不是 :/data），否则宿主机 scp 进来的文件容器内路径对不上。

**N8. NPU 单卡就够，别开 8 卡 DDP**
瓶颈是 CPU 采集（192 核全喂 64 worker），学习端只占 16% AICore。8 卡只有在上千环境规模才有意义。

**N9. torch_npu 序列化污染 checkpoint**
NPU 存的 last.pt 含 torch_npu 类引用，无 torch_npu 的机器（Mac）`torch.load` 直接 `ModuleNotFoundError: No module named 'torch_npu'`。**必须在 NPU 上先抽纯策略**（见 §4），不要直接把 last.pt 拿去 Mac 用。

---

## 4. checkpoint 导出 → Mac 渲染（验证过的流程）

### 4.1 NPU 上抽取纯 CPU 策略权重

```bash
ssh root@119.8.36.80 'docker exec sonic-train-zhouzhi python3 -c "
import torch
ckpt = torch.load(\"/data/z00666713/GR00T-WholeBodyControl/logs_rl/TRL_G1_Stub/stub_train_mujoco_ft-20260815_214321/last.pt\", map_location=\"cpu\", weights_only=False)
out = {\"policy_state_dict\": {k: v.detach().cpu() for k, v in ckpt[\"policy_state_dict\"].items()},
       \"global_step\": ckpt[\"state\"].global_step}
torch.save(out, \"/data/z00666713/policy_only.pt\")
print(\"SAVED, step:\", out[\"global_step\"])"
'
scp root@119.8.36.80:/data/z00666713/policy_only.pt <本地仓库>/policy_only.pt   # ~100MB
```

注意先确认 last.pt 不在写入中（`ls -la` 看 mtime 距今 >1 分钟；每 50 迭代存一次）。

### 4.2 Mac 渲染

```bash
cd /Users/max/code/ai/embody/GR00T-WholeBodyControl-feature-mujoco-training
MUJOCO_GL=cgl \
SONIC_CKPT=$PWD/policy_only.pt \
.venv_sim/bin/python scripts/record_walk.py 1000 /Users/max/code/ai/embody/g1_walk_ft.mp4
```

- **环境必须是 `.venv_sim`**（repo 里的虚拟环境，py3.10 + mujoco/torch/imageio 齐全）；系统 `python3` 没有 imageio，`~/.local/bin/python3.11` 没有 mujoco
- `MUJOCO_GL=cgl`：Mac 离屏渲染后端
- `SONIC_CKPT` 环境变量：指定权重（不设则默认 `sonic_release/last.pt` 官方权重），train_mujoco_sonic.py:31
- 参数：步数（默认 1000=20 秒）、输出路径
- 脚本自带：跟随相机、参考动作相位锁定、上身运动学覆盖

### 4.3 Mac 侧坑

**M1. `No module named 'imageio'` / `'mujoco'`**：用错解释器，必须 `.venv_sim/bin/python`。
**M2. `No module named 'torch_npu'`**：见 N9，先在 NPU 抽纯策略。
**M3. 渲染中杀进程**：video 是 imageio 直写，正常结束才有完整索引（和 GPU 侧 ffmpeg moov 问题同理）。

---

## 5. 效果基线（供后续对照）

微调曲线（256 envs × 24 steps/迭代，MuJoCo 物理，动作跟踪奖励）：

| 迭代 | 奖励 | 回合长度 |
|---|---|---|
| 0（官方零样本） | -104 | 7 步（0.14s） |
| 100 | +11 | 9.7 |
| 544 | +366 | 95 |
| 1003 | +1219 | 321（6.4s） |
| 1600 | — | Mac 渲染 1000 步无终止，走 63.6m |

健康指标参考：噪声 std 稳定 ~0.39（无塌缩）、熵稳定微升、每迭代 ~2.9s。

---

## 6. 快速重启清单（断电/换机后照做）

1. NPU 容器在跑？`docker ps | grep sonic-train-zhouzhi`；训练在跑？`ps aux | grep -c "[t]rain_agent_trl"`（~66）
2. 训练挂了 → 看 `/data/z00666713/mujoco_ft*.log` 尾部 → 用 §2 命令续跑（显式 checkpoint 路径）
3. 30 分钟自动监控在 ZCode 定时任务里（含自动重启命令），无需手工盯
4. 出视频 → §4 两步
5. GPU 侧（Isaac 对比实验）→ 见《GPU环境搭建指导-IsaacSim.md》
