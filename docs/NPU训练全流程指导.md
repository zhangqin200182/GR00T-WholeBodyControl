# SONIC 训练全流程指导（NPU 训练 + CPU 推理 + 服务器离屏渲染）

> 2026-08-25 实测搭建于 192.168.0.47（8× 昇腾 910B3 / 192 核 CPU / aarch64）。
> 环境已全部就绪并通过训练冒烟验证（2 迭代 RC=0）。新同事按本文档即可跑通三件事：
> **NPU 上的微调训练、CPU/MuJoCo 推理评估、服务器上离屏渲染视频（导回本地查看）**。
> 配套历史文档：《NPU微调与Mac渲染指导.md》（旧机 aura-6 版，本文档为其 47 迁移版）。

---

## 0. 架构总览（谁在哪干什么）

```
NPU 服务器 192.168.0.47（容器 sonic-train，镜像 sonic-train:v14-base）
 ├─ 学习端：npu:0 单卡（37M 策略只占 ~4.6G/64G HBM，AICore ~16%）
 ├─ 物理端：CPU 进程跑 MuJoCo（192 核，训满配 +mujoco_workers=64）
 └─ 每 50 迭代自动存 checkpoint → logs_rl/TRL_G1_Stub/<run>/last.pt

你的电脑
 ├─ ssh 连服务器（经跳板 119.8.234.170 或内网直连，见 §1）
 └─ 渲染好的 mp4 从服务器 scp 回本地查看（见 §4.2）
```

**全流程图**（训练主循环 + 训练后评估/渲染）：

```
═══════════════════════ 训练（§2，仿真与学习同时在线滚动）═══════════════════════

  ┌─ NPU npu:0（学习端）───────────────────────────────┐
  │  策略网络 Actor（37M）                               │
  │  ① 读观测：actor_obs 930D + tokenizer 1761D          │
  │  ② 出动作：256 env × 29D                             │
  │  ④ 攒满 32 步 × 256 env → PPO 更新一次（= 1 迭代）    │
  └───────┬──────────────────────────────▲─────────────┘
      动作│ 共享内存（SHM 零拷贝）            │观测 + reward
          ▼                              │
  ┌─ CPU 64 个 MuJoCo worker 进程（物理端）─────────────┐
  │  ③ 每 worker 管 4 个 env：动作 → PD 力矩 → 物理推进    │
  │     （2ms × 10 子步 = 20ms 控制周期）→ 新观测 + reward │
  │  reward 参照：33 段动捕（sample_data/robot_filtered） │
  └────────────────────────────────────────────────┘
          │ 每 50 迭代
          ▼
  checkpoint 落盘：logs_rl/TRL_G1_Stub/<run>/last.pt
  曲线：tensorboard（<run>/tb/，见 §2.4）

═══════════ 训练后（§3/§4，同一脚本 record_walk.py，不占 NPU）═══════════

  checkpoint ──► CPU 单进程 MuJoCo 确定性 rollout ─┬─► 指标数字（§3 评估）
                                                  └─► OSMesa 渲染 mp4
                                                      ─► scp 回本地（§4）
```

对应章节：§2 训练 = 上图上半部分；§3/§4 = 下半部分。

## 1. 连接服务器

47 无公网 IP，经跳板机 119.8.234.170 转发。在**自己电脑**的 `~/.ssh/config` 加：

```
Host npu47
  HostName 192.168.0.47
  ProxyJump root@119.8.234.170
  User root
```

之后 `ssh npu47` 一条命令直达（密码找管理员要；建议把自己公钥追加进两台的
`~/.ssh/authorized_keys` 实现免密）。

⚠️ 如果你电脑上有 Surge/Clash 类代理（TUN/增强模式），SSH 可能被 fake-IP 劫持
（症状：解析到 198.18.x.x、TCP 通但 SSH banner 超时）。解法：给 `*.1617k.com` /
119.8.234.170 加 DIRECT 规则，或临时关闭增强模式。

## 2. 训练（NPU 学习端 + CPU 仿真端，容器已就绪）

> 训练 = **仿真与学习同时在线滚动**：NPU 上跑策略网络（学习端），64 个
> MuJoCo worker 进程在 CPU 上持续跑物理仿真（物理端），两者每步交替——
> 仿真产生数据，学习端消费数据更新策略。所以"CPU 上的仿真"就在本章里，
> 不是后面的章节。

### 2.1 进入容器

```bash
ssh npu47
docker exec -it sonic-train bash    # 容器常驻（--restart unless-stopped），直接进
cd /data/sonic/GR00T-WholeBodyControl
```

容器关键信息：
- 镜像 `sonic-train:v14-base`（含 torch 2.7.1+cpu / torch_npu 2.7.1.post2 / mujoco 3.11.0 / CANN 8.5.2）
- 启动参数：`--runtime=ascend --shm-size 32g --network host -v /data:/data -e ASCEND_VISIBLE_DEVICES=0-7`
- 代码在共享盘 `/data/sonic/GR00T-WholeBodyControl`（**勿放容器内路径**，容器重建不丢）
- 官方权重：`sonic_release/last.pt`（SONIC 官方 release，step 41550）
- 动捕数据：`sample_data/robot_filtered/`（33 个 PKL：210531 行走集 + bones_round3）

### 2.2 冒烟验证（新环境第一次跑，先跑这个）

```bash
SONIC_MUJOCO_ENV=1 WANDB_MODE=disabled python3 gear_sonic/train_agent_trl.py \
  +exp=stub_train exp_var=smoke checkpoint=sonic_release/last.pt \
  algo.config.num_learning_iterations=2 num_envs=8 +mujoco_workers=4 use_wandb=false
```

判据：跑完 2 迭代退出（RC=0），日志出现 `Learning iteration 1/2` 和 `Mean rewards`。
输出目录 `logs_rl/TRL_G1_Stub/stub_train_smoke-<时间戳>/`。

### 2.3 正式微调（从官方权重起步）

```bash
SONIC_MUJOCO_ENV=1 WANDB_MODE=disabled nohup python3 gear_sonic/train_agent_trl.py \
  +exp=stub_train exp_var=mujoco_ft \
  checkpoint=sonic_release/last.pt \
  algo.config.num_learning_iterations=5000 \
  num_envs=256 +mujoco_workers=64 use_wandb=false \
  > /data/sonic/ft_run1.log 2>&1 &
```

要素说明：
- `SONIC_MUJOCO_ENV=1`：切到 MuJoCo CPU 物理（不用 Isaac）
- `WANDB_MODE=disabled` + `use_wandb=false`：不上传 Weights & Biases 云端
  （47 无公网，连不上 wandb 服务器）。只是关掉云日志，训练照常跑，
  本地日志和 tensorboard 曲线（§2.4）不受影响；有公网的环境想去掉这两个开关即可
- `+exp=stub_train`：PPO/模型超参（继承官方 ppo_im_phc，未改）
- `checkpoint=sonic_release/last.pt`：起点权重；**首次从官方权重不加 resume**
  （不恢复官方优化器状态，全新 PPO 进程）
- **续训**（接着自己的 checkpoint）：加 `+resume=true` 并把 checkpoint 指向
  `logs_rl/TRL_G1_Stub/<run>/last.pt`
- NPU 自动检测：脚本强制 fp32（NPU 无 bf16 torch.normal），device npu:0
- 资源参考：37M 模型 + 256 env + 64 worker ≈ 单卡 NPU ~16% AICore + 64 CPU 核

**与其他任务共存**（2026-08-26 实测）：47 是多租户机器，别的容器可能也在用
NPU（当时 `cybergym-baseline-zhouzhi` 占着全部 8 卡：每卡 ~31G HBM、
AICore 88-99%）。昇腾容器间默认**没有隔离**：

- **AICore 是时间片共享**：多方同卡只是排队轮执行、互相拖慢，不会崩——
  本文档的冒烟 + 微调流程就是在对方满负载下跑通的（~4-5 秒/迭代）
- **HBM 是先占先得**：已分配的显存不会被挤掉，但双方合计超过 64G 时
  后申请者直接 OOM 退出。起训练前 `npu-smi info` 看一眼每卡剩余即可
  （SONIC 只要几个 G，当时每卡还剩 ~30G，余量充足）
- 换卡躲不开（对方 8 卡均沾）；要硬隔离需管理员用 vNPU 模板重建容器，
  一般没必要

### 2.4 监控

```bash
ps aux | grep -c "[t]rain_agent_trl"     # ~66 进程 = 正常
grep -E "Mean rewards|Mean length" /data/sonic/ft_run1.log | tail -4
npu-smi info                              # 看 npu:0 占用
ls -lt logs_rl/TRL_G1_Stub/               # checkpoint 落盘
```

**tensorboard 看曲线**：训练事件实时写在 `logs_rl/TRL_G1_Stub/<run>/tb/`。
47 无公网、容器又是 host 网络，用 ssh 端口转发在本地浏览器看：

```bash
# 终端 A（服务器）：容器内起 tensorboard
docker exec sonic-train tensorboard \
  --logdir /data/sonic/GR00T-WholeBodyControl/logs_rl/TRL_G1_Stub --port 6006
# 终端 B（自己电脑）：经跳板转发 6006 端口
ssh -L 6006:localhost:6006 npu47
# 浏览器打开 http://localhost:6006
```

关键指标：
- `objective/length`：每 episode 存活步数，**微调是否有效的主判据**
  （官方权重基线 ~3-4 步，微调有效应持续上升）
- `objective/rewards`：每 episode 总 reward（基线约 -400 量级）
- `policy/approxkl_avg`、`loss/entropy_avg`、`loss/policy_avg`：PPO 健康度
- `lr`：adaptive 学习率变化

不想占服务器端口的话，也可以把 `<run>/tb/` 整个 scp 回本地，
`tensorboard --logdir` 指向本地目录看。

停止训练：`pkill -f train_agent_trl`（checkpoint 每 50 迭代已自动保存）。

### 2.5 （可选）GPU 上做仿真（Isaac Sim 原版链路）

本文档默认 **CPU 仿真**（`SONIC_MUJOCO_ENV=1`，MuJoCo 在 47 的 192 核 CPU
上跑）——这是本仓推荐的 NPU 单机方案（性能对比与设计依据见
`docs/NPU_GPU_Remote_Training_Design.md` §0）。需要 GPU 仿真时走 SONIC 官方
的 Isaac Sim 链路：

**什么时候需要**
- 要大吞吐：GPU 上 PhysX 可并行数千 env（官方满配 num_envs=4096、建议 64+ GPU），
  远超 CPU MuJoCo 的 256 env 规模
- 要与官方权重的训练物理完全一致：release 策略就是在 Isaac Sim 里训的，
  在 Isaac 下表现正常；MuJoCo 的物理差距会让它 3-4 步就倒（见 §3 基线）

**前提**
- x86 + NVIDIA GPU 机器。**47 跑不了**：aarch64 + 昇腾 NPU，Isaac Sim 不支持
- Isaac Lab 2.3.2 需单独安装（pip 装不了）：先按官方指导装
  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
  再 `pip install -e "gear_sonic/[training]"`

**跑法**（GPU 机器上，README 官方命令；**不设 `SONIC_MUJOCO_ENV`**）：

```bash
accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    +checkpoint=sonic_release/last.pt \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered
```

（数据准备：Bones-SEED 动捕集下载与转换见仓库 README "SONIC Training" 一节；
多机训练、评估、ONNX 导出见官方文档站。）

**与本文档链路的差异速查**

| | 本文档（默认） | GPU/Isaac 链路 |
|---|---|---|
| 物理仿真 | MuJoCo CPU（47 的 192 核） | Isaac Sim PhysX（GPU） |
| 学习端 | 47 的 npu:0 单卡 | GPU 机多卡（accelerate） |
| 评估 | `scripts/record_walk.py`（CPU） | `gear_sonic/eval_agent_trl.py`（硬依赖 isaaclab，47 跑不了） |
| 渲染 | OSMesa CPU 离屏 | Isaac 摄像头（`enable_cameras`） |
| 环境变量 | `SONIC_MUJOCO_ENV=1` | 不设 |

GPU 物理 + NPU 学习分离的远程方案（GPU 机跑仿真、NPU 机跑训练）是备选路径，
设计与网络开销分析见 `docs/NPU_GPU_Remote_Training_Design.md` §1-9。

## 3. CPU 推理评估（训练环节之外，不占 NPU）

> 本章**不在训练环节里**：训练中的 CPU 仿真已包含在 §2（那 64 个 MuJoCo
> worker 进程）。本章是训练之外，拿产出的 checkpoint 单独跑仿真看指标。
>
> **本章目的**：用数字回答"这个 checkpoint 好不好、微调有没有效"——
> ① 验收：length 是否比官方基线（~3-4 步）持续上升，决定继续训还是停；
> ② 选型：多个 run / 多个迭代之间挑最好的 checkpoint 交付或续训；
> ③ 体检：评估口径与训练完全一致，可交叉验证训练是否健康（例如这次实测
> 发现的"reward 升、length 降"早死退化，就是评估口径下才看得清的）。

训练产出的 checkpoint 用 `scripts/record_walk.py` 评估——它在容器内 CPU 上
跑确定性 rollout（action_mean），打印每个 episode 的 reward / length /
结束原因，并顺带渲染出视频（评估 + §4 渲染一步到位）：

```bash
docker exec -it sonic-train bash
cd /data/sonic/GR00T-WholeBodyControl
MUJOCO_GL=osmesa python3 scripts/record_walk.py \
  --ckpt logs_rl/TRL_G1_Stub/<run>/last.pt \
  --episodes 3 --out /data/sonic/renders/eval.mp4
```

注意：§2.2 的冒烟只跑 2 迭代（不到 50 迭代的保存间隔），run 目录里**不会落
checkpoint**；评估对象用官方权重 `sonic_release/last.pt` 或 §2.3 微调产物的
run 目录（`logs_rl/TRL_G1_Stub/` 下按时间戳找 `last.pt`）。

**基线判据**（2026-08-25 实测）：官方权重 `sonic_release/last.pt` 在 MuJoCo
物理下 mean length ≈ 4 步、mean reward ≈ -453（与训练 rollout 的 tensorboard
`objective/length` 3.3-3.5 一致——官方策略在 MuJoCo 里站不住，这正是微调
要解决的）。微调后的 checkpoint 应以 **length 持续上升**为有效信号；
脚本的评估口径与训练一致（alive_bonus=0.0、确定性动作）。

要点：`SONIC_MUJOCO_ENV=1` 下 rollout 全在 CPU（MuJoCo worker 进程），
NPU 只跑学习端——所以"推理"在训练机 CPU 上天然可用，无需单独环境。
若要在**自己电脑的 CPU** 上推理：把策略权重抽出来（见 §4.3），装
mujoco + torch（CPU 版）即可跑，同一个 `scripts/record_walk.py` 直接可用
（Mac 上把 `MUJOCO_GL` 换成 `cgl`，模型/数据路径用 `--model-xml` /
`--pkl-dir` 指向本地）。

## 4. 渲染视频（默认在服务器容器内，导回本地查看）

> §3 与 §4 的区别：**§3 评估要的是数字**（reward/length，判断 checkpoint 好坏），
> **§4 渲染要的是画面**（mp4，直观检查动作质量）。两者用同一个
> `scripts/record_walk.py`——它一次运行同时产出两者，区别只在参数：
> §3 用默认模式（倒地即停，口径与训练一致）；§4 想要长视频加 `--keep-going`
> （倒地重开继续录，见 §4.1）。

### 4.1 容器内离屏渲染（默认，2026-08-25 实测）

服务器无图形栈，用 OSMesa 软件渲染（libosmesa 已随环境装好，见踩坑 4；
已用官方权重实测 1280×720 出片正常）。渲染全在 CPU
（MuJoCo rollout + OSMesa 出帧），不占 NPU：

```bash
docker exec -it sonic-train bash
cd /data/sonic/GR00T-WholeBodyControl
MUJOCO_GL=osmesa python3 scripts/record_walk.py \
  --ckpt logs_rl/TRL_G1_Stub/<run>/last.pt --out /data/sonic/renders/walk.mp4
```

- 官方权重与自训 checkpoint 直接可用（脚本内置 load_release 兼容加载），**无需抽权重**
- 多 episode 时输出自动拆成 `walk_ep1.mp4`、`walk_ep2.mp4`……
- **视频时长**：默认 episode 结束（倒地/跑完动作）就停。官方权重或微调早期的
  checkpoint 在 MuJoCo 下只能站 ~3-4 步，录出来只有几帧——要长视频加
  `--keep-going`：倒地后自动重新开场继续录，录满 `--max-steps` 为止
  （50fps 实时，300 步 ≈ 6 秒、1000 步 ≈ 20 秒）：

  ```bash
  MUJOCO_GL=osmesa python3 scripts/record_walk.py \
    --ckpt logs_rl/TRL_G1_Stub/<run>/last.pt \
    --episodes 1 --max-steps 1000 --keep-going --out /data/sonic/renders/walk.mp4
  ```

  日志按"命"统计（每次开场一条命），评估口径与默认模式一致。
- 不要试 EGL：47 是昇腾 NPU 机器，没有 EGL GL 栈，osmesa 是唯一可用后端

### 4.2 导出到本地看结果

```bash
scp npu47:/data/sonic/renders/walk.mp4 ~/Desktop/
```

（经跳板 scp 直传即可；也可用 VS Code Remote / sftp 拖文件。）

### 4.3 （可选）Mac 本地渲染

仅在需要脱离服务器、在 Mac 上独立调策略时需要。NPU 容器里的
checkpoint 含 torch_npu 状态，Mac 的 CPU torch 直接 load 会报错，
需先在服务器容器内导出纯权重：

```bash
docker exec sonic-train python3 -c "
import torch
ckpt = torch.load('logs_rl/TRL_G1_Stub/<run>/last.pt', map_location='cpu')
# 按模型结构抽出 policy state_dict（参考仓库内既有抽取脚本），
# 保存为 policy_sd.pt（纯 CPU tensor）"
```

然后 `scp npu47:/data/sonic/GR00T-WholeBodyControl/.../policy_sd.pt ~/`，
在 Mac 上：

```bash
cd GR00T-WholeBodyControl    # 本地 clone 的 feature/mujoco-training
python3 -m venv .venv_sim && source .venv_sim/bin/activate
pip install mujoco torch numpy   # Apple Silicon 直接装
MUJOCO_GL=cgl python3 scripts/record_walk.py --ckpt policy_sd.pt --out walk.mp4
```

（`MUJOCO_GL=cgl` 走 Apple OpenGL 离屏渲染。）

## 5. 环境重建参考（若容器/机器丢失）

全部物资在共享盘 `/data/images/`（SFS，所有节点可见）：

| 文件 | 用途 |
|---|---|
| `sonic-train-base-cann852-v14.tar.gz`（8.4G）| 基础镜像（torch/torch_npu/mujoco/CANN 全套）|
| `sonic-train-requirements.txt`（172 行）| 应用层依赖配方（当年容器实导）|

重建步骤：

```bash
docker load -i /data/images/sonic-train-base-cann852-v14.tar.gz
docker tag k3-train:cann852-v14 sonic-train:v14-base
docker run -itd --name sonic-train --runtime=ascend --shm-size 32g \
  --network host -v /data:/data -e ASCEND_VISIBLE_DEVICES=0-7 \
  --restart unless-stopped --entrypoint bash sonic-train:v14-base -c "sleep infinity"
# 依赖（注意配方里 CANN 本地 whl 行和 git+ 包要特殊处理，见踩坑 2/3）：
docker exec sonic-train bash -c "apt-get install -y libgl1 libglib2.0-0 libosmesa6-dev && \
  pip3 install \$(grep -E '^[a-zA-Z0-9_.-]+==' /data/images/sonic-train-requirements.txt) -i https://pypi.tuna.tsinghua.edu.cn/simple && \
  pip3 install --no-deps 'smplx @ git+https://github.com/ZhengyiLuo/smplx.git@a5b8e4ac14f79f3f33fd2cf2a16e6f507146b813' 'smpl_sim @ git+https://github.com/ZhengyiLuo/SMPLSim.git@b5c08720503ad5fff64050c4d289c42d947fcf8d'"
```

重建后还必须补两个坑的修复（症状/原因见 §6），漏了跑不起来：

```bash
# 坑 5：STL 网格是 LFS 指针，必须换成真文件
cd /data/sonic/GR00T-WholeBodyControl && git lfs pull   # 或从旧真库拷 meshes/*.STL
# 坑 6：代码里硬编码的容器根绝对路径软链
docker exec sonic-train bash -c "ln -sf /data/sonic/GR00T-WholeBodyControl/gear_sonic_deploy /gear_sonic_deploy && \
  ln -sf /data/sonic/GR00T-WholeBodyControl/sample_data /sample_data"
```

## 6. 踩坑全记录（47 搭建实测，每个都踩过）

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | 容器缺 `ASCEND_VISIBLE_DEVICES` | torch_npu 加载 backend 失败 | docker run 加 `-e ASCEND_VISIBLE_DEVICES=0-7` |
| 2 | freeze 配方里的本地 whl | pip OSError `/root/selfgz...whl` 不存在 | 只装 `pkg==ver` 格式行（CANN 本地包镜像已自带）|
| 3 | smplx/smpl_sim 非 pypi 版 | ResolutionImpossible / 找不到 | 按 freeze 里的 `git+...@commit` 装 + `--no-deps` |
| 4 | 系统库 libGL 缺失 | `import cv2/mediapy` 报 libGL.so.1 | `apt-get install -y libgl1 libglib2.0-0 libosmesa6-dev` |
| 5 | clone 不带 LFS | MuJoCo 报 "STL number of faces ... ASCII file" | 从旧真库拷 meshes/*.STL（或 git lfs pull）|
| 6 | 代码里的容器根绝对路径 | 找不到 `/gear_sonic_deploy`、`/sample_data` | 容器内 `ln -s /data/sonic/GR00T-WholeBodyControl/gear_sonic_deploy /gear_sonic_deploy` 和 `ln -s .../sample_data /sample_data` |
| 7 | Mac 拷来的数据带 AppleDouble | `._*.pkl` 混入（66 个文件一半是垃圾）| 加载逻辑已过滤 `._`；自己拷数据用 `COPYFILE_DISABLE=1 tar` |
| 8 | ENTRYPOINT 拼接 | 新起容器 `cannot execute binary file` | 该镜像 ENTRYPOINT=python3，run 时用 `--entrypoint bash` |
| 9 | 代理劫持 SSH | 解析 198.18.x.x、banner 超时 | 代理加 DIRECT 规则或关增强模式 |

## 7. 常用路径速查

```
容器内代码/数据根   /data/sonic/GR00T-WholeBodyControl
官方权重            sonic_release/last.pt（step 41550）
训练输出            logs_rl/TRL_G1_Stub/<run>/
动捕 PKL            sample_data/robot_filtered/{210531,bones_round3}/
MuJoCo 模型         gear_sonic_deploy/g1/g1_29dof_v17.xml
渲染输出            /data/sonic/renders/（OSMesa 离屏渲染产物，scp 回本地看）
镜像归档            /data/images/sonic-train-base-cann852-v14.tar.gz
依赖配方            /data/images/sonic-train-requirements.txt
旧环境参考（aura-6） /data/z00666713/GR00T-WholeBodyControl（原始训练现场）
```
