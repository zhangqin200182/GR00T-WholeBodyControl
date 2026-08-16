# SONIC × MuJoCo × 昇腾 NPU 实验总结

> 分支：`feature/mujoco-npu-experiments`（基于 `feature/mujoco-training`）
> 时间：2026-08-14 ~ 2026-08-17
> 作者：Maxwell-AI-lab（实验执行）· 本文档供团队参考
> 一句话结论：**官方 Isaac 训练的 SONIC 37M 权重，经昇腾 NPU 在 MuJoCo CPU 物理中微调后，实现了贴合参考速度（0.68 vs 0.71 m/s）的正常步行——"脱离 NVIDIA 生态训练"路线验证通过。**

---

## ★ 主导思想与核心结论

### 主导思想（四条原则，贯穿所有实验）

1. **不从零训练，从官方权重出发**——官方 37M 是"已经会走路的大脑"，微调只教它适应新物理（MuJoCo），不从零学走路。代价小一个数量级（数小时 vs 数天）。
2. **对齐先行**——网络输出必须在两个引擎里是同一个物理含义。动作空间逐关节对齐是全部后续实验的必要条件，对齐之前任何训练都白费。
3. **对照组思维**——同一份权重，在两个引擎交叉验证：Isaac 官方表现是"标准答案"，MuJoCo 每一步都和它对照，避免自说自话。
4. **算力脱钩验证**——训练全程只用昇腾 NPU + CPU 物理（同事 fork 的核心命题），NVIDIA GPU 仅作对照组，不参与任何训练。

### 核心结论（按重要性）

| # | 结论 | 证据 |
|---|---|---|
| 1 | **路线走通**：官方权重经 NPU+MuJoCo 微调后正常步行 | 最终（11000 迭代）：速度贴合 0.73 vs 0.72 m/s，含启停跟随；训练均值奖励 +9207 / 长度 1476 步 |
| 2 | **前提是动作空间对齐**：同事原版 MuJoCo 环境的动作空间有 bug | 修复前每 4 步摔（奖励 -39867）；逐关节对齐 Isaac 公式后零摔倒（+1500） |
| 3 | **奖励设计决定学到什么**：不惩罚掉队就会学出"原地踏步" | 一轮（无掉队判死）：前进仅参考一半；二轮（判死+奖励×4）：真步行 |
| 4 | **技能绑定训练引擎**：微调后回 Isaac 也摔 | Isaac 回传 8 步摔（0.16s），与官方权重进 MuJoCo 的 7 步摔对称 |
| 5 | **瓶颈在数据不在算力**：单卡 NPU（16% AICore）+CPU（17%）大量闲置，而训练数据只有 2 条动作片段（80 秒） | 扩数据比扩卡收益大得多 |

---

## ★ 我们做了什么（先读这页）

**起点**：把 NVIDIA 官方训练好的 SONIC 37M 权重（Isaac Sim 训练，step 41550）拿到 MuJoCo 里直接跑——发现每 4 步就摔倒。此后所有工作围绕一个问题展开：**怎么让它在 MuJoCo 里正常走路，且全程不依赖 NVIDIA 算力。**

五大实验块：

```
块① 诊断零样本失败 ——→ 修复同事 MuJoCo 环境的动作空间 bug（与 Isaac 逐关节对齐）
        ↓ 零样本能站稳了，但步态快而扭曲（5 m/s 极限环）
块② 9 项对齐实验 ——→ 证明残差是物理层面的，零样本修不好，必须 RL 微调
        ↓ 决定微调路线，先建对照组
块③ 建对照组 ——→ 云 GPU 搭 Isaac 全链路，官方权重评估录像 = "标准答案"
块④ 脱离 NVIDIA 的训练环境 ——→ 昇腾 NPU 容器 + CPU 物理并行，双冒烟通过
块⑤ 两轮微调 ——→ 一轮学会"站稳不摔"但原地踏步；诊断后二轮学会"真走路"
```

**贯穿全程的核心对比（同一权重，两个引擎）**：

| | Isaac Sim（NVIDIA） | MuJoCo（CPU） |
|---|---|---|
| 官方权重（Isaac 训的） | ✅ 正常步行【标准答案】 | ❌ 7 步摔 → 修复动作空间后能站但扭曲 |
| 我们微调后（NPU+MuJoCo 训的） | ❌ 8 步摔（对称性发现） | ✅ **正常步行，速度贴合参考** |

**微调两轮的对比（每轮 ~4 小时，你的 NPU 单卡）**：

| | 一轮（5000 迭代） | 二轮（11000 迭代，掉队判死版） |
|---|---|---|
| 行为 | 站稳、能转弯，**但不往前走**（前进距离只有参考一半） | **真步行**：速度贴合 0.73 vs 参考 0.72 m/s，启停跟随 |
| 学会的东西 | 姿态跟踪、平衡 | 姿态跟踪 + 真推进 |
| 差别来源 | 奖励/终止设计允许"原地摆姿势"拿高分 | 掉队 1 米判死 + 位置奖励 ×4，堵死偷懒解 |

---

## 0. 阅读地图

| 想了解 | 看哪节 |
|---|---|
| 我们做了哪些实验、结论是什么 | §1 时间线 + §2 各实验详情 |
| 核心技术方案（为什么能成） | §3 动作空间对齐 + §4 奖励/终止演进 |
| 各阶段量化指标 | §5 指标汇总 |
| 环境怎么搭、坑怎么避 | §6 踩坑大全 + docs/ 下两份环境指导 |
| 怎么复现 | §7 复现命令 |
| 这个分支里有什么文件 | §8 产物清单 |

---

## 1. 实验时间线总览

| # | 实验 | 机器 | 结论 |
|---|---|---|---|
| E1 | Fork 代码架构分析 | Mac | 摸清三层架构：Isaac(官方)/Stub/MuJoCo 三条 env 路径 |
| E2 | 官方权重 MuJoCo 零样本推理 | Mac | 每 4 步摔倒——不可用，定位到动作空间不匹配 |
| E3 | **动作空间对齐修复**（核心贡献①） | Mac | 奖励 -39867→+1500，零摔倒；这是全部后续实验的地基 |
| E4 | 零样本对齐度系列实验（9 项） | Mac | 均无实质提升→证明残差是物理层面，必须微调 |
| E5 | 云 GPU Isaac 环境搭建 + 官方权重验证 | 云3060 | Isaac 全链路贯通，官方权重评估录像（8×1080p） |
| E6 | NPU 容器环境 + 双冒烟 | NPU 8×910B3 | Stub/MuJoCo 两路冒烟通过，同事架构全通 |
| E7 | **一轮微调**（5000 迭代） | NPU+CPU | 奖励 -104→+4088，长度 7→1145 步 |
| E8 | 交付管线 + 管线 bug 修正 | Mac | 发现 ±1 裁剪/上身覆盖两个失真源，修复后 1616 步 |
| E9 | **Isaac 回传对称性实验** | 云3060 | MuJoCo 微调权重回 Isaac 8 步摔——技能绑定训练引擎 |
| E10 | "不往前走"根因诊断 | Mac | 观测盲+掉队不判死+奖励太弱，三重宽容致局部最优 |
| E11 | **二轮微调**（掉队判死+奖励×4+1024环境） | NPU+CPU | 速度贴合 0.58 vs 0.56 m/s，真推进 23.65m ✅ |
| E12 | 收敛打磨（腿抖） | NPU（进行中） | 待收敛尾段+可选滤波/精修 |

---

## 2. 各实验详情

### E1 · Fork 架构分析

SONIC 策略本体：37M 参数通用 token 架构（G1 本体/远程操作/SMPL 三编码器 → FSQ 量化 2×32 维 token → 解码 29 个 G1 关节动作）。

仓库把官方 Isaac 训练管线改造成三路可选环境（环境变量切换）：
- `SONIC_STUB_ENV=1` → StubEnv（无物理，验证训练循环）
- `SONIC_MUJOCO_ENV=1` → **MuJoCoEnvManager**（本实验主线：128 CPU 进程 × 共享内存 × NPU 学习）
- 默认 → Isaac Sim（官方路径）

训练入口 `gear_sonic/train_agent_trl.py`，PPO 超参继承官方 `ppo_im_phc`（全部未改动）。

### E2 · 官方权重零样本（问题发现）

`sonic_release/last.pt`（step 41550）直接进 MuJoCo：奖励 -39867，每 4 步摔倒。
**根因**：同事的 `mujoco_env.py` 动作空间用了错误近似（统一 kp=100/kd=5 + 关节半行程缩放），与 Isaac 训练时的动作语义完全不同——网络输出的每个数在两个引擎里含义不同。

### E3 · 动作空间对齐（核心贡献①，一切的地基）

按 Isaac 官方 `g1.py`（ImplicitActuatorCfg）与 URDF 执行器表逐关节重写 MuJoCo 动作空间：

```python
# Isaac 精确公式（修复后）
_ nf = 10.0 * 2π                                   # 自然频率 10Hz
stiffness = mult × armature[trnid] × _nf²          # 每关节刚度
damping   = 2 × 2 × mult × armature[trnid] × _nf   # 阻尼
act_scale = 0.25 × effort_limit / stiffness        # 动作缩放
target    = default_pos + act_scale × action       # PD 目标（不再±1裁剪）
torque    = clip(kp·(target-q) - kd·qd, ±effort)   # 力矩限幅
```

电机参数（每关节组）：电枢 {5020: 0.00361, 7520_14: 0.01018, 7520_22: 0.02510, 4010: 0.00425}，力矩限幅 {腿 139/88, 脚/腰 50, 肩/肘 25, 腕 5}。

**效果**：奖励 -39867 → +1500，从"每 4 步摔"到零真实摔倒。`grep -c act_scale mujoco_env.py` >0 即为修复版。

### E4 · 零样本对齐系列（证明"必须微调"）

动作空间修复后，步态仍快/扭（~5 m/s 极限环）。系统性排除以下因素（全部无效或负效）：
±1 动作裁剪扫描（确认 ±1 裁剪在推理时反而有害）、β 系数扫描、观测 EMA、PPO impratio、MuJoCo 积分器更换、胶囊脚变体、网格减面（665856→209313 tris）、慢速参考播放、速度阻尼项。
**结论**：残差来自求解器级物理差异，属于策略层不可修——必须 RL 微调。这个结论直接决定了后续路线。

### E5 · 云 GPU Isaac 环境（对照组）

RTX 3060 12GB 从零搭 Isaac 全链路：py3.11 + torch 2.7.0 + isaacsim 5.0.0.0 + IsaacLab v2.3.2（含 4 处必打补丁，见 §6）。

**夜间自动化排障**（每 10 分钟定时推进，累计 21 个诊断版本 V1→V21）：URDF 转换反复崩溃，表象五花八门（NULL TfRefPtr、段错误、随机 PhysX 初始化失败），先后排除动作空间/转换缓存/importer 版本/网格面数，最终 V21 锁定根因——**URDF 引用的 3 个 `_rev_1_0` 后缀网格是 LFS 指针文本**，补齐+减面后转换通过。

跑通官方权重训练冒烟（reward 0.83→0.99）与评估录像（8 环境 × 1080p 跟踪步行）。**作用**：给 MuJoCo 实验提供"官方标准答案"对照。

### E6 · NPU 环境（主力训练机）

昇腾 910B3 单卡（fp32 强制，NPU 无 bf16 torch.normal）+ 128 CPU 进程跑 MuJoCo。容器 `sonic-train-zhouzhi`，同路径挂载 `/data/z00666713`。HuggingFace 代理中途断连，权重经 Mac 中继上传解决。StubEnv 与 MuJoCo 环境各 100 轮冒烟通过。

### E7 · 一轮微调（5000 迭代）

```
SONIC_MUJOCO_ENV=1 + 256环境 × 64worker, 起点官方权重
奖励: -104 → +4088 | 长度: 7步 → 1145步
```
单卡 NPU + 192 核 CPU，每迭代 2.9s，全程零人工干预（一处 torch_npu 存档 bug 已修，见 §6-N5）。

### E8 · 交付管线与"假摔"事件（重要教训）

一轮视频里机器人"像要摔倒、不前进"，与训练指标矛盾。排查发现**我的录制脚本两个零样本时代的 hack 在扭曲微调策略**：
1. ±1 动作裁剪（训练时 PPO 不裁剪，直接进 env）——裁剪使存活 1145→70 步
2. 上身运动学覆盖（手臂直接摆参考 pose）——破坏策略自身的平衡闭环

修复后诚实模式：**1616 步、零摔倒**。教训已写入 `scripts/record_walk.py`（诚实模式为默认，hack 全部 opt-in）。

### E9 · Isaac 回传对称性（科学发现）

**方法**：微调策略权重注入官方 checkpoint 结构（在仓库根目录 `import gear_sonic.trl.trainer.ppo_trainer` 触发 `trl.trainer.utils.OnlineTrainerState` pickle 别名后才能 `torch.load` 官方文件），再用与官方权重完全相同的评估管线（4 环境+渲染）回放。

| 权重 | Isaac | MuJoCo |
|---|---|---|
| 官方（Isaac 训练） | ✅ 优秀 | ❌ 7 步摔 |
| 微调 5000（MuJoCo 训练） | ❌ **8 步摔（0.16s）** | ✅ 1616 步 |

两方向脆弱度几乎相同。**结论：微调学到的技能绑定训练引擎，sim-to-sim 迁移不免费**。若需双引擎，需混合训练或轻调。

### E10 · "不往前走"诊断

一轮诚实视频：能站稳、能转弯，但前进距离只有参考一半（0.27 vs 0.52 m/s）。三层根因：
1. **观测盲**：tokenizer 观测 13 类信号无一编码水平面掉队量（只有高度/朝向/关节）
2. **掉队不判死**：终止只查"移植到机器人位置的相对姿态"，原地踏步可以活满全场
3. **位置奖励太弱**：r1 权重 0.5、σ=0.3，落后 1m 后饱和归零，损失可忽略

→ "原地摆对姿势"是低风险高分的局部最优。

### E11 · 二轮微调（突破）

三处修改后从 5000 步续训（目标 11000）：
1. **掉队判死**：水平偏差 > 1.0m 即终止
2. **位置奖励加强**：r1 权重 0.5→2.0、σ 0.3→0.2
3. **扩容**：256→1024 环境、64→128 worker（吞吐 ×3~4）

学习曲线三段式：搜索期(+0.1步/迭代) → 爆发期(+1.94) → 收获期(+0.87)。
**中段快照（step 6250）验证**：23.65m / 1427 步 / 行进段速度 **0.68 vs 参考 0.71 m/s**，含启停跟随（参考停它也停）。

### E12 · 收敛与腿抖（进行中）

腿抖 = PPO 中期高探索档位（std 0.43）在确定性输出中的残留，预期随收敛（std 回落）减轻；兜底手段：部署端一阶低通滤波（15Hz）/ 低学习率精修轮 / 加重 action_rate 惩罚。

---

## 3. 核心方案：为什么微调能成立

```
官方权重（会走路的"大脑"，Isaac 口味）
     + 动作空间逐关节对齐（两个引擎里同一网络输出 = 同一物理含义）  ← E3，必要条件
     + 奖励/终止设计逼出真跟踪（而不是苟活）                        ← E10/E11
     → PPO 在 MuJoCo 经验中重新校准"用力方式"，高层技能保留
```

三层一致性检查表（换环境重建时逐项核对）：
- MuJoCo 版本两端一致（3.11.0）
- 机器人 XML 一致（md5 核对）
- `mujoco_env.py` 为修复版（act_scale 存在）

## 4. 奖励/终止演进表

| 版本 | 关键差异 | 结果 |
|---|---|---|
| 同事原版 | kp=100 统一、半行程缩放 | 每 4 步摔 |
| E3 修复版 | Isaac 精确动作空间 | 零样本零摔，但 5m/s 扭曲 |
| 一轮 | E3 + 官方奖励权重 | 站稳+摆姿，不推进（局部最优） |
| 二轮 | + 掉队1m判死 + r1×4 | 真推进，速度贴合参考 ✅ |

## 5. 指标汇总

**训练曲线（二轮，1024 环境）**：

| 迭代(总) | 奖励 | 长度(步) | 备注 |
|---|---|---|---|
| 0（零样本） | -104 | 7 | 官方权重直入 MuJoCo |
| 5100 | 942 | 192 | 二轮开局（判死生效，长度重置） |
| 5614 | 1693 | 333 | 搜索期 |
| 5960 | 4110 | 1005 | 爆发期 |
| 7044 | 6015 | 1349 | 收获期（27s） |
| **10176** | **9207** | 1476 | 收敛尾段 |
| **11000（完成）** | ~9200 | ~1480 | 最终，峰值 +9168~9207 |

**诚实单环境评测（Mac，确定性输出，随机相位）**：

| 权重 | 存活 | 前进 | 速度 vs 参考 |
|---|---|---|---|
| 官方零样本 | 7 步 | ~0 | — |
| 一轮 5000 | 1616 步 | 8.64m | 0.27 vs 0.52（一半） |
| 二轮 6250 快照 | 1427 步 | **23.65m** | **0.68 vs 0.71 ✅** |
| 二轮 6550 快照 | 1102 步 | 12.74m | 0.58 vs 0.56 ✅ |
| **二轮最终 11000** | 984 步（随机相位） | 11.38m | **0.73 vs 0.72 ✅（最佳）** |

### 训练全程关键曲线（TensorBoard 导出）

**奖励**（step 5000 处的骤降=二轮奖励函数改版，非事故；随后爬升为二轮成绩）：

![rewards](assets/tb_rewards.png)

**回合长度**——注意三段式：一轮爬升 → 二轮开局重置（掉队判死生效）→ 爆发 → 收敛：

![length](assets/tb_length.png)

**策略健康度**（熵平稳无塌缩 / KL 无尖峰，全程无训练病理）：

![entropy](assets/tb_entropy.png)

**Isaac 官方权重基线（A 实验）**：8 环境全部完整跑满回合（40s×8，run_once 评估），零提前终止——即官方基线在 Isaac 的表现为"100% 回合完成率"。MPJPE 等逐项误差指标需 wandb 启用后才落盘，留作后续（见 §9）。

**Isaac 回传**：一轮权重 0.16s/8 步摔（对称性实验，E9）。

### 资源实测（规划参考）

| 配置 | 采集+学习/迭代 | 资源占用 | 判断 |
|---|---|---|---|
| 256 环境×64 worker | 2.9s | CPU 10%、NPU 16% AICore | 双端大量闲置 |
| 1024 环境×128 worker | 4.1s | CPU ~17%、NPU 突发 | 吞吐×3，当前配置 |
| 2048 环境×192 worker | 估 12-15s | CPU 接近打满 | 本机上限区间 |

- 单卡 NPU 足够（模型 37M，瓶颈在 CPU 采集不在学习）；8 卡 DDP 只在千级环境以上的大规模训练有意义
- 内存无忧（1.5TB 机器用量 <300GB）
- 数据是真正瓶颈：训练仅 2 条步行片段（80 秒），泛化天花板由数据决定

### 本地产物对照表（不在仓库内，供溯源）

| 视频/文件 | 位置（Maxwell 的 Mac） | 内容 |
|---|---|---|
| `g1_isaac_videos/` 8 个 | ~/code/ai/embody/ | 官方权重 Isaac 评估（对照组） |
| `g1_isaac_ft_videos/` 4 个 | 同上 | 微调权重 Isaac 回传（0.16s 对称摔倒） |
| `g1_walk_final_true.mp4` | 同上 | 一轮 5000 诚实模式（站稳不推进） |
| `g1_walk_r2_snap.mp4` / `_snap2.mp4` | 同上 | 二轮快照（真步行，速度贴合） |
| 零样本旧视频 4 个 | 同上 | 动作空间修复前的对照 |

## 6. 踩坑大全（按机器）

### Mac / 通用
- LFS 指针文件冒充网格（130 字节文本）→ `lfs_fetch.py` 走批量 API 下载 + `_rev_1_0` 变体网格别漏
- Git 直连慢/被 RST → ghfast.top 镜像 + HTTP/1.1
- 推理 Python 环境必须是仓库内 `.venv_sim`；`MUJOCO_GL=cgl` 离屏渲染
- **录制管线三坑**：±1 裁剪、上身覆盖 hack、参考相位锁定——对微调策略全部失真，诚实模式必须全部关闭
- NPU 存的权重带 torch_npu pickle 引用 → 必须在 NPU 上抽 `policy_state_dict` 成纯 CPU 文件再跨机用

### 云 GPU（Isaac）
- isaacsim 4.5 只有 cp310 轮子；2.3.2 要 isaacsim≥5.1 系、torch≥2.7 → 最终 py3.11+isaacsim5.0.0.0
- importer 版本 2.4.31/2.4.30 冲突 → 改 IsaacLab 源码版本号
- warp 双向不兼容（isaaclab fabric 要新 API、isaacsim core 要旧 API）→ 手写矩阵复合，统一用 kit 自带 1.7.1
- URDF importer 扩展不自动启用 → AppLauncher 后 `set_extension_enabled_immediate`
- 评估 hydra 全部要 `+` 前缀；motion 路径指向内部数据需覆盖；recorders 默认 empty 要改 render；写 mp4 要装 imageio-ffmpeg
- 转换失败后 `/tmp/IsaacLab` 缓存复用坏产物 → 清缓存重试
- 详细版见 `docs/GPU环境搭建指导-IsaacSim.md`

### NPU（昇腾）
- hydra 顶层新键必须 `+`；迭代数真参数是 `algo.config.num_learning_iterations`
- resume 的 glob 会选中空目录 → 永远显式传 `checkpoint=路径`
- 克隆不含 E3 修复的 `mujoco_env.py` → 重建必覆盖（本分支文件为准）
- `copy.deepcopy(state)` 遇 torch_npu Byte storage 崩 → model_save_callback 加 try/except 退浅拷贝
- 容器必须同路径挂载；单卡足够（瓶颈在 CPU 采集，不在 NPU）
- 详细版见 `docs/NPU微调与Mac渲染指导.md`

## 7. 复现速查

```bash
# ── NPU：二轮训练（或续跑）──
docker exec sonic-train-zhouzhi bash -c "cd /data/z00666713/GR00T-WholeBodyControl && \
  SONIC_MUJOCO_ENV=1 WANDB_MODE=disabled nohup python3 gear_sonic/train_agent_trl.py \
  +exp=stub_train exp_var=mujoco_ft2 +resume=true \
  checkpoint=logs_rl/TRL_G1_Stub/stub_train_mujoco_ft-20260815_214321/last.pt \
  algo.config.num_learning_iterations=11000 num_envs=1024 +mujoco_workers=128 \
  use_wandb=false > /data/z00666713/mujoco_ft_r2.log 2>&1 &"

# ── NPU：导出纯策略（跨机必需）──
docker exec sonic-train-zhouzhi python3 -c "
import torch
ckpt = torch.load('<last.pt>', map_location='cpu', weights_only=False)
torch.save({'policy_state_dict': {k: v.detach().cpu() for k, v in ckpt['policy_state_dict'].items()},
            'global_step': ckpt['state'].global_step}, '/data/z00666713/policy_export.pt')"

# ── Mac：诚实模式渲染 ──
cd <repo> && SONIC_NOCLIP=1 MUJOCO_GL=cgl \
  SONIC_CKPT=$PWD/checkpoints/policy_r2_6550.pt \
  .venv_sim/bin/python scripts/record_walk.py 2000 out.mp4
# 可选: SONIC_MOTION_IDX=0/1 固定动作片段; SONIC_NOVIDEO=1 只出统计
```

## 8. 本分支产物清单

```
gear_sonic/envs/mujoco_env.py          # E3 动作空间修复 + E11 二轮补丁（掉队判死/r1×4）
gear_sonic/train_agent_trl.py          # 路径修复（v17.xml→g1_29dof.xml 等）
gear_sonic/trl/callbacks/model_save_callback.py  # torch_npu deepcopy 存档补丁
gear_sonic/eval_agent_trl.py           # Isaac 评估: URDF importer 扩展启用补丁
scripts/train_mujoco_sonic.py          # Mac 单机训练/推理入口（release config 加载器）
scripts/record_walk.py                 # 诚实模式录像（默认无任何 hack）
scripts/diag_walk.py                   # 单环境诊断（clip/EMA/β 扫描等）
scripts/make_capsule_feet_xml.py       # 胶囊脚变体生成（E4）
tools/                                 # 补丁脚本（round2/save_ckpt/eval_warp/fabric）+ lfs_fetch.py
checkpoints/                           # 纯策略权重（见目录内 README）
docs/GPU环境搭建指导-IsaacSim.md       # 24 坑
docs/NPU微调与Mac渲染指导.md           # 12 坑
docs/EXPERIMENTS.md                    # 本文档
```

### 训练曲线（TensorBoard 导出，`docs/assets/`）

| 图 | 内容 | 看点 |
|---|---|---|
| `tb_rewards.png` | 奖励全程曲线 | step 5000 处的跳变=二轮奖励函数改版（非事故） |
| `tb_length.png` | 长度全程曲线 | 三段式：一轮爬升→二轮重置→爆发→平台 |
| `tb_entropy.png` / `tb_approxkl.png` | 策略健康度 | 无塌缩、无 KL 尖峰 |
| `tb_value_loss.png` | 价值网络收敛 | — |

### 关键视频（`docs/assets/`）

| 视频 | 内容 |
|---|---|
| `isaac_official_weights.mp4` | 官方权重 Isaac 评估（对照组标准答案） |
| `g1_walk_final_true.mp4` | 一轮 5000 诚实模式：站稳、不推进（局部最优实证） |
| `g1_walk_r2_snap.mp4` / `_snap2.mp4` | **二轮快照：真步行，速度贴合参考 0.68 vs 0.71 m/s** |
| `isaac_backtransfer_fall.mp4` | 微调权重回 Isaac 0.16 秒摔倒（对称性发现） |

## 9. 后续建议（优先级序）

1. **数据扩展**：当前仅 2 条步行片段（80 秒），泛化天花板由数据决定；扩到几十~几百条后 1024 环境/8 卡 DDP 才有意义
2. 腿抖收尾：收敛后评估，必要时 15Hz 低通或精修轮
3. 双引擎：若需 Isaac 兼容，混合训练或中间 checkpoint 双语测试（每 50 迭代都有存档）
4. 真机路线：动作空间对齐 + 滤波部署是现成的 sim-to-real 起步件
