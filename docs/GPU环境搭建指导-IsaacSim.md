# GPU 服务器 Isaac Sim 环境搭建指导（含全部踩坑记录）

> 2026-08-16 整理 | 机器：云 GPU（ssh.1617k.com:50555）RTX 3060 12GB
> 目标：在 Isaac Sim 里跑通 GR00T-WholeBodyControl（fork: zhangqin200182/feature/mujoco-training）的官方 SONIC 37M 权重训练/评估/录像
> 结果：全链路贯通 ✅ 官方权重加载验证 ✅ 1080p 评估录像 ✅

---

## 1. 最终可用版本清单（已验证）

| 组件 | 版本 | 来源/路径 |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS | 云主机自带 |
| GPU 驱动 | 580.105.08（RTX 3060 12GB） | 云平台预装 |
| Python | 3.11.15 | conda env `isaac` @ `/opt/Anaconda3/envs/isaac` |
| PyTorch | 2.7.0 (+torchvision 0.22.0) | TUNA 镜像 |
| Isaac Sim | **5.0.0.0**（pip 版） | `pypi.nvidia.com` |
| IsaacLab | v2.3.2（pip 包名 isaaclab 0.54.2，源码安装） | `/root/IsaacLab` |
| warp-lang | 1.16.0 | pip（注意：kit 内实际用自带 omni.warp 1.7.1，见坑 F2） |
| imageio-ffmpeg | 0.6.0 | pip（录像必需） |
| 代码仓 | `/root/GR00T-WholeBodyControl` | ghfast.top 镜像克隆（完整历史） |
| 官方权重 | `sonic_release/last.pt`（step 41550）+ config.yaml | HuggingFace |

**磁盘**：全部装完约占 40~50GB（isaacsim pip 版 ~20GB + torch + 仓库 + 权重），建议预留 60GB+。

---

## 2. 搭建步骤（按序）

### 2.1 网络与 Git

```bash
# GitHub 直连极慢/被重置 → 全局走 ghfast.top 镜像
git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/"
# HTTP/2 常被中间设备 RST → 降级 HTTP/1.1
git config --global http.version HTTP/1.1

# 完整历史克隆（不要 --depth=1，后续可能要查历史）
cd /root && git clone https://github.com/zhangqin200182/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl && git checkout feature/mujoco-training
```

### 2.2 Python 环境

```bash
conda create -n isaac python=3.11 -y
conda activate isaac   # 或直接用绝对路径 /opt/Anaconda3/envs/isaac/bin/python
```

⚠️ 见坑 A1：isaacsim 4.5 只有 cp310 轮子；5.x 需要 py3.11。直接上 py3.11 + 5.0。

### 2.3 PyTorch（务必走国内镜像）

```bash
pip install torch==2.7.0 torchvision==0.22.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

⚠️ 见坑 A3：IsaacLab v2.3.2 的 setup.py 要求 torch>=2.7；从 pytorch.org 官方源下载会死锁。

### 2.4 Isaac Sim（pip 版）

```bash
pip install isaacsim==5.0.0.0 --extra-index-url https://pypi.nvidia.com
```

### 2.5 IsaacLab v2.3.2（源码安装 + 两处必打补丁）

```bash
cd /root && git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && git checkout v2.3.2
pip install -e source/isaaclab --no-build-isolation
```

**补丁 ①**（importer 版本号对齐）：isaaclab 2.3.2 的 URDF 转换代码请求 importer 2.4.31，但 isaacsim 5.0 自带 2.4.30：
```bash
grep -rl "isaacsim.asset.importer.urdf-2.4.31" /root/IsaacLab/source/ | xargs sed -i 's/2\.4\.31/2.4.30/g'
```

**补丁 ②**（warp 1.7.1 兼容，`source/isaaclab/isaaclab/utils/warp/fabric.py`）：
把 `wp.transform_compose(...)`（需要 warp≥1.16，kit 只带 1.7.1）换成手工矩阵复合：
```python
rot = wp.quat_to_matrix(rotation)
... wp.mat44(
    rot[0,0]*scale[0], rot[0,1]*scale[1], rot[0,2]*scale[2], position[0],
    rot[1,0]*scale[0], rot[1,1]*scale[1], rot[1,2]*scale[2], position[1],
    rot[2,0]*scale[0], rot[2,1]*scale[1], rot[2,2]*scale[2], position[2],
    0.0, 0.0, 0.0, 1.0)
```
布局与 warp 1.16 的 transform_compose 逐元素一致（已对照其源码 docstring 验证）。

### 2.6 其余依赖

```bash
pip install h5py rich tensordict vector_quantize_pytorch imageio-ffmpeg -i https://pypi.tuna.tsinghua.edu.cn/simple
# flatdict 构建失败时（坑 D2）：
pip install "setuptools<81" toml -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flatdict --no-build-isolation
# ⚠️ 必须卸载 usd-core（与 isaacsim 自带 pxr 冲突，坑 D4）：
pip uninstall -y usd-core
```

### 2.7 LFS 网格文件（最容易翻车的一步）

克隆下来的 `gear_sonic/data/assets/robot_description/urdf/g1/meshes/` 里是 **130 字节的 LFS 指针文本**而不是 STL。用批量 API 脚本 `lfs_fetch.py`（Mac 上有成品）逐个换回真文件：

- 原理：读指针里的 oid/sha256 → `POST <lfs-server>/info/lfs/objects/batch` 拿下载链接 → 下载 + sha256 校验
- 共需 60+ 个 STL，**还有 3 个 `_rev_1_0` 后缀的变体网格**（URDF 引用了但容易漏）
- 可选：对大网格减面（我们 665856→209313 三角形），降低转换器负担

### 2.8 官方权重

`sonic_release/` 下需要 `last.pt`（469MB）+ `config.yaml`（28KB），从 HuggingFace 官方仓库下载（LFS 同样可能要批量 API 处理）。

---

## 3. 踩坑全记录（症状 → 根因 → 解法）

### A. 版本矩阵类

**A1. `isaacsim==4.5 在 py3.11 装不上（No matching distribution）**
4.5 的轮子只有 cp310；5.x 才有 cp311。直接 py3.11 + isaacsim 5.0，不要来回折腾。

**A2. isaaclab 2.3.2 与 isaacsim 5.1 的 importer 冲突**
2.3.2 源码里写死了 importer 2.4.31（5.1 分支的），pip 装的 isaacsim 自带 2.4.30 → 启动即版本不匹配。解法见补丁①（统一改成 2.4.30）。

**A3. IsaacLab 2.3.2 强制 torch>=2.7，官方 torch 源下载死锁**
装 torch 一律走 TUNA。曾组合：py3.10+4.5（被 A1 否）、py3.11+5.1（被 A2 拖）、最终 py3.11+5.0.0.0+torch2.7.0 ✅。

### B. Git/网络类

**B1. 克隆挂死/HTTP2 RST** → ghfast 镜像 + HTTP/1.1（见 2.1）。
**B2. SMPLSim 等直接从 GitHub拉源码的包装不动** → 全局 insteadOf 镜像对 git+https 也生效。

### C. LFS 类（夜间排查最久的根因）

**C1. URDF 转换崩溃：`NULL TfRefPtr<UsdStage>` / 段错误 / PhysX 初始化随机崩**
根因就是网格是 LFS 指针——USD 舞台引用了空文件。这个坑会伪装成各种底层崩溃，换了 4 个版本组合才定位到。**先查网格文件是不是 130 字节文本**：
```bash
find gear_sonic/data/assets -name "*.stl" -size -1k | head   # 有输出就是指针
```

**C2. `_rev_1_0` 变体网格缺失**
URDF 里 `g1_hand_rev_1_0.stl` 等引用和主网格不同名，单独漏掉 3 个，转换照样崩。批量下载后用 URDF 引用清单核对。

### D. 包安装类

**D1. wandb UsageError** → 运行加 `use_wandb=false` + 环境变量 `WANDB_MODE=disabled`。
**D2. flatdict 构建失败（pkg_resources）** → `setuptools<81` + `--no-build-isolation`，主环境先装好 toml。
**D3. EULA 交互卡死（EOFError）** → `yes Yes | <命令>` 管道喂进去。
**D4. `pxr` 段错误（extension class wrapper）** → pip 的 usd-core 和 isaacsim 内置 pxr 冲突，卸载 usd-core。
**D5. 缺 h5py/rich/tensordict/torchvision/vector_quantize_pytorch** → 按需补装。

### E. URDF 转换类

**E1. `ImportConfig has no set_merge_fixed_ignore_inertia`（4.5/5.0 没这 API）**
`IsaacLab/source/isaaclab/isaaclab/sim/converters/urdf_converter.py` 里加 hasattr 防御。

**E2. 转换缓存复用坏产物**
转换失败后 `/tmp/IsaacLab` 下的 USD 缓存会被复用，改完网格还崩 → `rm -rf /tmp/IsaacLab/*` 再重试。

### F. 运行时类

**F1. `ModuleNotFoundError: No module named 'isaacsim.asset'`**
URDF importer 扩展没随 experience 自动启用。在 `simulation_app = app_launcher.app` 之后补：
```python
import omni.kit.app
_ext_mgr = omni.kit.app.get_app_interface().get_extension_manager()
_ext_mgr.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
```
`train_agent_trl.py` 和 `eval_agent_trl.py` 都要加（评估脚本当时漏了，白崩一次）。

**F2. warp 版本两头不兼容（开相机渲染时）**
- isaaclab 的 fabric.py 要 `wp.transform_compose`（只有 warp≥1.16 有）
- isaacsim core 的 `wp.types.array` 注解要旧 API（1.16 里已移除）
- kit 里 `import warp` 固定解析到自带 omni.warp 1.7.1，**预导入 pip 版 warp 1.16 也行不通**（isaacsim core 立刻炸）
- 唯一干净解法：补丁②，手写矩阵复合，全进程统一用 1.7.1

### G. 评估/录像类

**G1. Hydra `+` 前缀**：`base_eval.yaml` 只定义了 `checkpoint`，其它键（num_envs/headless/run_once/use_wandb/manager_env.config.render_results...）全部要 `+key=value` 新增语法，否则 `Could not override` 直接退出。

**G2. release 配置指向未公开数据**：`sonic_release/config.yaml` 里 `motion_file: data/motion_lib_bones_seed/...` 是 NVIDIA 内部库，公开包里没有 → CLI 覆盖成样例数据：
```
+manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered/210531
+manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered
```

**G3. 录像黑屏/不出文件三连**：
① `manager_env.config.render_results=True` 才会建 eval_camera（否则 `save_video_path` 静默失效）；
② `base_eval.yaml` 默认 `manager_env/recorders: empty`，0 个 recorder = 不写任何视频 → 改成 `manager_env/recorders: render`；
③ 写 mp4 需要 `imageio-ffmpeg`（pip 装，自带静态 ffmpeg，无需系统安装）。

**G4. 录制中的 mp4 拷出来打不开**：ffmpeg 的 moov 索引在退出时才写入。正常跑完没问题；提前要片段就 kill 进程，ffmpeg 收到 stdin EOF 会自动 finalize，文件即可用（校验 `moov`+`mdat` 标记存在）。

**G5. `python: command not found`**：非交互 ssh 没激活 conda → 一律用绝对路径 `/opt/Anaconda3/envs/isaac/bin/python`。

### H. 运维类

**H1. `pkill/pgrep -f` 自杀**：如果命令行里带着目标脚本名（比如 grep 的路径参数），pgrep 会匹配到执行中的远端 shell 自己，整条命令无声失败 → 模式用 `[e]val_agent_trl` 括号写法，且杀进程命令不要和其他引用该名字的命令拼在同一条 ssh 里。
**H2. 后台驻留**：`nohup ... > log 2>&1 & echo PID=$!; disown` 组合；ssh 端建议加 `exit 0` 防挂起。

---

## 4. 验证过的命令

### 4.1 训练冒烟（官方权重加载 + 跑通）

```bash
cd /root/GR00T-WholeBodyControl
WANDB_MODE=disabled /opt/Anaconda3/envs/isaac/bin/python gear_sonic/train_agent_trl.py \
  +exp=sonic_release_test checkpoint=sonic_release/last.pt \
  num_envs=16 +max_train_steps=2 use_wandb=false   # exp 名以 config/exp/manager/universal_token/ 下实际文件为准
```
通过标志：日志出现 `Loaded checkpoint from step 41550`，两轮迭代无崩溃，reward ≈ 0.83→0.99。

### 4.2 官方权重评估录像（1080p × N 环境）

```bash
cd /root/GR00T-WholeBodyControl
WANDB_MODE=disabled nohup /opt/Anaconda3/envs/isaac/bin/python gear_sonic/eval_agent_trl.py \
  checkpoint=sonic_release/last.pt \
  +manager_env.config.render_results=True \
  +num_envs=8 +headless=true +run_once=true +use_wandb=false \
  +manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered/210531 \
  +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered \
  > /root/eval_video.log 2>&1 &
```
- 视频输出：`sonic_release/renderings_training/00000N.mp4`（每环境一个）
- 12GB 卡上 8 环境 × 1080p 相机 ≈ 10.7GB 显存，别再往上加
- 跑完自动退出（run_once）；中断则 kill 后文件仍可用（坑 G4）
- 40 秒动作 × 8 路渲染全程约 25~40 分钟，属正常

---

## 5. 遗留注意事项

1. **机器上已打的所有补丁清单**（换机器重建时照抄）：
   - `gear_sonic/train_agent_trl.py`、`gear_sonic/eval_agent_trl.py`：F1 的 importer 扩展启用
   - `IsaacLab/.../warp/fabric.py`：F2 补丁②
   - `IsaacLab/.../converters/urdf_converter.py`：E1 hasattr 防御 + 补丁①版本号
   - `gear_sonic/config/base_eval.yaml`：recorders: empty → render
2. **不要 `pip install -U` 任何 isaac* / warp / usd 相关包**，版本矩阵极脆弱。
3. GPU 云机小时计费，闲置时可以销毁；重建时按本文档 2.1→2.8 顺序约 1.5 小时（不含排坑）。
4. MuJoCo/NPU 侧（另一台 8×910B3 机器）的环境搭建是另一套流程，文档另行整理。
