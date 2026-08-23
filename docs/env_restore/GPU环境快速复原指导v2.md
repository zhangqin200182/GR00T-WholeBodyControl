# GPU 环境快速复原指导（v2，2026-08-23 实战总结）

> 适用：更换/新申请云 GPU 后，把 Isaac 采集/验证环境快速复原到可用状态。
> 实战记录：555→519 复原全程约 2.5 小时（含 9G 环境传输 80 分钟），
> 若按本文档避坑可压缩到 ~1.5 小时（传输时间为主）。
> 旧版文档：`GPU环境搭建指导-IsaacSim.md`（从零安装用，慢路）；
> 本文档走**零安装整体搬运**路线（快路，已两次验证）。

---

## 一、复原物资（Mac 本地，都在）

| 物资 | 路径 | 用途 |
|---|---|---|
| 20G conda 环境 | `gpu_backup_20260818/env_full/opt/` | 零安装环境（tar 后 ~9G）|
| 权重+数据+repo_state+脚本 | `gpu_backup_20260818/`（weights/data/scripts/repo_state）| 小件包 ~2.2G |
| 519 增量（R5 后） | `isaac_r5_delivery/`（脚本含全部 patch）| 最新脚本版本 |
| git 仓库 | `gwc-push/`（1.7G 含 .git，HEAD=c72806f+）| 比 backup 的 repo_state 新 |

## 二、标准复原流程（7 步）

```bash
# 0) SSH 免密（无 sshpass 时用 expect）
expect -c 'spawn ssh -p <PORT> root@<HOST> "mkdir -p /root/.ssh"
  expect "*assword*" { send "<PWD>\r" }'
cat ~/.ssh/id_ed25519.pub | ssh -p <PORT> root@<HOST> "cat >> /root/.ssh/authorized_keys"

# 1) 传小件（2.2G，~19 分钟 @2MB/s）
COPYFILE_DISABLE=1 tar czf /tmp/gpu_small.tar.gz -C gpu_backup_20260818 weights data scripts repo_state env
rsync -e "ssh -p <PORT>" /tmp/gpu_small.tar.gz root@<HOST>:/root/

# 2) 传仓库：GitHub 直连多半慢/断 → 直接本地 rsync（15 分钟）
rsync -a -e "ssh -p <PORT>" gwc-push/ root@<HOST>:/root/GR00T-WholeBodyControl/

# 3) IsaacLab：ghfast 代理 clone（新机自己拉，不占本地带宽）
git clone --filter=blob:none --no-checkout https://ghfast.top/https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && git checkout 37ddf626871758333d6ed89cf64ad702aef127d0

# 4) 传环境（9G tar，~80 分钟，与步骤 2/3 并行不得——串行省心）
COPYFILE_DISABLE=1 tar czf /tmp/gpu_env.tar.gz -C gpu_backup_20260818/env_full opt
rsync -e "ssh -p <PORT>" /tmp/gpu_env.tar.gz root@<HOST>:/root/
# 远端: tar xzf /root/gpu_env.tar.gz -C /   → /opt/Anaconda3/envs/isaac

# 5) 清理 AppleDouble 垃圾（必须！15.8 万个 ._* 文件会导致 Kit 加载 so 报 invalid ELF）
find /opt/Anaconda3/envs/isaac \( -name "._*" -o -name ".DS_Store" \) -delete
find /root/IsaacLab /root/GR00T-WholeBodyControl \( -name "._*" -o -name ".DS_Store" \) -delete

# 6) 打 IsaacLab 兼容 patch（urdf importer 版本差）
#    urdf_converter.py L143 的 set_merge_fixed_ignore_inertia 加 hasattr 守卫
#    （备份 env 是 importer 2.4.19，IsaacLab v2.3.2 期望 2.4.31）

# 7) 恢复仓库运行态 + 铺资产 + 打快照（防云实例重置！）
cd /root/GR00T-WholeBodyControl
git config --global --add safe.directory $(pwd)
tar xzf /root/gpu_restore/repo_state/modified_files.tgz   # base_eval.yaml + STL
mkdir -p sonic_release sample_data
cp /root/gpu_restore/weights/{last.pt,config.yaml} sonic_release/
cp -r /root/gpu_restore/data/sample_data_full/* sample_data/
cp isaac_r5_delivery/scripts/*.py gear_sonic/   # 最新脚本（含 R5 字段版 collect 等）
tar czf /root/GR00T_quick_backup.tgz --exclude=.git .   # 快照，防重置
```

## 三、验证判据（三层，由快到全）

1. **import 层**（秒级）：
   `python -c "import torch; print(torch.cuda.is_available()); import isaacsim"`
   `python -c "import isaaclab; print(isaaclab.__file__)"` → 必须指向 /root/IsaacLab
2. **P1 物理层**（~8 分钟）：`bash /root/run_p1.sh`
   → **27/27 npz + root_drift ≤3mm** = 环境等价（RESTORE.md 判据）
3. **P0 策略层**（~6 分钟）：单批 3 env × 500 步
   → 3 npz 产出且**文件大小 412066 字节与基准一致** = 策略链路等价

⚠️ P0 注意：环境变量名是 `COLLECT_OUT`/`COLLECT_CLIPS`/`COLLECT_POLICY`（json 数组
注意 shell 引号剥除，最好用 run_batched_r2.sh 包装）。

## 四、踩坑全记录（换机必读，每个都实测踩过）

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | Mac tar AppleDouble | Kit 报 `invalid ELF header` / UnicodeDecodeError | COPYFILE_DISABLE=1 打包；已污染则 find delete |
| 2 | GitHub 直连慢 | clone 超时/RPC 失败 | GR00T 用本地 rsync；IsaacLab 用 ghfast.top 代理 |
| 3 | urdf importer 版本差 | `no attribute set_merge_fixed_ignore_inertia` | hasattr patch（L143）|
| 4 | 云实例中途重置 | 恢复好的目录消失 | 铺完立即 tar 快照；crontab 无益（平台层问题）|
| 5 | git safe.directory | git 命令 fatal owner | `git config --global --add safe.directory` |
| 6 | LFS 指针文件 | URDF 转换 `Unsupported Format (/meshes/xx)` | 用录制环境真 STL 覆盖（同事包也踩过）|
| 7 | editable isaaclab | import 找不到 | IsaacLab 必须 checkout 37ddf6268 到 /root/IsaacLab |

## 五、IsaacLab 脚本层 API 坑（此版本 v2.3.2 特有，写脚本/跑同事包必读）

| 坑 | 解法 |
|---|---|
| `SimulationContext(sim_cfg, device=)` 不接受 | `sim_cfg.device = device` 后传单参 |
| `isaacsim.asset` 导不进 | AppLauncher 后 `set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)`（不带版本号）|
| AppLauncher 构造 | 用 `add_app_launcher_args + parse_known_args` 链（P1 同款），别用 dict |
| `robot.num_dof` 不存在 | `len(robot.joint_names)` |
| per-body 接触力 | ContactSensor 直连构造（prim 挂 `/World/envs/env_0/Robot/.*`）+ `_initialize_impl()` 手动初始化；MySceneCfg 不认 sensors 字段 |
| **裸循环 drive 零力矩** | **每控制步必须 `set_joint_position_target` + `write_data_to_sim`**（只写一次 → tau=0，实测）|
| 每档 `sim.reset()` | stop/play 丢 drive 状态 → 只 reset 一次 |
| `robot.data.net_forces_w` 不存在 | 只在 ContactSensor（纯法向力，无关节反力投影）|
| RigidContactView 无 get_contact_count | 此版本无 manifold 点数 API |

## 六、GPU 现状速查（2026-08-23）

- **519**（ssh.1617k.com:50519，免密已配）：RTX 3060 12GB，环境完整（P1/P0/R3 三链路验证过），R5 全部产物在 /root/isaac_r5_delivery.tgz + /tmp/isaac_settle_probe*，快照 /root/GR00T_quick_backup.tgz
- 旧 555 已释放；Mac 备份 = gpu_backup_20260818（8/18 快照）+ isaac_r5_delivery（8/23 增量）
- 关键版本：isaacsim 5.0.0.0 / IsaacLab v2.3.2@37ddf6268 / torch 2.7.0+cu126 / RTX 3060 同款同驱动 580.105.08
