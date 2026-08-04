# PhysX 5 引擎切换 — Task 拆解

> 基于 `docs/physx-engine-switch-implementation.md` v2 | 2026-08-02

## 依赖关系总览

```
T4 (Python FK) 1d ──────────────────────┐
                                         │
T1 (编译环境) 0.5d                       │
  └─ T2 (pybind11 API) 2d               │
       └─ T3 (MJCF→PhysX) 2d ──────────┤
                                         ├─ T5 (PhysXEnv) 2d
                                         │    ├─ T6 (ref PD) 1d ★ ─┐
                                         │    └─ T7 (EnvManager) 1d ┤
                                         │                           └─ T8 (训练) 1d

总工时 ~10.5 天，关键路径 (T1→T2→T3→T5→T6→T8) ~8.5 天日历时间
```

---

## T1：NPU 服务器编译环境确认 + pybind11 编译验证

**目标**：确认 NPU 容器上 pybind11 + PhysX 5 可以编译链接，产出 `physx_core.pyd`。

**前置**：无

**步骤**：

1. SSH 到 NPU 服务器 `113.46.41.54`，进入 `sonic-train` 容器
2. 确认编译工具链：`gcc --version`, `cmake --version`
3. 确认 PhysX 5 库文件存在：`ls /opt/physx/physx/bin/linux.aarch64/release/`
4. `pip install pybind11`
5. 写一个最小 `physx_bindings.cpp`（只暴露 `init_foundation` + `create_scene` + `simulate` + `fetch_results`）
6. `cmake && make -j64` 编译
7. `python -c "import physx_core; print('OK')"` 通过

**产出**：

| 文件 | 说明 |
|---|---|
| `gear_sonic/envs/physx/CMakeLists.txt` | aarch64 构建配置（5 个 PhysX 库） |
| `gear_sonic/envs/physx/physx_bindings.cpp` | 最小封装（~50 行，仅 scene 创建/步进） |

**验收**：

- [ ] `import physx_core` 不报错
- [ ] `scene = px.create_scene(gravity=(0,0,-9.81))` 返回有效对象
- [ ] `scene.simulate(0.002); scene.fetch_results()` 不 crash

**估时**：0.5 天（编译环境问题可能延长）

---

## T2：pybind11 完整 API 封装

**目标**：封装全部 ~25 个 PhysX API 函数（按实施方案 §3.1 清单），C++ 侧 quat 统一为 `[w,x,y,z]`。

**前置**：T1

**步骤**：

1. 扩展 `physx_bindings.cpp`，逐步加入：
   - `PxArticulationReducedCoordinate` 的创建和 link/joint 添加
   - 状态读写（get/set joint positions, velocities, root pose, link pose）
   - `PxArticulationDrive`（set targets + params）
   - 碰撞几何（sphere, box, capsule, plane）
   - 材质和接触查询
2. 每个 API 组写单元测试（`scripts/test_physx_api.py`）
3. C++ 侧 quat 转换统一在返回 `PxQuat` 的 lambda 中处理：

   ```cpp
   // PxQuat(x,y,z,w) → numpy [w,x,y,z]
   auto q = link->getGlobalPose().q;
   return py::array_t<float>({q.w, q.x, q.y, q.z});
   ```

**产出**：

| 文件 | 说明 |
|---|---|
| `gear_sonic/envs/physx/physx_bindings.cpp` | 完整封装（~400 行） |
| `scripts/test_physx_api.py` | API 单元测试（~100 行） |

**验收**：

- [ ] 全部 25 个 API 函数可调用，参数类型正确
- [ ] `art.get_link_world_pose(idx)` 返回的四元数通过 `assert quat_mul(q, quat_inv(q)) ≈ [1,0,0,0]`
- [ ] 1000 次 `simulate + fetchResults` 循环无 crash、无内存泄漏

**估时**：2 天

---

## T3：MJCF→PhysX 转换器

**目标**：解析 G1 MJCF XML，调用 T2 的 API 创建完整的 `PxArticulation`（运动学树 + 质量惯性 + 碰撞几何 + PD 驱动参数）。

**前置**：T2

**步骤**：

1. 从 NPU 服务器拉取训练实际使用的 XML：`g1_29dof_v17.xml`
2. 用 `xml.etree.ElementTree` 解析 XML
3. 按实施方案 §4.4 碰撞几何清单创建 shape：
   - 独立 body 碰撞 mesh：22 个（从 STL 文件读顶点 → `PxConvexMeshCooking`）
   - torso 额外子 geom：5 个（logo/head/waist_support/rubber_hand×2）
   - shoulder capsule：4 个（cylinder 近似，含 `--collision-mode` 切换）
   - foot sphere/box：取决于 v17 XML 实际内容
   - ground plane：1 个
4. 创建运动学链：29 个 revolute joint + 1 个浮基，含 joint damping/frictionloss
5. 设置每关节 PD drive params（kp=100, kd=5, force_limit 来自 `actuatorfrcrange`）
6. 验证：加载 G1 → 设初始 pose → 跑 1000 步物理不掉

**产出**：

| 文件 | 说明 |
|---|---|
| `gear_sonic/envs/physx/physx_loader.py` | MJCF→PhysX 转换器（~400 行） |

**验收**：

- [ ] `art = load_g1_from_mjcf("g1_29dof_v17.xml")` 返回 articulation，`num_joints == 29`
- [ ] 碰撞几何数量和类型与 XML 一致（mesh≥27, capsule≥4, sphere/box≥8）
- [ ] 1000 步 `simulate(0.002)` 无 crash、无 NaN、关节角在合理范围

**估时**：2 天（STL mesh cooking 可能踩坑）

---

## T4：Python FK 模块

**目标**：实现纯 Python 前向运动学（替代 `mj_kinematics`）。给定 `(root_pos, root_quat, 29 joint_angles)` → 所有 link 世界位姿。

**前置**：无（可与 T2/T3 并行，可在本地开发）

**步骤**：

1. 从 MJCF XML 解析运动学链（body 名、parent 索引、local pos/quat、joint axis）
2. 实现 `G1ForwardKinematics` 类：
   - `compute(root_pos, root_quat, joint_angles)` → `[(pos3, quat4), ...]`
   - `get_link_pose(poses, name)` → `(pos3, quat4)`
3. 处理 revolute joint：绕 axis 旋转 angle，累积到 world transform
4. 浮基作为 root transform 传入
5. 验证：用 MuJoCo 生成 100 帧 gold 数据（已知 qpos → `mj_kinematics` → xpos/xquat），逐 frame 逐 body 对比

**产出**：

| 文件 | 说明 |
|---|---|
| `gear_sonic/envs/physx/physx_fk.py` | Python FK 实现（~150 行） |
| `scripts/verify_physx_fk.py` | FK 精度验证（~60 行） |

**验收**：

- [ ] 14 个 BODY_NAMES 的位姿全部可计算
- [ ] 和 MuJoCo `mj_kinematics` 的位姿误差 < 1e-6 per body per frame（100 帧随机 qpos）
- [ ] 性能 < 0.01ms per call（远小于 physics step 的 0.2ms）

**估时**：1 天

---

## T5：PhysXEnv 单环境实现

**目标**：实现 `PhysXEnv`（~500 行），完全兼容 `MuJoCoEnv` 的 API：`reset()` → obs dict，`step(action)` → (obs, reward, done, info)。

**前置**：T2, T3, T4

**步骤**：

1. 以 `mujoco_env.py` 的结构为模板
2. 替换物理层（~60 行 API 调用替换）：
   - `__init__`：`mjModel/mjData` → `PxScene + PxArticulation`，`_ensure_foundation()` 进程级单例
   - `_physics_step`：`mj_step` → `scene.simulate(native_dt) + fetchResults`
   - `_pd_control`：手动 `torque = kp*(target-q) - kd*qvel` → `art.set_joint_drive_targets(target)`（PhysX 内置 PD，力和接触同时求解）
   - `_compute_ref_body_state`：`mj_kinematics` → `self._fk.compute(root_pos, root_quat, ref_joint_angles)`
   - 所有状态查询：`data.qpos[7:]` → `art.get_joint_positions()` 等
3. 移植所有 MDP 函数（实施方案 §5.4 清单——~280 行直接搬）
4. 去掉 PKL quat `[x,y,z,w] → [w,x,y,z]` 的转换（PhysX C++ 侧已统一约定）
5. 确认 `mujoco_math.py` 7 个 quat 函数**零改动**直接 import 通过
6. 终止阈值：PhysX 下用 **Isaac 原始严格值**（`ORI_THRESH=0.2`, `ANK_POS_THRESH=0.2`, `ANK_H_MULT=1.0`），因为 PhysX 精度应和 Isaac 同级

**产出**：

| 文件 | 说明 |
|---|---|
| `gear_sonic/envs/physx_env.py` | 单 PhysX 环境（~500 行） |

**验收**：

- [ ] `env = PhysXEnv(xml, pkl_dir)` 初始化成功，无报错
- [ ] `obs = env.reset()` 返回 `{"actor_obs": (930,), "critic_obs": (1645,), "tokenizer": (1761,)}`
- [ ] `obs, reward, done, info = env.step(np.zeros(29))` 不 crash
- [ ] 200 步随机 action 循环：0 crash、reward 为标量 float、done rate < 100%
- [ ] `mujoco_math.py` 7 个函数在 PhysX 四元数上全部验证通过

**估时**：2 天

---

## T6：ref PD 精度验证 ★ 最关键

**目标**：用 100 episode 纯 ref PD 量化 PhysX 的跟踪精度，验证 α < 0.002、存活 > 80 步。**这是整个引擎切换的决策门。**

**前置**：T5

**步骤**：

1. 写 `scripts/test_physx_ref_pd.py`
2. ref PD 逻辑：

   ```python
   env = PhysXEnv(model_xml, pkl_dir)
   for ep in range(100):
       obs = env.reset()
       for step in range(500):
           # 正确做法：将参考 motion 关节角转为 action 空间
           ref_qpos = env.get_current_ref_qpos()      # 当前帧参考关节角 (29,)
           action = (ref_qpos - env.jm) / env.jh      # 归一化到 [-1,1]
           obs, reward, done, info = env.step(action)
           if done:
               break
       survivals.append(step + 1)
   ```

3. 收集指标：
   - α = mean(|root_pos - ref_root_pos|) per step（前 20 步平均）
   - 存活步数分布（mean ± std, min, max, histogram）
   - 每步 reward 分解（12 项分别统计 mean/std）
4. **对照组**：同样的 ref PD 在 MuJoCo 上跑 100 episode，得到基线数据
5. 输出对比表：α、存活步数、reward 各项

**产出**：

| 文件 | 说明 |
|---|---|
| `scripts/test_physx_ref_pd.py` | ref PD 精度测试（~100 行） |
| ref PD 指标报告 | α、存活、reward 逐项对比 |

**验收标准**：

| 指标 | MuJoCo 基线 | PhysX 目标 | 状态 |
|---|---|---|---|
| ref PD α (per-step drift) | 0.013 | **< 0.002** | 待测 |
| ref PD 存活 (宽松阈值) | 21 步 | **> 80 步** | 待测 |
| ref PD 存活 (Isaac 严格阈值 ORI=0.2, ANK=0.2) | < 5 步 | **> 50 步** | 待测 |
| reward per step | ~4.5 | > 4.5 | 待测 |

**如果验收不通过**：

1. 检查 kp/kd 是否和 Isaac Sim 一致
2. 检查求解器参数（TGS solver、position_iters=8）
3. 检查碰撞几何是否正确转换（特别是脚部——v17 XML 可能是 box 脚而不是 sphere）
4. 检查 contact_offset 设置
5. 对比 PhysX 和 MuJoCo 的 per-joint torque 轨迹（同 motion、同 action）
6. 如果上述全部正确但仍不达标：可能是 PhysX ReducedCoordinate Articulation 的某些内部行为差异，需要查 PhysX 文档或 NVIDIA 论坛

**估时**：1 天（如果 α 达标）；3-5 天（如果不达标，需要排查调试）

---

## T7：PhysXEnvManager 多进程并行

**目标**：实现 `PhysXEnvManager`，支持 4096 envs × 160 workers 并行，接口和 `MuJoCoEnvManager` 完全一致。

**前置**：T5

**步骤**：

1. 创建 `gear_sonic/envs/physx_env_manager.py`（~200 行）
2. 从 `mujoco_env_manager.py` import 共享组件：

   ```python
   from gear_sonic.envs.mujoco_env_manager import (
       EnvSharedMemory, _to_numpy,
       OBS_DIM, ACT_DIM, OBS_BYTES, ACT_BYTES,
       REW_BYTES, DONE_BYTES, TIMEOUT_BYTES, ORIG_DONE_BYTES,
   )
   ```

3. 实现 `_physx_worker_loop`：和 `_worker_loop` 结构完全相同，仅 `MuJoCoEnv` → `PhysXEnv`
4. 实现 `PhysXEnvManager` 类：
   - `__init__`：计算 env 分配 → 创建 SHM → 创建 Barrier → spawn `_physx_worker_loop`
   - `step()`、`reset()`、`reset_all()`、`close()`、`_handle_worker_crash()` 全部复用 MuJoCo 版本的结构
   - `_EnvStub` 内嵌类提供 `observation_space` 和 `action_space` 供 trainer 初始化
5. 小规模测试：64 envs × 2 workers，100 steps
6. 规模测试：4096 envs × 160 workers，100 steps

**产出**：

| 文件 | 说明 |
|---|---|
| `gear_sonic/envs/physx_env_manager.py` | 多进程管理器（~200 行） |

**验收**：

- [ ] 64 env × 2 worker：100 步，0 crash
- [ ] 4096 env × 160 worker：100 步，0 crash，内存 < 16GB
- [ ] `step()` 返回的 `(obs_dict, rewards, dones, infos)` 格式和 `MuJoCoEnvManager` 一致
- [ ] Barrier sync 时间 < 60s timeout

**估时**：1 天

---

## T8：训练 smoke test + 入口集成

**目标**：PhysX + PPO 训练跑通，reward 趋势可见。

**前置**：T6（ref PD 已达标）, T7

**步骤**：

1. 在 `train_agent_trl.py` 加入 `SONIC_PHYSX_ENV` 分支（~10 行）：

   ```python
   if os.environ.get("SONIC_PHYSX_ENV"):
       from gear_sonic.envs.physx_env_manager import PhysXEnvManager
       env = PhysXEnvManager(
           num_envs=local_envs,
           num_workers=getattr(config, "mujoco_workers", 160) // accelerator.num_processes,
           model_xml="/gear_sonic_deploy/g1/g1_29dof_v17.xml",
           pkl_dir="/sample_data/robot_filtered",
           env_config=OmegaConf.create({
               "alive_bonus": 0.0,
               "ignore_terminations": True,
           }),
       )
   ```

2. 确认 `+exp=stub_train` 配置适用于 PhysX（obs dim、action dim 等）
3. 小规模 smoke test：64 env × 2 worker
   - BC warmup：100 iter（从 pretrained checkpoint 启动）
   - PPO：100 iter
4. 检查 TensorBoard：reward 趋势、entropy、length、approxkl、NaN 计数、g1_recon
5. 如果小规模正常：扩展到 4096 env × 160 worker，500 iter

**产出**：

| 文件 | 说明 |
|---|---|
| `train_agent_trl.py` | +10 行（`SONIC_PHYSX_ENV` 分支） |
| 训练日志 + TensorBoard | smoke test 指标截图 |

**验收**：

- [ ] BC warmup 100 iter：loss 下降、不 crash
- [ ] PPO 100 iter：reward 趋势正向、entropy < 30 且下降、0 NaN、g1_recon < 0.08
- [ ] 4096 env × 160 worker：训练启动不 OOM、不 crash、fps 在可接受范围

**启动命令**：

```bash
SONIC_PHYSX_ENV=1 accelerate launch gear_sonic/train_agent_trl.py \
  +exp=stub_train num_envs=4096 headless=True use_wandb=False \
  algo.config.num_learning_iterations=500 \
  algo.config.init_at_random_ep_len=False \
  algo.trl.bf16=False algo.trl.fp16=False \
  checkpoint=/sonic-data/sonic_release/last.pt \
  use_manager_env=False sim_type=mujoco base_dir=/sonic-data/logs_rl \
  project_name=TRL_G1_Track callbacks.model_save.save_frequency=500
```

**估时**：1 天

---

## Task 汇总

| Task | 名称 | 估时 | 前置 | 可并行 | 位置 |
|---|---|---|---|---|---|
| T1 | 编译环境确认 + 最小 pybind11 | 0.5d | — | — | 服务器 |
| T2 | pybind11 完整 API 封装 | 2d | T1 | T4 | 服务器 |
| T3 | MJCF→PhysX 转换器 | 2d | T2 | T4 | 服务器 |
| T4 | Python FK 模块 | 1d | — | T2,T3 | 本地/服务器 |
| T5 | PhysXEnv 单环境 | 2d | T2,T3,T4 | — | 服务器 |
| T6 | ref PD 精度验证 ★ | 1d | T5 | T7 | 服务器 |
| T7 | PhysXEnvManager 多进程 | 1d | T5 | T6 | 服务器 |
| T8 | 训练 smoke test | 1d | T6,T7 | — | 服务器 |

**总估时**：~10.5 天（不含 T6 调试时间）

**关键路径**：T1 → T2 → T3 → T5 → T6 → T8

**可并行**：T4（Python FK）和 T2/T3 完全独立；T6（ref PD）和 T7（EnvManager）可同时在不同进程跑

---

## 启动前检查清单

在开始 T1 之前，需要确认：

- [ ] NPU 服务器 `113.46.41.54` 可 SSH 访问，`sonic-train` 容器运行中
- [ ] `g1_29dof_v17.xml` 在服务器上存在：`ls /gear_sonic_deploy/g1/g1_29dof_v17.xml`
- [ ] PhysX 5 库文件路径确认：`ls /opt/physx/physx/bin/linux.aarch64/release/`
- [ ] 编译工具链确认：`gcc --version`, `cmake --version`
- [ ] pybind11 可安装：`pip install pybind11`
- [ ] Motion PKL 数据可访问：`ls /sample_data/robot_filtered/`
- [ ] Pretrained checkpoint 存在：`ls /sonic-data/sonic_release/last.pt`

## 文件产出总览

| 文件 | Task | 行数 | 说明 |
|---|---|---|---|
| `gear_sonic/envs/physx/CMakeLists.txt` | T1 | ~40 | aarch64 构建配置 |
| `gear_sonic/envs/physx/physx_bindings.cpp` | T1→T2 | ~400 | pybind11 C++ 封装 |
| `gear_sonic/envs/physx/physx_loader.py` | T3 | ~400 | MJCF→PhysX 转换器 |
| `gear_sonic/envs/physx/physx_fk.py` | T4 | ~150 | Python 前向运动学 |
| `gear_sonic/envs/physx_env.py` | T5 | ~500 | 单 PhysX 环境 |
| `gear_sonic/envs/physx_env_manager.py` | T7 | ~200 | 多进程管理器 |
| `gear_sonic/train_agent_trl.py` | T8 | +10 | 训练入口修改 |
| `scripts/test_physx_api.py` | T2 | ~100 | API 单元测试 |
| `scripts/verify_physx_fk.py` | T4 | ~60 | FK 精度验证 |
| `scripts/test_physx_ref_pd.py` | T6 | ~100 | ref PD 精度测试 |

**总计**：~1960 行新代码 + 10 行修改

## 参考文档

- `docs/physx-engine-switch-design.md` — 引擎切换设计 v2（原始决策文档）
- `docs/physx-engine-switch-implementation.md` — 详细实施方案 v2（本文档的父文档）
- `docs/mujoco-vs-isaac-precision-analysis.md` — 精度分析报告
- Memory: `[[physx-engine-switch-design]]`, `[[status-and-next-steps]]`
