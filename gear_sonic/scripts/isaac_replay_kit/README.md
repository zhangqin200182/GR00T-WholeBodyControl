# Isaac 侧一步重放包（给 GPU+Isaac 机执行）

## 目的

在 **Isaac 官方环境自己** 上做一步植物保真重放：把 npz 录制的状态精确放回官方仿真，喂同一步的命令，跑一个控制步（4×0.005s），量 Isaac 自身的残差量级。用于判决我们 NPU 侧 PhysX/MuJoCo 一步残差（step_test12：AIR 全模式 0.003-0.007、GROUND 全模式 ≈0.010 且 zero 对照同量级 → 接触轮是最后主项）的来源：

- 若 Isaac 自回放 ≈ 我方水平（AIR 0.003-0.007、GROUND ≈0.010）→ 残差是协议固有（录制噪声/一步瞬态），我方 plant 无罪；
- 若 Isaac GROUND ≈0.003-0.005（显著低于我方）→ 我方接触几何有真实差距 ~0.005-0.007（foot 几何轮）。

## 前置条件

- 就是之前生成这 12 个 release npz 的那台机器（有 isaaclab + 官方 env + 本仓库，能 import `gear_sonic.envs.manager_env`）。
- 无需训练栈、无需策略权重、无需渲染。headless 即可。

## 执行

```bash
cd <仓库根目录>
python gear_sonic/scripts/isaac_replay_kit/isaac_one_step_replay.py \
    --npz-dir gear_sonic/scripts/isaac_replay_kit/data \
    --out /tmp/isaac_replay_results
```

可选参数：`--start 100 --end 200`（窗口，默认 100..199）、`--clips 某个文件名子串`（只跑部分 clip）、`--modes target,affine,torque,target_lag`（默认全跑）、`--device cuda`。

每个 clip 4 个模式 × 100 步 × 4 子步，12 个 clip 总共约 2 万物理步，单卡几分钟量级。

## 接触配置探针（probe_contact_cfg.py，顺手跑一下）

```bash
python gear_sonic/scripts/isaac_replay_kit/probe_contact_cfg.py \
    --out /tmp/isaac_replay_results/probe.txt
```

打印官方场景的实际内容（只需 spawn，不仿真）：机器人每个碰撞 shape 的 prim 类型与尺寸（胶囊的 radius/height/PhysX halfHeight、mesh 的 convex 近似模式）、contactOffset/restOffset、脚部碰撞 shape 绑定的材质摩擦、ground plane 的类型/尺寸/contactOffset/restOffset/材质。预期（我方本地已从代码/资产核实，此探针用于机器侧终审）：
- 机器人碰撞 shape = **胶囊**（7 根：5 杆 + 2 销），contactOffset=0.02、restOffset=0.001（isaaclab 默认，官方 env 零覆盖）
- ground plane = 网格四边形，contactOffset=0.02、restOffset=0.001，材质 1.0/1.0
- 若材质显示 "unbound (scene default)" → 继承场景默认（1.0/1.0 multiply），与官方随机化（见下）不矛盾——随机化在 env event 层每 episode 覆盖
- ⚠️ 材质随机化：官方 env 的 `events/tracking/base`（eval 也启用）含 `physics_material` 项——startup 模式每 episode 对 robot 全身 body 随机 static [0.3,1.6] / dynamic [0.3,1.2] / restitution [0,0.5]（64 buckets）。此前 round-2 A2 读到的"脚摩擦 0.787/0.531/0.163"= **一次随机采样，不是固定值**；12 个 npz clip 的摩擦也各自随机（npz 未记录）。场景默认 1.0/1.0 只对未绑定材质的 shape 生效，机器人 body 在 episode 启动时被覆盖。

把 probe.txt 连同结果目录一起打 tgz 回传即可。

## 四个模式（协议依据，见下"已验证的通道语义"）

| 模式 | 喂什么 | 对应我方哪条测试 |
|---|---|---|
| `target` | 关节目标 = `joint_target[k+1]`（step k 真实激活的目标，逐位） | A 测试（喂 target 走官方 PD） |
| `affine` | 关节目标 = deploy 仿射 `offset + scale·action_raw[k]`（isaaclab 序） | A 测试的交叉验证（验证仿射映射） |
| `torque` | PD 归零 + 前馈力矩 = `applied_torque[k]` | B 测试（纯力矩） |
| `zero` | 不喂任何命令，PD 归零（纯状态设置+步进） | 我方 zero 对照（GROUND 0.0106 / AIR 0.0021） |
| `target_lag` | 关节目标 = `joint_target[k]`（滞后通道，对照组） | 预期残差显著大于 `target` |

所有模式：状态逐帧 = npz 第 k 帧（qpos/qvel/root；根速度用 k−1→k 差分，npz 无根速度通道），跑恰好一个控制步后与 npz 第 k+1 帧比较。

注意：`torque` 模式通过 Isaac 自身的 `set_joint_effort_target` 喂力矩（robot 序 = isaaclab 序 = npz 序，全程无任何转序）——这正是我方修复索引 bug 后的 B 协议，可直接对比。`torque` 与 `zero` 成对：若该机器 isaaclab 的 ImplicitActuator 不把 effort target 当纯前馈，`torque` 会退化为 `zero`，两数接近即可自证，把两个数都发回即可。

机器人资产 = 官方 `G1_CYLINDER_MODEL_12_DEX_CFG`（main.urdf，与 npz 录制环境同源），脚部碰撞 = 官方资产自带几何：ankle_roll 显式 **7 圆柱 = 5 胶囊杆**（y −0.018→+0.018，r 0.008/0.01，l 0.167-0.186，rpy −90° y 轴沿脚前后向）**+ 2 踝轴销**（(0.075,±0.026,−0.025)，r=0.01，l=0.05），`replace_cylinders_with_capsules=True`（g1.py L202）→ 全部转胶囊（PhysX halfHeight = h/2 − r）；ankle_pitch 无 collision 标签→visual mesh 凸包（36×13mm 踝罩，底面比胶囊高 ~26mm，永不触地）。**Isaac 脚底 = 纯胶囊弧面，无任何平面**。**不是**我方 XML 的 box 脚——故意如此：先量官方自洽度，再谈几何差距。

## 已验证的通道语义（写入协议的原因）

- **joint_target 通道滞后一帧**：recorded[t] ≈ true[t−1]（corr 0.999256）。step k 的真实目标 = `joint_target[k+1]` ≈ `offset + scale·action_raw[k]`（err 0.005 rad，corr 0.99993，已在数据上验证）。所以 `target` 模式喂 [k+1]，`affine` 模式喂 action[k]。
- **applied_torque 通道无滞后**：trq[t] 与 state[t] 配对（用官方 kp/kd 重建 PD 扭矩，corr 0.97，mean err 0.38 N·m）。所以 `torque` 模式喂 trq[k]。
- 官方 PD = 扭矩域 ImplicitActuator（kp/kd 表在 g1.py，armature 在动力学里）；`torque` 模式通过 set_gains(0,0) + 目标=当前状态 双保险归零 PD。
- 控制步 = sim_dt 0.005 × decimation 4 = 0.02s，目标在 4 个子步内保持（官方 env 同）。

## 输出与回传

`/tmp/isaac_replay_results/<clip名>/`：
- `dq_<mode>.npy`（100×29 逐关节残差）、`dv_<mode>.npy`、`dtau_<mode>.npy`、`droot_<mode>.npy`（根位置/姿态误差）、`dtgt_affine.npy`（仿射 vs 逐位目标的差，应 ~0.005 rad）
- `summary.txt`：每模式全局 med/mean/max |dq|、med|dtau|、根误差、逐关节 median |dq| 与 signed bias

**回传**：`tar czf isaac_replay_results.tgz /tmp/isaac_replay_results` 把 tgz 发回即可（结果目录很小，几 MB）。

## 判读参考

- 我方同协议对照值（NPU 侧，2026-08-22 step_test12，力矩索引 bug 修复后同配置，窗口 100-199）：
  - B（torque，按 isaaclab 序原样喂 applied_torque[k]）：AIR **0.0071** / GROUND **0.0114**
  - zero（只设状态、不喂力矩，PD 归零）：AIR **0.0021** / GROUND **0.0106**
  - A-old（ACCEL kp=100/kd=5 跟踪 target）：AIR 0.0035 / GROUND 0.0094（全表最低）
  - A-force（官方扭矩域增益 FORCE 模式）：AIR 0.0063 / GROUND 0.0101
  - → GROUND 全模式 ≈0.010 且 zero 对照同量级 → **主项 = 接触轮（GROUND），PD 增益轮已证伪**；AIR 已到 0.0035
- 判决（对照 Isaac 自回放）：
  - Isaac GROUND ≈0.010 → 接触残差是协议固有（录制噪声/一步接触瞬态），我方脚几何无罪；
  - Isaac GROUND ≈0.003-0.005 → 我方接触几何有真实差距 ~0.005-0.007；
  - Isaac AIR ≈0.003-0.007 → 本体通道已对齐，剩余即通道本底。
- `target_lag` 应显著大于 `target`（滞后效应对照）；若两者接近，说明该 clip 的滞后语义不同，把两个数字都发回即可。
- `dtau_*`：Isaac 自身应用力矩 vs npz applied_torque[k+1] 的差。target/affine 模式若 ~0.4 N·m 量级 = 通道噪声本底（与我方数据侧重建一致）；显著大于此 = 需要检查 PD 表或目标语义。

## 可能的兼容性问题

- 若报 `ImplicitActuator has no attribute set_gains`（旧版 isaaclab）：把 `zero_actuator_gains/restore_actuator_gains` 里的 `act.set_gains(...)` 换成直接改写 stiffness/damping 张量并调用 `robot.write_actuator_stiffness_to_sim/write_actuator_damping_to_sim`（如存在）。torque 模式同时已有"目标=当前状态"兜底，PD 项仍会归零（速度目标随状态演化，不完全精确，把结果注记一下）。
- 若 `AppLauncher({"headless": True})` 报参数错误：改成 `AppLauncher(args=argparse.Namespace(headless=True))` 或按该机器 isaaclab 版本的脚本惯例。
- 结果如有异常（如所有模式残差都 ~0.1+），先跑单个 clip：`--clips A509`。
