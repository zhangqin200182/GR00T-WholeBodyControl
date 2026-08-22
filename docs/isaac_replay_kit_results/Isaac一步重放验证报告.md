# Isaac 一步重放验证报告（同事 replay kit，2026-08-22）

> 执行环境：GPU 519（新复原云机，与原 555 采集环境等价——P1 重放 27/27 通过、
> root 漂移 2.67mm；isaacsim 5.0.0.0 / IsaacLab v2.3.2@37ddf6268 / RTX 3060 同款同驱动）。
> 数据：kit 自带 12 个 npz（round-2 重采版 release clips，ep 后缀 00-11）。
> 协议：每 clip 取窗口 [100,200) 共 100 步，逐步把 npz 状态写回 Isaac、按模式注入
> 驱动、推一个控制步（实测精确 0.0200s），对比下一步状态的残差。

---

## 1. 全量结果（12 clips × 5 模式）

| 模式 | med\|dq\| 均值 | min | max | med\|dtau\| 均值 (N·m) | med root_err |
|---|---|---|---|---|---|
| target | **0.00494** | 0.00357 | 0.00591 | 0.72 | ~0.0014 m |
| affine | **0.00507** | 0.00359 | 0.00601 | 0.72 | ~0.0014 m |
| torque (B协议) | **0.00333** | 0.00284 | 0.00387 | 0.38 | ~0.0016 m |
| target_lag | **0.00759** | 0.00556 | 0.00909 | 2.31 | ~0.0020 m |
| zero | **0.00277** | 0.00239 | 0.00330 | 1.28 | ~0.0022 m |

逐 clip 明细见各 `summary.txt`；逐关节残差见 `dq_<mode>.npy`（100×29）。

## 2. 按同事判读标准的结论

同事 NPU 侧对照（2026-08-22 step_test12，力矩索引 bug 修复后）：
- B（torque）GROUND **0.0114**；zero GROUND **0.0106**

本侧 Isaac 自回放（全部为触地行进 clip，即 GROUND 域）：
- torque **0.00333**、zero **0.00277**、target **0.00494**

→ 落在判读标准 **"Isaac GROUND ≈0.003-0.005 → 我方（PhysX 侧）接触几何有真实差距
~0.005-0.008"** 的区间。Isaac 自回放残差比 PhysX 侧低约 3-4×，说明：
1. 协议固有噪声（录制噪声 + 一步接触瞬态）在 Isaac 侧只有 ~0.003；
2. PhysX 侧多出的 ~0.007-0.008 不能用协议噪声解释 → **指向 PhysX 脚部接触几何
   与 Isaac 官方胶囊脚的真实差距**（Isaac 脚 = 纯胶囊弧面，非 box）。

## 3. 协议自检项（全部通过）

- 步长实测：一个控制步推进 **0.0200s**（4×0.005 子步，精确）
- target_lag (0.0076) **显著大于** target (0.0049)：滞后语义对照成立
  （joint_target 通道滞后一帧的结论在 Isaac 侧重放中复现）
- affine ≈ target（0.0051 vs 0.0049）：action 仿射重建目标语义正确
- torque < target：力矩前馈（set_joint_effort_target）比 PD 目标跟踪更贴近记录，
  ImplicitActuator 的 effort target 在此环境表现为纯前馈（未退化）
- zero 最小（0.0028）：无驱动状态本底
- med|dtau|：torque 模式 0.38 N·m ≈ README 预期的 ~0.4 通道噪声本底；
  target 模式 0.72 N·m（同量级略高）
- root_err 全模式 ≤0.0022 m

## 4. 执行中的兼容性 patch（4 项，均已验证不影响结果语义）

1. `SimulationContext(sim_cfg, device=)` → 此版 IsaacLab 不接受 device kwarg，
   改 `sim_cfg.device = device; SimulationContext(sim_cfg)`
2. AppLauncher 构造 → 改用 `add_app_launcher_args + parse_known_args` 链（与采集脚本同款），
   并在其后显式 `set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)`
   （IsaacLab v2.3.2 硬编码要求 importer 2.4.31，本环境为 2.4.19——555 备份时点的版本；
   无版本号启用即可）
3. **kit 的 68 个 STL 全是 Git LFS 指针文件**（同事打包未拉 LFS）→ 用录制环境同源
   真资产覆盖（main.urdf 用仓库相对路径版，与 npz 录制环境一致）
4. **两个结果正确性 bug**（影响数字，非仅兼容性）：
   a. 循环内从不调 `robot.update(SIM_DT)` → data buffer 不刷新（修复前五模式数字完全相同）
   b. 从不调 `robot.write_data_to_sim()` → set_*_target 只写 buffer 未落物理
   （修复前 zero 与 target 无分化）。修复后冒烟即呈现全部判读特征。

patch 后脚本随包附上：`isaac_one_step_replay_patched.py`（probe 脚本仅做了 1/2 两项）。
**注意：这两处（4a/4b）在同事原开发环境若同样缺失，其 NPU/Isaac 对照数字可能也受影响，
建议同事侧核对。**

## 5. probe.txt 说明

接触配置探针输出 `total capsules: 0`——与 555 时代 round-2 spec dump 的已知现象一致
（IsaacLab URDF 导入的 collision 在 USD 层遍历不可见，需读 URDF 源）。此为 USD 层视图
问题，不影响物理（contact sensor 数据正常）。脚部真实几何以 round-2 specs.txt 的
URDF 7 圆柱→胶囊为准。

## 6. 环境注记

非原机 555（已释放），为等价复原环境（同版本栈/同卡/同驱动，P1 逐位等价验证）。
一处已知差异：IsaacLab urdf_converter 加了 hasattr 守卫（importer 2.4.19 缺
set_merge_fixed_ignore_inertia，该方法只设置默认值）。P0 采集 npz 与原机逐字节同大小，
说明 URDF 转换产物一致。

## 7. 交付物

- `isaac_replay_results.tgz`（2.4M）— 回传同事：12 clips 结果目录（dq/dv/dtau/droot npy
  × 5 模式 + summary.txt）+ probe.txt + patched 脚本
- 本报告
- `isaac_one_step_replay_patched.py` — 4 项 patch 后的可复现脚本
