# Isaac 侧统一需求清单（第二轮，2026-08-19）

> 前一轮协议（`physx-isaac-data-collection.md`）除下述变更外全部有效。
> 本轮共 2 项工作 + 1 项校准信息。A 和 C 是一次性的；B 是一次重采。

---

## A. 规格提取（一次性，产出 `specs.txt` 回传）

**用途**：定位"踝 pitch corr 0.05-0.08"的植物闭环分歧。我们的脚碰撞是 MJCF box（16cm×7cm），接触参数自设——需要你们 Isaac 侧的真实值逐项对比。

| # | 要什么 | 来源/查询方式 |
|---|--------|--------------|
| A1 | **左脚碰撞体的几何规格**：形状类型（box/capsule/convex mesh）、尺寸（半边长或半径/半高）、相对脚/踝的局部位姿（pos+quat）| G1 USD 资产的 foot collision prim；或 Isaac Lab 里 `scene` 查 articulation 的 shape；或 `main.urdf` 的碰撞几何。输出我们可直接构造 PhysX 形状的参数 |
| A2 | **摩擦参数**：地面 plane 的 static/dynamic friction；脚的摩擦（若 per-shape 单独设置）| Isaac env 的 ground plane 配置 + articulation 材质配置 |
| A3 | **场景接触参数**：bounceThresholdVelocity、frictionType（patch / one-directional / two-directional）、contactOffset 默认值、restOffset | omni.physx 场景描述/默认值；或直接告知"全部为 PhysX 默认" |
| A4 | （校准用）**你们"腿 corr 0.67-0.85"那张表的确切运行命令与配置**：哪个 npz、drive 配置（FORCE+CFG？）、vel_iters、步数、以及 corr 的计算口径（对齐方式/窗口）| 日志/历史命令 |

**执行方式**：可以跑我们提供的 `spec_dump.py` 骨架（在你们的 Isaac 环境里，按注释适配 API），或按上表手工提取写进 specs.txt。**只要 A1-A4 的最终数值**，格式不限。

---

## B. release P0 重采（修 obs 侧注入，一次重采）

**背景**：上轮采集中 COLLECT_CLIPS 只覆盖了**记录侧** command term，策略实际消费的 motion sampling 未被覆盖——批内三 env 的 qpos 差异模式跨批逐位重复（策略每批消费同样的 3 条片段）。PD 数据有效；release 数据无法用于逐集裁决。

**变更点**（其余协议——t=0、终止中和、500 步、字段规范、joint_names——全部不变）：

| # | 要求 |
|---|------|
| B1 | COLLECT_CLIPS 同时覆盖**策略 obs 消费的 motion sampling**（确保策略输入里的动捕 = 记录的动捕）|
| B2 | 每 env 记录**实际消费的动捕 index/名称**（新字段或 sidecar 文件）|
| B3 | obs_step0 在**定义明确的时刻**捕获：reset 后第一步、**逐集导出**（每 episode 一个 930 维 actor obs），替代上轮的"首 env 的第 0 步" |
| B4 | 12 片段 × release + PD 各 1 集，其余与上轮一致（PD 上轮数据仍有效，若方便也重采一次保持批次一致）|

**回传**：与上轮相同的打包结构 + `specs.txt`（A 项）+ B2 的消费 index 文件。

---

## C. 校准信息（和 B 一起回传即可）

- 上轮 release P0 采集脚本里 **motion sampling 的调用点**（哪个函数/哪行）——方便我们确认 B1 的修法覆盖了正确的位置。
- 若 P1 数据近期有重新采集的可能（上轮 P1 的记录通道有已知自洽性问题，你们已发现），**P1 不重采也行**——D1 结论已不依赖它；仅注明即可。

---

## 已撤销（不需要做）

- ~~29 关节运行时 stiffness/damping dump~~：D1 已证运行时增益 = CFG 公式（τ = armature·ω²·e − 2ζ·armature·ω·q̇，ω=2π·10Hz，ζ=2.0）逐位精确，无需 dump。
- ~~applied_torque 字段语义澄清~~：已解决（单样本滞后记录，字段 = 驱动原始力矩）。
- ~~smoke npz~~：可忽略。

---

**优先级**：A 是当前引擎分歧排查的阻塞项（半天内可完成）；B 是 obs 层判决的阻塞项（一次重采）。两者并行最好。
