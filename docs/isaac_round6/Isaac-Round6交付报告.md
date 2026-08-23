# Isaac Round-6 交付报告（接触层收口：restOffset 三判据 + collider dump）

> 日期：2026-08-23。环境同 R5（GPU 519 等价环境）。
> 按 R6 文档 A→B→C→D 执行。**A+B 判决数据干净；C 发现两个重大协议/场景事实（见下）；D 闭环。**

---

## A. 双场景运行时 offset 回读 → **两场景 restOffset 全 0；contactOffset 不同**

读取途径：`ArticulationView.get_contact_offsets()/get_rest_offsets()`（运行时真值，
45 shapes；非 USD 层）。shape↔link 映射见 `shape_map.txt`（URDF collision-tag 数
恰 45=shape 数；**左脚 capsules=shapes 3-9，右脚=12-18**，torso=19-22）。

| 场景 | restOffset | contactOffset | 材质表 |
|---|---|---|---|
| 1 settle（MySceneCfg）| **全 0**（45/45）| 0.0020-0.0029 | 全默认 1.0/1.0/0（scene 默认）|
| 2 官方 env（ModularTrackingEnvCfg hydra 链）| **全 0**（45/45）| **0.0005-0.0029** | per-shape 多样（随机化/URDF 材质）|

**关键细节**：官方 env 的 shape 9-16（左脚末 1 个 + 右脚前 5 个 capsule）的
contactOffset = **0.000491**（0.49mm），其余 2.0-2.9mm——**官方 env 对部分脚
shape 的运行时参数确有改写**（材质随机化事件连带 or 独立），settle 场景无此现象。

**判决（按 R6 文档预测表）**：官方 env pair restOffset = 0（两场景一致）→
**站距非 restOffset 驱动** → 走 C 分支（曲线形状定案）。同时"settle 与官方 env
碰撞参数不同"的猜想部分成立：**不在 restOffset（都 0），在 contactOffset/材质管线**。

## B. 官方 env pinned +0mm（root_z=0.7845）→ **双足 0N，脚底悬空 ~12mm**

- 四个 foot body 全 0N；ankle_roll 中心 z=0.059-0.062 → capsule 脚底 ≈ +0.012（离地 12mm）
- 根因几何：default_joint_pos（URDF 微屈站姿）+ root 0.7845 下脚底在空中；
  **官方 env 行走时脚贴地靠 motion 帧姿态 + 策略，非 default spawn 姿态**
- **重要参照系修正**：触地起点 = root ≈ 0.7845-0.012 ≈ **0.7725**（-12mm 档），
  R5 settle 场景的"悬空 5-6mm"（root 0.76 参照）与此几何自洽
- 判决：**官方 env 侧 0N@悬空 → R≈0 分支与 A 互证**

## C. 深档 pinned → 两个重大发现 + 伪影注明

### 发现 1（场景级）：settle 场景（裸 MySceneCfg）的 capsule 脚从不与地面碰撞

锁关节修正版（每控制步重写 root+joint state）深档 -6~-40mm：**settle 场景
全部 0N，-40mm 档脚中心压到 0.019（穿地 ~16mm）仍 0N**。回查 R5 数据同样成立：
倒地时 93N 在 ankle_pitch（visual hull），**ankle_roll capsule 力在 settle 场景
从未非零**。裸 MySceneCfg 的 terrain（collision_group=-1）与 capsule 脚的
碰撞对疑似被过滤——**settle 系列探针（R5+R6C 的 settle 版）的接触数据全部无效**，
请在官方 env 场景解读（我们的 C 深档已改官方 env 场景重跑）。

### 发现 2（协议级）：不锁关节时腿被重力压塌、锁关节时有 depenetration 瞬态

- 软 PD（不锁）：-40mm 档腿塌 40mm，脚在世界系不动、永不触地（PD 拉不过重力）
- 锁关节（每步重写 state）：脚位置精确跟随，但**每控制步重写会打断接触求解**，
  出现非单调力（见下表 -40 档 0N = 弹离伪影，同你们 probe41 的"重写伪影"族）

### 官方 env 场景深档表（root pin @ 0.7845 基准，锁关节，尾 10 步均值）

| pin z | 名义档 | ankle_roll z | F 左 | F 右 |
|---|---|---|---|---|
| 0.7785 | -6mm | 0.0561 | 0 | 0 |
| 0.7745 | -10mm | 0.0521 | 0 | 0 |
| 0.7705 | -14mm | 0.0481 | 0 | 0 |
| 0.7645 | -20mm | 0.0421 | **0** | **155.8** |
| 0.7595 | -25mm | 0.0390 | **120.4** | **118.5** |
| 0.7445 | -40mm | 0.0221 | 0（弹离伪影）| 0（弹离伪影）|

可用的定性结论：官方 env capsule 脚在深穿透区确能出力（118-156N 量级，
与你们 probe41 的 150-250N 深区平坦段同族），但**干净力-深度曲线需要
"平衡起点+稳态读数"协议迭代**（如你们 probe42 的 200 步平衡法移植），
本轮时间所限未再迭代，npz 全序列（100 步×4 脚 body 位姿+力）已随包回传，
瞬态波形可供判读。

## D. collider dump → 两 link 无自有 collider（merged 实证）

- URDF 源：**waist_roll_link / left_shoulder_pitch_link 均无 collision tag**
  （24 个零 collision links 之一，见 shape_map.txt 全清单）
- importer：转换日志明示 merged 进父级；**USD stage 遍历：两 link 子树零几何 prim**
  （collider_dump.txt）
- → R5 squat 的 waist_roll 182N / shoulder 182N / 臂 ~10N：力在 PhysX 里归属
  merged 后的 shape（torso 4-shape 凸包等），sensor 按 link 名报告。
  **注意矛盾点**：link_paths 里 waist_roll 仍是独立 link（30 之一）但无自有
  shape，其 182N 报告的具体 shape 归属建议你们侧按 body 序对一下
- 过滤设置：raw PhysX pair filter API 不暴露；URDF importer 默认 = 父子对过滤

## 交付物

```
isaac_r6_delivery/
├── offsets_scene1.txt / offsets_scene2.txt   A：两场景 45-shape 运行时值全表
├── shape_map.txt                            shape↔link 映射（脚=3-9/12-18）
├── official_tiers/                          B+C：官方 env pinned 档位 npz+summary（7 组）
├── deep/                                    C：settle 场景深档（脚不碰地实证，注明无效）
├── collider_dump.txt                        D：空遍历 = merged 无 shape
├── scripts/                                 4 个探针脚本（可复现）
└── logs/
```

## 一句话总结（可转述）

两场景 restOffset 都是 0（站距非 restOffset 驱动，转 C 分支）；官方 env 部分
脚 capsule 的 contactOffset 被改到 0.49mm（运行时差异实锤）；官方 env 默认
spawn 脚悬空 12mm（触地在 -12mm 档）；settle 场景 capsule 脚从不碰地（该场景
接触数据全废，已换官方 env 场景补跑，深区力 118-156N 与你们同族但重写伪影
需协议迭代）；waist_roll/shoulder 无自有 collider（merged，R5 力是父级归属）。
