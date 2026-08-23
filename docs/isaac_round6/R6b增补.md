
---

# R6b 增补（地面运行时参数 + sep 精算 + 0.000491 来源）

## 地面运行时读取结果

- TerrainImporter **无任何 view**（plane/collision_view/root_physx_view/view 属性均不存在，
  data 上无 offset 属性）——地面是纯 USD prim，无 isaaclab 对象包装，get_contact_offsets()
  路线对地面不可用
- USD authored：`/World/ground/terrain/mesh (Mesh) contactOffset=-inf restOffset=-inf`
  （哨兵值 = 未设置，运行时用 PhysX 默认）
- **但请看下节**：机制实质闭环的证据在实测力数据里

## Sole separation 精算表（各档，sole = ankle_roll 中心 − 0.033~0.035，capsule r 0.008/0.010 两口径）

| pin z | sole_sep L | F_L | sole_sep R | F_R |
|---|---|---|---|---|
| 0.7845 (+0mm) | +24~29mm | 0 | +24~26mm | 0 |
| 0.7785 (-6) | +18~23mm | 0 | +18~20mm | 0 |
| 0.7745 (-10) | +14~19mm | 0 | +14~16mm | 0 |
| 0.7705 (-14) | +10~15mm | 0 | +10~12mm | 0 |
| 0.7645 (-20) | +7.1~9.1mm | 0 | **+4.3~6.3mm** | **155.8N** |
| 0.7595 (-25) | **+4.0~6.0mm** | **120.4N** | **+4.0~6.0mm** | **118.5N** |
| 0.7445 (-40) | −11~−13mm | 0(伪影) | −14~−16mm | 0(伪影) |

**决定性模式**：全部有力样本的 sole_sep ∈ [+4.0, +6.3]mm；分离 ≥+7.1mm 的样本全部 0N。
→ **官方 env 的 pair 接触生成阈值（contactOffset 语义）实测 ≈ 4.3-6mm**，
与你们预期的 ground co≈0.0042 / 站距 4.3-4.5mm 精确同区——**机制实质闭环**：
受载站距由 pair contactOffset 决定（robot 侧 co 0.49-2.9mm + 地面侧共同生效，
pair 取合成值 ~4-6mm），restOffset 两场景均 0 不参与。
（-20 档左脚 +7.1mm 无力 → 阈值上界 ~7mm；-40 深穿透 0N 为重写伪影，见 R6 主报告。）

## 0.000491 来源排查

- isaaclab 源码无该字面量（grep 空）；urdf_converter 无 contact_offset 逻辑
- **physics_material 随机化 event 实锤存在**（官方 env，startup 模式）：
  `randomize_rigid_body_material`（body_names='.*' 全身），static [0.3,1.6] /
  dynamic [0.3,1.2] / restitution [0.0,0.5]，num_buckets=64 ——官方 env 材质表
  多样的来源（也是 R5 friction_sample 的实值范围）；该 event 只写材质不改 offset
- 结论：0.000491 推断为 isaacsim/PhysX 层按 shape 几何自动计算的 contactOffset
  （capsule/convex 尺寸相关），非任何 cfg/event 显式设置

## 顺手项

- solver 运行时迭代数：PhysxSceneAPI 的 iteration attr 读取失败（此版本 attr 名
  不匹配，已尝试遍历 Get*Iteration*）；USD 层证据已有（R5 A3：solverType=TGS，
  max/minPositionIteration=255/1 ——场景上下限非实际迭代数）。g1.py 的 8/4 是
  articulation solver_position/velocity_iteration_count（关节求解器迭代），
  与场景 TGS 迭代是两层参数
