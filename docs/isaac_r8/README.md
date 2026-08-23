# R8 obs-dict dump（喂给 release 策略的完整 obs，f0 逐位对比用）

- patch：isaac_baseline_collect.py 主循环 rollout 前 dump obs_dict（step0 时打印 key 清单）
- 采集：R5 同款 env 变量（release / 500 步 / batch1 3 clips：A050/A476/A232 / 3 env）
- 产物：500 × obs_pre_policy_release_step{000-499}.npy（18M tar）
  每文件 = dict{key: (3, dim) float32}，np.load(..., allow_pickle=True).item() 读取
- **obs key 清单（实测）**：
  - actor_obs  (3, 930)  float32
  - critic_obs (3, 1645) float32
  - tokenizer  (3, 1761) float32
- env0 = f0 逐位对比目标
