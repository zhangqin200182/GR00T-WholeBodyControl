# Isaac 基准采集脚本（2026-08-18 交付包配套）

- `isaac_baseline_collect.py` — P0 采集：12 片段 × {release, PD} × 免终止 500 步 × 16 字段
  （基于 eval_agent_trl.py 启动序列改造；终止运行时中和、顺序配对、t=0 强制、PD 恒等翻译）
- `isaac_p1_drive.py` — P1 单关节驱动响应：3 关节 × 阶跃/正弦 27 组，root 固定+其余关节锁定，200Hz
- `run_batched.sh` — P0 分批 wrapper（4×3 环境，clean→bounded run→clean）
- `run_p1.sh` / `run_collect.sh` — 单次运行 wrapper

产物：isaac_baseline_20260818.tar.gz（md5 4245c358579a5d0e989ae1abd702d41c）
运行环境：Isaac Sim 5.1 / IsaacLab 0.54.2 / RTX 3060 12GB
注意：脚本需放在 gear_sonic/ 下运行（hydra config_path 相对解析）
