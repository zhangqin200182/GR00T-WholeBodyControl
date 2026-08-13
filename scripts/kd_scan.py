#!/usr/bin/env python3
"""kd inference-only scan — from A1 checkpoint, pure rollout, no training."""
import subprocess, re, os

CKPT = "/sonic-data/logs_rl/TRL_G1_Track/stub_train_test-20260719_071131/last.pt"
LOG_BASE = "/tmp/kd_scan_results"
KD_VALUES = [2, 3, 5, 8, 10, 15]
NUM_ENVS = 4096
TIMEOUT = 600

os.makedirs(LOG_BASE, exist_ok=True)

# Backup original training script
with open("/gear_sonic/train_agent_trl.py") as f:
    ORIGINAL = f.read()

results = []

for kd in KD_VALUES:
    print(f"\n--- kd={kd} ---")

    # Inject kd_override into env_config via string replace
    # Handle both clean line (...0.5},) and already-modified line (...0.5, "kd_override": X},)
    import re as _re
    modified = _re.sub(
        r'"alive_bonus": 0\.5(, "kd_override": [0-9.]+)?\}',
        f'"alive_bonus": 0.5, "kd_override": {kd}}}',
        ORIGINAL
    )
    with open("/gear_sonic/train_agent_trl.py", "w") as f:
        f.write(modified)

    log = os.path.join(LOG_BASE, f"kd_{kd}.log")
    env = os.environ.copy()
    env.setdefault("SONIC_MUJOCO_ENV", "1")

    cmd = [
        "accelerate", "launch", "/gear_sonic/train_agent_trl.py",
        "+exp=stub_train",
        f"num_envs={NUM_ENVS}",
        "headless=True",
        "use_wandb=False",
        "algo.config.num_learning_iterations=1",
        "algo.config.init_at_random_ep_len=False",
        "algo.trl.bf16=False",
        "algo.trl.fp16=False",
        f"checkpoint={CKPT}",
        "use_manager_env=False",
        "sim_type=mujoco",
        f"base_dir={LOG_BASE}/runs/kd_{kd}",
        "project_name=TRL_G1_KdScan",
    ]

    try:
        with open(log, "w") as f:
            subprocess.run(cmd, cwd="/", env=env, stdout=f,
                          stderr=subprocess.STDOUT, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {TIMEOUT}s")

    # Parse metrics from TensorBoard events
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import glob as gb
    import time

    length = reward = entropy = noise = None
    errors = 0

    # Wait a bit for TB flush
    time.sleep(5)
    tb_dirs = gb.glob(f"{LOG_BASE}/runs/kd_{kd}/TRL_G1_KdScan/*/tb")
    if tb_dirs:
        try:
            ea = EventAccumulator(tb_dirs[0])
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(tag)
                if not events:
                    continue
                v = events[-1].value
                if "length" in tag.lower():
                    length = v
                elif "reward" in tag.lower():
                    reward = v
                elif "entropy" in tag.lower():
                    entropy = v
                elif "noise" in tag.lower():
                    noise = v
        except Exception as e:
            print(f"  TB read failed: {e}")
    else:
        print("  No TB dir found, checking stdout...")
        with open(log) as f:
            text = f.read()
        m = re.search(r"Mean length:\s*([0-9.]+)", text)
        if m: length = float(m.group(1))
        m = re.search(r"Mean rewards:\s*(-?[0-9.]+)", text)
        if m: reward = float(m.group(1))
        m = re.search(r"Mean entropy:\s*([0-9.]+)", text)
        if m: entropy = float(m.group(1))
        m = re.search(r"noise std:\s*([0-9.]+)", text)
        if m: noise = float(m.group(1))

    # Also check for errors in stdout (handle binary chars from box-drawing)
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="replace") as f:
            text = f.read()
        errors += text.count("Traceback") + text.count("ChildFailedError")

    results.append(dict(kd=kd, length=length, reward=reward,
                       entropy=entropy, noise=noise, errors=errors))
    print(f"  kd={kd}  len={length}  rew={reward}  ent={entropy}  "
          f"noise={noise}  err={errors}")

# Restore original
with open("/gear_sonic/train_agent_trl.py", "w") as f:
    f.write(ORIGINAL)

# Summary
print("\n" + "=" * 60)
print(f"{'kd':<6} {'length':<10} {'reward':<10} {'entropy':<10} {'noise':<10}")
print("-" * 60)
for r in results:
    print(f"{r['kd']:<6} {str(r['length']):<10} {str(r['reward']):<10} "
          f"{str(r['entropy']):<10} {str(r['noise']):<10}")

# Best kd by length
valid = [r for r in results if r['length'] is not None]
if valid:
    best = max(valid, key=lambda r: r['length'])
    print(f"\nBest kd by length: {best['kd']} "
          f"(len={best['length']:.2f}, rew={best['reward']:.2f})")
print("done.")
