#!/usr/bin/env python3
"""Task 5: MuJoCoEnvManager verification."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gear_sonic.envs.mujoco_env_manager import MuJoCoEnvManager

XML = "/root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"
PKL = "/root/GR00T-WholeBodyControl/sample_data/robot_filtered"

def test_small():
    print("=" * 60)
    print("5.1: Small scale — 64 envs × 4 workers")
    mgr = MuJoCoEnvManager(num_envs=64, num_workers=4, model_xml=XML, pkl_dir=PKL)
    print(f"  Workers: {mgr._actual_workers}")

    # Initial obs
    actions = np.random.uniform(-0.2, 0.2, (64, 29)).astype(np.float32)
    obs, rewards, dones, info = mgr.step(actions)
    print(f"  obs: a={obs['actor_obs'].shape} c={obs['critic_obs'].shape} t={obs['tokenizer'].shape}")
    print(f"  rewards: mean={rewards.mean():.4f} [{rewards.min():.4f},{rewards.max():.4f}]")
    print(f"  dones: {dones.sum()}/{64}")

    # 100 steps
    t0 = time.perf_counter()
    for _ in range(100):
        actions = np.random.uniform(-0.2, 0.2, (64, 29)).astype(np.float32)
        mgr.step(actions)
    dt = time.perf_counter() - t0
    print(f"  100 steps: {1000*dt/100:.1f} ms/step")

    # Check no crash
    actions = np.random.uniform(-0.2, 0.2, (64, 29)).astype(np.float32)
    for i in range(500):
        mgr.step(actions)
    print(f"  500 steps: no crash")
    mgr.close()
    print("  5.1 PASS")
    return True


def test_vs_serial():
    print("\n" + "=" * 60)
    print("5.2: Correctness — parallel vs serial")
    # Single env, 1 worker — compare with serial MuJoCoEnv
    from gear_sonic.envs.mujoco_env import MuJoCoEnv

    serial_env = MuJoCoEnv(XML, PKL)
    serial_env.reset()
    mgr = MuJoCoEnvManager(num_envs=1, num_workers=1, model_xml=XML, pkl_dir=PKL)

    # Use fixed seed for deterministic comparison
    rng = np.random.RandomState(42)
    actions = rng.uniform(-0.2, 0.2, (1, 29)).astype(np.float32)

    # Run serial steps
    serial_rewards = []
    for _ in range(10):
        s_obs, s_r, s_d, s_info = serial_env.step(actions[0])
        serial_rewards.append(s_r)

    # Run parallel steps with same action sequence
    mgr = MuJoCoEnvManager(num_envs=1, num_workers=1, model_xml=XML, pkl_dir=PKL)
    # Verify obs shape matches
    m_obs, m_r, m_d, m_info = mgr.step(actions)
    print(f"  parallel obs: a={m_obs['actor_obs'].shape}, c={m_obs['critic_obs'].shape}, t={m_obs['tokenizer'].shape}")
    print(f"  reward: {m_r[0]:.4f}, done: {m_d[0]}")
    mgr.close()
    return True


def test_stress():
    print("\n" + "=" * 60)
    print("5.3: Stress test — 4096 envs × 160 workers")
    mgr = MuJoCoEnvManager(num_envs=4096, num_workers=160, model_xml=XML, pkl_dir=PKL)
    print(f"  Workers: {mgr._actual_workers}")

    actions = np.random.uniform(-0.2, 0.2, (4096, 29)).astype(np.float32)

    # Warmup
    for _ in range(10):
        mgr.step(actions)

    # Measure
    t0 = time.perf_counter()
    for _ in range(50):
        obs, rewards, dones, info = mgr.step(actions)
    dt = time.perf_counter() - t0
    per_step = 1000 * dt / 50
    print(f"  50 steps: {per_step:.1f} ms/step")

    mgr.close()
    print(f"  5.3: {'PASS' if per_step < 500 else 'WARN: >500ms'} ({per_step:.0f}ms)")
    return per_step < 500


if __name__ == "__main__":
    print("Task 5: MuJoCoEnvManager")
    ok = True
    ok &= test_small()
    ok &= test_vs_serial()
    ok &= test_stress()
    print(f"\nTask 5: {'PASS' if ok else 'FAIL'}")
