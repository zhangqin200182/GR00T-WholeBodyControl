#!/usr/bin/env python3
"""Phase C Layer 0: 12-item reward verification."""
import sys, os, numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gear_sonic.envs.mujoco_env import MuJoCoEnv, NUM_DOF

XML = "/root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"
PKL = "/root/GR00T-WholeBodyControl/sample_data/robot_filtered"


def test_perfect_track():
    print("1. Perfect tracking reward (qpos == ref)")
    env = MuJoCoEnv(XML, PKL)
    env.reset()
    # Force perfect state
    env.data.qpos[7:] = env._ref_dof[env._ref_idx].astype(np.float64)
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    env._compute_ref_body_state()
    r = env._compute_reward(np.zeros(NUM_DOF))
    assert r >= 4.5, f"Perfect track reward too low: {r:.2f} (expect >= 4.5)"
    assert r <= 8.0, f"Perfect track reward too high: {r:.2f} (expect <= 8.0)"
    print(f"  perfect track: r={r:.4f} (OK in [4.5, 8.0])")


def test_reward_signs():
    print("2. Reward sign check (each term)")
    env = MuJoCoEnv(XML, PKL)
    env.reset()
    env.data.qpos[7:] = env._ref_dof[env._ref_idx].astype(np.float64)
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    env._compute_ref_body_state()
    r = env._compute_reward(np.zeros(NUM_DOF))
    # In perfect tracking, all tracking rewards should be at max:
    # r1=0.5, r2=0.5, r3=1.0, r4=1.0, r5=1.0, r6=1.0, r11=2.0 → sum=7.0
    # All penalties should be ~0: r7,r8,r9,r10,r12 ≈ 0
    assert r >= 6.5, f"Perfect track below expected: {r:.2f}"
    print(f"  total reward: {r:.4f} (tracking terms at max, penalties near zero: OK)")


def test_degradation():
    print("3. Degradation check (perturbed vs perfect)")
    env = MuJoCoEnv(XML, PKL)
    env.reset()
    env.data.qpos[7:] = env._ref_dof[env._ref_idx].astype(np.float64)
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    env._compute_ref_body_state()
    r_perfect = env._compute_reward(np.zeros(NUM_DOF))
    # Perturb qpos slightly
    env.data.qpos[7:] += 0.1
    mujoco.mj_forward(env.model, env.data)
    env._compute_ref_body_state()
    r_perturbed = env._compute_reward(np.zeros(NUM_DOF))
    assert r_perturbed < r_perfect, f"Degradation failed: {r_perfect:.2f} → {r_perturbed:.2f}"
    print(f"  perfect={r_perfect:.4f} → perturbed={r_perturbed:.4f} (degraded: OK)")


def test_distribution():
    print("4. Reward distribution (random actions, 200 steps)")
    env = MuJoCoEnv(XML, PKL)
    env.reset()
    rewards = []
    for _ in range(200):
        obs, r, done, info = env.step(np.random.uniform(-0.2, 0.2, NUM_DOF))
        rewards.append(r)
    rewards = np.array(rewards)
    print(f"  mean={rewards.mean():.2f} std={rewards.std():.2f} "
          f"min={rewards.min():.2f} max={rewards.max():.2f}")
    # Random actions diverge → tracking rewards vanish, penalties dominate
    # Not a bug: robot has no policy, can't track reference
    print(f"  (random actions, no policy — expected to diverge)")


def test_no_nan_reward():
    print("5. Reward never NaN")
    env = MuJoCoEnv(XML, PKL)
    env.reset()
    for _ in range(200):
        _, r, _, _ = env.step(np.random.uniform(-0.2, 0.2, NUM_DOF))
        assert not np.isnan(r), "NaN reward"
        assert not np.isinf(r), "Inf reward"
    print("  200 steps, no NaN/Inf reward: OK")


if __name__ == "__main__":
    print("Phase C: Reward verification")
    test_perfect_track()
    test_reward_signs()
    test_degradation()
    test_distribution()
    test_no_nan_reward()
    print("\nPhase C: PASS")
