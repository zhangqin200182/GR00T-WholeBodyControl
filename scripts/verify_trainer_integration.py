#!/usr/bin/env python3
"""Task 7 prep: verify MuJoCoEnvManager ↔ ppo_trainer interface alignment."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gear_sonic.envs.mujoco_env_manager import MuJoCoEnvManager
from gear_sonic.envs.mujoco_env import NUM_DOF

XML = "/root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"
PKL = "/root/GR00T-WholeBodyControl/sample_data/robot_filtered"


def test_policy_state_dict():
    """Verify env.step(policy_state_dict) — the exact call ppo_trainer makes."""
    print("1. policy_state_dict interface")
    mgr = MuJoCoEnvManager(num_envs=4, num_workers=2, model_xml=XML, pkl_dir=PKL)

    # ppo_trainer calls: policy_state_dict = policy_step(obs) → env.step(policy_state_dict)
    policy_state_dict = {
        "actions": np.random.uniform(-0.2, 0.2, (4, NUM_DOF)).astype(np.float32),
        "action_mean": np.zeros((4, NUM_DOF), dtype=np.float32),
        "log_prob": np.zeros(4, dtype=np.float32),
    }
    obs, rewards, dones, infos = mgr.step(policy_state_dict)

    assert obs["actor_obs"].shape == (4, 930), f"actor shape {obs['actor_obs'].shape}"
    assert obs["critic_obs"].shape == (4, 1645)
    assert obs["tokenizer"].shape == (4, 1761)
    assert "time_outs" in infos, "time_outs missing from infos"
    assert "terminal_obs" in infos, "terminal_obs missing from infos"
    print(f"  obs keys: {sorted(obs.keys())}")
    print(f"  infos keys: {sorted(infos.keys())}")
    print(f"  rewards: {rewards.shape}, dones: {dones.sum()}/{len(dones)}")
    mgr.close()
    print("  PASS")


def test_ppo_trainer_loop():
    """Simulate a mini rollout exactly like ppo_trainer does."""
    print("\n2. Trainer-compatible rollout loop")
    mgr = MuJoCoEnvManager(num_envs=16, num_workers=2, model_xml=XML, pkl_dir=PKL)

    # Initial step to get obs
    dummy = {"actions": np.random.uniform(-0.2, 0.2, (16, NUM_DOF)).astype(np.float32)}
    obs_dict, _, dones, _ = mgr.step(dummy)
    print(f"  init obs: {[f'{k}:{v.shape}' for k,v in obs_dict.items()]}")

    num_steps = 4
    for step in range(num_steps):
        # Simulate policy_step (just random actions for test)
        policy_state_dict = {
            "actions": np.random.uniform(-0.2, 0.2, (16, NUM_DOF)).astype(np.float32),
        }
        obs_dict, rewards, dones, infos = mgr.step(policy_state_dict)

        # Verify each step
        assert not np.any(np.isnan(obs_dict["actor_obs"])), f"NaN in actor_obs step {step}"
        assert "time_outs" in infos
        assert rewards.shape == (16,)
        assert dones.shape == (16,)
        if step == num_steps - 1:
            print(f"  step {step}: r={rewards.mean():.4f}, dones={dones.sum()}, "
                  f"to={infos['time_outs'].sum()}, terminal={infos['terminal_obs'] is not None}")

    mgr.close()
    print("  PASS")


if __name__ == "__main__":
    test_policy_state_dict()
    test_ppo_trainer_loop()
    print("\nTask 7 prep: PASS — interface ready for ppo_trainer")
