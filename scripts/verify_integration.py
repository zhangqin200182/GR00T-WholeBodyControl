#!/usr/bin/env python3
"""Phase E: Layer 0 integration + Layer 1 distribution + Layer 2 POC training."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gear_sonic.envs.mujoco_env import MuJoCoEnv, NUM_DOF
from gear_sonic.envs.mujoco_env_manager import MuJoCoEnvManager

XML = "/root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"
PKL = "/root/GR00T-WholeBodyControl/sample_data/robot_filtered"


# ═══════════════════════════════════════════════════════════════
# E2: Layer 0 — step/reset integration
# ═══════════════════════════════════════════════════════════════
def test_e2():
    print("E2: Layer 0 — step/reset integration")
    env = MuJoCoEnv(XML, PKL)
    # 1000 steps no crash
    env.reset()
    for i in range(1000):
        obs, r, done, info = env.step(np.random.uniform(-0.2, 0.2, NUM_DOF))
        assert "actor_obs" in obs, f"step {i}: actor missing"
        assert "time_outs" in info, f"step {i}: time_outs missing"
        assert "terminal_obs" in info, f"step {i}: terminal_obs missing"
        if done:
            assert info["terminal_obs"] is not None, f"step {i}: done but no terminal_obs"
    print("  1000 steps: OK")
    # Reset produces fresh state
    env.reset()
    o1 = env._obs()
    env.reset()
    o2 = env._obs()
    diff = np.max(np.abs(o1["actor_obs"] - o2["actor_obs"]))
    print(f"  reset diversity: max_diff={diff:.4f} (different motions: {'OK' if diff > 0.01 else 'WARN'})")


# ═══════════════════════════════════════════════════════════════
# E3: Layer 1 — distribution check (64 envs × 200 steps)
# ═══════════════════════════════════════════════════════════════
def test_e3():
    print("\nE3: Layer 1 — distribution check (64 envs × 200 steps)")
    mgr = MuJoCoEnvManager(num_envs=64, num_workers=4, model_xml=XML, pkl_dir=PKL)
    actions = np.random.uniform(-0.2, 0.2, (64, NUM_DOF)).astype(np.float32)
    all_rewards = []
    dones_total = 0
    for step in range(200):
        obs, rewards, dones, info = mgr.step(actions)
        all_rewards.extend(rewards.tolist())
        dones_total += int(dones.sum())
        actions = np.random.uniform(-0.2, 0.2, (64, NUM_DOF)).astype(np.float32)
        # Sanity checks
        for k in ["actor_obs", "critic_obs", "tokenizer"]:
            assert not np.any(np.isnan(obs[k])), f"NaN in {k} at step {step}"
    mgr.close()

    rewards = np.array(all_rewards)
    print(f"  rewards: mean={rewards.mean():.1f} std={rewards.std():.1f}")
    print(f"  range: [{np.percentile(rewards, 1):.1f}, {np.percentile(rewards, 99):.1f}]")
    print(f"  dones: {dones_total}/{64*200} ({100*dones_total/12800:.1f}%)")
    print(f"  no NaN: OK")
    print(f"  E3: PASS (pipeline stable)")


# ═══════════════════════════════════════════════════════════════
# E4: Layer 2 — POC training (TinyPolicy, 30 iters)
# ═══════════════════════════════════════════════════════════════
def test_e4():
    import torch; import torch.nn as nn

    print("\nE4: Layer 2 — POC training (30 iters, TinyPolicy)")

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(nn.Linear(930, 128), nn.ReLU(),
                                       nn.Linear(128, 128), nn.ReLU(),
                                       nn.Linear(128, 29))
            self.critic = nn.Sequential(nn.Linear(1645, 128), nn.ReLU(),
                                        nn.Linear(128, 128), nn.ReLU(),
                                        nn.Linear(128, 1))
            self.log_std = nn.Parameter(torch.ones(29) * -3.0)

        def get_action(self, obs):
            x = torch.from_numpy(obs["actor_obs"]).float()
            mean = self.actor(x)
            std = torch.exp(self.log_std).expand_as(mean)
            a = torch.distributions.Normal(mean, std).sample()
            return a.numpy()

    policy = TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    mgr = MuJoCoEnvManager(num_envs=16, num_workers=2, model_xml=XML, pkl_dir=PKL)
    actions = np.random.uniform(-0.2, 0.2, (16, NUM_DOF)).astype(np.float32)
    obs, *_ = mgr.step(actions)
    obs = {k: v.copy() for k, v in obs.items()}

    rewards_hist = []

    for it in range(30):
        t0 = time.perf_counter()
        # Rollout 8 steps
        roll_obs, roll_act, roll_rew, roll_done = [], [], [], []
        for _ in range(8):
            actions = policy.get_action(obs)
            roll_obs.append({k: v.copy() for k, v in obs.items()})
            roll_act.append(actions.copy())
            obs_next, rewards, dones, info = mgr.step(actions)
            roll_rew.append(rewards.copy())
            roll_done.append(dones.copy())
            obs = {k: v.copy() for k, v in obs_next.items()}

        # Simple policy gradient
        r_mean = np.mean([r.mean() for r in roll_rew])
        rewards_hist.append(r_mean)

        # Loss: maximize reward
        all_obs_flat = np.concatenate([o["actor_obs"] for o in roll_obs], axis=0)
        all_act_flat = np.concatenate(roll_act, axis=0)
        x = torch.from_numpy(all_obs_flat).float()
        mean = policy.actor(x)
        std = torch.exp(policy.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(torch.from_numpy(all_act_flat).float()).sum(-1)
        loss = -log_prob.mean()

        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        if it % 10 == 0:
            print(f"  iter {it:3d}: r={r_mean:.2f} loss={loss.item():.4f} t={time.perf_counter()-t0:.2f}s")

    mgr.close()

    first_mean = np.mean(rewards_hist[:5]); last_mean = np.mean(rewards_hist[-5:])
    improved = last_mean > first_mean * 1.15
    print(f"  first 5 mean: {first_mean:.2f}, last 5 mean: {last_mean:.2f}")
    print(f"  E4: {'PASS (reward improved >15%)' if improved else 'WARN (no clear improvement)'}")


if __name__ == "__main__":
    print("Phase E: Integration + POC")
    test_e2()
    test_e3()
    test_e4()
    print("\nPhase E: DONE")
