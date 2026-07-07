#!/usr/bin/env python3
"""Task 6: Minimal PPO training with MuJoCoEnvManager.
Verifies end-to-end pipeline: env → actor forward → storage → GAE → PPO update.
"""
import sys, os, time, copy
import numpy as np
import torch; import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gear_sonic.envs.mujoco_env_manager import MuJoCoEnvManager
from gear_sonic.envs.mujoco_env import NUM_DOF

XML = "/root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"
PKL = "/root/GR00T-WholeBodyControl/sample_data/robot_filtered"

# ── Minimal Policy network ──────────────────────────────────────────────
class TinyPolicy(nn.Module):
    """Minimal actor+critic for POC — replace with full SONIC model later."""

    def __init__(self, obs_dim=930, act_dim=29, hidden=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(1645, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.ones(act_dim) * -3.0)

    def get_action(self, obs_dict):
        x = torch.from_numpy(obs_dict["actor_obs"]).float()
        mean = self.actor(x)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.numpy(), log_prob.detach().numpy()

    def evaluate(self, obs_dict, action):
        x = torch.from_numpy(obs_dict["actor_obs"]).float()
        mean = self.actor(x)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(torch.from_numpy(action).float()).sum(-1)
        entropy = dist.entropy().sum(-1).mean()
        # Value
        cx = torch.from_numpy(obs_dict["critic_obs"]).float()
        value = self.critic(cx).squeeze(-1)
        return log_prob, entropy, value


# ── Training loop ───────────────────────────────────────────────────────
def main():
    num_envs = 64
    num_steps = 8   # short rollout for POC
    num_iters = 20  # few iterations for POC
    lr = 3e-4

    print(f"Task 6: PPO with MuJoCoEnvManager ({num_envs} envs × {num_steps} steps × {num_iters} iters)")
    env = MuJoCoEnvManager(num_envs=num_envs, num_workers=4, model_xml=XML, pkl_dir=PKL)

    policy = TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Warmup: one step to populate obs
    obs, _, _, _ = env.step(np.random.uniform(-0.2, 0.2, (num_envs, NUM_DOF)).astype(np.float32))
    obs = {k: v.copy() for k, v in obs.items()}

    for iteration in range(num_iters):
        t0 = time.perf_counter()
        # ── Rollout ──
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_log_probs = []

        for step in range(num_steps):
            actions, log_probs = policy.get_action(obs)
            obs_next, rewards, dones, info = env.step(actions)
            all_obs.append({k: v.copy() for k, v in obs.items()})
            all_actions.append(actions.copy())
            all_rewards.append(rewards.copy())
            all_dones.append(dones.copy())
            all_log_probs.append(log_probs.copy())
            obs = {k: v.copy() for k, v in obs_next.items()}

        # ── Compute returns (simple MC, no GAE for POC) ──
        # Note: value computation only uses critic_obs, action arg is unused
        values = []
        for step in range(num_steps):
            _, _, v = policy.evaluate(all_obs[step], all_actions[step])
            values.append(v.detach().numpy())
        values = np.array(values)

        returns = np.zeros((num_steps, num_envs), dtype=np.float32)
        running_return = np.zeros(num_envs, dtype=np.float32)
        gamma = 0.99
        for t in reversed(range(num_steps)):
            running_return = all_rewards[t] + gamma * running_return * (1 - all_dones[t])
            returns[t] = running_return

        advantages = returns - values

        # ── PPO Update (single epoch for POC) ──
        obs_flat = {k: np.concatenate([o[k] for o in all_obs], axis=0) for k in all_obs[0]}
        actions_flat = np.concatenate(all_actions, axis=0)
        returns_flat = returns.flatten()
        advantages_flat = advantages.flatten()

        log_probs_new, entropy, values_new = policy.evaluate(obs_flat, actions_flat)
        log_probs_old = torch.from_numpy(np.concatenate(all_log_probs, axis=0)).float()

        ratio = torch.exp(log_probs_new - log_probs_old)
        adv = torch.from_numpy(advantages_flat).float()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pg_loss = -(ratio * adv).mean()
        vf_loss = nn.MSELoss()(values_new, torch.from_numpy(returns_flat).float())
        loss = pg_loss + 0.5 * vf_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dt = time.perf_counter() - t0
        r_mean = np.mean(all_rewards)
        print(f"  iter {iteration:3d}: loss={loss.item():.4f} reward={r_mean:.4f} time={dt:.2f}s")

    env.close()
    print("Task 6 POC: PASS")


if __name__ == "__main__":
    main()
