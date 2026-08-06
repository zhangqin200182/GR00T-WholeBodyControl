#!/usr/bin/env python3
"""PhysX Direct API PPO training — multi-env parallel via PhysXEnvManager.

Adapted from train_physx_ppo_direct.py.
Uses SHM+Barrier pattern (same as MuJoCoEnvManager) for 4096-env training.
"""
import sys, os, time, argparse

_repo = "/root/GR00T-WholeBodyControl"
sys.path.insert(0, _repo)
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs"))
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx"))
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx", "build"))

# CRITICAL: import manager BEFORE torch (fork safety — avoids HCCL inheritance)
from gear_sonic.envs.physx_env_manager import PhysXEnvManager

import numpy as np
import torch
import torch.nn as nn

# ── Config ───────────────────────────────────────────────────────────────
XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
VF_COEF = 1.0
ENT_COEF = 0.01
NUM_EPOCHS = 2
NUM_MINI_BATCHES = 4
LR = 3e-4
MAX_GRAD_NORM = 1.0

# Small defaults for smoke test; scale via --envs/--workers
NUM_ENVS = 16
NUM_WORKERS = 4
NUM_STEPS = 24
NUM_ITERS = 5
BC_WARMUP_ITERS = 2

OBS_DIM = 930
ACT_DIM = 29
CRITIC_OBS_DIM = 1645


class SmokeConfig:
    max_episode_length = 500
    ignore_terminations = True
    alive_bonus = 0.0


# ── Policy ───────────────────────────────────────────────────────────────
class TinyPolicy(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ACT_DIM),
        )
        self.critic = nn.Sequential(
            nn.Linear(CRITIC_OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.ones(ACT_DIM) * -2.0)

    def forward(self, obs_dict):
        x = obs_dict["actor_obs"]
        mean = self.actor(x)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def evaluate_actions(self, actor_obs, actions):
        mean = self.actor(actor_obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1).mean()
        return log_prob, entropy

    def get_value(self, critic_obs):
        return self.critic(critic_obs).squeeze(-1)


# ── GAE (vectorized over envs) ───────────────────────────────────────────
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """rewards, values, dones: (num_steps, num_envs). last_value: (num_envs,)."""
    num_steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(num_envs)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nxt_val = last_value
        else:
            nxt_val = values[t + 1]
        nxt_nonterm = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * nxt_val * nxt_nonterm - values[t]
        gae = delta + gamma * lam * nxt_nonterm * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=NUM_ENVS)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--steps", type=int, default=NUM_STEPS)
    parser.add_argument("--iters", type=int, default=NUM_ITERS)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    print("=" * 60)
    print("PhysX Direct API PPO — Parallel Multi-Env")
    print(f"  {args.envs} envs × {args.workers} workers")
    print(f"  {args.iters} iters × {args.steps} steps, lr={args.lr}")
    print("=" * 60)

    # ── Create PhysX environment manager ──
    print(f"\n[1] Creating PhysXEnvManager ({args.envs} envs, {args.workers} workers)...",
          flush=True)
    t0 = time.time()
    manager = PhysXEnvManager(args.envs, args.workers, XML, PKL, env_config=SmokeConfig())
    print(f"  OK ({time.time()-t0:.1f}s)", flush=True)

    # ── Create policy ──
    print("\n[2] Creating policy...", flush=True)
    policy = TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  OK — {n_params:,} parameters", flush=True)

    # ── Reward normalization state ──
    reward_mean = 0.0
    reward_var = 1.0
    reward_beta = 0.01

    # ── Training loop ──
    print(f"\n[3] Training ({args.iters} iters)...", flush=True)
    print("-" * 60)

    obs = manager.reset_all()
    total_env_steps = 0

    for iteration in range(args.iters):
        t_start = time.time()

        # Storage buffers: (steps, envs, dim)
        actor_obs_buf = torch.zeros(args.steps, args.envs, OBS_DIM)
        critic_obs_buf = torch.zeros(args.steps, args.envs, CRITIC_OBS_DIM)
        action_buf = torch.zeros(args.steps, args.envs, ACT_DIM)
        log_prob_buf = torch.zeros(args.steps, args.envs)
        reward_buf = torch.zeros(args.steps, args.envs)
        done_buf = torch.zeros(args.steps, args.envs)
        value_buf = torch.zeros(args.steps, args.envs)

        # ── Rollout ──
        for step in range(args.steps):
            with torch.no_grad():
                action, log_prob = policy(obs)
                value = policy.get_value(obs["critic_obs"])

            action_np = action.numpy()
            next_obs, rewards, dones, infos = manager.step({"actions": action_np})
            total_env_steps += args.envs

            actor_obs_buf[step] = obs["actor_obs"]
            critic_obs_buf[step] = obs["critic_obs"]
            action_buf[step] = action
            log_prob_buf[step] = log_prob
            reward_buf[step] = rewards
            done_buf[step] = dones.float()
            value_buf[step] = value

            obs = next_obs

        # ── Reward normalization ──
        batch_mean = reward_buf.mean().item()
        batch_var = reward_buf.var().item() if reward_buf.var().item() > 1e-8 else 1.0
        reward_mean = (1 - reward_beta) * reward_mean + reward_beta * batch_mean
        reward_var = (1 - reward_beta) * reward_var + reward_beta * batch_var
        norm_rewards = (reward_buf - reward_mean) / (np.sqrt(reward_var) + 1e-8)
        norm_rewards = torch.clamp(norm_rewards, -10.0, 10.0)

        # ── Diagnostics before training ──
        nan_obs = torch.isnan(actor_obs_buf).any().item()
        nan_act = torch.isnan(action_buf).any().item()
        nan_rew = torch.isnan(reward_buf).any().item()

        # ── Skip training during BC warmup ──
        if iteration < BC_WARMUP_ITERS:
            elapsed = time.time() - t_start
            mean_reward = reward_buf.mean().item()
            mean_value = value_buf.mean().item()
            mean_done = done_buf.mean().item()
            print(f"Iter {iteration+1:3d}: "
                  f"rew={mean_reward:7.2f} val={mean_value:6.2f} "
                  f"done={mean_done:.3f} "
                  f"NaN=obs:{nan_obs},act:{nan_act},rew:{nan_rew} "
                  f"[BC warmup] {elapsed:.1f}s", flush=True)
            if nan_obs or nan_act or nan_rew:
                print("  FATAL: NaN detected during warmup!", flush=True)
                manager.close()
                return
            continue

        # ── Compute GAE ──
        with torch.no_grad():
            last_value = policy.get_value(obs["critic_obs"])
        advantages, returns = compute_gae(
            norm_rewards, value_buf, done_buf, last_value, GAMMA, LAM)
        returns = torch.clamp(returns, -100.0, 100.0)

        # ── Flatten (steps, envs, ...) → (steps * envs, ...) for PPO update ──
        actor_obs_flat = actor_obs_buf.reshape(-1, OBS_DIM)
        critic_obs_flat = critic_obs_buf.reshape(-1, CRITIC_OBS_DIM)
        action_flat = action_buf.reshape(-1, ACT_DIM)
        log_prob_flat = log_prob_buf.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)

        # ── PPO update ──
        total_samples = args.steps * args.envs
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(NUM_EPOCHS):
            indices = torch.randperm(total_samples)
            batch_size = max(1, total_samples // NUM_MINI_BATCHES)
            for mb in range(NUM_MINI_BATCHES):
                idx = indices[mb * batch_size:(mb + 1) * batch_size]
                if len(idx) == 0:
                    continue

                new_log_probs, entropy = policy.evaluate_actions(
                    actor_obs_flat[idx], action_flat[idx])
                new_values = policy.get_value(critic_obs_flat[idx])

                ratio = torch.exp(new_log_probs - log_prob_flat[idx])
                adv = advantages_flat[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss = 0.5 * (returns_flat[idx] - new_values).pow(2).mean()
                loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # ── Logging ──
        elapsed = time.time() - t_start
        mean_reward = reward_buf.mean().item()
        mean_value = value_buf.mean().item()
        mean_done = done_buf.mean().item()
        fps = total_env_steps / elapsed if elapsed > 0 else 0
        print(f"Iter {iteration+1:3d}: "
              f"rew={mean_reward:7.2f} val={mean_value:6.2f} "
              f"done={mean_done:.3f} "
              f"pg={total_pg_loss/n_updates:.4f} "
              f"vf={total_vf_loss/n_updates:.4f} "
              f"H={total_entropy/n_updates:.4f} "
              f"NaN=obs:{nan_obs},act:{nan_act},rew:{nan_rew} "
              f"fps={fps:.0f} "
              f"{elapsed:.1f}s", flush=True)

        if nan_obs or nan_act or nan_rew:
            print("  FATAL: NaN detected! Aborting.", flush=True)
            manager.close()
            return

    # ── Summary ──
    print("-" * 60)
    print(f"\nPhysX Direct API PPO Parallel Smoke Test PASSED")
    print(f"  {args.envs} envs × {args.steps} steps × {args.iters} iters")
    print(f"  Total env steps: {total_env_steps}")
    print(f"  Final reward mean: {reward_mean:.3f}")

    manager.close()


if __name__ == "__main__":
    main()
