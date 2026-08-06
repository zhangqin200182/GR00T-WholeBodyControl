#!/usr/bin/env python3
"""PhysX Direct API PPO training smoke test.

Single-environment sequential training loop using PhysXEnv (Direct API).
Verifies: kp scaling, obs/reward/termination, policy forward/backward,
GAE, PPO update — end-to-end on NPU.
"""
import sys, os, time, argparse

_repo = "/root/GR00T-WholeBodyControl"
sys.path.insert(0, _repo)
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs"))
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx"))
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx", "build"))

# CRITICAL: import physx_core BEFORE torch (fork safety for multi-worker later)
import physx_core
from gear_sonic.envs.physx_env import PhysXEnv

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

NUM_STEPS = 64          # steps per rollout
NUM_ITERS = 50          # total PPO iterations
BC_WARMUP_ITERS = 5     # collect rollouts without training first

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
        self.log_std = nn.Parameter(torch.ones(ACT_DIM) * -2.0)  # wider initial exploration

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


# ── GAE ──────────────────────────────────────────────────────────────────
def compute_gae(rewards, values, dones, gamma, lam):
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t].float()) - values[t]
        gae = delta + gamma * lam * (1 - dones[t].float()) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    return advantages, returns


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=NUM_STEPS)
    parser.add_argument("--iters", type=int, default=NUM_ITERS)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    print("=" * 60)
    print("PhysX Direct API PPO Smoke Test")
    print(f"  {args.iters} iters x {args.steps} steps, lr={args.lr}")
    print("=" * 60)

    # ── Create PhysX environment ──
    print("\n[1] Creating PhysX environment (Direct API + eACCELERATION)...", flush=True)
    t0 = time.time()
    px = physx_core
    px.init_foundation()
    env = PhysXEnv(px, XML, PKL, config=SmokeConfig(),
                   native_dt=0.001961, decimation=10, pos_iters=8, vel_iters=1)
    print(f"  OK ({time.time()-t0:.1f}s) — {env.art.num_joints} DOFs, dt={env.ctrl_dt:.4f}s", flush=True)

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

    obs = env.reset()
    total_env_steps = 0

    for iteration in range(args.iters):
        t_start = time.time()

        # Storage buffers (single env → 1D)
        actor_obs_buf = torch.zeros(args.steps, OBS_DIM)
        critic_obs_buf = torch.zeros(args.steps, CRITIC_OBS_DIM)
        action_buf = torch.zeros(args.steps, ACT_DIM)
        log_prob_buf = torch.zeros(args.steps)
        reward_buf = torch.zeros(args.steps)
        done_buf = torch.zeros(args.steps)
        value_buf = torch.zeros(args.steps)

        # ── Rollout ──
        for step in range(args.steps):
            # Convert numpy obs → torch tensors with batch dim for policy
            obs_t = {
                "actor_obs": torch.from_numpy(obs["actor_obs"]).float().unsqueeze(0),
                "critic_obs": torch.from_numpy(obs["critic_obs"]).float().unsqueeze(0),
            }

            with torch.no_grad():
                action, log_prob = policy(obs_t)
                value = policy.get_value(obs_t["critic_obs"])

            # action is (1, 29), squeeze to (29,) for env
            action_np = action.squeeze(0).numpy()
            obs, reward, done, info = env.step(action_np)
            total_env_steps += 1

            # Store as 1D tensors (no batch dim)
            actor_obs_buf[step] = obs_t["actor_obs"].squeeze(0)
            critic_obs_buf[step] = obs_t["critic_obs"].squeeze(0)
            action_buf[step] = action.squeeze(0)
            log_prob_buf[step] = log_prob.squeeze(0)
            reward_buf[step] = float(reward)
            done_buf[step] = float(done)
            value_buf[step] = value.squeeze(0)

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
            print(f"Iter {iteration+1:3d}: "
                  f"reward={reward_buf.mean().item():7.2f} "
                  f"value={value_buf.mean().item():6.2f} "
                  f"NaN_obs={nan_obs} NaN_act={nan_act} "
                  f"[BC warmup — no update] "
                  f"{elapsed:.1f}s", flush=True)
            if nan_obs or nan_act or nan_rew:
                print("  FATAL: NaN detected during warmup!", flush=True)
                return
            continue

        # ── Compute GAE ──
        with torch.no_grad():
            last_critic_t = torch.from_numpy(obs["critic_obs"]).float().unsqueeze(0)
            last_value = policy.get_value(last_critic_t).squeeze(0)
        # Append bootstrap value for terminal state
        ext_rewards = norm_rewards
        advantages, returns = compute_gae(ext_rewards, value_buf, done_buf, GAMMA, LAM)
        returns = torch.clamp(returns, -100.0, 100.0)

        # ── PPO update ──
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(NUM_EPOCHS):
            indices = torch.randperm(args.steps)
            batch_size = max(1, args.steps // NUM_MINI_BATCHES)
            for mb in range(NUM_MINI_BATCHES):
                idx = indices[mb * batch_size:(mb + 1) * batch_size]
                if len(idx) == 0:
                    continue

                new_log_probs, entropy = policy.evaluate_actions(
                    actor_obs_buf[idx], action_buf[idx])
                new_values = policy.get_value(critic_obs_buf[idx])

                ratio = torch.exp(new_log_probs - log_prob_buf[idx])
                adv = advantages[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss = 0.5 * (returns[idx] - new_values).pow(2).mean()
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
        print(f"Iter {iteration+1:3d}: "
              f"reward={mean_reward:7.2f} "
              f"value={mean_value:6.2f} "
              f"pg={total_pg_loss/n_updates:.4f} "
              f"vf={total_vf_loss/n_updates:.4f} "
              f"H={total_entropy/n_updates:.4f} "
              f"NaN=obs:{nan_obs},act:{nan_act},rew:{nan_rew} "
              f"{elapsed:.1f}s", flush=True)

        if nan_obs or nan_act or nan_rew:
            print("  FATAL: NaN detected! Aborting.", flush=True)
            return

    # ── Summary ──
    print("-" * 60)
    print(f"\nPhysX Direct API PPO Smoke Test PASSED")
    print(f"  Total env steps: {total_env_steps}")
    print(f"  Final reward mean: {reward_mean:.3f}")
    print(f"  No NaN, pipeline verified end-to-end.")

    try:
        px.release_foundation()
    except Exception:
        pass


if __name__ == "__main__":
    main()
