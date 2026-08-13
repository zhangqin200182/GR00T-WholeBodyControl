#!/usr/bin/env python3
"""PhysX PPO training smoke test — minimal training loop with GAE.

IMPORTANT: PhysXEnvManagerOv must be imported BEFORE torch.
The manager forks worker processes, and children must inherit
ovphysx C++ runtime without Ascend HCCL (torch) interference.
"""
import sys, os, time

sys.path.insert(0, "/root/GR00T-WholeBodyControl")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx")

# CRITICAL: import manager BEFORE torch
from gear_sonic.envs.physx_env_manager_ov import PhysXEnvManagerOv

import numpy as np
import torch
import torch.nn as nn

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

# --- Hyperparams (tiny for smoke test) ---
GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
VF_COEF = 1.0  # higher weight to train critic faster
ENT_COEF = 0.01
NUM_EPOCHS = 2
NUM_MINI_BATCHES = 2
LR = 3e-4

NUM_ENVS = 2
NUM_STEPS_PER_ENV = 32  # short rollout
TOTAL_STEPS = NUM_ENVS * NUM_STEPS_PER_ENV
NUM_ITERS = 5

OBS_DIM = 930
ACT_DIM = 29
CRITIC_OBS_DIM = 1645


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
        self.log_std = nn.Parameter(torch.ones(ACT_DIM) * -3.0)

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


def compute_gae(rewards, values, dones, gamma, lam):
    """Compute GAE and returns."""
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t].float()) - values[t]
        gae = delta + gamma * lam * (1 - dones[t].float()) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    return advantages, returns


def _prepare_usd(robot_usd_path):
    output_path = f"/tmp/g1_combined_{os.getpid()}.usda"
    with open(robot_usd_path, "r") as f:
        original = f.read()
    world_open = 'def Xform "World"'
    insert_pos = original.index(world_open) + len(world_open)
    brace_pos = original.index("{", insert_pos)
    SCENE = """
    def PhysicsScene "physicsScene"
    {
        float3 gravity = (0, 0, -9.81)
    }
    def Xform "GroundPlane"
    {
        quatf xformOp:orient = (1, 0, 0, 0)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        def Plane "CollisionPlane" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            uniform token axis = "Z"
            uniform token purpose = "guide"
        }
    }
"""
    combined = original[:brace_pos + 1] + SCENE + original[brace_pos + 1:]
    with open(output_path, "w") as f:
        f.write(combined)
    return output_path


class SmokeConfig:
    max_episode_length = 200
    ignore_terminations = True
    alive_bonus = 0.0


def main():
    print("=" * 60)
    print("PhysX PPO Training Smoke Test")
    print("=" * 60)

    # Prepare USD
    print("\nPreparing USD...", flush=True)
    usd_path = _prepare_usd("/root/GR00T-WholeBodyControl/g1_29dof_physx_v3.usda")
    print(f"USD: {usd_path}", flush=True)

    # Create env manager
    print(f"\nCreating PhysXEnvManagerOv ({NUM_ENVS} envs x {NUM_ENVS} workers)...", flush=True)
    manager = PhysXEnvManagerOv(
        num_envs=NUM_ENVS,
        num_workers=NUM_ENVS,  # 1 env per worker
        robot_usd_path=usd_path,
        model_xml=XML,
        pkl_dir=PKL,
        env_config=SmokeConfig(),
    )

    # Create policy
    policy = TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    # Training loop
    print(f"\nStarting training: {NUM_ITERS} iters x {NUM_STEPS_PER_ENV} steps x {NUM_ENVS} envs")
    print("-" * 60)

    # Running reward statistics for normalization (prevents critic explosion)
    reward_mean = 0.0
    reward_var = 1.0
    reward_beta = 0.01  # smoothing factor

    for iteration in range(NUM_ITERS):
        t0 = time.time()

        # Storage
        actor_obs_buf = torch.zeros(NUM_STEPS_PER_ENV, NUM_ENVS, OBS_DIM)
        critic_obs_buf = torch.zeros(NUM_STEPS_PER_ENV, NUM_ENVS, CRITIC_OBS_DIM)
        action_buf = torch.zeros(NUM_STEPS_PER_ENV, NUM_ENVS, ACT_DIM)
        log_prob_buf = torch.zeros(NUM_STEPS_PER_ENV, NUM_ENVS)
        reward_buf = torch.zeros(NUM_STEPS_PER_ENV, NUM_ENVS)
        done_buf = torch.zeros(NUM_STEPS_PER_ENV, NUM_ENVS)
        value_buf = torch.zeros(NUM_STEPS_PER_ENV, NUM_ENVS)

        # Get initial observation
        obs = manager.reset_all()

        for step in range(NUM_STEPS_PER_ENV):
            # Forward policy
            with torch.no_grad():
                actor_obs = obs["actor_obs"]
                critic_obs = obs["critic_obs"]
                actions, log_probs = policy(obs)
                values = policy.get_value(critic_obs)

            # Step environments
            obs_dict, rewards, dones, infos = manager.step({
                "actions": actions.numpy()
            })

            # Store
            actor_obs_buf[step] = actor_obs
            critic_obs_buf[step] = critic_obs
            action_buf[step] = actions
            log_prob_buf[step] = log_probs
            reward_buf[step] = rewards
            done_buf[step] = dones.float()
            value_buf[step] = values

            obs = obs_dict

        # Compute GAE with normalized rewards
        raw_rewards = reward_buf.clone()
        # Update running stats
        batch_mean = raw_rewards.mean().item()
        batch_var = raw_rewards.var().item()
        reward_mean = (1 - reward_beta) * reward_mean + reward_beta * batch_mean
        reward_var = (1 - reward_beta) * reward_var + reward_beta * batch_var
        # Normalize rewards to ~N(0,1) scale
        norm_rewards = (raw_rewards - reward_mean) / (np.sqrt(reward_var) + 1e-8)
        norm_rewards = torch.clamp(norm_rewards, -10.0, 10.0)

        # Compute GAE with normalized rewards
        with torch.no_grad():
            last_values = policy.get_value(obs["critic_obs"])
        advantages, returns = compute_gae(norm_rewards, value_buf, done_buf, GAMMA, LAM)
        # Clip returns to prevent critic explosion
        returns = torch.clamp(returns, -100.0, 100.0)

        # Flatten for PPO update
        actor_obs_flat = actor_obs_buf.view(-1, OBS_DIM)
        critic_obs_flat = critic_obs_buf.view(-1, CRITIC_OBS_DIM)
        actions_flat = action_buf.view(-1, ACT_DIM)
        log_probs_flat = log_prob_buf.view(-1)
        advantages_flat = advantages.view(-1)
        returns_flat = returns.view(-1)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        # PPO update
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(NUM_EPOCHS):
            indices = torch.randperm(TOTAL_STEPS)
            batch_size = TOTAL_STEPS // NUM_MINI_BATCHES
            for mb in range(NUM_MINI_BATCHES):
                idx = indices[mb * batch_size:(mb + 1) * batch_size]

                new_log_probs, entropy = policy.evaluate_actions(
                    actor_obs_flat[idx], actions_flat[idx])
                new_values = policy.get_value(critic_obs_flat[idx])

                # Ratio
                ratio = torch.exp(new_log_probs - log_probs_flat[idx])

                # Clipped loss
                adv = advantages_flat[idx]
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                vf_loss = 0.5 * (returns_flat[idx] - new_values).pow(2).mean()

                # Total loss
                loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # Logging
        elapsed = time.time() - t0
        mean_reward = reward_buf.mean().item()
        mean_value = value_buf.mean().item()
        nan_actions = torch.isnan(action_buf).any().item()

        print(f"Iter {iteration+1:2d}: "
              f"reward={mean_reward:7.2f} "
              f"value={mean_value:6.2f} "
              f"pg_loss={total_pg_loss/n_updates:.4f} "
              f"vf_loss={total_vf_loss/n_updates:.4f} "
              f"entropy={total_entropy/n_updates:.4f} "
              f"NaN={nan_actions} "
              f"time={elapsed:.1f}s",
              flush=True)

        if nan_actions:
            print("  WARNING: NaN actions detected!", flush=True)

    manager.close()
    print(f"\n*** PhysX PPO SMOKE TEST COMPLETED ***")
    print(f"Verified: env manager + policy forward + GAE + PPO update on ovphysx backend")


if __name__ == "__main__":
    main()
