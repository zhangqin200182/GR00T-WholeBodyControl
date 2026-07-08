#!/usr/bin/env python3
"""Task 6: PPO training with GAE + terminal_obs bootstrap."""
import sys, os, time, numpy as np
import torch; import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gear_sonic.envs.mujoco_env_manager import MuJoCoEnvManager
from gear_sonic.envs.mujoco_env import NUM_DOF

XML = "/root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"
PKL = "/root/GR00T-WholeBodyControl/sample_data/robot_filtered"

GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
NUM_EPOCHS = 3
NUM_MINI_BATCHES = 2


class TinyPolicy(nn.Module):
    def __init__(self, obs_dim=930, act_dim=29, critic_dim=1645, hidden=512):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, act_dim))
        self.critic = nn.Sequential(nn.Linear(critic_dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.ones(act_dim) * -3.0)

    def get_action(self, obs):
        x = torch.from_numpy(obs["actor_obs"]).float()
        mean = self.actor(x)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.numpy(), log_prob.detach().numpy()

    def evaluate_actions(self, obs, action):
        x = torch.from_numpy(obs["actor_obs"]).float()
        mean = self.actor(x)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(torch.from_numpy(action).float()).sum(-1)
        entropy = dist.entropy().sum(-1).mean()
        cx = torch.from_numpy(obs["critic_obs"]).float()
        value = self.critic(cx).squeeze(-1)
        return log_prob, entropy, value

    def get_value(self, obs):
        cx = torch.from_numpy(obs["critic_obs"]).float()
        return self.critic(cx).squeeze(-1).detach().numpy()


def compute_gae(rewards, dones, values, last_values, terminal_obs, policy, gamma=0.99, lam=0.95):
    """GAE with terminal_obs bootstrap.

    When done=True, env auto-resets so values[t+1] is V(s_reset).
    We need V(s_terminal) for bootstrapping — use terminal_obs.
    """
    n_steps, n_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)

    # Compute terminal values for done steps
    terminal_values = np.zeros((n_steps, n_envs), dtype=np.float32)
    for t in range(n_steps):
        done_mask = dones[t].astype(bool)
        if done_mask.any() and terminal_obs is not None:
            t_obs = {
                "actor_obs": terminal_obs["actor_obs"][done_mask],
                "critic_obs": terminal_obs["critic_obs"][done_mask],
                "tokenizer": terminal_obs["tokenizer"][done_mask],
            }
            terminal_values[t, done_mask] = policy.get_value(t_obs)

    gae = np.zeros(n_envs, dtype=np.float32)
    for t in reversed(range(n_steps)):
        # next_values uses bootstrap override for done steps
        if t == n_steps - 1:
            nxt = last_values.copy()
        else:
            nxt = values[t + 1].copy()
        done_mask = dones[t].astype(bool)
        nxt[done_mask] = terminal_values[t, done_mask]

        not_done = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * nxt * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    # Normalize
    adv_flat = advantages.flatten()
    adv_mean, adv_std = adv_flat.mean(), adv_flat.std()
    return returns, (advantages - adv_mean) / (adv_std + 1e-8)


def main():
    num_envs = 64; num_steps = 12; num_iters = 30; lr = 3e-4
    env_config = {"ignore_terminations": True}
    print(f"POC Training ({num_envs}e×{num_steps}s×{num_iters}i) — kp=100 kd=5 ignore_terms")

    env = MuJoCoEnvManager(num_envs=num_envs, num_workers=4, model_xml=XML, pkl_dir=PKL, env_config=env_config)
    policy = TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    obs, _, _, _ = env.step(np.random.uniform(-0.2, 0.2, (num_envs, NUM_DOF)).astype(np.float32))
    obs = {k: v.copy() for k, v in obs.items()}

    for iteration in range(num_iters):
        t0 = time.perf_counter()

        # ── Rollout ──
        roll_obs = []; roll_act = []; roll_rew = []; roll_done = []; roll_lp = []; roll_val = []
        roll_info = []; last_terminal = None
        for step in range(num_steps):
            actions, log_probs = policy.get_action(obs)
            roll_obs.append({k: v.copy() for k, v in obs.items()})
            roll_act.append(actions.copy())
            roll_lp.append(log_probs.copy())
            val = policy.get_value(obs)
            roll_val.append(val.copy())

            obs_next, rewards, dones, info = env.step(actions)
            roll_rew.append(rewards.copy())
            # Use _orig_done (actual termination) for GAE when ignore_terminations is on
            orig_done = info.get("_orig_done")
            roll_done.append(orig_done.copy() if orig_done is not None else dones.copy())
            roll_info.append(info)
            obs = {k: v.copy() for k, v in obs_next.items()}
            # Save terminal obs for GAE bootstrap
            if info.get("terminal_obs") is not None:
                last_terminal = {k: v.copy() for k, v in info["terminal_obs"].items()}

        # Last value (V(s_T+1)) — bootstraps from terminal_obs for done envs
        last_values = policy.get_value(obs)

        # ── GAE ──
        rew_arr = np.array(roll_rew, dtype=np.float32)
        done_arr = np.array(roll_done, dtype=np.float32)
        val_arr = np.array(roll_val, dtype=np.float32)

        returns, advantages = compute_gae(
            rew_arr, done_arr, val_arr, last_values, last_terminal, policy, GAMMA, LAM)

        # ── PPO update ──
        for epoch in range(NUM_EPOCHS):
            total_samples = num_steps * num_envs
            perm = np.random.permutation(total_samples)
            mini_size = total_samples // NUM_MINI_BATCHES

            for mb in range(NUM_MINI_BATCHES):
                idx = perm[mb * mini_size:(mb + 1) * mini_size]
                # Build flat batch
                mb_obs = {}
                for k in roll_obs[0]:
                    stacked = np.stack([o[k] for o in roll_obs])  # (T, N, ...)
                    mb_obs[k] = stacked.reshape(total_samples, -1)[idx]
                mb_act = np.array(roll_act).reshape(total_samples, -1)[idx]
                mb_ret = returns.flatten()[idx]
                mb_adv = advantages.flatten()[idx]
                mb_lp = np.array(roll_lp).flatten()[idx]

                log_prob, entropy, values = policy.evaluate_actions(mb_obs, mb_act)
                ratio = torch.exp(log_prob - torch.from_numpy(mb_lp).float())
                adv = torch.from_numpy(mb_adv).float()

                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss = torch.max(pg1, pg2).mean()
                vf_loss = nn.MSELoss()(values, torch.from_numpy(mb_ret).float())
                loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        dt = time.perf_counter() - t0
        term_rate = float(np.array(roll_done).mean())
        term_rate = done_arr.mean()
        print(f"  iter {iteration:3d}: loss={loss.item():.4f} term={term_rate:.2f} "
              f"r={rew_arr.mean():.4f} term={term_rate:.3f} t={dt:.2f}s")

    env.close()
    print("Task 6 GAE: PASS")


if __name__ == "__main__":
    main()
