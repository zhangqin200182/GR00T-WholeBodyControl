#!/usr/bin/env python3
"""T6: ref PD precision validation — PhysX vs MuJoCo baseline."""
import sys, os, time, numpy as np

sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
import physx_core as px
import physx_env
PhysXEnv = physx_env.PhysXEnv

XML_PATH = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL_DIR  = "/sample_data/robot_filtered"
N_EPISODES = 100
MAX_STEPS  = 500

def run_ref_pd(n_episodes=N_EPISODES):
    px.init_foundation()
    env = PhysXEnv(px, XML_PATH, PKL_DIR)

    survivals = []
    alphas = []  # per-step drift (first 20 steps)
    rewards = []

    t0 = time.time()
    for ep in range(n_episodes):
        env.reset()
        ep_rewards = []
        ep_drifts = []
        for step in range(MAX_STEPS):
            ref_qpos = env.get_current_ref_qpos()
            action = (ref_qpos - env.jm) / env.jh
            obs, reward, done, info = env.step(action.astype(np.float64))

            # Per-step drift: root pos error
            root_pos = env.art.get_root_world_pose()[0]
            ref_root = env._ref_root_pos()
            drift = np.linalg.norm(root_pos - ref_root)
            ep_drifts.append(drift)
            ep_rewards.append(reward)

            if done:
                break

        survivals.append(step + 1)
        # α: mean drift over first 20 steps (or fewer if terminated early)
        n_drift = min(20, len(ep_drifts))
        alphas.append(np.mean(ep_drifts[:n_drift]))
        rewards.append(np.mean(ep_rewards))

        if (ep + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  ep {ep+1}/{n_episodes}: survival={survivals[-1]}, "
                  f"α={alphas[-1]:.4f}, reward={rewards[-1]:.2f}, "
                  f"{elapsed:.0f}s elapsed")

    px.release_foundation()
    elapsed = time.time() - t0

    surv = np.array(survivals)
    alphas_arr = np.array(alphas)
    rewards_arr = np.array(rewards)

    print(f"\n=== ref PD Results ({n_episodes} episodes, {elapsed:.0f}s) ===")
    print(f"Survival: mean={surv.mean():.1f} std={surv.std():.1f} "
          f"min={surv.min()} max={surv.max()}")
    print(f"Survival distribution: ≤5={np.sum(surv<=5)} "
          f"≤20={np.sum(surv<=20)} ≤50={np.sum(surv<=50)} "
          f"≤100={np.sum(surv<=100)} >100={np.sum(surv>100)}")
    print(f"α (per-step drift, first 20 steps): mean={alphas_arr.mean():.4f} "
          f"std={alphas_arr.std():.4f} min={alphas_arr.min():.4f}")
    print(f"Mean reward per step: {rewards_arr.mean():.2f}")

    # Acceptance criteria
    alpha_pass = alphas_arr.mean() < 0.002
    survival_pass = surv.mean() > 80
    print(f"\nα < 0.002: {'PASS' if alpha_pass else 'FAIL'} "
          f"(mean α = {alphas_arr.mean():.4f})")
    print(f"survival > 80: {'PASS' if survival_pass else 'FAIL'} "
          f"(mean survival = {surv.mean():.1f})")

    return surv, alphas_arr, rewards_arr


if __name__ == "__main__":
    run_ref_pd(20)  # quick test first; increase to 100 for full validation
