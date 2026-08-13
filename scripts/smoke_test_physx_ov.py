#!/usr/bin/env python3
"""Smoke test: PhysXEnvOv with ref PD policy — verify no NaN, valid rewards.

Usage (on NPU server):
    cd /root/GR00T-WholeBodyControl
    python3 scripts/smoke_test_physx_ov.py

Pre-requisites:
    - g1_29dof_physx_v3.usda in current directory
    - /sample_data/robot_filtered with motion PKLs
    - /gear_sonic_deploy/g1/g1_29dof_v17.xml for FK
"""

import os, sys, time

# Ensure gear_sonic is importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_this_dir)
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "gear_sonic", "envs"))
sys.path.insert(0, os.path.join(_repo_root, "gear_sonic", "envs", "physx"))

# LD_LIBRARY_PATH for ovphysx shared libs — MUST be set before importing numpy or ovphysx
for p in [
    "/usr/local/python3.11.15/lib/python3.11/site-packages/ovphysx/lib",
    "/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin",
    "/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin/plugins",
    "/usr/lib/aarch64-linux-gnu",
]:
    e = os.environ.get("LD_LIBRARY_PATH", "")
    if p not in e:
        os.environ["LD_LIBRARY_PATH"] = f"{p}:{e}"

import numpy as np
import ovphysx, ovstage
from physx_env_ov import PhysXEnvOv, prepare_usd

# ── Config ─────────────────────────────────────────────────────────────
ROBOT_USD = os.path.join(_repo_root, "g1_29dof_physx_v3.usda")
MODEL_XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL_DIR = "/sample_data/robot_filtered"
N_EPISODES = 5
MAX_STEPS = 500


def main():
    # Prepare combined USD (robot + PhysicsScene + GroundPlane)
    combined_usd = prepare_usd(ROBOT_USD)
    print(f"Combined USD: {combined_usd}")

    # Create PhysX and Stage
    ovphysx.PhysX.set_cpu_mode(True)
    px = ovphysx.PhysX()
    stage = ovstage.Stage("smoke_test")

    # Create env
    t0 = time.time()
    env = PhysXEnvOv(px, stage, combined_usd, MODEL_XML, PKL_DIR,
                     native_dt=0.001961, decimation=17)
    print(f"Env init: {time.time() - t0:.1f}s")

    survivals = []
    mean_rewards = []
    alphas = []
    nan_count = 0

    t0 = time.time()
    for ep in range(N_EPISODES):
        obs = env.reset()

        ep_rewards = []
        ep_pos_errs = []

        for step in range(MAX_STEPS):
            # Ref PD policy: perfect tracking of reference
            ref_qpos = env.get_current_ref_qpos()
            action = (ref_qpos - env.jm) / env.jh
            action = np.clip(action, -1, 1)

            obs, reward, done, info = env.step(action.astype(np.float64))

            # Check for NaN
            if np.isnan(reward) or np.any(np.isnan(obs["actor_obs"])):
                nan_count += 1
                print(f"  ep {ep} step {step}: NaN detected! reward={reward}")
                break

            ep_rewards.append(reward)

            # Track root pos error for alpha metric
            root_pos, root_quat = env._read_root_pose()
            ref_root_pos = env._ref_root_pos()
            pos_err = np.linalg.norm(root_pos - ref_root_pos)
            ep_pos_errs.append(pos_err)

            if done:
                break

        survivals.append(step + 1)
        mean_rewards.append(np.mean(ep_rewards) if ep_rewards else 0)
        alphas.append(np.mean(ep_pos_errs[:min(20, len(ep_pos_errs))]) if ep_pos_errs else float('nan'))

        print(f"  ep {ep+1}/{N_EPISODES}: survival={survivals[-1]:3d}  "
              f"mean_reward={mean_rewards[-1]:.2f}  alpha={alphas[-1]:.4f}  "
              f"NaN={nan_count}")

    elapsed = time.time() - t0

    # ── Summary ────────────────────────────────────────────────────────
    surv = np.array(survivals)
    alphas_arr = np.array(alphas)
    rewards_arr = np.array(mean_rewards)

    print(f"\n=== Smoke Test Results ({N_EPISODES} episodes, {elapsed:.0f}s) ===")
    print(f"Survival: mean={surv.mean():.1f} std={surv.std():.1f}  "
          f"min={surv.min()} max={surv.max()}")
    print(f"Alpha (first 20 steps root pos err): mean={alphas_arr.mean():.4f}  "
          f"std={alphas_arr.std():.4f}")
    print(f"Mean reward: {rewards_arr.mean():.2f}  std={rewards_arr.std():.2f}")
    print(f"NaN count: {nan_count}")

    # Pass/fail criteria
    if nan_count > 0:
        print("\nFAIL: NaN detected!")
    elif surv.mean() < 50:
        print(f"\nWARNING: Low survival ({surv.mean():.0f} steps avg)")
    else:
        print("\nPASS: No NaN, reasonable survival")

    # Cleanup
    env.close()
    stage.destroy()
    px.release()
    print("Cleanup done.")


if __name__ == "__main__":
    main()
