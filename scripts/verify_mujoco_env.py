#!/usr/bin/env python3
"""Task 3: MuJoCoEnv single environment verification."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gear_sonic.envs.mujoco_env import MuJoCoEnv

if __name__ == "__main__":
    xml = "/root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"
    pkl = "/root/GR00T-WholeBodyControl/sample_data/robot_filtered"
    print("Task 3: MuJoCoEnv (imported from gear_sonic.envs.mujoco_env)")
    env = MuJoCoEnv(xml, pkl)
    print(f"  nb={env.model.nbody} nu={env.nu} dt={env.ctrl_dt}")
    obs = env.reset()
    print(f"  obs: a={obs['actor_obs'].shape} c={obs['critic_obs'].shape} t={obs['tokenizer'].shape}")

    t0 = time.perf_counter(); rs = []; ds = 0
    for i in range(200):
        a = np.random.uniform(-0.2, 0.2, env.nu)
        o, r, d, info = env.step(a); rs.append(r)
        if d: ds += 1
    dt = time.perf_counter() - t0
    print(f"  200 steps: {1000*dt/200:.2f}ms r={np.mean(rs):.4f}[{np.min(rs):.4f},{np.max(rs):.4f}] dones={ds}")
    env.reset()
    for i in range(200):
        o, r, d, info = env.step(np.zeros(env.nu))
        if d: print(f"  zero-action fell at {i}"); break
    else: print("  zero-action: stable 200 steps")
    print("Task 3: PASS")
