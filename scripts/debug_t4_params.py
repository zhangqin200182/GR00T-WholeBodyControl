"""T4: Test ref PD precision with different solver parameters."""
import sys, os, time, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
import physx_core as px
from physx_env import PhysXEnv

XML_PATH = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL_DIR  = "/sample_data/robot_filtered"

def run_ref_pd(label, n_episodes=20, max_steps=200, **kwargs):
    """Run ref PD tracking and report survival + alpha."""
    px.init_foundation()
    env = PhysXEnv(px, XML_PATH, PKL_DIR, **kwargs)
    survivals = []
    alphas = []

    t0 = time.time()
    for ep in range(n_episodes):
        env.reset()
        ep_drifts = []
        for step in range(max_steps):
            ref_qpos = env.get_current_ref_qpos()
            action = (ref_qpos - env.jm) / env.jh
            obs, reward, done, info = env.step(action.astype(np.float64))

            rp = env.art.get_root_world_pose()[0]
            ref_root = env._ref_root_pos()
            ep_drifts.append(np.linalg.norm(rp - ref_root))

            if done:
                break
        survivals.append(step + 1)
        n_drift = min(20, len(ep_drifts))
        alphas.append(np.mean(ep_drifts[:n_drift]))

    px.release_foundation()
    elapsed = time.time() - t0
    surv = np.array(survivals)
    alpha = np.mean(alphas)
    print(f"  {label}: surv={surv.mean():.1f}±{surv.std():.1f}, α={alpha:.6f} "
          f"(max_surv={surv.max()}, ≤5={np.sum(surv<=5)}, {elapsed:.0f}s)")
    return surv.mean(), alpha

print("=== T4: Solver parameter sweep ===\n")

# Baseline
run_ref_pd("baseline                        ", 20, pos_iters=8, vel_iters=1, native_dt=0.002, decimation=10)

# More solver iterations
run_ref_pd("pos16/vel2                      ", 20, pos_iters=16, vel_iters=2, native_dt=0.002, decimation=10)
run_ref_pd("pos32/vel4                      ", 20, pos_iters=32, vel_iters=4, native_dt=0.002, decimation=10)

# Smaller timestep (same 20ms control)
run_ref_pd("dt=1ms, dec=20                  ", 20, pos_iters=8, vel_iters=1, native_dt=0.001, decimation=20)
run_ref_pd("dt=1ms, dec=20, pos16/vel2      ", 20, pos_iters=16, vel_iters=2, native_dt=0.001, decimation=20)

# Smaller timestep + more iters
run_ref_pd("dt=0.5ms, dec=40                ", 20, pos_iters=8, vel_iters=1, native_dt=0.0005, decimation=40)

print("\nDone.")
