"""Test v9 with DOF2ACT-remapped targets. Drives are enabled now."""
import sys, os, numpy as np
for p in ['/usr/local/python3.11.15/lib/python3.11/site-packages/ovphysx/lib',
          '/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin',
          '/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin/plugins',
          '/usr/lib/aarch64-linux-gnu']:
    e = os.environ.get('LD_LIBRARY_PATH', '')
    if p not in e: os.environ['LD_LIBRARY_PATH'] = f'{p}:{e}'

sys.path.insert(0, "/root/GR00T-WholeBodyControl")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx")

import ovphysx, ovstage
from gear_sonic.envs.physx_env_ov import PhysXEnvOv, prepare_usd, DOF2ACT, ACT2DOF

class SmokeConfig:
    max_episode_length = 500
    ignore_terminations = True
    alive_bonus = 0.0

ROBOT = "/root/GR00T-WholeBodyControl/g1_29dof_physx_v9.usda"
XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

ovphysx.PhysX.set_cpu_mode(True)
px = ovphysx.PhysX()
stage = ovstage.Stage("v9_remap")
usd = prepare_usd(ROBOT)
env = PhysXEnvOv(px, stage, usd, XML, PKL, config=SmokeConfig())

obs = env.reset()

ref_time = env._ref_time - env.ctrl_dt
ref_idx = max(0, min(int(ref_time * env._ref_fps), len(env._ref_dof) - 1))
ref_q0 = env._ref_dof[ref_idx].astype(np.float64)

# Test 1: Self-consistent targets with DOF2ACT remapping
print("=== Test 1: Self-consistent targets (DOF2ACT remapped) ===")
env._write_joint_positions(ref_q0.astype(np.float32))
env._write_joint_velocities(np.zeros(env.nu, dtype=np.float32))
env._px.update_articulations_kinematic()

# Read back in DOF order
pos_dof = env._read_joint_positions()
# Write same values as targets (same remapping)
env._write_joint_targets(pos_dof)

for _ in range(env.decimation):
    env._px.step(env.native_dt)
env._px.wait_all()

q = env._read_joint_positions()
delta = np.abs(q - pos_dof)
print(f"Max delta after self-consistent: {delta.max():.6f}")
for j in range(29):
    if delta[j] > 0.01:
        print(f"  Joint {j:2d}: q={q[j]:+.4f} target={pos_dof[j]:+.4f} delta={delta[j]:.4f}")

# Test 2: Ref PD tracking with CORRECT DOF2ACT remapping
print("\n=== Test 2: Ref PD tracking (correct DOF2ACT remapping) ===")
# Reset again
obs = env.reset()
ref_time = env._ref_time - env.ctrl_dt
ref_idx = max(0, min(int(ref_time * env._ref_fps), len(env._ref_dof) - 1))
ref_q0 = env._ref_dof[ref_idx].astype(np.float64)

errors = []
n_frames = 50

for i in range(n_frames):
    # Get ref target for next frame
    next_time = env._ref_time + env.ctrl_dt
    next_idx = min(int(next_time * env._ref_fps), len(env._ref_dof) - 1)
    ref_target = env._ref_dof[next_idx].astype(np.float64)

    # Write targets via DOF2ACT (correct remapping)
    env._write_joint_targets(ref_target)

    # Step
    for _ in range(env.decimation):
        env._px.step(env.native_dt)
    env._px.wait_all()

    # Advance time
    env._advance_motion_time()

    # Read actual and compute error
    q_actual = env._read_joint_positions()
    errors.append(q_actual - ref_target)

errors = np.array(errors)
alpha = float(np.sqrt(np.mean(errors ** 2)))
print(f"alpha = {alpha:.6f} rad")
print(f"alpha vs Isaac target (0.002): {alpha/0.002:.1f}x")

if alpha < 0.002:
    print("EXCELLENT: matches Isaac target!")
elif alpha < 0.01:
    print("GOOD: within 5x Isaac target")
elif alpha < 0.02:
    print("OK: near bare-SDK plateau")
else:
    print(f"BELOW target: need further investigation")

# Per-joint breakdown
print(f"\nPer-joint RMS error:")
for j in range(29):
    rms = np.sqrt(np.mean(errors[:, j] ** 2))
    bar = "#" * min(80, int(rms * 2000))
    if rms > 0.01:
        bar = bar + " ***" if rms > 0.05 else bar
    print(f"  [{j:2d}] rms={rms:.6f} {bar}")

env.close()
px.release()
print("\n*** DONE ***")
