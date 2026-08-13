"""Verify corrected DOF2ACT mapping with per-DOF perturbation and ref PD tracking."""
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
from gear_sonic.envs.physx_env_ov import PhysXEnvOv, prepare_usd, TensorBinding, DOF2ACT, ACT2DOF
from ovphysx.types import TensorType

class SmokeConfig:
    max_episode_length = 500
    ignore_terminations = True
    alive_bonus = 0.0

ROBOT = "/root/GR00T-WholeBodyControl/g1_29dof_physx_v9.usda"
XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

print(f"DOF2ACT = {DOF2ACT.tolist()}")
print(f"ACT2DOF = {ACT2DOF.tolist()}")

ovphysx.PhysX.set_cpu_mode(True)
px = ovphysx.PhysX()
stage = ovstage.Stage("verify_map")
usd = prepare_usd(ROBOT)
env = PhysXEnvOv(px, stage, usd, XML, PKL, config=SmokeConfig())
obs = env.reset()

def step_cycle():
    for _ in range(env.decimation):
        env._px.step(env.native_dt)
    env._px.wait_all()

# ── Test 1: Per-DOF self-consistency ──
print("\n=== Test 1: Per-DOF perturbation self-consistency ===")
ref_time = env._ref_time - env.ctrl_dt
ref_idx = max(0, min(int(ref_time * env._ref_fps), len(env._ref_dof) - 1))
ref_q0 = env._ref_dof[ref_idx].astype(np.float64)

match_count = 0
for dof_i in range(env.nu):
    env._write_joint_positions(ref_q0.astype(np.float32))
    env._write_joint_velocities(np.zeros(env.nu, dtype=np.float32))
    env._px.update_articulations_kinematic()

    pos_dof = env._read_joint_positions()
    env._write_joint_targets(pos_dof)
    step_cycle()  # settle

    pos_before = env._read_joint_positions()
    perturbed = pos_before.copy()
    perturbed[dof_i] += 0.2
    env._write_joint_targets(perturbed)
    step_cycle()

    q = env._read_joint_positions()
    delta = np.abs(q - pos_before)
    worst = np.argmax(delta)
    ok = worst == dof_i
    if ok:
        match_count += 1
    status = "OK" if ok else f"MISMATCH (worst={worst})"
    if not ok or dof_i < 3 or dof_i > 25:
        print(f"  DOF[{dof_i:2d}] → tensor[{DOF2ACT[dof_i]:2d}] delta={delta.max():.4f} {status}")

print(f"\n  {match_count}/{env.nu} MATCH (expected 29/29)")

# ── Test 2: Self-consistent target hold ──
print("\n=== Test 2: Self-consistent hold (should be near-zero) ===")
obs = env.reset()
ref_time = env._ref_time - env.ctrl_dt
ref_idx = max(0, min(int(ref_time * env._ref_fps), len(env._ref_dof) - 1))
ref_q0b = env._ref_dof[ref_idx].astype(np.float64)

env._write_joint_positions(ref_q0b.astype(np.float32))
env._write_joint_velocities(np.zeros(env.nu, dtype=np.float32))
env._write_joint_targets(ref_q0b.astype(np.float32))
env._px.update_articulations_kinematic()

step_cycle()

pos_after = env._read_joint_positions()
delta_hold = np.abs(pos_after - ref_q0b)
print(f"  Max hold error: {delta_hold.max():.6f} rad")
for j in range(29):
    if delta_hold[j] > 0.005:
        print(f"    DOF[{j:2d}]: delta={delta_hold[j]:.4f}")

# ── Test 3: Ref PD tracking ──
print("\n=== Test 3: Ref PD tracking (aiming for alpha < 0.01) ===")
obs = env.reset()
n_frames = 80
errors = []

for i in range(n_frames):
    next_time = env._ref_time + env.ctrl_dt
    next_idx = min(int(next_time * env._ref_fps), len(env._ref_dof) - 1)
    ref_target = env._ref_dof[next_idx].astype(np.float64)
    env._write_joint_targets(ref_target)
    step_cycle()
    env._advance_motion_time()
    q_actual = env._read_joint_positions()
    errors.append(q_actual - ref_target)

errors = np.array(errors)
alpha = float(np.sqrt(np.mean(errors ** 2)))
print(f"\n  alpha = {alpha:.6f} rad")
print(f"  vs Isaac (0.002): {alpha/0.002:.1f}x")

if alpha < 0.005:
    print("  EXCELLENT!")
elif alpha < 0.01:
    print("  GOOD!")
elif alpha < 0.05:
    print("  OK")
else:
    print("  BELOW target")

print(f"\n  Per-DOF RMS error:")
for j in range(29):
    rms = np.sqrt(np.mean(errors[:, j] ** 2))
    bar = "#" * max(1, int(rms / 0.001))
    if rms > 0.01:
        bar = bar + " ***"
    print(f"    [{j:2d}] rms={rms:.6f} {bar}")

env.close()
px.release()
print("\n*** DONE ***")
