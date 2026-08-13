"""Test v10 USD (converter-generated) with identity DOF2ACT mapping."""
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

ROBOT = "/root/GR00T-WholeBodyControl/g1_29dof_physx_v10.usda"
XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

print(f"DOF2ACT = {DOF2ACT.tolist()}")

ovphysx.PhysX.set_cpu_mode(True)
px = ovphysx.PhysX()
stage = ovstage.Stage("test_v10")
usd = prepare_usd(ROBOT)
env = PhysXEnvOv(px, stage, usd, XML, PKL, config=SmokeConfig())
obs = env.reset()

def step_cycle():
    for _ in range(env.decimation):
        env._px.step(env.native_dt)
    env._px.wait_all()

# Verify DRIVE_MODEL
pattern = "/World/G1/*"
b_dm = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_DRIVE_MODEL))
dm = b_dm.read()
driven = [i for i in range(29) if np.any(dm[0, i, :] > 0)]
print(f"Driven DOF: {len(driven)}/29  indices={driven}")

# Test 1: Per-DOF self-consistency
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
    step_cycle()

    pos_before = env._read_joint_positions()
    perturbed = pos_before.copy()
    perturbed[dof_i] += 0.2
    env._write_joint_targets(perturbed)
    step_cycle()

    q = env._read_joint_positions()
    delta = np.abs(q - pos_before)
    worst = np.argmax(delta)
    ok = worst == dof_i
    if ok: match_count += 1
    if not ok:
        print(f"  DOF[{dof_i:2d}] MISMATCH: worst=DOF[{worst:2d}] delta={delta.max():.4f}")

print(f"  {match_count}/{env.nu} MATCH")

# Test 2: Self-consistent hold
print("\n=== Test 2: Self-consistent hold ===")
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
big = [(j, float(delta_hold[j])) for j in range(29) if delta_hold[j] > 0.01]
if big:
    for j, d in big[:10]:
        print(f"    DOF[{j:2d}]: delta={d:.4f}")

# Test 3: Ref PD tracking
print("\n=== Test 3: Ref PD tracking ===")
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
print(f"  alpha = {alpha:.6f} rad")
print(f"  vs Isaac (0.002): {alpha/0.002:.1f}x")

print(f"\n  Per-DOF RMS error:")
for j in range(29):
    rms = np.sqrt(np.mean(errors[:, j] ** 2))
    flag = " ***" if rms > 0.05 else (" !!!" if rms < 0.005 else "")
    print(f"    [{j:2d}] rms={rms:.6f}{flag}")

env.close()
px.release()
print("\n*** DONE ***")
