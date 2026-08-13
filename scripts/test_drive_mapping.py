"""Find correct DOF2ACT by probing driven tensor indices."""
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

ovphysx.PhysX.set_cpu_mode(True)
px = ovphysx.PhysX()
stage = ovstage.Stage("drive_map")
usd = prepare_usd(ROBOT)
env = PhysXEnvOv(px, stage, usd, XML, PKL, config=SmokeConfig())
obs = env.reset()

pattern = "/World/G1/*"

# 1. Check DRIVE_MODEL: which tensor indices have active drives?
b_dm = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_DRIVE_MODEL))
dm = b_dm.read()
print("DRIVE_MODEL shape:", dm.shape)
driven = []
for i in range(dm.shape[1]):
    if np.any(dm[0, i, :] > 0):
        driven.append(i)
print(f"Driven tensor indices ({len(driven)}): {driven}")
print(f"Non-driven indices ({29 - len(driven)}): {[i for i in range(29) if i not in driven]}")

# 2. Check stiffness for driven vs non-driven
b_stiff = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_STIFFNESS))
stiff = b_stiff.read().ravel()
print(f"\nStiffness: min={stiff.min():.1f} max={stiff.max():.1f}")

print(f"\nDetailed per-index:")
for i in range(29):
    driven_flag = "DRIVEN" if i in driven else "BASE"
    print(f"  tensor[{i:2d}]: stiff={stiff[i]:8.1f}  dm={dm[0,i,:]}  {driven_flag}")

# 3. Try writing targets directly to raw tensor indices
# Start from zero-ish position for safety
print("\n=== Direct raw-index target write test ===")

def step_cycle():
    for _ in range(env.decimation):
        env._px.step(env.native_dt)
    env._px.wait_all()

# First, soft-land: write current positions as targets to all 29 indices
pos_raw = env._b_pos.read().ravel().copy()
env._b_tgt.write(pos_raw.astype(np.float32).reshape(1, -1))
# Step once to sync drives to current pos
step_cycle()
print("Soft-landed: drives synced to current positions")

# Now perturb ONE driven index at a time
# Only test a few driven indices to avoid explosion
for test_idx in driven[:3]:  # first 3 driven
    pos_start = env._b_pos.read().ravel().copy()
    tgt = pos_start.copy()
    tgt[test_idx] += 0.1  # small perturbation
    env._b_tgt.write(tgt.astype(np.float32).reshape(1, -1))
    step_cycle()
    pos_end = env._b_pos.read().ravel()
    delta = np.abs(pos_end - pos_start)
    # Sort deltas
    sorted_idx = np.argsort(-delta)
    print(f"\nPerturbing tensor[{test_idx:2d}] by +0.1:")
    for j in sorted_idx[:8]:
        if delta[j] > 1e-6:
            marker = "← SELF" if j == test_idx else ""
            print(f"  tensor[{j:2d}] delta={delta[j]:.6f} {marker}")
    if delta[test_idx] < 1e-6:
        print(f"  *** NO delta at perturbed index! Max elsewhere: [{sorted_idx[0]}]={delta[sorted_idx[0]]:.6f}")

# 4. Map: which DOF (PKL order) does each tensor index correspond to?
# Method: read joint positions in DOF order, perturb one tensor index,
# and see which DOF moved
print("\n=== Tensor-index → DOF mapping ===")
idx_to_dof = {}
for test_idx in driven[:6]:
    # Reset to ref pose
    ref_time = env._ref_time - env.ctrl_dt
    ref_idx = max(0, min(int(ref_time * env._ref_fps), len(env._ref_dof) - 1))
    ref_q0 = env._ref_dof[ref_idx].astype(np.float64)

    env._write_joint_positions(ref_q0.astype(np.float32))
    env._write_joint_velocities(np.zeros(env.nu, dtype=np.float32))
    env._px.update_articulations_kinematic()

    # Write raw pos as targets
    pos_raw = env._b_pos.read().ravel().copy()
    env._b_tgt.write(pos_raw.astype(np.float32).reshape(1, -1))
    # Wait for settle
    for _ in range(3):
        step_cycle()

    pos_before = env._read_joint_positions()
    pos_raw = env._b_pos.read().ravel().copy()
    tgt = pos_raw.copy()
    tgt[test_idx] += 0.15
    env._b_tgt.write(tgt.astype(np.float32).reshape(1, -1))
    step_cycle()
    pos_after = env._read_joint_positions()

    delta_dof = np.abs(pos_after - pos_before)
    worst_dof = np.argmax(delta_dof)
    print(f"  tensor[{test_idx:2d}] → DOF[{worst_dof:2d}] delta={delta_dof.max():.4f}")

    if delta_dof.max() > 0.001:
        idx_to_dof[test_idx] = worst_dof

env.close()
px.release()
print(f"\nMapping so far: {idx_to_dof}")
print("\n*** DONE ***")
