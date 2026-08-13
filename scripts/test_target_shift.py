"""Test: shift target indices by 6 to skip floating base DOF."""
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
from gear_sonic.envs.physx_env_ov import PhysXEnvOv, prepare_usd, TensorBinding
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
stage = ovstage.Stage("shift_test")
usd = prepare_usd(ROBOT)
env = PhysXEnvOv(px, stage, usd, XML, PKL, config=SmokeConfig())
obs = env.reset()

# Read position and check: are the first 6 entries zero/small (base DOF)?
pos_raw = env._b_pos.read().ravel()
print("Position tensor values:")
for i in range(29):
    print(f"  pos[{i:2d}] = {pos_raw[i]:+.4f}")

# Check stiffness: are first 6 entries non-zero?
pattern = "/World/G1/*"
b_stiff = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_STIFFNESS))
stiff = b_stiff.read().ravel()
print(f"\nStiffness values (first 10): {stiff[:10]}")
print(f"Non-zero stiff entries: {np.count_nonzero(stiff > 1.0)} / {len(stiff)}")

# Check DRIVE_MODEL
b_dm = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_DRIVE_MODEL))
dm = b_dm.read()
print(f"\nDRIVE_MODEL shape: {dm.shape}")
for i in range(29):
    vals = dm[0, i, :]
    if np.any(vals > 0):
        print(f"  dm[{i:2d}]: {vals}")

# Test: write targets to indices 6:29 (actuated DOF only, skip base)
# Use DOF2ACT but shift by +6
# Hypothesis: the DRIVES are at tensor indices 6:29 (23 driven DOF)
# while tensor indices 0:5 are the 6 base DOF (no drives)

# Build shifted DOF2ACT accounting for base DOF
# The FIRST driven DOF (first non-base, at tensor index 6) should get
# the data for DOF order joint 0

print("\n=== Test: shifted target mapping (skip base DOF) ===")
ref_time = env._ref_time - env.ctrl_dt
ref_idx = max(0, min(int(ref_time * env._ref_fps), len(env._ref_dof) - 1))
ref_q0 = env._ref_dof[ref_idx].astype(np.float64)

# Current DOF2ACT maps DOF→tensor_index
# If base occupies tensor[0:6], the actual drives are at tensor[6:29]
# For a DOF index d, we need to map to the drive index D = DOF2ACT[d]
# But if D < 6, that's a base DOF (no drive)
# So the actual drive for DOF d is at some DIFFERENT tensor index

# Let's find the correct mapping empirically:
# Perturb each DOF and see which DRIVE index responds

def step_cycle():
    for _ in range(env.decimation):
        env._px.step(env.native_dt)
    env._px.wait_all()

# Hypothesis test: the drives are at tensor indices 6:29 (23 drives)
# And DOF indices 0:22 map to these 23 drives
# The DOF2ACT maps the FIRST 23 DOF to some tensor indices
# Some of which are 0:5 (base, no drives) and some are 6:29 (drives)
#
# Instead of this complexity, let's try a different approach:
# Write targets DIRECTLY in raw tensor order, but map DOF→tensor manually

# First: find out which tensor indices are base DOF (6 entries)
# The base DOF should have stiffness=0 or drive_model=0
# Even if stiffness is set, drive_model should be 0 for base DOF

base_dof_indices = []
driven_dof_indices = []
for i in range(29):
    if np.any(dm[0, i, :] > 0):
        driven_dof_indices.append(i)
    else:
        base_dof_indices.append(i)

print(f"Base DOF indices (no drive): {base_dof_indices}")
print(f"Driven DOF indices (has drive): {driven_dof_indices}")

# Now map: DOF data[joint_k] → tensor[driven_dof_indices[k]]
# But we also need to know which DOF corresponds to which drive
# This requires an additional mapping

# Even simpler test: write targets to ALL 29 tensor indices,
# but set indices 0:5 to their current position (no perturbation)
# and only perturb indices 6:29 for driven DOF

# Actually, let me try the simplest possible test:
# Write ref_q0 to target via DOF2ACT, plus a shift
# If base DOF are at tensor[0:6], then actuated DOF start at tensor[6]
# DOF2ACT maps some actuated DOF to tensor[0:5] (base), which is wrong
# We need a NEW mapping: NEW_DOF2ACT[d] = DOF2ACT[d] - shift_for_base_reorder

# The positions of driven DOF in the tensor are at indices where dm > 0
# These are: [some subset of 0:29]
# Let me just try: map DOF→tensor using ONLY the 23 driven indices

# Create a new target tensor filled with current positions
pos_tensor = env._b_pos.read().ravel().copy()
tgt_tensor = pos_tensor.copy()

# Perturb one driven index and check
for test_act in driven_dof_indices[:5]:
    env._write_joint_positions(ref_q0.astype(np.float32))
    env._write_joint_velocities(np.zeros(env.nu, dtype=np.float32))
    env._px.update_articulations_kinematic()

    pos_t = env._b_pos.read().ravel().copy()
    tgt_t = pos_t.copy()
    tgt_t[test_act] += 0.2  # perturb specific driven DOF

    env._b_tgt.write(tgt_t.astype(np.float32).reshape(1, -1))
    step_cycle()

    q = env._b_pos.read().ravel()
    delta = np.abs(q - pos_t)
    worst = np.argmax(delta)
    print(f"Perturb driven[{test_act:2d}]: worst tensor[{worst:2d}] delta={delta.max():.4f} {'SELF' if worst == test_act else 'OTHER'}")

env.close()
px.release()
print("\n*** DONE ***")
