"""Test v8 USD with PhysXEnvOv — check if drives now work."""
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
from ovphysx.types import TensorType
from gear_sonic.envs.physx_env_ov import PhysXEnvOv, prepare_usd, DOF2ACT, ACT2DOF

class SmokeConfig:
    max_episode_length = 500
    ignore_terminations = True
    alive_bonus = 0.0

ROBOT = "/root/GR00T-WholeBodyControl/g1_29dof_physx_v8.usda"
XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

ovphysx.PhysX.set_cpu_mode(True)
px = ovphysx.PhysX()
stage = ovstage.Stage("v8_test")
usd = prepare_usd(ROBOT)
env = PhysXEnvOv(px, stage, usd, XML, PKL, config=SmokeConfig())

# Check DRIVE_MODEL
from gear_sonic.envs.physx_env_ov import TensorBinding
pattern = "/World/G1/*"
b_dm = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_DRIVE_MODEL))
dm = b_dm.read()
print(f'DRIVE_MODEL shape: {dm.shape}')
print(f'Non-zero entries: {np.count_nonzero(dm)} / {dm.size}')
print(f'Values:')
for i in range(min(29, dm.shape[1])):
    print(f'  dof[{i:2d}]: {dm[0,i,:]}')

# Now test PD hold
obs = env.reset()
ref_time = env._ref_time - env.ctrl_dt
ref_idx = max(0, min(int(ref_time * env._ref_fps), len(env._ref_dof) - 1))
ref_q0 = env._ref_dof[ref_idx].astype(np.float64)

print(f'\n=== PD hold test with v8 ===')
env._write_joint_positions(ref_q0.astype(np.float32))
env._write_joint_velocities(np.zeros(env.nu, dtype=np.float32))
env._write_joint_targets(ref_q0.astype(np.float64))
env._px.update_articulations_kinematic()

# Step 1 cycle
for _ in range(env.decimation):
    env._px.step(env.native_dt)
env._px.wait_all()
q1 = env._read_joint_positions()
err1 = np.abs(q1 - ref_q0)
print(f'After 1 cycle: max |q - ref| = {err1.max():.6f}')
for j in range(29):
    if err1[j] > 0.01:
        print(f'  Joint {j:2d}: err={err1[j]:.4f}')

# Step 10 more cycles with target re-set
for _ in range(10):
    env._write_joint_targets(ref_q0.astype(np.float64))
    for _ in range(env.decimation):
        env._px.step(env.native_dt)
    env._px.wait_all()
q10 = env._read_joint_positions()
err10 = np.abs(q10 - ref_q0)
print(f'\nAfter 10 more cycles: RMS jerr = {np.sqrt(np.mean((q10 - ref_q0)**2)):.6f}')

# Self-consistent test
print(f'\n=== Self-consistent target test ===')
env._write_joint_positions(ref_q0.astype(np.float32))
env._px.update_articulations_kinematic()
pos_raw = env._b_pos.read()
env._b_tgt.write(pos_raw)

for _ in range(env.decimation):
    env._px.step(env.native_dt)
env._px.wait_all()
q_self = env._read_joint_positions()
err_self = np.abs(q_self - ref_q0)
print(f'Max error: {err_self.max():.6f} (should be ~0 if drives work)')

env.close()
px.release()
print('\n*** DONE ***')
