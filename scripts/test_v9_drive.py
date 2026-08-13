"""Test v9 USD: drives should be enabled."""
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
from gear_sonic.envs.physx_env_ov import PhysXEnvOv, prepare_usd, DOF2ACT, ACT2DOF, TensorBinding

class SmokeConfig:
    max_episode_length = 500
    ignore_terminations = True
    alive_bonus = 0.0

ROBOT = "/root/GR00T-WholeBodyControl/g1_29dof_physx_v9.usda"
XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

ovphysx.PhysX.set_cpu_mode(True)
px = ovphysx.PhysX()
stage = ovstage.Stage("v9_test")
usd = prepare_usd(ROBOT)
env = PhysXEnvOv(px, stage, usd, XML, PKL, config=SmokeConfig())

# Check DRIVE_MODEL
pattern = "/World/G1/*"
b_dm = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_DRIVE_MODEL))
dm = b_dm.read()
nz = np.count_nonzero(dm)
print(f'DRIVE_MODEL: {nz} / {dm.size} non-zero')
if nz > 0:
    print('DRIVES ENABLED!')
else:
    print('DRIVES STILL DISABLED')

# PD hold test regardless
obs = env.reset()
ref_time = env._ref_time - env.ctrl_dt
ref_idx = max(0, min(int(ref_time * env._ref_fps), len(env._ref_dof) - 1))
ref_q0 = env._ref_dof[ref_idx].astype(np.float64)

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
print(f'PD hold after 1 cycle: max err = {err1.max():.6f}')

# Self-consistent test
env._write_joint_positions(ref_q0.astype(np.float32))
env._px.update_articulations_kinematic()
pos_raw = env._b_pos.read()
env._b_tgt.write(pos_raw)

for _ in range(env.decimation):
    env._px.step(env.native_dt)
env._px.wait_all()
q_self = env._read_joint_positions()
err_self = np.abs(q_self - ref_q0)
print(f'Self-consistent: max err = {err_self.max():.6f} ({("PASS" if err_self.max() < 0.01 else "FAIL")})')

env.close()
px.release()
print('DONE')
