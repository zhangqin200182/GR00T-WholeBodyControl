"""Definitive mapping: perturb ONE DOF's target, see which joint moves."""
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
stage = ovstage.Stage("map_target")
usd = prepare_usd(ROBOT)
env = PhysXEnvOv(px, stage, usd, XML, PKL, config=SmokeConfig())

obs = env.reset()

ref_time = env._ref_time - env.ctrl_dt
ref_idx = max(0, min(int(ref_time * env._ref_fps), len(env._ref_dof) - 1))
ref_q0 = env._ref_dof[ref_idx].astype(np.float64)

def step_cycle(env):
    for _ in range(env.decimation):
        env._px.step(env.native_dt)
    env._px.wait_all()

print("=== Per-DOF target mapping (drives enabled) ===")
print(f"{'DOF':>4s} {'DOF2ACT':>8s} {'worst DOF':>10s} {'max delta':>10s} {'MATCH?'}")
print("-" * 50)

match_count = 0
for dof_i in range(env.nu):
    # Reset to ref_q0 each time
    env._write_joint_positions(ref_q0.astype(np.float32))
    env._write_joint_velocities(np.zeros(env.nu, dtype=np.float32))
    env._px.update_articulations_kinematic()

    # Write self-consistent targets via DOF2ACT
    pos_dof = env._read_joint_positions()
    env._write_joint_targets(pos_dof)

    # Now perturb ONE DOF's target
    perturbed = pos_dof.copy()
    perturbed[dof_i] += 0.2
    env._write_joint_targets(perturbed)

    step_cycle(env)

    q = env._read_joint_positions()
    delta = np.abs(q - pos_dof)
    worst = np.argmax(delta)
    match = "MATCH" if worst == dof_i else "*** MISMATCH ***"
    if worst == dof_i:
        match_count += 1
    print(f"{dof_i:4d} {DOF2ACT[dof_i]:8d} {worst:10d} {delta.max():10.4f} {match}")

print(f"\n{match_count}/{env.nu} DOF match via DOF2ACT")

# Also test direct raw-order perturbation
print("\n=== Raw-order perturbation (no remapping) ===")
match_count = 0
for act_i in range(env.nu):
    env._write_joint_positions(ref_q0.astype(np.float32))
    env._write_joint_velocities(np.zeros(env.nu, dtype=np.float32))
    env._px.update_articulations_kinematic()

    # Read raw positions, write directly as targets
    pos_raw = env._b_pos.read().ravel().copy()
    tgt_raw = pos_raw.copy()
    tgt_raw[act_i] += 0.2
    env._b_tgt.write(tgt_raw.astype(np.float32).reshape(1, -1))

    step_cycle(env)

    q_raw = env._b_pos.read().ravel()
    delta_raw = np.abs(q_raw[act_i] - pos_raw[act_i])
    # Also compute in DOF order
    q_dof = env._read_joint_positions()
    delta_dof = np.abs(q_dof - ref_q0)
    worst_dof = np.argmax(delta_dof)

    match = "MATCH" if worst_dof == act_i else ""
    if delta_dof.max() > 0.01:
        print(f"  act[{act_i:2d}] → dof[{worst_dof:2d}] delta={delta_dof.max():.4f} act_delta={delta_raw:.4f} {match}")

env.close()
px.release()
print("\n*** DONE ***")
