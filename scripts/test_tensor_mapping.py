"""Diagnose: do position and target tensors have the same shape?
If not, DOF2ACT mapping for targets is wrong."""
import sys, os
sys.path.insert(0, "/root/GR00T-WholeBodyControl")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx")

for p in ["/usr/local/python3.11.15/lib/python3.11/site-packages/ovphysx/lib",
          "/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin",
          "/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin/plugins",
          "/usr/lib/aarch64-linux-gnu"]:
    e = os.environ.get("LD_LIBRARY_PATH", "")
    if p not in e:
        os.environ["LD_LIBRARY_PATH"] = f"{p}:{e}"

import ovphysx, ovstage
import numpy as np
from gear_sonic.envs.physx_env_ov import PhysXEnvOv, DOF2ACT, ACT2DOF

class SmokeConfig:
    max_episode_length = 500
    ignore_terminations = True
    alive_bonus = 0.0

def _prepare_usd(robot_usd_path, output_path=None):
    if output_path is None:
        output_path = f"/tmp/g1_{os.getpid()}.usda"
    with open(robot_usd_path, "r") as f:
        original = f.read()
    world_open = 'def Xform "World"'
    insert_pos = original.index(world_open) + len(world_open)
    brace_pos = original.index("{", insert_pos)
    SCENE = """
    def PhysicsScene "physicsScene"
    {
        float3 gravity = (0, 0, -9.81)
    }
    def Xform "GroundPlane"
    {
        quatf xformOp:orient = (1, 0, 0, 0)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        def Plane "CollisionPlane" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            uniform token axis = "Z"
            uniform token purpose = "guide"
        }
    }
"""
    combined = original[:brace_pos + 1] + SCENE + original[brace_pos + 1:]
    with open(output_path, "w") as f:
        f.write(combined)
    return output_path

ROBOT = "/root/GR00T-WholeBodyControl/g1_29dof_physx_v7.usda"
XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

ovphysx.PhysX.set_cpu_mode(True)
px = ovphysx.PhysX()
stage = ovstage.Stage("tensor_map")
usd = _prepare_usd(ROBOT)
env = PhysXEnvOv(px, stage, usd, XML, PKL, config=SmokeConfig())
obs = env.reset()

# === KEY CHECK: tensor sizes ===
pos_raw = env._b_pos.read()
tgt_raw = env._b_tgt.read()
print(f"=== Tensor sizes ===")
print(f"_b_pos shape: {pos_raw.shape}  size: {pos_raw.size}")
print(f"_b_tgt shape: {tgt_raw.shape}  size: {tgt_raw.size}")
print(f"nu (env.nu): {env.nu}")
print(f"DOF2ACT: {DOF2ACT}")
print(f"DOF2ACT len: {len(DOF2ACT)}")
print(f"ACT2DOF: {ACT2DOF}")
print(f"ACT2DOF len: {len(ACT2DOF)}")

# Check if DOF2ACT values are all < pos tensor size
max_dof2act = max(DOF2ACT)
print(f"\nmax(DOF2ACT) = {max_dof2act}, pos size = {pos_raw.size}")
if max_dof2act >= pos_raw.size:
    print(f"*** ERROR: DOF2ACT[DOF]={max_dof2act} maps beyond pos tensor size {pos_raw.size}!")

# Check if DOF2ACT values are all < tgt tensor size
if max_dof2act >= tgt_raw.size:
    print(f"*** ERROR: DOF2ACT maps beyond TARGET tensor size!")

# === If sizes differ, the target tensor is the issue ===
# Let's read current positions and try different mapping approaches
print(f"\n=== Systematic target perturbation test ===")
ref_time = env._ref_time - env.ctrl_dt
ref_idx = max(0, min(int(ref_time * env._ref_fps), len(env._ref_dof) - 1))
ref_q0 = env._ref_dof[ref_idx].astype(np.float64)

def reset_to_ref():
    """Reset robot to ref_q0 with zero vel."""
    env._write_joint_positions(ref_q0.astype(np.float32))
    env._write_joint_velocities(np.zeros(env.nu, dtype=np.float32))
    env._px.update_articulations_kinematic()

def step_one_cycle():
    for _ in range(env.decimation):
        env._px.step(env.native_dt)
    env._px.wait_all()

# Test A: For each joint in DOF order, perturb one actuator index
# and see which DOF moves most. This maps act -> dof.
print(f"\n=== Test A: Map actuator index -> DOF index ===")
act2dof_map = {}
for act_idx in range(pos_raw.size):
    reset_to_ref()
    # Write targets = current positions (self-consistent)
    pos_vals = env._b_pos.read().ravel().copy()
    tgt_vals = pos_vals.copy()
    # Perturb ONE actuator index
    tgt_vals[act_idx] += 0.5
    env._b_tgt.write(tgt_vals.astype(np.float32).reshape(1, -1))

    step_one_cycle()

    q = env._read_joint_positions()
    delta = np.abs(q - ref_q0)
    worst_dof = np.argmax(delta)
    worst_delta = delta[worst_dof]

    if worst_delta > 0.02:
        act2dof_map[act_idx] = (worst_dof, worst_delta)
        print(f"  act[{act_idx}] → dof[{worst_dof}] delta={worst_delta:.4f}")

# Print the full mapping
print(f"\n=== Discovered act→dof mapping ({len(act2dof_map)} entries) ===")
for act_idx in sorted(act2dof_map.keys()):
    dof_idx, delta = act2dof_map[act_idx]
    print(f"  act[{act_idx:2d}] → dof[{dof_idx:2d}]  delta={delta:.4f}")

# Test B: Verify DOF2ACT by perturbing ONE DOF index (current method)
print(f"\n=== Test B: Verify DOF2ACT (dof order perturbation) ===")
for dof_idx in range(env.nu):
    reset_to_ref()
    act_idx = DOF2ACT[dof_idx]

    # Read position in actuator order
    pos_act = env._b_pos.read().ravel().copy()
    tgt_act = pos_act.copy()
    tgt_act[act_idx] += 0.5
    env._b_tgt.write(tgt_act.astype(np.float32).reshape(1, -1))

    step_one_cycle()

    q = env._read_joint_positions()
    delta = np.abs(q - ref_q0)
    worst_dof = np.argmax(delta)
    worst_delta = delta[worst_dof]

    if worst_dof != dof_idx or worst_delta > 0.1:
        print(f"  dof[{dof_idx:2d}] → act[{act_idx:2d}] → moves dof[{worst_dof:2d}] by {worst_delta:.4f} {'*** MISMATCH' if worst_dof != dof_idx else 'large'}")

# Test C: If tgt tensor is smaller, try direct sequential mapping
if tgt_raw.size != pos_raw.size:
    print(f"\n=== Test C: Direct sequential mapping (tgt size={tgt_raw.size} != pos size={pos_raw.size}) ===")
    # Hypothesis: target tensor only has driven DOF (excluding base)
    # Write targets sequentially matching DOF order
    for seq_idx in range(min(tgt_raw.size, env.nu)):
        reset_to_ref()
        tgt_vals = np.zeros(tgt_raw.size, dtype=np.float32)
        # Set self-consistent targets first
        for j in range(tgt_raw.size):
            tgt_vals[j] = env._b_pos.read().ravel()[j]
        tgt_vals[seq_idx] += 0.5
        env._b_tgt.write(tgt_vals.reshape(1, -1))

        step_one_cycle()

        q = env._read_joint_positions()
        delta = np.abs(q - ref_q0)
        worst_dof = np.argmax(delta)
        worst_delta = delta[worst_dof]

        if worst_delta > 0.02:
            print(f"  tgt_seq[{seq_idx:2d}] → dof[{worst_dof:2d}] delta={worst_delta:.4f}")

# Test D: Self-consistent test with correct mapping
print(f"\n=== Test D: Self-consistent with per-index matching ===")
reset_to_ref()
pos_act = env._b_pos.read().ravel().copy()
env._b_tgt.write(pos_act.astype(np.float32).reshape(1, -1))

step_one_cycle()

q = env._read_joint_positions()
delta = np.abs(q - ref_q0)
print(f"Self-consistent (raw pos→tgt): max delta = {delta.max():.6f}")
if delta.max() > 0.01:
    print("  Joints with significant movement:")
    for j in range(env.nu):
        if delta[j] > 0.01:
            print(f"    dof[{j:2d}]: delta={delta[j]:.4f}")

env.close()
px.release()
print("\n*** DONE ***")
