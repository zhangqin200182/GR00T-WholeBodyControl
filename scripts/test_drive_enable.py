"""Fix test: enable drives via DRIVE_MODEL tensor, verify PD control works."""
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
from ovphysx.types import TensorType
from gear_sonic.envs.physx_env_ov import PhysXEnvOv, DOF2ACT, ACT2DOF

class SmokeConfig:
    max_episode_length = 500
    ignore_terminations = True
    alive_bonus = 0.0

class TensorBinding:
    def __init__(self, binding):
        self._b = binding
        self._buf = np.zeros(binding.shape, dtype=np.float32)
    def read(self):
        self._b.read(self._buf)
        return self._buf.copy()
    def write(self, data):
        self._buf[:] = data
        self._b.write(self._buf)
    def destroy(self):
        self._b.destroy()

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

pattern = "/World/G1/*"

# Check joint axes from USD to determine which drive axis to enable
print("=== Joint axes from USD ===")
import re
with open(ROBOT) as f:
    usd_text = f.read()
joint_blocks = re.findall(r'def PhysicsRevoluteJoint "([^"]+)".*?physics:axis = "(\w)"', usd_text, re.DOTALL)
print(f"Found {len(joint_blocks)} joints")
for name, axis in joint_blocks:
    print(f"  {name:40s} axis={axis}")

# Load env
ovphysx.PhysX.set_cpu_mode(True)
px = ovphysx.PhysX()
stage = ovstage.Stage("fix_test")
usd = _prepare_usd(ROBOT)
env = PhysXEnvOv(px, stage, usd, XML, PKL, config=SmokeConfig())

# Create direct DRIVE_MODEL tensor
b_dm = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_DRIVE_MODEL))
dm = b_dm.read()
print(f"\nDRIVE_MODEL shape: {dm.shape}")

# Enable all drives: set to 1 (eFORCE) for all entries
# The shape is (1, 29, 3) — 29 DOF × 3 components per drive
# The 3 components likely correspond to the 3 possible axes (X/Y/Z)
# For a revolute joint with axis A, we need dm[axis] = 1
# Let's try setting ALL to 1 first to see if it works
dm_on = np.ones_like(dm)
b_dm.write(dm_on)

# Verify
dm_check = b_dm.read()
print(f"After write: {np.count_nonzero(dm_check)} non-zero")

# Reset env
obs = env.reset()

ref_time = env._ref_time - env.ctrl_dt
ref_idx = max(0, min(int(ref_time * env._ref_fps), len(env._ref_dof) - 1))
ref_q0 = env._ref_dof[ref_idx].astype(np.float64)

print(f"\n=== Test 1: PD hold at ref_q0 with drives enabled ===")
env._write_joint_positions(ref_q0.astype(np.float32))
env._write_joint_velocities(np.zeros(env.nu, dtype=np.float32))
env._write_joint_targets(ref_q0.astype(np.float64))
env._px.update_articulations_kinematic()

q0_check = env._read_joint_positions()
print(f"Before step: max |q - ref_q0| = {np.abs(q0_check - ref_q0).max():.6f}")

# Step 1 cycle
for _ in range(env.decimation):
    env._px.step(env.native_dt)
env._px.wait_all()

q1 = env._read_joint_positions()
err1 = np.abs(q1 - ref_q0)
print(f"After 1 cycle: max |q - ref_q0| = {err1.max():.6f}")
for j in range(29):
    if err1[j] > 0.01:
        print(f"  Joint {j:2d}: q={q1[j]:+.4f} ref={ref_q0[j]:+.4f} err={err1[j]:.4f}")

# Step 10 more cycles
for _ in range(10):
    env._write_joint_targets(ref_q0.astype(np.float64))
    for _ in range(env.decimation):
        env._px.step(env.native_dt)
    env._px.wait_all()
q10 = env._read_joint_positions()
err10 = np.abs(q10 - ref_q0)
print(f"\nAfter 10 more cycles: max |q - ref_q0| = {err10.max():.6f}")
print(f"  RMS jerr = {np.sqrt(np.mean((q10 - ref_q0)**2)):.6f}")

# Test 2: Self-consistent target test (target = position in raw order)
print(f"\n=== Test 2: Self-consistent targets ===")
env._write_joint_positions(ref_q0.astype(np.float32))
env._write_joint_velocities(np.zeros(env.nu, dtype=np.float32))
env._px.update_articulations_kinematic()

pos_raw = env._b_pos.read()
env._b_tgt.write(pos_raw)  # write same values to target

for _ in range(env.decimation):
    env._px.step(env.native_dt)
env._px.wait_all()

q_self = env._read_joint_positions()
err_self = np.abs(q_self - ref_q0)
print(f"Max error after self-consistent targets: {err_self.max():.6f}")
for j in range(29):
    if err_self[j] > 0.01:
        print(f"  Joint {j:2d}: q={q_self[j]:+.4f} ref={ref_q0[j]:+.4f} err={err_self[j]:.4f}")

# Test 3: Per-DOF perturbation
print(f"\n=== Test 3: Per-DOF perturbation ===")
for dof_idx in [2, 5, 18, 25, 11]:
    env._write_joint_positions(ref_q0.astype(np.float32))
    env._write_joint_velocities(np.zeros(env.nu, dtype=np.float32))
    env._px.update_articulations_kinematic()

    act_idx = DOF2ACT[dof_idx]
    pos_raw = env._b_pos.read().ravel().copy()
    tgt_raw = pos_raw.copy()
    tgt_raw[act_idx] += 0.5
    env._b_tgt.write(tgt_raw.astype(np.float32).reshape(1, -1))

    for _ in range(env.decimation):
        env._px.step(env.native_dt)
    env._px.wait_all()

    q = env._read_joint_positions()
    delta = np.abs(q - ref_q0)
    worst = np.argmax(delta)
    print(f"  dof[{dof_idx:2d}]→act[{act_idx:2d}] +0.5: worst dof[{worst:2d}] delta={delta.max():.4f}  "
          f"{'MATCH' if worst == dof_idx else '*** MISMATCH'}")

env.close()
b_dm.destroy()
px.release()
print("\n*** DONE ***")
