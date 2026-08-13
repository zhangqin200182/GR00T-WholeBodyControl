"""Check drive model, stiffness, damping, max force tensors.
Are drives actually configured in the loaded USD articulation?"""
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

ovphysx.PhysX.set_cpu_mode(True)
px = ovphysx.PhysX()
stage = ovstage.Stage("drive_cfg")
usd = _prepare_usd(ROBOT)
ovstage.population.open_usd(stage, usd, ordinal=1, domains=ovstage.PopulationDomain.PHYSICS)
px.attach_ovstage(stage, read_ordinal=1)
px.wait_all()

pattern = "/World/G1/*"

# Read all dof-level tensors
for tt_name in ['ARTICULATION_DOF_DRIVE_MODEL', 'ARTICULATION_DOF_STIFFNESS',
                'ARTICULATION_DOF_DAMPING', 'ARTICULATION_DOF_MAX_FORCE',
                'ARTICULATION_DOF_POSITION', 'ARTICULATION_DOF_POSITION_TARGET',
                'ARTICULATION_DOF_VELOCITY', 'ARTICULATION_DOF_VELOCITY_TARGET']:
    tt = getattr(TensorType, tt_name)
    try:
        b = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=tt))
        val = b.read()
        print(f"{tt_name}: shape={val.shape} values={np.array2string(val.ravel(), precision=2, max_line_width=120)}")
        b.destroy()
    except Exception as e:
        print(f"{tt_name}: ERROR - {e}")

# Now write zeros to target and then read back to confirm write took effect
print("\n=== Write test: can we write to POSITION_TARGET? ===")
b_pos = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION))
b_tgt = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION_TARGET))

# Read initial
pos0 = b_pos.read()
tgt0 = b_tgt.read()
print(f"Initial pos: {pos0.ravel()[:6]}...")
print(f"Initial tgt: {tgt0.ravel()[:6]}...")

# Write a known pattern
test_tgt = np.ones(29, dtype=np.float32) * 0.5
b_tgt.write(test_tgt.reshape(1, -1))

# Read back immediately
tgt1 = b_tgt.read()
print(f"After write tgt: {tgt1.ravel()[:6]}...")
print(f"Write successful: {np.allclose(tgt1.ravel(), 0.5)}")

# Step and check if targets persisted
px.step(0.001)
px.wait_all()
tgt2 = b_tgt.read()
print(f"After 1 step tgt: {tgt2.ravel()[:6]}...")

# Now try: step with targets = current position (self-consistent)
print("\n=== Self-consistent test with direct tensor write ===")
# Reset targets to current positions
px.update_articulations_kinematic()
pos_now = b_pos.read()
b_tgt.write(pos_now)

# Step one native_dt
px.step(0.001)
px.wait_all()

pos_after = b_pos.read()
delta = np.abs(pos_after.ravel() - pos_now.ravel())
print(f"Max position change: {delta.max():.6f}")
print(f"Position delta per DOF: {np.array2string(delta, precision=4, max_line_width=160)}")

# Check which joints have drives enabled (drive_model != 0 means enabled)
b_dm = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_DRIVE_MODEL))
dm = b_dm.read().ravel()
print(f"\nDrive model (0=off, 1=force, 2=acceleration): {dm}")
n_driven = np.sum(dm > 0)
print(f"Number of driven DOF: {n_driven}")

b_pos.destroy(); b_tgt.destroy(); b_dm.destroy()
px.release()
print("\n*** DONE ***")
