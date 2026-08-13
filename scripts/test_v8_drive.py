"""Quick check: does v8 USD with drive:type=force enable drives?"""
import sys, os, numpy as np
for p in ['/usr/local/python3.11.15/lib/python3.11/site-packages/ovphysx/lib',
          '/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin',
          '/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin/plugins',
          '/usr/lib/aarch64-linux-gnu']:
    e = os.environ.get('LD_LIBRARY_PATH', '')
    if p not in e: os.environ['LD_LIBRARY_PATH'] = f'{p}:{e}'

import ovphysx, ovstage
from ovphysx.types import TensorType

class TB:
    def __init__(self, b): self._b = b; self._buf = np.zeros(b.shape, dtype=np.float32)
    def read(self): self._b.read(self._buf); return self._buf.copy()

ROBOT = "/root/GR00T-WholeBodyControl/g1_29dof_physx_v8.usda"
with open(ROBOT) as f: original = f.read()
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
usd = f'/tmp/g1_v8_{os.getpid()}.usda'
with open(usd, 'w') as f: f.write(combined)

ovphysx.PhysX.set_cpu_mode(True)
px = ovphysx.PhysX()
stage = ovstage.Stage('v8_test')
ovstage.population.open_usd(stage, usd, ordinal=1, domains=ovstage.PopulationDomain.PHYSICS)
px.attach_ovstage(stage, read_ordinal=1)
px.wait_all()

pattern = '/World/G1/*'

# Check DRIVE_MODEL
b_dm = TB(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_DRIVE_MODEL))
dm = b_dm.read()
print(f'DRIVE_MODEL shape: {dm.shape}')
print(f'Non-zero entries: {np.count_nonzero(dm)} / {dm.size}')
print(f'Values: {np.array2string(dm.ravel(), precision=1, max_line_width=250)}')

# Check stiffness
b_stiff = TB(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_STIFFNESS))
print(f'Stiffness (first 10): {b_stiff.read().ravel()[:10]}')

px.release()
print('DONE')
