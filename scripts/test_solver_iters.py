"""Test PD tracking with different solver iteration counts."""
import sys, os, numpy as np, tempfile
for p in ['/usr/local/python3.11.15/lib/python3.11/site-packages/ovphysx/lib',
          '/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin',
          '/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin/plugins',
          '/usr/lib/aarch64-linux-gnu']:
    e = os.environ.get('LD_LIBRARY_PATH', '')
    if p not in e: os.environ['LD_LIBRARY_PATH'] = f'{p}:{e}'

sys.path.insert(0, "/root/GR00T-WholeBodyControl")

import ovphysx, ovstage
from ovphysx.types import TensorType
from gear_sonic.envs.physx_env_ov import TensorBinding

ROBOT = "/root/GR00T-WholeBodyControl/g1_29dof_physx_v9.usda"
with open(ROBOT) as rf:
    original = rf.read()

world_match = 'def Xform "World"'
world_pos = original.index(world_match)
brace_pos = original.index("{", world_pos)

configs = [
    ("pos4_vel1", 4, 1),
    ("pos8_vel1", 8, 1),
    ("pos16_vel4", 16, 4),
    ("pos32_vel8", 32, 8),
]

native_dt = 0.001961
decimation = 17
ctrl_dt = native_dt * decimation
act2dof = np.arange(29, dtype=np.int32)
dof2act = np.arange(29, dtype=np.int32)
pattern = "/World/G1/*"


def test_tracking(px, b_pos, b_tgt, b_vel):
    def _read_dof():
        return b_pos.read().ravel()[act2dof].astype(np.float64)
    def _write_dof(q):
        b_pos.write(q[dof2act].astype(np.float32).reshape(1, -1))
    def _write_tgt(t):
        b_tgt.write(t[dof2act].astype(np.float32).reshape(1, -1))
    def _step(n=decimation):
        for _ in range(n):
            px.step(native_dt)
        px.wait_all()

    zero = np.zeros(29, dtype=np.float64)
    _write_dof(zero)
    b_vel.write(np.zeros((1, 29), dtype=np.float32))
    _write_tgt(zero)
    px.update_articulations_kinematic()
    _step(5)

    n_frames = 80
    errors = []
    t = 0.0
    for i in range(n_frames):
        t += ctrl_dt
        ref_target = np.zeros(29, dtype=np.float64)
        ref_target[0] = 0.1 * np.sin(2 * np.pi * 1.0 * t)
        _write_tgt(ref_target)
        _step()
        errors.append(_read_dof()[0] - ref_target[0])

    errors = np.array(errors)
    rms = float(np.sqrt(np.mean(errors[-40:] ** 2)))
    peak = float(np.max(np.abs(errors[-40:])))
    return rms, peak


# Baseline first (no physics scene override)
print("Testing baseline (default solver)...")
ovphysx.PhysX.set_cpu_mode(True)
px = ovphysx.PhysX()
stage = ovstage.Stage("base")
ovstage.population.open_usd(stage, ROBOT, ordinal=1, domains=ovstage.PopulationDomain.PHYSICS)
px.attach_ovstage(stage, read_ordinal=1)
px.wait_all()

b_pos = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION))
b_tgt = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION_TARGET))
b_vel = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_VELOCITY))

rms, peak = test_tracking(px, b_pos, b_tgt, b_vel)
print(f"  baseline (default):  RMS={rms:.6f}  peak={peak:.6f}")

for b in [b_pos, b_tgt, b_vel]:
    try: b.destroy()
    except: pass
px.detach_ovstage()
px.release()

# Now test each config
for label, pos_iters, vel_iters in configs:
    physx_scene = f"""
    def PhysicsScene "physicsScene"
    {{
        int physxSolverPositionIterations = {pos_iters}
        int physxSolverVelocityIterations = {vel_iters}
    }}
"""
    combined = original[:brace_pos + 1] + physx_scene + original[brace_pos + 1:]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".usda", delete=False) as f:
        f.write(combined)
        usd_path = f.name

    try:
        ovphysx.PhysX.set_cpu_mode(True)
        px = ovphysx.PhysX()
        stage = ovstage.Stage(f"s_{label}")
        ovstage.population.open_usd(stage, usd_path, ordinal=1, domains=ovstage.PopulationDomain.PHYSICS)
        px.attach_ovstage(stage, read_ordinal=1)
        px.wait_all()

        b_pos = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION))
        b_tgt = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION_TARGET))
        b_vel = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_VELOCITY))

        rms, peak = test_tracking(px, b_pos, b_tgt, b_vel)
        print(f"  {label:12s} (pos={pos_iters:2d}, vel={vel_iters:2d}): RMS={rms:.6f}  peak={peak:.6f}")

        for b in [b_pos, b_tgt, b_vel]:
            try: b.destroy()
            except: pass
        px.detach_ovstage()
        px.release()
    finally:
        try: os.unlink(usd_path)
        except: pass

print()
print("*** DONE ***")
