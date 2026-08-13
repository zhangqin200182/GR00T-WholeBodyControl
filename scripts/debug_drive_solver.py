"""Debug PhysX drive model format and solver configuration."""
import sys, os, numpy as np
for p in ['/usr/local/python3.11.15/lib/python3.11/site-packages/ovphysx/lib',
          '/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin',
          '/usr/local/python3.11.15/lib/python3.11/site-packages/ovstage/bin/plugins',
          '/usr/lib/aarch64-linux-gnu']:
    e = os.environ.get('LD_LIBRARY_PATH', '')
    if p not in e: os.environ['LD_LIBRARY_PATH'] = f'{p}:{e}'

sys.path.insert(0, "/root/GR00T-WholeBodyControl")

import ovphysx, ovstage
from ovphysx.types import TensorType

ROBOT = "/root/GR00T-WholeBodyControl/g1_29dof_physx_v9.usda"

ovphysx.PhysX.set_cpu_mode(True)
px = ovphysx.PhysX()

# Check if PhysX object has solver config methods
print("=== PhysX API inspection ===")
for attr in dir(px):
    if any(k in attr.lower() for k in ['solver', 'iter', 'step', 'config', 'param',
                                         'pgs', 'tgs', 'position', 'velocity',
                                         'create', 'tensor', 'attach', 'update']):
        print(f"  px.{attr}")

# Check TensorTypes available
print("\n=== TensorType enum ===")
for attr in dir(TensorType):
    if 'DRIVE' in attr.upper() or 'SOLVER' in attr.upper() or 'TORQUE' in attr.upper() or 'FORCE' in attr.upper():
        print(f"  TensorType.{attr}")

stage = ovstage.Stage("debug")
ovstage.population.open_usd(stage, ROBOT, ordinal=1, domains=ovstage.PopulationDomain.PHYSICS)
px.attach_ovstage(stage, read_ordinal=1)
px.wait_all()

pattern = "/World/G1/*"

# Read ALL available tensor types to understand what we're working with
tensor_types_of_interest = {
    'DRIVE_MODEL': TensorType.ARTICULATION_DOF_DRIVE_MODEL,
    'STIFFNESS': TensorType.ARTICULATION_DOF_STIFFNESS,
    'DAMPING': TensorType.ARTICULATION_DOF_DAMPING,
    'MAX_FORCE': TensorType.ARTICULATION_DOF_MAX_FORCE,
    'POS_TARGET': TensorType.ARTICULATION_DOF_POSITION_TARGET,
    'VEL_TARGET': TensorType.ARTICULATION_DOF_VELOCITY_TARGET,
    'POSITION': TensorType.ARTICULATION_DOF_POSITION,
    'VELOCITY': TensorType.ARTICULATION_DOF_VELOCITY,
    'FORCE_LIMIT': TensorType.ARTICULATION_DOF_FORCE_LIMIT,
    'ACCUMULATED_FORCE': TensorType.ARTICULATION_DOF_ACCUMULATED_FORCE,
    'TORQUE': TensorType.ARTICULATION_DOF_TORQUE,
}

# Also try to read other potentially available types
for tname in dir(TensorType):
    if 'ARTICULATION' in tname.upper() and not tname.startswith('_'):
        if tname not in tensor_types_of_interest:
            tensor_types_of_interest[tname] = getattr(TensorType, tname)

from gear_sonic.envs.physx_env_ov import TensorBinding

for name, ttype in tensor_types_of_interest.items():
    try:
        b = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=ttype))
        val = b.read()
        shape_str = str(val.shape)
        if val.size <= 200:
            val_preview = str(val.ravel()[:min(val.size, 10)])
        else:
            val_preview = f"[{val.ravel()[0]:.4f}, ..., {val.ravel()[-1]:.4f}]"
        print(f"\n{name}: shape={shape_str}")
        print(f"  Values: {val_preview}")
        if val.size <= 30:
            print(f"  All: {val.ravel()}")
        b.destroy()
    except Exception as e:
        print(f"\n{name}: ERROR — {e}")

# Check if we can configure solver via USD physics scene or through API
# Try accessing articulation to get solver iteration counts
print("\n=== Inspecting stage for solver config ===")
# Look at what create_tensor_binding supports or if there's a way to set solver params

# Test sine tracking at ultra-low frequency to confirm bandwidth hypothesis
b_pos = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION))
b_tgt = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION_TARGET))
b_vel = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_VELOCITY))

act2dof = np.arange(29, dtype=np.int32)
dof2act = np.arange(29, dtype=np.int32)

def read_dof():
    return b_pos.read().ravel()[act2dof].astype(np.float64)
def write_dof(q):
    b_pos.write(q[dof2act].astype(np.float32).reshape(1, -1))
def write_tgt(t):
    b_tgt.write(t[dof2act].astype(np.float32).reshape(1, -1))

native_dt = 0.001961
decimation = 17
ctrl_dt = native_dt * decimation

def step(n=decimation):
    for _ in range(n):
        px.step(native_dt)
    px.wait_all()

# Test at 0.2 Hz (very slow)
print("\n=== Sine tracking at 0.2 Hz (ultra-slow) ===")
zero = np.zeros(29, dtype=np.float64)
write_dof(zero)
b_vel.write(np.zeros((1, 29), dtype=np.float32))
write_tgt(zero)
px.update_articulations_kinematic()
step(5)

n_frames = 100
errors_02 = []
t = 0.0
for i in range(n_frames):
    t += ctrl_dt
    ref_target = np.zeros(29, dtype=np.float64)
    ref_target[0] = 0.1 * np.sin(2 * np.pi * 0.2 * t)
    write_tgt(ref_target)
    step()
    errors_02.append(read_dof() - ref_target)

errors_02 = np.array(errors_02)
alpha_02 = float(np.sqrt(np.mean(errors_02[-50:, 0] ** 2)))
print(f"alpha DOF[0] at 0.2Hz: {alpha_02:.6f} rad")

# Test at 2 Hz
print("\n=== Sine tracking at 2 Hz ===")
write_dof(zero)
b_vel.write(np.zeros((1, 29), dtype=np.float32))
write_tgt(zero)
px.update_articulations_kinematic()
step(3)

n_frames = 100
errors_2 = []
t = 0.0
for i in range(n_frames):
    t += ctrl_dt
    ref_target = np.zeros(29, dtype=np.float64)
    ref_target[0] = 0.1 * np.sin(2 * np.pi * 2.0 * t)
    write_tgt(ref_target)
    step()
    errors_2.append(read_dof() - ref_target)

errors_2 = np.array(errors_2)
alpha_2 = float(np.sqrt(np.mean(errors_2[-50:, 0] ** 2)))
print(f"alpha DOF[0] at 2Hz: {alpha_2:.6f} rad")

# Test at 5 Hz
print("\n=== Sine tracking at 5 Hz ===")
write_dof(zero)
b_vel.write(np.zeros((1, 29), dtype=np.float32))
write_tgt(zero)
px.update_articulations_kinematic()
step(3)

n_frames = 150
errors_5 = []
t = 0.0
for i in range(n_frames):
    t += ctrl_dt
    ref_target = np.zeros(29, dtype=np.float64)
    ref_target[0] = 0.1 * np.sin(2 * np.pi * 5.0 * t)
    write_tgt(ref_target)
    step()
    errors_5.append(read_dof() - ref_target)

errors_5 = np.array(errors_5)
alpha_5 = float(np.sqrt(np.mean(errors_5[-50:, 0] ** 2)))
print(f"alpha DOF[0] at 5Hz: {alpha_5:.6f} rad")

print(f"\n=== Frequency response summary (DOF[0]) ===")
print(f"  0.2 Hz: α={alpha_02:.6f}")
print(f"  1.0 Hz: (from prior test) α≈0.047")
print(f"  2.0 Hz: α={alpha_2:.6f}")
print(f"  5.0 Hz: α={alpha_5:.6f}")

# Cleanup
for b in [b_pos, b_tgt, b_vel]:
    try: b.destroy()
    except: pass
px.detach_ovstage()
px.release()
print("\n*** DONE ***")
