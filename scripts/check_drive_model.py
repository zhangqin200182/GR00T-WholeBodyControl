"""Check and fix DRIVE_MODEL for position+velocity PD control."""
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

from gear_sonic.envs.physx_env_ov import TensorBinding, DOF2ACT, ACT2DOF
from ovphysx.types import TensorType
import ovphysx, ovstage

ROBOT = "/root/GR00T-WholeBodyControl/g1_29dof_physx_v9.usda"

ovphysx.PhysX.set_cpu_mode(True)
px = ovphysx.PhysX()
stage = ovstage.Stage("drive_check")

# Load v9 directly (with gravity, we only care about tensor values)
ovstage.population.open_usd(stage, ROBOT, ordinal=1, domains=ovstage.PopulationDomain.PHYSICS)
px.attach_ovstage(stage, read_ordinal=1)
px.wait_all()

pattern = "/World/G1/*"

# Check DRIVE_MODEL values
b_dm = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_DRIVE_MODEL))
dm = b_dm.read()
print(f"DRIVE_MODEL shape: {dm.shape}")
print(f"Unique DM values: {np.unique(dm)}")

# Show per-DOF drive model
print("\nPer-DOF DRIVE_MODEL [pos, vel, acc]:")
for i in range(29):
    vals = dm[0, i, :]
    if np.any(vals > 0):
        print(f"  DOF[{i:2d}]: {vals}")

# Are values 1.0 or something else?
# If 1.0 → position drive only → no damping!
# If 3.0 → position + velocity drive
all_ones = np.allclose(dm[dm > 0], 1.0)
print(f"\nAll non-zero DM values are 1.0: {all_ones}")

# Check stiffness/damping
b_stiff = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_STIFFNESS))
b_damp = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_DAMPING))
stiff = b_stiff.read().ravel()
damp = b_damp.read().ravel()

print(f"\nPer-DOF stiffness / damping:")
for i in range(29):
    print(f"  DOF[{i:2d}]: kp={stiff[i]:8.1f}  kd={damp[i]:8.4f}")

# Try writing velocity target to zero, maybe that's needed for damping
b_vtgt = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_VELOCITY_TARGET))
vtgt = b_vtgt.read()
print(f"\nVelocity target shape: {vtgt.shape}")
print(f"Velocity target: {vtgt.ravel()[:10]}")

# Try writing drive model to enable velocity drive (value 3 = pos+vel)
# PhysX convention: drive model bitmask
# bit 0 (value 1) = position drive
# bit 1 (value 2) = velocity drive
# So value 3 = pos + vel
print("\n=== Attempting to enable velocity drive (value=3) ===")
new_dm = dm.copy()
for i in range(29):
    for j in range(3):
        if new_dm[0, i, j] > 0:
            new_dm[0, i, j] = 3  # pos + vel drive
b_dm.write(new_dm)
px.wait_all()
# Read back to verify
dm2 = b_dm.read()
changed = np.count_nonzero(dm2 == 3.0)
print(f"DM entries changed to 3: {changed}/87")

# Also set velocity targets to 0
b_vtgt.write(np.zeros((1, 29), dtype=np.float32))
print("Velocity targets set to 0")

# Now test sine tracking with velocity drive enabled
# Set up identity DOF2ACT for testing
dof2act = np.arange(29, dtype=np.int32)
act2dof = np.arange(29, dtype=np.int32)

b_pos = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION))
b_tgt = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION_TARGET))
b_vel = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_VELOCITY))

def read_dof():
    return b_pos.read().ravel()[act2dof].astype(np.float64)
def write_dof(q):
    b_pos.write(q[dof2act].astype(np.float32).reshape(1, -1))
def write_tgt(t):
    b_tgt.write(t[dof2act].astype(np.float32).reshape(1, -1))

def step(n=17):
    for _ in range(n):
        px.step(0.001961)
    px.wait_all()

native_dt = 0.001961
ctrl_dt = native_dt * 17

# Reset to zero
zero = np.zeros(29, dtype=np.float64)
write_dof(zero)
b_vel.write(np.zeros((1, 29), dtype=np.float32))
write_tgt(zero)
px.update_articulations_kinematic()
step(3)

print("\n=== Sine tracking with velocity drive ===")
n_frames = 100
errors = []
t = 0.0
for i in range(n_frames):
    t += ctrl_dt
    ref_target = np.zeros(29, dtype=np.float64)
    ref_target[0] = 0.1 * np.sin(2 * np.pi * 1.0 * t)
    ref_target[6] = 0.1 * np.sin(2 * np.pi * 1.0 * t + 0.5)
    ref_target[18] = 0.1 * np.sin(2 * np.pi * 1.5 * t)
    write_tgt(ref_target)
    step()
    q_actual = read_dof()
    errors.append(q_actual - ref_target)

errors = np.array(errors)
driven = [0, 6, 18]
mask = np.zeros(29, dtype=bool); mask[driven] = True
alpha = float(np.sqrt(np.mean(errors[-50:, mask] ** 2)))
print(f"alpha (driven) = {alpha:.6f} rad")
for j in driven:
    rms = np.sqrt(np.mean(errors[-50:, j] ** 2))
    print(f"  DOF[{j:2d}] rms={rms:.6f}")

# Cleanup
for b in [b_pos, b_tgt, b_vel, b_dm, b_stiff, b_damp, b_vtgt]:
    try: b.destroy()
    except: pass
px.detach_ovstage()
px.release()
print("\n*** DONE ***")
