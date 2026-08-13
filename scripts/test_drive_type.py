"""Test PD tracking with acceleration-level drives vs force-level drives."""
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
with open(ROBOT) as f:
    usd_text = f.read()

# Generate "acceleration" variant
usd_accel = usd_text.replace(
    'drive:angular:physics:type = "force"',
    'drive:angular:physics:type = "acceleration"')

with tempfile.NamedTemporaryFile(mode="w", suffix=".usda", delete=False) as f:
    f.write(usd_accel)
    accel_path = f.name

native_dt = 0.001961
decimation = 17
ctrl_dt = native_dt * decimation
act2dof = np.arange(29, dtype=np.int32)
dof2act = np.arange(29, dtype=np.int32)
pattern = "/World/G1/*"


def run_test(label, usd_path):
    ovphysx.PhysX.set_cpu_mode(True)
    px = ovphysx.PhysX()
    stage = ovstage.Stage(f"test_{label}")
    ovstage.population.open_usd(stage, usd_path, ordinal=1,
                                 domains=ovstage.PopulationDomain.PHYSICS)
    px.attach_ovstage(stage, read_ordinal=1)
    px.wait_all()

    # Check drive model tensor
    b_dm = TensorBinding(px.create_tensor_binding(
        pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_DRIVE_MODEL))
    dm = b_dm.read()
    uniq = np.unique(dm.ravel())
    print(f"  DRIVE_MODEL unique: {uniq}")

    b_pos = TensorBinding(px.create_tensor_binding(
        pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION))
    b_tgt = TensorBinding(px.create_tensor_binding(
        pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION_TARGET))
    b_vel = TensorBinding(px.create_tensor_binding(
        pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_VELOCITY))

    def read_dof():
        return b_pos.read().ravel()[act2dof].astype(np.float64)

    def write_dof(q):
        b_pos.write(q[dof2act].astype(np.float32).reshape(1, -1))

    def write_tgt(t):
        b_tgt.write(t[dof2act].astype(np.float32).reshape(1, -1))

    def step(n=decimation):
        for _ in range(n):
            px.step(native_dt)
        px.wait_all()

    # === Test 1: Static hold ===
    zero = np.zeros(29, dtype=np.float64)
    write_dof(zero)
    b_vel.write(np.zeros((1, 29), dtype=np.float32))
    write_tgt(zero)
    px.update_articulations_kinematic()
    step(10)
    q = read_dof()
    print(f"  Hold max |q|: {np.max(np.abs(q)):.6f} rad")

    # === Test 2: Per-DOF perturbation ===
    match_count = 0
    for dof_i in range(29):
        write_dof(zero)
        write_tgt(zero)
        px.update_articulations_kinematic()
        step(1)
        pos_before = read_dof()
        tgt = np.zeros(29, dtype=np.float64)
        tgt[dof_i] = 0.5
        write_tgt(tgt)
        step(3)
        delta = np.abs(read_dof() - pos_before)
        worst = np.argmax(delta)
        if worst == dof_i:
            match_count += 1
        elif dof_i < 20:
            print(f"    DOF[{dof_i:2d}] MISMATCH: worst=DOF[{worst:2d}] delta={delta.max():.4f}")
    print(f"  PertMatch: {match_count}/29")

    # === Test 3: Sine tracking at 1Hz ===
    write_dof(zero)
    b_vel.write(np.zeros((1, 29), dtype=np.float32))
    write_tgt(zero)
    px.update_articulations_kinematic()
    step(5)

    n_frames = 80
    errors = []
    t = 0.0
    for i in range(n_frames):
        t += ctrl_dt
        ref = np.zeros(29, dtype=np.float64)
        ref[0] = 0.1 * np.sin(2 * np.pi * 1.0 * t)
        ref[6] = 0.1 * np.sin(2 * np.pi * 1.0 * t + 0.5)
        write_tgt(ref)
        step()
        errors.append(read_dof() - ref)

    errors = np.array(errors)
    alpha = float(np.sqrt(np.mean(errors[-40:] ** 2)))
    print(f"  Sine1Hz alpha: {alpha:.6f} rad")

    # Cleanup
    for b in [b_pos, b_tgt, b_vel, b_dm]:
        try: b.destroy()
        except: pass
    px.detach_ovstage()
    px.release()

    return alpha


print("=== FORCE drives ===")
a_force = run_test("force", ROBOT)

print("\n=== ACCELERATION drives ===")
a_accel = run_test("accel", accel_path)

print(f"\n=== Summary ===")
print(f"  Force:        alpha = {a_force:.6f}")
print(f"  Acceleration: alpha = {a_accel:.6f}")
print(f"  Improvement:  {a_force/a_accel:.1f}x")

os.unlink(accel_path)
print("\n*** DONE ***")
