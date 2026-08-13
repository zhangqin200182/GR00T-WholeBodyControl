"""Test PhysX ARTICULATION_GRAVITY_FORCE as feedforward for gravity compensation."""
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
from gear_sonic.envs.physx_env_ov import TensorBinding

ovphysx.PhysX.set_cpu_mode(True)
px = ovphysx.PhysX()
stage = ovstage.Stage("grav_feasibility")
ovstage.population.open_usd(stage, "/root/GR00T-WholeBodyControl/g1_29dof_physx_v9.usda",
                             ordinal=1, domains=ovstage.PopulationDomain.PHYSICS)
px.attach_ovstage(stage, read_ordinal=1)
px.wait_all()

pattern = "/World/G1/*"
b_grav = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_GRAVITY_FORCE))
b_pos = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION))
b_act = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_ACTUATION_FORCE))
b_tgt = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION_TARGET))
b_vel = TensorBinding(px.create_tensor_binding(pattern=pattern, tensor_type=TensorType.ARTICULATION_DOF_VELOCITY))

act2dof = np.arange(29, dtype=np.int32)
dof2act = np.arange(29, dtype=np.int32)
native_dt = 0.001961
decimation = 17

def read_grav():
    return b_grav.read().ravel()

def read_dof():
    return b_pos.read().ravel()[act2dof].astype(np.float64)

def write_dof(q):
    b_pos.write(q[dof2act].astype(np.float32).reshape(1, -1))

def write_tgt(t):
    b_tgt.write(t[dof2act].astype(np.float32).reshape(1, -1))

def write_act(tau):
    b_act.write(tau[dof2act].astype(np.float32).reshape(1, -1))

def step(n=decimation):
    for _ in range(n):
        px.step(native_dt)
    px.wait_all()

# ── Test 1: Gravity at different poses ──
print("=== Test 1: Gravity torque is pose-dependent ===")
poses = {
    "zero": np.zeros(29, dtype=np.float64),
    "stand": np.zeros(29, dtype=np.float64),
    "squat": np.zeros(29, dtype=np.float64),
}
poses["stand"][[0, 6]] = 0.3
poses["stand"][[3, 9]] = 0.6
poses["squat"][[0, 6]] = 0.8
poses["squat"][[3, 9]] = 1.2

for name, q in poses.items():
    write_dof(q)
    b_vel.write(np.zeros((1, 29), dtype=np.float32))
    px.update_articulations_kinematic()
    px.wait_all()
    tau_g = read_grav()[6:35]
    print(f"  {name:6s}: norm={np.linalg.norm(tau_g):6.1f}  max={np.max(np.abs(tau_g)):6.1f}  DOFs with |tau|>1: {np.sum(np.abs(tau_g)>1)}")
    big = [(j, tau_g[j]) for j in range(29) if abs(tau_g[j]) > 5]
    for j, v in big:
        print(f"    DOF[{j:2d}]: {v:8.1f} Nm")

# ── Test 2: Gravity comp static hold ──
print("\n=== Test 2: Gravity compensation + PD static hold ===")
zero = np.zeros(29, dtype=np.float64)

# 2a: PD only (no gravity comp) — baseline
write_dof(zero)
b_vel.write(np.zeros((1, 29), dtype=np.float32))
write_tgt(zero)
write_act(zero)
px.update_articulations_kinematic()
step(10)
print(f"  PD only: max |q| = {np.max(np.abs(read_dof())):.6f}")

# 2b: Gravity comp + PD
write_dof(zero)
b_vel.write(np.zeros((1, 29), dtype=np.float32))
write_tgt(zero)
grav0 = read_grav()[6:35].astype(np.float64)
write_act(-grav0)
px.update_articulations_kinematic()
step(20)
q = read_dof()
print(f"  Grav+PD: max |q| = {np.max(np.abs(q)):.6f}")
bad = [(j, float(q[j])) for j in range(29) if abs(q[j]) > 0.005]
for j, v in bad:
    print(f"    DOF[{j:2d}]: q={v:.4f}")

# ── Test 3: Sine tracking with gravity comp ──
print("\n=== Test 3: Sine tracking (1 Hz, A=0.1) with gravity comp ===")
write_dof(zero)
b_vel.write(np.zeros((1, 29), dtype=np.float32))
write_tgt(zero)
grav0 = read_grav()[6:35]
write_act(-grav0)
px.update_articulations_kinematic()
step(5)

n_frames = 80
errors = []
t = 0.0
ctrl_dt = native_dt * decimation
for i in range(n_frames):
    t += ctrl_dt
    ref = np.zeros(29, dtype=np.float64)
    ref[0] = 0.1 * np.sin(2 * np.pi * 1.0 * t)  # hip_pitch
    write_tgt(ref)
    step()
    errors.append(read_dof() - ref)

errors = np.array(errors)
alpha_tail = float(np.sqrt(np.mean(errors[20:] ** 2)))
print(f"  alpha (all DOF): {alpha_tail:.6f} rad")
print(f"  alpha DOF[0] only: {np.sqrt(np.mean(errors[20:, 0]**2)):.6f} rad")

# ── Test 4: Sine tracking WITHOUT gravity comp (baseline) ──
print("\n=== Test 4: Sine tracking WITHOUT gravity comp ===")
write_dof(zero)
b_vel.write(np.zeros((1, 29), dtype=np.float32))
write_tgt(zero)
write_act(zero)
px.update_articulations_kinematic()
step(5)

errors2 = []
t = 0.0
for i in range(n_frames):
    t += ctrl_dt
    ref = np.zeros(29, dtype=np.float64)
    ref[0] = 0.1 * np.sin(2 * np.pi * 1.0 * t)
    write_tgt(ref)
    step()
    errors2.append(read_dof() - ref)

errors2 = np.array(errors2)
alpha2 = float(np.sqrt(np.mean(errors2[20:] ** 2)))
print(f"  alpha (all DOF): {alpha2:.6f} rad")
print(f"  alpha DOF[0] only: {np.sqrt(np.mean(errors2[20:, 0]**2)):.6f} rad")

# ── Test 5: Full ref PD tracking with gravity comp ──
print("\n=== Test 5: Ref PD tracking with gravity comp ===")
# Use synthetic step targets to simulate ref tracking
write_dof(zero)
b_vel.write(np.zeros((1, 29), dtype=np.float32))
write_tgt(zero)
grav0 = read_grav()[6:35]
write_act(-grav0)
px.update_articulations_kinematic()
step(5)

# Track alternating targets (±0.2 rad on hips, ±0.5 on knees)
n_cycles = 10
errors3 = []
hold_errors = []
for cycle in range(n_cycles):
    # Target A
    tgt_a = np.zeros(29, dtype=np.float64)
    tgt_a[0] = 0.3; tgt_a[6] = 0.3
    tgt_a[3] = 0.6; tgt_a[9] = 0.6
    tgt_a[15] = -0.5; tgt_a[22] = -0.5
    write_tgt(tgt_a)
    # Apply gravity comp at target pose
    write_dof(tgt_a); px.update_articulations_kinematic(); px.wait_all()
    grav_a = read_grav()[6:35]
    write_dof(zero); px.update_articulations_kinematic(); px.wait_all()  # reset (approximate)
    write_act(-grav_a)
    for _ in range(10):
        step()
    q_a = read_dof()
    errors3.append(q_a - tgt_a)

    # Target B
    tgt_b = np.zeros(29, dtype=np.float64)
    tgt_b[1] = 0.2; tgt_b[7] = -0.2
    write_tgt(tgt_b)
    write_act(grav0)  # wrong sign intentionally — using old grav
    for _ in range(10):
        step()
    q_b = read_dof()
    errors3.append(q_b - tgt_b)

errors3 = np.array(errors3)
alpha3 = float(np.sqrt(np.mean(errors3 ** 2)))
print(f"  alpha (step tracking, 29 DOF): {alpha3:.6f}")
print(f"  Max per-DOF RMS:")
for j in range(29):
    rms = np.sqrt(np.mean(errors3[:, j] ** 2))
    if rms > 0.01:
        print(f"    DOF[{j:2d}]: rms={rms:.4f}")

# Cleanup
for b in [b_grav, b_pos, b_act, b_tgt, b_vel]:
    try: b.destroy()
    except: pass
px.detach_ovstage()
px.release()
print("\n*** DONE ***")
