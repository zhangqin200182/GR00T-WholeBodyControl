"""Verify ARTICULATION_JACOBIAN for gravity compensation feasibility."""
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
stage = ovstage.Stage("grav_test")
ovstage.population.open_usd(stage, "/root/GR00T-WholeBodyControl/g1_29dof_physx_v9.usda",
                             ordinal=1, domains=ovstage.PopulationDomain.PHYSICS)
px.attach_ovstage(stage, read_ordinal=1)
px.wait_all()

pattern = "/World/G1/*"

# ── Read key tensors ──
b_jac = TensorBinding(px.create_tensor_binding(pattern=pattern,
                       tensor_type=TensorType.ARTICULATION_JACOBIAN))
b_mass = TensorBinding(px.create_tensor_binding(pattern=pattern,
                       tensor_type=TensorType.ARTICULATION_BODY_MASS))
b_grav = TensorBinding(px.create_tensor_binding(pattern=pattern,
                       tensor_type=TensorType.ARTICULATION_GRAVITY_FORCE))
b_act = TensorBinding(px.create_tensor_binding(pattern=pattern,
                       tensor_type=TensorType.ARTICULATION_DOF_ACTUATION_FORCE))
b_tgt = TensorBinding(px.create_tensor_binding(pattern=pattern,
                       tensor_type=TensorType.ARTICULATION_DOF_POSITION_TARGET))
b_pos = TensorBinding(px.create_tensor_binding(pattern=pattern,
                       tensor_type=TensorType.ARTICULATION_DOF_POSITION))
b_vel = TensorBinding(px.create_tensor_binding(pattern=pattern,
                       tensor_type=TensorType.ARTICULATION_DOF_VELOCITY))

jac = b_jac.read()      # (1, 180, 35) = batch, 6*30 links, 6+29 DOF
mass = b_mass.read()     # (1, 30)
grav_force = b_grav.read()  # (1, 35) — pre-computed gravity forces

print(f"Jacobian shape: {jac.shape}")
print(f"Mass shape: {mass.shape}")
print(f"Gravity force shape: {grav_force.shape}")
print(f"Gravity force[6:16] (joint DOFs): {grav_force.ravel()[6:16]}")

# ── Check ACTUATION_FORCE writability ──
try:
    b_act.write(np.zeros((1, 29), dtype=np.float32))
    print("ACTUATION_FORCE is WRITABLE ✓")
except Exception as e:
    print(f"ACTUATION_FORCE write error: {e}")

# ── Compute gravity torque from Jacobian ──
J = jac[0]                # (180, 35)
J_joints = J[:, 6:]       # (180, 29) — joint columns only
print(f"J_joints shape: {J_joints.shape}")

# Build gravity wrench per link
# Spatial velocity: [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]
# Spatial force:    [torque_x, torque_y, torque_z, force_x, force_y, force_z]
# Gravity at CoM: pure force in -Z, no torque
g = 9.81
wrench = np.zeros(180, dtype=np.float64)
for k in range(30):
    mk = float(mass[0, k])
    wrench[k * 6 + 5] = -mk * g  # z-force component

# tau_g = J^T @ wrench (power: P = wrench @ v = wrench @ J @ qdot)
tau_g_computed = J_joints.T @ wrench  # (29,)
print(f"\nComputed tau_g (first 10): {tau_g_computed[:10]}")
print(f"PhysX tau_g (first 10):     {grav_force.ravel()[6:16]}")

# Compare
err = tau_g_computed - grav_force.ravel()[6:]
print(f"\nComputation error RMS: {np.sqrt(np.mean(err**2)):.4f} Nm")
print(f"Max abs error: {np.max(np.abs(err)):.4f} Nm")

# ── Per-joint gravity torque ──
print(f"\nPer-joint gravity torque (computed):")
for j in range(29):
    big = " ***" if abs(tau_g_computed[j]) > 30 else ""
    print(f"  DOF[{j:2d}]: {tau_g_computed[j]:8.2f} Nm{big}")

# ── Quick test: enable gravity compensation in a static hold ──
print("\n=== Gravity compensation test: static hold ===")
native_dt = 0.001961
decimation = 17

act2dof = np.arange(29, dtype=np.int32)
dof2act = np.arange(29, dtype=np.int32)

def read_dof():
    return b_pos.read().ravel()[act2dof].astype(np.float64)
def write_dof(q):
    b_pos.write(q[dof2act].astype(np.float32).reshape(1, -1))
def write_tgt(t):
    b_tgt.write(t[dof2act].astype(np.float32).reshape(1, -1))
def write_actuation(tau):
    b_act.write(tau[dof2act].astype(np.float32).reshape(1, -1))
def step(n=decimation):
    for _ in range(n):
        px.step(native_dt)
    px.wait_all()

# Start at zero pose
zero = np.zeros(29, dtype=np.float64)
write_dof(zero)
b_vel.write(np.zeros((1, 29), dtype=np.float32))
write_tgt(zero)
write_actuation(np.zeros(29, dtype=np.float64))
px.update_articulations_kinematic()
step(5)

# Apply gravity compensation
write_actuation(tau_g_computed)
print(f"Applied gravity compensation. tau_g[0]={tau_g_computed[0]:.1f}, tau_g[14]={tau_g_computed[14]:.1f}")

# Let settle
for _ in range(10):
    step()

q_no_pd = read_dof()
print(f"\nGravity comp only (no PD), after settle:")
for j in range(29):
    if abs(q_no_pd[j]) > 0.01:
        print(f"  DOF[{j:2d}]: q={q_no_pd[j]:.4f}")

# Now test: gravity comp + PD should hold at zero perfectly
write_dof(zero)
b_vel.write(np.zeros((1, 29), dtype=np.float32))
write_tgt(zero)
px.update_articulations_kinematic()
step(3)

# Apply gravity comp plus PD target at zero
write_actuation(tau_g_computed)
step(5)

q_comp = read_dof()
hold_err = np.abs(q_comp - zero)
print(f"\nGravity comp + PD (target=0), after settle:")
print(f"  Max hold error: {hold_err.max():.6f} rad")
bad = [(j, float(hold_err[j])) for j in range(29) if hold_err[j] > 0.01]
if bad:
    for j, e in bad:
        print(f"  DOF[{j:2d}]: err={e:.4f}")

# Now test: gravity comp + PD target at non-zero reference posture
# Use a realistic standing pose
print(f"\n=== Perturbation test with gravity comp ===")
standing_q = np.zeros(29, dtype=np.float64)
# Typical standing: slight hip pitch backward, knees slightly bent
standing_q[0] = 0.2   # left hip pitch backward
standing_q[3] = 0.4   # left knee bend
standing_q[6] = 0.2   # right hip pitch
standing_q[9] = 0.4   # right knee bend
standing_q[15] = -0.3 # left shoulder pitch (arm down)
standing_q[22] = -0.3 # right shoulder pitch

# Write standing pose, let gravity comp stabilize
write_dof(zero)
write_tgt(standing_q)
b_vel.write(np.zeros((1, 29), dtype=np.float32))
px.update_articulations_kinematic()

# Apply gravity comp (computed at zero pose — will be slightly wrong)
write_actuation(tau_g_computed)
for _ in range(10):
    step()

q_result = read_dof()
print(f"Standing target achieved:")
for j in range(29):
    err = abs(q_result[j] - standing_q[j])
    if standing_q[j] != 0 or err > 0.01:
        tgt = standing_q[j]
        act = q_result[j]
        print(f"  DOF[{j:2d}]: target={tgt:.3f}  actual={act:.4f}  err={err:.4f}")

# Cleanup
for b in [b_jac, b_mass, b_grav, b_act, b_tgt, b_pos, b_vel]:
    try: b.destroy()
    except: pass
px.detach_ovstage()
px.release()
print("\n*** DONE ***")
