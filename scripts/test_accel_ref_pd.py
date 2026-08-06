"""Test eACCELERATION on harder (high-amplitude) motion segments."""
import sys, os, numpy as np, glob, joblib, time, xml.etree.ElementTree as ET

_repo = "/root/GR00T-WholeBodyControl"
sys.path.insert(0, _repo)
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx"))
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx", "build"))

import physx_core, physx_loader

px = physx_core
native_dt = 0.001961
decimation = 17
NU = 29

pkls = [p for p in glob.glob("/sample_data/robot_filtered/**/*.pkl", recursive=True)
        if not os.path.basename(p).startswith("._")]
data = joblib.load(pkls[0])
mname = list(data.keys())[0]; motion = data[mname]
dof_ref = motion["dof"]; root_trans = motion["root_trans_offset"]; root_rot = motion["root_rot"]

xml_path = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
tree = ET.parse(xml_path)
actuator = tree.getroot().find("actuator")
motor_order = [m.get("joint", "") for m in actuator.findall("motor") if m.get("joint", "")]
base_kp = np.zeros(NU, dtype=np.float64)
for i, jname in enumerate(motor_order):
    kp, _ = physx_loader._isaac_pd_gains(jname)
    base_kp[i] = kp

LEGS = list(range(0, 12))
WAIST = list(range(12, 15))
ARMS = list(range(15, 29))


def run_test(start_frame, n_frames, kp_leg, kp_arm, kd_frac, force_limit, pos_iters=8, vel_iters=1):
    px.init_foundation()
    art = physx_loader.load_g1(px, xml_path, pos_iters=pos_iters, vel_iters=vel_iters)
    scene = px.create_scene(gravity=np.array([0., 0., -9.81], dtype=np.float32),
                            solver_type="TGS")
    mat = scene.create_material(0.6, 0.5, 0.0)
    scene.add_ground_plane(mat, np.array([0., 0., 1.], dtype=np.float32))
    scene.add_articulation(art)

    for i in range(NU):
        m = kp_arm if i in ARMS else kp_leg
        kp = base_kp[i] * m
        kd = np.sqrt(kp) * kd_frac
        art.set_joint_drive_params(i, float(kp), float(kd), float(force_limit))

    ref_q0 = dof_ref[start_frame]
    rr0 = root_rot[start_frame]
    root_quat0 = np.array([rr0[3], rr0[0], rr0[1], rr0[2]], dtype=np.float32)
    art.set_root_world_pose(root_trans[start_frame].astype(np.float32), root_quat0)
    art.set_joint_positions(ref_q0.astype(np.float32))
    art.set_joint_velocities(np.zeros(NU, dtype=np.float32))
    art.set_joint_drive_targets(ref_q0.astype(np.float32))

    for _ in range(100):
        scene.simulate(native_dt)
        scene.fetch_results()

    errors = []
    for i in range(n_frames):
        ref_idx = start_frame + i + 1
        ref_target = dof_ref[ref_idx].astype(np.float32)
        rr = root_rot[ref_idx]
        root_quat = np.array([rr[3], rr[0], rr[1], rr[2]], dtype=np.float32)
        art.set_root_world_pose(root_trans[ref_idx].astype(np.float32), root_quat)
        art.set_joint_drive_targets(ref_target)
        for _ in range(decimation):
            scene.simulate(native_dt)
            scene.fetch_results()
        errors.append(art.get_joint_positions().astype(np.float64) - ref_target.astype(np.float64))

    try: px.release_foundation()
    except: pass

    errors = np.array(errors)
    if np.any(np.isnan(errors)):
        return None
    use_last = min(50, n_frames)
    alpha = float(np.sqrt(np.mean(errors[-use_last:] ** 2)))
    leg_rms = float(np.sqrt(np.mean(errors[-use_last:, LEGS] ** 2)))
    waist_rms = float(np.sqrt(np.mean(errors[-use_last:, WAIST] ** 2)))
    arm_rms = float(np.sqrt(np.mean(errors[-use_last:, ARMS] ** 2)))
    return alpha, leg_rms, waist_rms, arm_rms


# Focus on hardest segment (frames 400-500) with optimized params
print("=== Hard segment (frames 400-500, max_abs=1.37 rad) ===")

tests = [
    # (label, kp_leg_mul, kp_arm_mul, kd_frac, force_limit, pos_iters)
    ("Baseline: Leg 10k× Arm 300k× kd=0.5", 10000, 300000, 0.5, 5000, 8),
    ("Leg 20k× Arm 300k× kd=0.4", 20000, 300000, 0.4, 5000, 8),
    ("Leg 20k× Arm 500k× kd=0.4", 20000, 500000, 0.4, 5000, 8),
    ("Leg 50k× Arm 500k× kd=0.3", 50000, 500000, 0.3, 5000, 8),
    ("Leg 50k× Arm 1M× kd=0.3", 50000, 1000000, 0.3, 5000, 8),
    ("Leg 100k× Arm 1M× kd=0.3", 100000, 1000000, 0.3, 5000, 8),
    # Try with higher force_limit
    ("Leg 50k× Arm 500k× kd=0.3 FL=20000", 50000, 500000, 0.3, 20000, 8),
    # Try higher solver iters
    ("Leg 50k× Arm 500k× kd=0.3 it=32/4", 50000, 500000, 0.3, 5000, 32),
    # Try with v17-style per-joint scaling (ankles & waist need more)
]

for label, kpl, kpa, kdf, fl, pi in tests:
    r = run_test(400, 100, kpl, kpa, kdf, fl, pos_iters=pi)
    if r:
        a, l, w, ar = r
        print(f"  {label}")
        print(f"    alpha={a:.6f}  leg={l:.4f}  waist={w:.4f}  arm={ar:.4f}  ({a/0.002:.1f}x)")
    else:
        print(f"  {label}: NaN!")

# Also: test with DIFFERENT joint-group multipliers
# Ankles & waist have base_kp=28.5, less than arms' 14.3. But they're performing worse.
# The issue might be gravity loading, not kp.
# Let me try: classify by joint ROLE, not by body region
# High-load: ankles (support body weight), waist (heavy torso)
# Low-load: elbows, wrists

# Map joints to custom multipliers based on loading
print("\n=== Per-joint-type scaling (frames 400-500) ===")

# Classify by Isaac base_kp + loading
def classify_joint(idx, jname):
    """Return multiplier based on joint loading."""
    if "ankle" in jname:
        return 50000  # High load (support body + dynamic)
    elif "hip" in jname and "yaw" in jname:
        return 20000  # Medium load
    elif "hip" in jname:
        return 10000  # Low load (pitch/roll, axis-aligned)
    elif "knee" in jname:
        return 10000  # Low load
    elif "waist" in jname:
        return 50000  # High load (heavy torso)
    elif "shoulder" in jname and ("pitch" in jname or "yaw" in jname):
        return 200000  # Medium load (against gravity)
    elif "shoulder" in jname:
        return 200000  # Roll
    elif "elbow" in jname:
        return 100000  # Lower load
    elif "wrist" in jname:
        return 200000  # Low load but low inertia
    return 200000

r = run_test(400, 100, 0, 0, 0.5, 5000)  # placeholder

# Build per-joint multipliers
px.init_foundation()
art = physx_loader.load_g1(px, xml_path, pos_iters=8, vel_iters=1)
scene = px.create_scene(gravity=np.array([0., 0., -9.81], dtype=np.float32), solver_type="TGS")
mat = scene.create_material(0.6, 0.5, 0.0)
scene.add_ground_plane(mat, np.array([0., 0., 1.], dtype=np.float32))
scene.add_articulation(art)

for i in range(NU):
    jname = motor_order[i]
    m = classify_joint(i, jname)
    kp = base_kp[i] * m
    kd = np.sqrt(kp) * 0.4
    art.set_joint_drive_params(i, float(kp), float(kd), 5000.0)

ref_q0 = dof_ref[400]
rr0 = root_rot[400]
root_quat0 = np.array([rr0[3], rr0[0], rr0[1], rr0[2]], dtype=np.float32)
art.set_root_world_pose(root_trans[400].astype(np.float32), root_quat0)
art.set_joint_positions(ref_q0.astype(np.float32))
art.set_joint_velocities(np.zeros(NU, dtype=np.float32))
art.set_joint_drive_targets(ref_q0.astype(np.float32))
for _ in range(100):
    scene.simulate(native_dt); scene.fetch_results()

errors = []
for i in range(100):
    ref_idx = 400 + i + 1
    ref_target = dof_ref[ref_idx].astype(np.float32)
    rr = root_rot[ref_idx]
    root_quat = np.array([rr[3], rr[0], rr[1], rr[2]], dtype=np.float32)
    art.set_root_world_pose(root_trans[ref_idx].astype(np.float32), root_quat)
    art.set_joint_drive_targets(ref_target)
    for _ in range(decimation):
        scene.simulate(native_dt); scene.fetch_results()
    errors.append(art.get_joint_positions().astype(np.float64) - ref_target.astype(np.float64))

try: px.release_foundation()
except: pass

errors = np.array(errors)
alpha = float(np.sqrt(np.mean(errors[-50:] ** 2)))
leg_rms = float(np.sqrt(np.mean(errors[-50:, LEGS] ** 2)))
waist_rms = float(np.sqrt(np.mean(errors[-50:, WAIST] ** 2)))
arm_rms = float(np.sqrt(np.mean(errors[-50:, ARMS] ** 2)))
print(f"  Per-joint-type scaling:")
print(f"    alpha={alpha:.6f}  leg={leg_rms:.4f}  waist={waist_rms:.4f}  arm={arm_rms:.4f}  ({alpha/0.002:.1f}x)")

# Per-DOF
print(f"  Per-DOF:")
for j in range(NU):
    rms = np.sqrt(np.mean(errors[-50:, j] ** 2))
    flag = " ***" if rms > 0.005 else ""
    print(f"    [{j:2d}] {motor_order[j]:30s} rms={rms:.6f}{flag}")

print("\n*** DONE ***")
