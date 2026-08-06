#!/usr/bin/env python3
"""P2.1: Sweep kd damping ratio (ζ) on PhysX Direct API, measure α on Hard segment.

ζ = kd / (2 * sqrt(kp)) — currently 0.4.
Tests ζ ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0}
"""
import sys, os, numpy as np, glob, joblib, xml.etree.ElementTree as ET

_repo = "/root/GR00T-WholeBodyControl"
sys.path.insert(0, _repo)
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx"))
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx", "build"))
import physx_core, physx_loader

px = physx_core
XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
native_dt = 0.001961
decimation = 17
NU = 29

# Load motion data
pkls = [p for p in glob.glob("/sample_data/robot_filtered/**/*.pkl", recursive=True)
        if not os.path.basename(p).startswith("._")]
data = joblib.load(pkls[0])
mname = list(data.keys())[0]; motion = data[mname]
dof_ref = motion["dof"]; root_trans = motion["root_trans_offset"]; root_rot = motion["root_rot"]

# Parse motor order from XML
tree = ET.parse(XML)
motor_order = [m.get("joint") for m in tree.getroot().find("actuator").findall("motor")]

# Isaac base PD gains
ISAAC_PD = {
    "hip_pitch": (99.1, 6.3), "hip_roll": (99.1, 6.3), "knee": (99.1, 6.3),
    "hip_yaw": (40.2, 2.6),
    "ankle_pitch": (28.5, 1.8), "ankle_roll": (28.5, 1.8),
    "waist_roll": (28.5, 1.8), "waist_pitch": (28.5, 1.8), "waist_yaw": (40.2, 2.6),
    "shoulder_pitch": (14.3, 0.9), "shoulder_roll": (14.3, 0.9),
    "shoulder_yaw": (14.3, 0.9), "elbow": (14.3, 0.9),
    "wrist_roll": (14.3, 0.9), "wrist_pitch": (16.8, 1.1), "wrist_yaw": (16.8, 1.1),
}

ACCEL_KP_SCALE = {
    "hip": 10000, "knee": 10000, "ankle": 10000,
    "waist": 10000,
    "shoulder": 200000, "elbow": 200000, "wrist": 200000,
}


def get_kp(jname):
    for pat, (kp, _) in ISAAC_PD.items():
        if pat in jname:
            for sp, scale in ACCEL_KP_SCALE.items():
                if sp in jname:
                    return kp * scale
    return 100 * 10000


def run_zeta(zeta, start_frame, n_frames):
    """Run ref PD with given ζ, return (alpha, leg_alpha, waist_alpha, arm_alpha)."""
    xml_path = XML

    # Load G1 with default loader (we'll override PD gains)
    art = physx_loader.load_g1(px, xml_path, pos_iters=8, vel_iters=1)
    scene = px.create_scene(gravity=np.array([0., 0., -9.81], dtype=np.float32), solver_type="TGS")
    mat = scene.create_material(0.6, 0.5, 0.0)
    scene.add_ground_plane(mat, np.array([0., 0., 1.], dtype=np.float32))
    scene.add_articulation(art)

    # Override PD gains with new ζ
    for i, jname in enumerate(motor_order):
        kp = get_kp(jname)
        kd = 2 * zeta * np.sqrt(kp)
        art.set_joint_drive_params(i, float(kp), float(kd), 5000.0)

    # Init
    ref_q0 = dof_ref[start_frame]
    rr0 = root_rot[start_frame]
    root_quat0 = np.array([rr0[3], rr0[0], rr0[1], rr0[2]], dtype=np.float32)
    art.set_root_world_pose(root_trans[start_frame].astype(np.float32), root_quat0)
    art.set_joint_positions(ref_q0.astype(np.float32))
    art.set_joint_velocities(np.zeros(NU, dtype=np.float32))
    art.set_joint_drive_targets(ref_q0.astype(np.float32))
    # Warmup
    for _ in range(100):
        scene.simulate(native_dt); scene.fetch_results()

    errors = []
    for i in range(n_frames):
        ref_idx = start_frame + i + 1
        ref_target = dof_ref[ref_idx].astype(np.float32)
        rr = root_rot[ref_idx]
        root_quat = np.array([rr[3], rr[0], rr[1], rr[2]], dtype=np.float32)
        art.set_root_world_pose(root_trans[ref_idx].astype(np.float32), root_quat)
        art.set_joint_drive_targets(ref_target)
        for _ in range(decimation):
            scene.simulate(native_dt); scene.fetch_results()
        errors.append(art.get_joint_positions().astype(np.float64) - ref_target.astype(np.float64))

    errors = np.array(errors)
    if np.any(np.isnan(errors)):
        return None
    use_last = min(50, n_frames)
    alpha = float(np.sqrt(np.mean(errors[-use_last:] ** 2)))
    leg = float(np.sqrt(np.mean(errors[-use_last:, :12] ** 2)))
    waist = float(np.sqrt(np.mean(errors[-use_last:, 12:15] ** 2)))
    arm = float(np.sqrt(np.mean(errors[-use_last:, 15:] ** 2)))
    return alpha, leg, waist, arm


ZETA_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
HARD_START = 400
TEST_FRAMES = 100

px.init_foundation()  # once per process

print(f"{'ζ':>6s}  {'α':>8s}  {'leg_α':>8s}  {'waist_α':>8s}  {'arm_α':>8s}  {'vs ζ=0.4':>8s}")
print("-" * 58)

results = []
for zeta in ZETA_VALUES:
    r = run_zeta(zeta, HARD_START, TEST_FRAMES)
    if r is None:
        print(f"{zeta:6.2f}  {'NaN!':>8s}")
        results.append((zeta, None))
        continue
    alpha, leg, waist, arm = r
    vs_baseline = alpha / 0.0055 if zeta != 0.4 else 1.0  # 0.0055 is ζ=0.4 baseline
    print(f"{zeta:6.2f}  {alpha:8.5f}  {leg:8.4f}  {waist:8.4f}  {arm:8.4f}  {vs_baseline:8.2f}x")
    results.append((zeta, alpha, leg, waist, arm))

# Best ζ
valid = [(r[0], r[1]) for r in results if r[1] is not None]
if valid:
    best = min(valid, key=lambda x: x[1])
    print(f"\nBest ζ: {best[0]:.1f} (α={best[1]:.5f})")

print("DONE")
