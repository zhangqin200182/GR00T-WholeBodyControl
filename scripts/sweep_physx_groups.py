#!/usr/bin/env python3
"""P2.2: Per-group kp scale + ζ optimization.

Tests combinations of kp_scale and ζ per joint group.
Measures α on Hard segment (frame 400-500).
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

pkls = [p for p in glob.glob("/sample_data/robot_filtered/**/*.pkl", recursive=True)
        if not os.path.basename(p).startswith("._")]
data = joblib.load(pkls[0])
mname = list(data.keys())[0]; motion = data[mname]
dof_ref = motion["dof"]; root_trans = motion["root_trans_offset"]; root_rot = motion["root_rot"]

tree = ET.parse(XML)
motor_order = [m.get("joint") for m in tree.getroot().find("actuator").findall("motor")]

ISAAC_PD = {
    "hip_pitch": (99.1, 6.3), "hip_roll": (99.1, 6.3), "knee": (99.1, 6.3),
    "hip_yaw": (40.2, 2.6),
    "ankle_pitch": (28.5, 1.8), "ankle_roll": (28.5, 1.8),
    "waist_roll": (28.5, 1.8), "waist_pitch": (28.5, 1.8), "waist_yaw": (40.2, 2.6),
    "shoulder_pitch": (14.3, 0.9), "shoulder_roll": (14.3, 0.9),
    "shoulder_yaw": (14.3, 0.9), "elbow": (14.3, 0.9),
    "wrist_roll": (14.3, 0.9), "wrist_pitch": (16.8, 1.1), "wrist_yaw": (16.8, 1.1),
}

# Default kp scales (baseline)
BASE_SCALES = {"leg": 10000, "waist": 10000, "arm": 200000}
BASE_ZETA = {"leg": 0.4, "waist": 0.4, "arm": 0.4}

HARD_START = 400
TEST_FRAMES = 100

def classify_joint(jname):
    if any(p in jname for p in ["hip", "knee", "ankle"]):
        return "leg"
    if "waist" in jname:
        return "waist"
    return "arm"

def get_isaac_kp(jname):
    for pat, (kp, _) in ISAAC_PD.items():
        if pat in jname:
            return kp
    return 100.0

def run_config(name, scales, zetas, start_frame, n_frames):
    art = physx_loader.load_g1(px, XML, pos_iters=8, vel_iters=1)
    scene = px.create_scene(gravity=np.array([0.,0.,-9.81], dtype=np.float32), solver_type="TGS")
    mat = scene.create_material(0.6, 0.5, 0.0)
    scene.add_ground_plane(mat, np.array([0.,0.,1.], dtype=np.float32))
    scene.add_articulation(art)

    for i, jname in enumerate(motor_order):
        group = classify_joint(jname)
        kp = get_isaac_kp(jname) * scales[group]
        kd = 2 * zetas[group] * np.sqrt(kp)
        art.set_joint_drive_params(i, float(kp), float(kd), 5000.0)

    ref_q0 = dof_ref[start_frame]
    rr0 = root_rot[start_frame]
    root_quat0 = np.array([rr0[3], rr0[0], rr0[1], rr0[2]], dtype=np.float32)
    art.set_root_world_pose(root_trans[start_frame].astype(np.float32), root_quat0)
    art.set_joint_positions(ref_q0.astype(np.float32))
    art.set_joint_velocities(np.zeros(NU, dtype=np.float32))
    art.set_joint_drive_targets(ref_q0.astype(np.float32))
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


px.init_foundation()

# ── Phase 1: Baseline ──
print("=== Phase 1: Baseline (uniform ζ=0.4) ===")
r = run_config("baseline", BASE_SCALES, BASE_ZETA, HARD_START, TEST_FRAMES)
if r:
    bl_a, bl_l, bl_w, bl_ar = r
    print(f"Baseline: α={bl_a:.5f}  leg={bl_l:.4f}  waist={bl_w:.4f}  arm={bl_ar:.4f}")
print()

# ── Phase 2: Per-group ζ (fixed kp scales) ──
print("=== Phase 2: Per-group ζ sweep (fixed kp scales) ===")
print(f"{'ζ_leg':>7s} {'ζ_waist':>9s} {'ζ_arm':>7s}  {'α':>8s}  {'leg':>8s}  {'waist':>8s}  {'arm':>8s}")
print("-" * 66)

# From the uniform sweep: legs like high ζ, waist likes low ζ, arms moderate
ZETA_LEG = [0.8, 1.0, 1.2]
ZETA_WAIST = [0.2, 0.3, 0.4]
ZETA_ARM = [0.6, 0.8, 1.0]

best_result = None
best_alpha = 999

for zl in ZETA_LEG:
    for zw in ZETA_WAIST:
        for za in ZETA_ARM:
            zetas = {"leg": zl, "waist": zw, "arm": za}
            r = run_config(f"ζ_l={zl}_w={zw}_a={za}", BASE_SCALES, zetas, HARD_START, TEST_FRAMES)
            if r is None:
                print(f"{zl:7.1f} {zw:9.1f} {za:7.1f}  {'NaN!':>8s}")
                continue
            a, l, w, ar = r
            marker = " <--" if a < best_alpha else ""
            if a < best_alpha:
                best_alpha = a
                best_result = (zl, zw, za, a, l, w, ar)
            print(f"{zl:7.1f} {zw:9.1f} {za:7.1f}  {a:8.5f}  {l:8.4f}  {w:8.4f}  {ar:8.4f}{marker}")

if best_result:
    zl, zw, za, a, l, w, ar = best_result
    print(f"\nBest per-group ζ: leg={zl}, waist={zw}, arm={za} → α={a:.5f}")

# ── Phase 3: kp scale sweep (best ζ) ──
print(f"\n=== Phase 3: kp scale sweep (best per-group ζ) ===")
print(f"{'kp_leg':>7s} {'kp_waist':>9s} {'kp_arm':>7s}  {'α':>8s}  {'leg':>8s}  {'waist':>8s}  {'arm':>8s}")
print("-" * 66)

BEST_ZETA = {"leg": best_result[0], "waist": best_result[1], "arm": best_result[2]} if best_result else BASE_ZETA

# Test variations around baseline
scales_to_test = [
    ("baseline", {"leg": 10000, "waist": 10000, "arm": 200000}),
    ("ankle ×2", {"leg": 10000, "waist": 10000, "arm": 200000}),  # same as baseline for now
    ("leg ×1.5", {"leg": 15000, "waist": 10000, "arm": 200000}),
    ("leg ×0.5", {"leg": 5000, "waist": 10000, "arm": 200000}),
    ("waist ×2", {"leg": 10000, "waist": 20000, "arm": 200000}),
    ("arm ×1.5", {"leg": 10000, "waist": 10000, "arm": 300000}),
    ("arm ×0.5", {"leg": 10000, "waist": 10000, "arm": 100000}),
]

for name, scales in scales_to_test:
    r = run_config(name, scales, BEST_ZETA, HARD_START, TEST_FRAMES)
    if r is None:
        print(f"  {name:12s}  {'NaN!':>8s}")
        continue
    a, l, w, ar = r
    marker = " <--" if a < best_alpha else ""
    if a < best_alpha:
        best_alpha = a
    print(f"  {name:12s}  {a:8.5f}  {l:8.4f}  {w:8.4f}  {ar:8.4f}{marker}")

print(f"\n=== Summary ===")
print(f"Baseline (uniform ζ=0.4): α={bl_a:.5f}")
print(f"Best per-group configuration: α={best_alpha:.5f}")
if bl_a > 0:
    print(f"Improvement: {bl_a/best_alpha:.2f}×")
print("DONE")
