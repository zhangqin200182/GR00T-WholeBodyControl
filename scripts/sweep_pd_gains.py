"""Task 2: G1 PD gain sweep — find optimal kp/kd for ref tracking accuracy.

Uses fixed-base (zero-G + per-step root reset) to isolate joint tracking
from root dynamics.  Scans kp_mult ∈ [1..50], kd_mult = √kp_mult (ζ const),
measures steady-state α per combination.
"""
import sys, numpy as np
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx/build")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs")
sys.path.insert(0, "/root/GR00T-WholeBodyControl/gear_sonic/envs/physx")
import physx_core as px, physx_loader, physx_fk
import joblib, glob, os, time, xml.etree.ElementTree as ET

XML = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
PKL = "/sample_data/robot_filtered"

# ── Joint name → index map (from MJCF actuator order) ──
_tree = ET.parse(XML)
_actuator = _tree.getroot().find("actuator")
JOINT_NAMES = [m.get("joint", "") for m in _actuator.findall("motor")]
assert len(JOINT_NAMES) == 29, f"Expected 29 motors, got {len(JOINT_NAMES)}"

# Isaac baseline PD by joint group
_ISAAC_KP_KD = {
    # joint_pattern: (kp, kd, force_limit)
    "hip_pitch": (99.1, 6.3), "hip_roll": (99.1, 6.3), "knee": (99.1, 6.3),
    "hip_yaw": (40.2, 2.6),
    "ankle_pitch": (28.5, 1.8), "ankle_roll": (28.5, 1.8),
    "waist_roll": (28.5, 1.8), "waist_pitch": (28.5, 1.8), "waist_yaw": (40.2, 2.6),
    "shoulder_pitch": (14.3, 0.9), "shoulder_roll": (14.3, 0.9),
    "shoulder_yaw": (14.3, 0.9), "elbow": (14.3, 0.9),
    "wrist_roll": (14.3, 0.9), "wrist_pitch": (16.8, 1.1), "wrist_yaw": (16.8, 1.1),
}

def get_isaac_kp_kd(jname):
    for pattern, (kp, kd) in _ISAAC_KP_KD.items():
        if pattern in jname:
            return kp, kd
    return 100.0, 5.0


# ── Sweep config ──
KP_MULTS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
KD_MULTS = [1.0, 1.4, 2.2, 3.2,  4.5,  7.1]   # √kp_mult, ζ constant

N_REF = 20          # reference frames
DECIMATION = 17     # 0.0333/0.002
SIM_DT = 0.002
TRANSIENT = 5       # skip first N substeps per ref frame


def run_one(art, fk, scene, ref_root, ref_qpos, kp_mult, kd_mult):
    """Run 20-ref-frame tracking with given multipliers, return mean α."""
    # Set per-joint PD gains
    for j_idx in range(29):
        jname = JOINT_NAMES[j_idx]
        kp_base, kd_base = get_isaac_kp_kd(jname)
        art.set_joint_drive_params(j_idx,
                                   kp_base * kp_mult,
                                   kd_base * kd_mult,
                                   500.0)  # high force limit for sweep

    # Warmup
    art.set_root_world_pose(ref_root, np.array([1,0,0,0], dtype=np.float32))
    art.set_joint_positions(ref_qpos[0].astype(np.float32))
    art.set_joint_velocities(np.zeros(29, dtype=np.float32))
    art.set_joint_drive_targets(ref_qpos[0].astype(np.float32))
    for _ in range(20):
        art.set_root_world_pose(ref_root, np.array([1,0,0,0], dtype=np.float32))
        scene.simulate(SIM_DT); scene.fetch_results()

    all_errs = []
    for ref_i in range(N_REF):
        target = ref_qpos[ref_i + 1].astype(np.float32)
        art.set_joint_drive_targets(target)
        frame_errs = []
        for sub in range(DECIMATION):
            art.set_root_world_pose(ref_root, np.array([1,0,0,0], dtype=np.float32))
            scene.simulate(SIM_DT); scene.fetch_results()
            actual = art.get_joint_positions()
            if np.any(np.isnan(actual)):
                return float('inf'), f"NaN at ref {ref_i}.{sub}"
            e = np.linalg.norm(actual - target)
            frame_errs.append(e)
        # Steady-state: last N steps of each held period
        steady = frame_errs[TRANSIENT:]
        all_errs.extend(steady)

    mean_e = np.mean(all_errs)
    alpha = mean_e / np.sqrt(29)
    return alpha, None


# ── Load data ──
print("Loading data & model...")
pk = joblib.load(glob.glob(os.path.join(PKL, "**/*.pkl"), recursive=True)[0])
if "dof" not in pk: pk = list(pk.values())[0]
start = 500
ref_qpos = pk["dof"][start:start+N_REF+1].astype(np.float64)
ref_root = pk["root_trans_offset"][start].astype(np.float32)

print(f"Ref: {N_REF} frames from idx {start}, root={ref_root}")

# ── Sweep ──
print(f"\n{'='*70}")
print(f"Sweeping {len(KP_MULTS)} kp x {len(KP_MULTS)} kd combinations...")
print(f"{'='*70}")
print(f"{'kp_mult':>8} {'kd_mult':>8} {'α':>10} {'status':>15}")
print(f"{'-'*8} {'-'*8} {'-'*10} {'-'*15}")

results = []
for kpm, kdm in zip(KP_MULTS, KD_MULTS):
    t0 = time.time()
    px.init_foundation()
    art = physx_loader.load_g1(px, XML, pos_iters=8, vel_iters=1)
    fk = physx_fk.G1ForwardKinematics(XML)
    scene = px.create_scene(gravity=np.array([0,0,0], dtype=np.float32))
    scene.add_articulation(art)

    alpha, err_msg = run_one(art, fk, scene, ref_root, ref_qpos, kpm, kdm)
    px.release_foundation()

    status = err_msg if err_msg else f"{time.time()-t0:.1f}s"
    print(f"{kpm:8.1f} {kdm:8.1f} {alpha:10.6f} {status:>15}")
    if err_msg is None:
        results.append((kpm, kdm, alpha))

# ── Report ──
print(f"\n{'='*70}")
print(f"RESULTS (sorted by α)")
print(f"{'='*70}")
results.sort(key=lambda x: x[2])

for rank, (kpm, kdm, alpha) in enumerate(results[:5]):
    flag = "✅ < 0.01" if alpha < 0.01 else ("⚠️  < 0.05" if alpha < 0.05 else "❌")
    print(f"  #{rank+1}: kp×{kpm:.0f}  kd×{kdm:.1f}  →  α={alpha:.6f}  {flag}")

best_kpm, best_kdm, best_alpha = results[0]
print(f"\nBest: kp_mult={best_kpm:.0f}, kd_mult={best_kdm:.1f}, α={best_alpha:.6f}")

# ── Stability test (gravity + ground) for top-3 ──
print(f"\n{'='*70}")
print(f"STABILITY TEST (gravity, ground plane, PD hold, 50 steps)")
print(f"{'='*70}")
for rank, (kpm, kdm, alpha) in enumerate(results[:3]):
    px.init_foundation()
    art = physx_loader.load_g1(px, XML, pos_iters=8, vel_iters=1)
    scene = px.create_scene(gravity=np.array([0,0,-9.81], dtype=np.float32))
    mat = scene.create_material(0.6, 0.5, 0.0)
    scene.add_ground_plane(mat, np.array([0,0,1], dtype=np.float32))
    scene.add_articulation(art)

    # Apply PD gains
    for j_idx in range(29):
        jname = JOINT_NAMES[j_idx]
        kp_base, kd_base = get_isaac_kp_kd(jname)
        art.set_joint_drive_params(j_idx, kp_base * kpm, kd_base * kdm, 500.0)

    root_start = np.array([0.0, 0.0, 0.85], dtype=np.float32)
    art.set_root_world_pose(root_start, np.array([1,0,0,0], dtype=np.float32))
    art.set_joint_positions(ref_qpos[0].astype(np.float32))
    art.set_joint_velocities(np.zeros(29, dtype=np.float32))
    art.set_joint_drive_targets(ref_qpos[0].astype(np.float32))

    survived = 50
    for step in range(50):
        rp = np.array(art.get_root_world_pose()[0])
        jp = art.get_joint_positions()
        if np.any(np.isnan(jp)) or np.any(np.isnan(rp)):
            survived = step; break
        scene.simulate(SIM_DT); scene.fetch_results()
    status = f"✅ stable ({survived}/50)" if survived == 50 else f"❌ NaN @ step {survived}"
    print(f"  #{rank+1}  kp×{kpm:.0f} kd×{kdm:.1f}:  {status}")
    px.release_foundation()

print("\nDone")
