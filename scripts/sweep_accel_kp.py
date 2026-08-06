"""Sweep stiffness values for eACCELERATION to find right scale."""
import sys, os, numpy as np, time, xml.etree.ElementTree as ET

_repo = "/root/GR00T-WholeBodyControl"
sys.path.insert(0, _repo)
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx"))
sys.path.insert(0, os.path.join(_repo, "gear_sonic", "envs", "physx", "build"))

import physx_core, physx_loader

px = physx_core
native_dt = 0.001961
NU = 29

# Get joint names from MJCF motor order
xml_path = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"
tree = ET.parse(xml_path)
actuator = tree.getroot().find("actuator")
motor_order = [m.get("joint", "") for m in actuator.findall("motor") if m.get("joint", "")]

base_kp = np.zeros(NU, dtype=np.float64)
base_kd = np.zeros(NU, dtype=np.float64)
for i, jname in enumerate(motor_order):
    kp, kd = physx_loader._isaac_pd_gains(jname)
    base_kp[i] = kp
    base_kd[i] = kd

print(f"Motor order ({len(motor_order)} DOFs):")
for i in range(NU):
    print(f"  [{i:2d}] {motor_order[i]:30s}  kp={base_kp[i]:6.1f}  kd={base_kd[i]:4.1f}")


def test_single_config(kp_mult, kd_frac):
    px.init_foundation()
    art = physx_loader.load_g1(px, xml_path, pos_iters=8, vel_iters=1)
    scene = px.create_scene(gravity=np.array([0., 0., -9.81], dtype=np.float32))
    mat = scene.create_material(0.6, 0.5, 0.0)
    scene.add_ground_plane(mat, np.array([0., 0., 1.], dtype=np.float32))
    scene.add_articulation(art)

    # Override PD params
    for i in range(NU):
        kp = base_kp[i] * kp_mult
        kd = np.sqrt(kp) * kd_frac
        art.set_joint_drive_params(i, float(kp), float(kd), 5000.0)

    # Init
    art.set_root_world_pose(np.array([0., 0., 0.9], dtype=np.float32),
                            np.array([1., 0., 0., 0.], dtype=np.float32))
    art.set_joint_positions(np.zeros(NU, dtype=np.float32))
    art.set_joint_velocities(np.zeros(NU, dtype=np.float32))
    art.set_joint_drive_targets(np.zeros(NU, dtype=np.float32))
    for _ in range(50):
        scene.simulate(native_dt)
        scene.fetch_results()
    q = art.get_joint_positions()
    if np.any(np.isnan(q)):
        try: px.release_foundation()
        except: pass
        return [999., 999., 999.], 999., True

    # Test: Static hold on 3 representative DOFs
    test_dofs = [0, 6, 15]  # left_hip_yaw, right_hip_yaw, left_shoulder_pitch
    hold_errs = []
    for d in test_dofs:
        tgt = np.zeros(NU, dtype=np.float32)
        tgt[d] = 0.1
        art.set_joint_drive_targets(tgt)
        for _ in range(80):
            scene.simulate(native_dt)
            scene.fetch_results()
        q = art.get_joint_positions()
        hold_errs.append(abs(float(q[d]) - 0.1))

    # Test: Sine tracking (DOF[0], 1Hz, 0.1 amplitude)
    art.set_joint_positions(np.zeros(NU, dtype=np.float32))
    art.set_joint_velocities(np.zeros(NU, dtype=np.float32))
    art.set_joint_drive_targets(np.zeros(NU, dtype=np.float32))
    for _ in range(30):
        scene.simulate(native_dt)
        scene.fetch_results()

    ctrl_dt = native_dt * 17
    t = 0.0
    sine_errs = []
    for _ in range(40):
        t += ctrl_dt
        ref = np.zeros(NU, dtype=np.float64)
        ref[0] = 0.1 * np.sin(2 * np.pi * 1.0 * t)
        art.set_joint_drive_targets(ref.astype(np.float32))
        for _ in range(17):
            scene.simulate(native_dt)
            scene.fetch_results()
        q = art.get_joint_positions()
        sine_errs.append(float(q[0]) - ref[0])

    try: px.release_foundation()
    except: pass

    rms = float(np.sqrt(np.mean(np.array(sine_errs[-20:])**2)))
    is_nan = any(np.isnan(e) for e in hold_errs) or np.isnan(rms)
    return hold_errs, rms, is_nan


print(f"\n{'kp_mult':>8s}  {'DOF0 err':>9s}  {'DOF6 err':>9s}  {'DOF15err':>9s}  {'sine RMS':>9s}  {'status'}")
print("-" * 75)

for m in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
    try:
        h, rms, nan = test_single_config(m, kd_frac=0.7)
        status = "NaN!" if nan else ""
        print(f"{m:8d}  {h[0]:9.4f}  {h[1]:9.4f}  {h[2]:9.4f}  {rms:9.4f}  {status}")
    except Exception as e:
        print(f"{m:8d}  {'ERR':>9s}  {'ERR':>9s}  {'ERR':>9s}  {'ERR':>9s}  {str(e)[:60]}")
    time.sleep(0.1)

print("\n*** DONE ***")
