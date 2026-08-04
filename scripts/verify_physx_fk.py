#!/usr/bin/env python3
"""Verify Python FK against MuJoCo mj_kinematics."""
import sys, os, numpy as np

# Need MuJoCo installed on the server
try:
    import mujoco
except ImportError:
    print("SKIP: mujoco not installed")
    sys.exit(0)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gear_sonic.envs.physx.physx_fk import G1ForwardKinematics

XML_PATH = "/gear_sonic_deploy/g1/g1_29dof_v17.xml"

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Load Python FK
fk = G1ForwardKinematics(XML_PATH)
print(f"Python FK: {fk.num_links()} links")

# Test 1: identity pose
root_pos = np.array([0.0, 0.0, 0.8], dtype=np.float64)
root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
joint_angles = np.zeros(29, dtype=np.float64)

# MuJoCo FK
data.qpos[:3] = root_pos
data.qpos[3:7] = root_quat  # MuJoCo convention matches [w,x,y,z]
data.qpos[7:] = joint_angles
mujoco.mj_kinematics(model, data)

# Python FK
py_poses = fk.compute(root_pos, root_quat, joint_angles)

# Compare all 14 tracked bodies
max_err = 0.0
for name in G1ForwardKinematics.TRACKED_BODIES:
    mj_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    py_idx = fk.get_link_index(name)

    mj_pos = data.xpos[mj_idx].copy()
    mj_quat = data.xquat[mj_idx].copy()  # [w,x,y,z]

    py_pos, py_quat = py_poses[py_idx]

    pos_err = np.linalg.norm(mj_pos - py_pos)
    # Quaternion error: 2*arccos(|dot|)
    dot = abs(np.dot(mj_quat, py_quat))
    dot = min(dot, 1.0)
    quat_err = 2.0 * np.arccos(dot)

    max_err = max(max_err, pos_err)
    if pos_err > 1e-6 or quat_err > 1e-5:
        print(f"  {name}: pos_err={pos_err:.2e}, quat_err={quat_err:.2e}")

print(f"Test 1 (identity pose): max_err={max_err:.2e}")

# Test 2: random joint angles (100 frames)
np.random.seed(42)
errs = []
for frame in range(100):
    root_pos = np.array([0.0, 0.0, 0.8], dtype=np.float64)
    root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    joint_angles = np.random.uniform(-1.0, 1.0, 29).astype(np.float64)

    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat
    data.qpos[7:] = joint_angles
    mujoco.mj_kinematics(model, data)

    py_poses = fk.compute(root_pos, root_quat, joint_angles)

    frame_err = 0.0
    for name in G1ForwardKinematics.TRACKED_BODIES:
        mj_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        py_idx = fk.get_link_index(name)
        mj_pos = data.xpos[mj_idx]
        py_pos, _ = py_poses[py_idx]
        frame_err = max(frame_err, np.linalg.norm(mj_pos - py_pos))
    errs.append(frame_err)

errs = np.array(errs)
print(f"Test 2 (100 random frames): max={errs.max():.2e}, mean={errs.mean():.2e}, >1e-6: {(errs>1e-6).sum()}")

if errs.max() < 1e-6:
    print("T4: PASS (FK accuracy < 1e-6)")
else:
    print(f"T4: FAIL (max error {errs.max():.2e} > 1e-6)")
