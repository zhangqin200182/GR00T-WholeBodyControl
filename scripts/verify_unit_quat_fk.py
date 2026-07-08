#!/usr/bin/env python3
"""Phase A verification: quaternion math + FK reference body."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gear_sonic.envs.mujoco_math import (
    quat_mul, quat_inv, quat_apply, quat_error_magnitude,
    quat_to_matrix, subtract_frame_transforms, quat_diff_to_angvel,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def test_quat_mul_inv():
    """quat_mul(q, quat_inv(q)) ≈ identity."""
    q = np.array([0.5, 0.5, 0.5, 0.5])
    q = q / np.linalg.norm(q)
    result = quat_mul(q, quat_inv(q))
    err = np.max(np.abs(result - IDENTITY))
    assert err < 1e-6, f"quat_mul(quat_inv) error={err}"
    print("  quat_mul + quat_inv: OK")


def test_quat_apply():
    """quat_apply with known rotation."""
    q = np.array([0.0, 0.0, 0.0, 1.0])  # 180° around Z
    v = np.array([1.0, 0.0, 0.0])
    result = quat_apply(q, v)
    expected = np.array([-1.0, 0.0, 0.0])
    err = np.max(np.abs(result - expected))
    assert err < 1e-6, f"quat_apply error={err}"
    print("  quat_apply: OK")


def test_quat_error():
    """quat_error_magnitude with identical quaternions = 0."""
    q = IDENTITY
    err = quat_error_magnitude(q, q)
    assert abs(err) < 1e-10, f"quat_error_magnitude(self)={err}"
    print("  quat_error_magnitude: OK")


def test_quat_to_matrix():
    """quat_to_matrix(identity) = I."""
    m = quat_to_matrix(IDENTITY)
    err = np.max(np.abs(m - np.eye(3)))
    assert err < 1e-6, f"quat_to_matrix(identity) error={err}"
    print("  quat_to_matrix: OK")


def test_subtract_frame():
    """subtract_frame_transforms with same frame = (0, identity)."""
    t01 = np.array([1.0, 2.0, 3.0])
    q01 = IDENTITY
    t_rel, q_rel = subtract_frame_transforms(t01, q01, t01, q01)
    assert np.max(np.abs(t_rel)) < 1e-10, f"t_rel={t_rel}"
    assert np.max(np.abs(q_rel - IDENTITY)) < 1e-10, f"q_rel={q_rel}"
    print("  subtract_frame_transforms: OK")


def test_quat_diff():
    """quat_diff_to_angvel with same quaternion = zero."""
    q = np.tile(IDENTITY, (4, 1))
    vel = quat_diff_to_angvel(q, q, 0.02)
    assert np.max(np.abs(vel)) < 1e-6, f"angvel={vel}"
    print("  quat_diff_to_angvel: OK")


def test_fk_ref_body():
    """MuJoCo FK reference body computation."""
    import mujoco
    xml = "/root/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"

    model = mujoco.MjModel.from_xml_path(xml)
    ref_data = mujoco.MjData(model)

    # Set known joint angles
    ref_data.qpos[7:] = 0.1  # slight joint bend
    ref_data.qvel[:] = 0
    mujoco.mj_kinematics(model, ref_data)

    # BODY_NAMES from design doc
    BODY_NAMES = [
        "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
        "torso_link", "left_shoulder_roll_link", "left_elbow_link",
        "left_wrist_yaw_link", "right_shoulder_roll_link",
        "right_elbow_link", "right_wrist_yaw_link",
    ]

    body_idx = {}
    for name in BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, f"Body '{name}' not found in model"
        body_idx[name] = bid

    # Check all 14 bodies have non-zero positions
    for name in BODY_NAMES:
        xpos = ref_data.xpos[body_idx[name]]
        norm = np.linalg.norm(xpos)
        assert norm > 1e-6, f"{name} xpos is zero vector"
    print(f"  FK ref body: OK ({len(BODY_NAMES)} bodies, all non-zero)")

    # Verify pelvis position
    pelvis_pos = ref_data.xpos[body_idx["pelvis"]]
    print(f"  pelvis pos: ({pelvis_pos[0]:.3f}, {pelvis_pos[1]:.3f}, {pelvis_pos[2]:.3f})")
    assert abs(pelvis_pos[2] - 0.793) < 0.1, f"pelvis z={pelvis_pos[2]:.3f}"

    # Future FK test
    for i in range(10):
        ref_data.qpos[7:] = 0.1 + i * 0.02
        mujoco.mj_kinematics(model, ref_data)
        z = ref_data.xpos[body_idx["pelvis"]][2]
        assert not np.isnan(z), f"Future FK frame {i} is NaN"
    print("  Future FK (10 frames): OK (no NaN)")


if __name__ == "__main__":
    print("Phase A verification: quat + FK")
    test_quat_mul_inv()
    test_quat_apply()
    test_quat_error()
    test_quat_to_matrix()
    test_subtract_frame()
    test_quat_diff()
    test_fk_ref_body()
    print("\nPhase A: PASS")
