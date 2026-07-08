"""Quaternion math utilities for MuJoCoEnv.  Pure numpy, no isaaclab/torch."""
import numpy as np


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication (w,x,y,z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (inverse for unit quaternions)."""
    inv = q.copy()
    inv[..., 1:] *= -1
    return inv


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q."""
    qv = np.zeros(v.shape[:-1] + (4,), dtype=v.dtype)
    qv[..., 1:] = v
    qv_rot = quat_mul(quat_mul(q, qv), quat_inv(q))
    return qv_rot[..., 1:]


def quat_error_magnitude(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular error between two quaternions (radians)."""
    dot = abs(np.dot(q1, q2))
    dot = min(dot, 1.0)
    return float(2.0 * np.arccos(dot))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def subtract_frame_transforms(
    t01: np.ndarray, q01: np.ndarray,
    t02: np.ndarray, q02: np.ndarray,
) -> tuple:
    """Transform (t02,q02) into frame 1 coordinates. Returns (t_rel, q_rel)."""
    q01_inv = quat_inv(q01)
    q_rel = quat_mul(q01_inv, q02)
    t_rel = quat_apply(q01_inv, t02 - t01)
    return t_rel, q_rel


def quat_diff_to_angvel(q_prev: np.ndarray, q_curr: np.ndarray, dt: float) -> np.ndarray:
    """Approximate angular velocity from consecutive quaternions. (N,4)→(N,3)."""
    dq = quat_mul(q_curr, quat_inv(q_prev))
    ang_vel = 2.0 * dq[..., 1:] / dt
    return np.clip(ang_vel, -100, 100)
