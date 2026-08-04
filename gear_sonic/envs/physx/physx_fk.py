#!/usr/bin/env python3
"""Pure-Python forward kinematics for G1 robot.

Replaces mj_kinematics: given (root_pos, root_quat [w,x,y,z], 29 joint_angles),
computes world poses for all 14 tracked bodies.

All quaternion math matches mujoco_math.py conventions: [w,x,y,z].
"""
import xml.etree.ElementTree as ET
import numpy as np

# Quaternion helpers (same convention as mujoco_math.py)
def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def _quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def _quat_rotate(q, v):
    """Rotate vector v by quaternion q."""
    qv = np.array([0, v[0], v[1], v[2]])
    r = _quat_mul(_quat_mul(q, qv), _quat_inv(q))
    return r[1:]


class G1ForwardKinematics:
    """Forward kinematics for the G1 robot (29-DOF reduced-coordinate model).

    Usage:
        fk = G1ForwardKinematics("g1_29dof_v17.xml")
        poses = fk.compute(root_pos, root_quat, joint_angles)
        # poses[i] = (pos[3], quat[4]) for each tracked body
    """

    # 14 body names tracked by SONIC (order matters for observation building)
    TRACKED_BODIES = (
        "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
        "torso_link", "left_shoulder_roll_link", "left_elbow_link",
        "left_wrist_yaw_link", "right_shoulder_roll_link",
        "right_elbow_link", "right_wrist_yaw_link",
    )

    def __init__(self, xml_path):
        """Parse kinematic chain from MJCF XML.

        Args:
            xml_path: Path to g1_29dof_v17.xml
        """
        tree = ET.parse(xml_path)
        worldbody = tree.find("worldbody")
        pelvis = worldbody.find("body")
        if pelvis is None:
            raise ValueError("no root body in XML")

        # Build kinematic chain: for each body, store:
        #   name, parent_idx, local_pos, local_quat [w,x,y,z], joint_axis
        self.link_names = []
        self.parents = []
        self.local_pos = []
        self.local_quat = []
        self.joint_axes = []       # None for root
        self._has_joint = []       # True if body has a consumable hinge joint

        self._parse_body(pelvis, -1)

        # Map tracked body name → link index
        self._name_to_idx = {}
        for i, name in enumerate(self.link_names):
            self._name_to_idx[name] = i

        # Map tracked body name → order in TRACKED_BODIES
        self._tracked_indices = []
        for name in self.TRACKED_BODIES:
            if name in self._name_to_idx:
                self._tracked_indices.append(self._name_to_idx[name])
            else:
                raise ValueError(f"tracked body '{name}' not found in XML")

    def _parse_body(self, elem, parent_idx):
        """Recursively parse body tree."""
        name = elem.get("name", f"body_{len(self.link_names)}")
        idx = len(self.link_names)
        self.link_names.append(name)
        self.parents.append(parent_idx)
        self.local_pos.append(self._parse_vec3(elem.get("pos", "0 0 0")))
        self.local_quat.append(self._parse_quat(elem.get("quat", "")))

        # Find joint (if any) to get axis. Bodies without a real hinge joint
        # (e.g., fixed attachments) don't consume a joint_angles slot.
        has_joint = False
        axis = np.array([0.0, 0.0, 1.0])  # default Z
        for child in elem:
            if child.tag == "joint":
                jtype = child.get("type", "hinge")
                if jtype != "free":
                    axis = self._parse_vec3(child.get("axis", "0 0 1"))
                    axis = axis / (np.linalg.norm(axis) + 1e-10)
                    has_joint = True
                break
        self.joint_axes.append(axis if parent_idx >= 0 else None)
        self._has_joint.append(has_joint)

        # Recurse into child bodies
        for child in elem:
            if child.tag == "body":
                self._parse_body(child, idx)

    @staticmethod
    def _parse_vec3(s):
        parts = s.strip().split()
        if len(parts) >= 3:
            return np.array([float(x) for x in parts[:3]], dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    @staticmethod
    def _parse_quat(s):
        """Parse MJCF quat [w,x,y,z]."""
        if not s:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        parts = s.strip().split()
        if len(parts) >= 4:
            return np.array([float(parts[0]), float(parts[1]),
                             float(parts[2]), float(parts[3])], dtype=np.float64)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # ── Public API ────────────────────────────────────────────────────

    def compute(self, root_pos, root_quat, joint_angles):
        """Compute world poses for all links.

        Args:
            root_pos:    (3,)  pelvis world position [x,y,z]
            root_quat:   (4,)  pelvis world orientation [w,x,y,z]
            joint_angles:(29,) joint angles in radians, order matching
                          the motor/joint order from the XML actuator list.

        Returns:
            list of (pos[3], quat[4]) for all links in creation order.
            quat is [w,x,y,z] convention.
        """
        n = len(self.link_names)
        world_pos = np.zeros((n, 3), dtype=np.float64)
        world_quat = np.zeros((n, 4), dtype=np.float64)

        # Root
        world_pos[0] = root_pos
        world_quat[0] = root_quat

        # Walk from root to leaves. joint_idx tracks position in the
        # joint_angles array; only bodies with real hinge joints consume a slot.
        joint_idx = 0
        for i in range(1, n):
            p = self.parents[i]
            p_pos, p_quat = world_pos[p], world_quat[p]
            local_pos = self.local_pos[i]
            local_quat = self.local_quat[i]

            if self._has_joint[i]:
                if joint_idx >= len(joint_angles):
                    raise IndexError(f"joint_angles too short: need at least {joint_idx+1}")
                angle = joint_angles[joint_idx]
                axis = self.joint_axes[i]
                rot_quat = _axis_angle_to_quat(axis, angle)
                local_quat = _quat_mul(local_quat, rot_quat)
                joint_idx += 1

            # World transform: T_world = T_parent * T_local
            world_pos[i] = p_pos + _quat_rotate(p_quat, local_pos)
            world_quat[i] = _quat_mul(p_quat, local_quat)

        return list(zip(world_pos, world_quat))

    def get_tracked_poses(self, root_pos, root_quat, joint_angles):
        """Compute world poses for the 14 TRACKED_BODIES.

        Returns:
            List of (pos[3], quat[4]) in TRACKED_BODIES order.
        """
        all_poses = self.compute(root_pos, root_quat, joint_angles)
        return [all_poses[i] for i in self._tracked_indices]

    def get_link_index(self, name):
        """Get link index from body name."""
        return self._name_to_idx[name]

    def num_links(self):
        return len(self.link_names)


def _axis_angle_to_quat(axis, angle):
    """Axis-angle → quaternion [w,x,y,z]."""
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s])
