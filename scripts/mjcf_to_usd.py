#!/usr/bin/env python3
"""MJCF → USD converter for ovphysx articulation loading.

Converts G1 MJCF XML to .usda with PhysicsArticulationRootAPI,
PhysicsRevoluteJoint + PhysicsDriveAPI, and PhysicsMassAPI schemas.

Key advantage over bare PhysX SDK: joint frame (localPos0/localRot0) is
decoupled from body position in the USD.  The PhysX-USD bridge reads these
as explicit setParentPose/setChildPose calls, bypassing the createLink
CoM-identity bug that causes α≈0.02 plateau.

Usage:
    python scripts/mjcf_to_usd.py \\
        gear_sonic/data/robots/g1/g1_29dof.xml \\
        -o g1_29dof_physx.usda

    # Then on NPU:
    # physx.add_usd("g1_29dof_physx.usda")
    # binding = physx.create_tensor_binding("/World/G1/**", TensorType.ARTICULATION_DOF_POSITION)
"""

import argparse
import xml.etree.ElementTree as ET
import numpy as np
import os
import sys
from typing import Optional


# ── Isaac Sim PD gains per joint group ──────────────────────────────
_ISAAC_PD = {
    "hip_pitch": (99.1, 6.3), "hip_roll": (99.1, 6.3), "knee": (99.1, 6.3),
    "hip_yaw": (40.2, 2.6),
    "ankle_pitch": (28.5, 1.8), "ankle_roll": (28.5, 1.8),
    "waist_roll": (28.5, 1.8), "waist_pitch": (28.5, 1.8), "waist_yaw": (40.2, 2.6),
    "shoulder_pitch": (14.3, 0.9), "shoulder_roll": (14.3, 0.9),
    "shoulder_yaw": (14.3, 0.9), "elbow": (14.3, 0.9),
    "wrist_roll": (14.3, 0.9), "wrist_pitch": (16.8, 1.1), "wrist_yaw": (16.8, 1.1),
}


def _isaac_pd(joint_name: str):
    for pattern, gains in _ISAAC_PD.items():
        if pattern in joint_name:
            return gains
    return (100.0, 5.0)


# ── MJCF parsing helpers ─────────────────────────────────────────────

def _parse_vec3(s: str):
    parts = s.strip().split()
    if len(parts) >= 3:
        return np.array([float(x) for x in parts[:3]], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def _parse_quat(s: str):
    """MuJoCo quat [w, x, y, z] → numpy."""
    parts = s.strip().split()
    if len(parts) >= 4:
        return np.array([float(parts[0]), float(parts[1]),
                         float(parts[2]), float(parts[3])], dtype=np.float64)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _parse_float(s: str, default: float = 0.0):
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def _quat_multiply(q1, q2):
    """Quaternion multiplication q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_inverse(q):
    """Quaternion conjugate (inverse for unit quat)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_rotate(q, v):
    """Rotate vector v by quaternion q: q * v * q⁻¹."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    q_inv = _quat_inverse(q)
    result = _quat_multiply(_quat_multiply(q, qv), q_inv)
    return np.array([result[1], result[2], result[3]])


def _axis_to_enum(v):
    """Map a 3D vector to nearest PhysX joint axis enum.
    Returns ("X"|"Y"|"Z", 0|1|2)."""
    v = np.abs(v)
    if v[0] >= v[1] and v[0] >= v[2]:
        return "X", 0
    elif v[1] >= v[2]:
        return "Y", 1
    else:
        return "Z", 2


# ── USD string formatting ────────────────────────────────────────────

def _fmt_vec3(v):
    return f"({v[0]:.6g}, {v[1]:.6g}, {v[2]:.6g})"


def _fmt_quat(q):
    """ovphysx quat format: (w, x, y, z) — note ovphysx uses WXYZ, not standard USD xyzw!"""
    return f"({q[0]:.8g}, {q[1]:.8g}, {q[2]:.8g}, {q[3]:.8g})"


def _fmt_quat_identity():
    return "(1, 0, 0, 0)"


# ── MJCF tree parsing ────────────────────────────────────────────────

class Geom:
    """Collision geometry from MJCF <geom> element."""
    __slots__ = ("geom_type", "pos", "quat", "radius", "half_height",
                 "box_size", "friction", "solref", "solimp")

    def __init__(self, geom_type: str):
        self.geom_type = geom_type  # "cylinder", "box", "sphere", "mesh"
        self.pos = np.zeros(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.radius = 0.0
        self.half_height = 0.0
        self.box_size = np.zeros(3)
        self.friction = (0.5, 0.01, 0.001)
        self.solref = (0.02, 1.0)    # MuJoCo default: (timeconst, dampratio)
        self.solimp = (0.9, 0.95, 0.001, 0.5, 2.0)  # MuJoCo default


class Body:
    __slots__ = ("name", "parent", "children", "pos", "quat",
                 "mass", "diag_inertia", "com_pos", "com_quat",
                 "joint_name", "joint_axis", "joint_range",
                 "actuatorfrcrange", "joint_frictionloss", "joint_damping",
                 "joint_axis_str", "depth", "index", "geoms")

    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.children = []
        self.pos = np.zeros(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.mass = 1.0
        self.diag_inertia = np.array([0.01, 0.01, 0.01])
        self.com_pos = np.zeros(3)
        self.com_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.joint_name = None
        self.joint_axis = None
        self.joint_axis_str = ""
        self.joint_range = (-1.57, 1.57)
        self.actuatorfrcrange = ""
        self.joint_frictionloss = 0.0
        self.joint_damping = 0.0
        self.depth = 0
        self.index = -1
        self.geoms = []


def _parse_mjcf_tree(xml_path: str) -> Body:
    """Parse MJCF and return the root Body tree."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> in MJCF XML")

    pelvis = worldbody.find("body")
    if pelvis is None:
        raise ValueError("No root <body> in <worldbody>")

    root_body = Body(pelvis.get("name", "pelvis"), None)

    def _parse_body(elem, body, depth):
        body.depth = depth
        body.pos = _parse_vec3(elem.get("pos", "0 0 0"))
        body.quat = _parse_quat(elem.get("quat", ""))

        for child in elem:
            tag = child.tag.lower()

            if tag == "inertial":
                body.mass = _parse_float(child.get("mass", "1.0"), 1.0)
                diag = child.get("diaginertia", "")
                if diag:
                    parts = diag.strip().split()
                    body.diag_inertia = np.array([float(x) for x in parts[:3]])
                ipos = child.get("pos", "")
                if ipos:
                    body.com_pos = _parse_vec3(ipos)
                iquat = child.get("quat", "")
                if iquat:
                    body.com_quat = _parse_quat(iquat)

            elif tag == "joint":
                jtype = child.get("type", "hinge")
                if jtype == "free":
                    body.joint_name = "floating_base_joint"
                    continue
                body.joint_name = child.get("name", "")
                body.joint_axis_str = child.get("axis", "0 0 1")
                body.joint_axis = _parse_vec3(body.joint_axis_str)
                range_str = child.get("range", "-1.57 1.57")
                parts = range_str.strip().split()
                body.joint_range = (
                    float(parts[0]) if len(parts) >= 1 else -1.57,
                    float(parts[1]) if len(parts) >= 2 else 1.57,
                )
                body.actuatorfrcrange = child.get("actuatorfrcrange", "")
                body.joint_frictionloss = _parse_float(child.get("frictionloss", "0"))
                body.joint_damping = _parse_float(child.get("damping", "0"))

            elif tag == "geom":
                # Parse collision geometry
                gtype = child.get("type", "sphere")
                contype = child.get("contype", "")
                conaffinity = child.get("conaffinity", "")
                # Skip visual-only geoms (contype="0" AND conaffinity="0")
                if contype == "0" and conaffinity == "0":
                    continue
                g = Geom(gtype)
                g.pos = _parse_vec3(child.get("pos", "0 0 0"))
                g.quat = _parse_quat(child.get("quat", ""))
                if gtype in ("cylinder", "capsule"):
                    size = child.get("size", "")
                    if size:
                        parts = size.strip().split()
                        g.radius = float(parts[0]) if len(parts) >= 1 else 0.01
                        g.half_height = float(parts[1]) if len(parts) >= 2 else 0.01
                elif gtype == "box":
                    size = child.get("size", "")
                    if size:
                        parts = size.strip().split()
                        g.box_size = np.array([float(p) for p in parts[:3]])
                elif gtype == "sphere":
                    size = child.get("size", "0.01")
                    g.radius = float(size.strip().split()[0])
                friction = child.get("friction", "")
                if friction:
                    parts = friction.strip().split()
                    g.friction = tuple(float(p) for p in parts[:3])
                solref = child.get("solref", "")
                if solref:
                    parts = solref.strip().split()
                    g.solref = (float(parts[0]), float(parts[1])) if len(parts) >= 2 else g.solref
                solimp = child.get("solimp", "")
                if solimp:
                    parts = solimp.strip().split()
                    if len(parts) >= 3:
                        g.solimp = tuple(float(p) for p in parts[:5])
                body.geoms.append(g)

            elif tag == "body":
                child_body = Body(child.get("name", f"body_{len(body.children)}"), body)
                _parse_body(child, child_body, depth + 1)
                body.children.append(child_body)

    _parse_body(pelvis, root_body, 0)

    # Assign flat indices via DFS (matching actuator order for joints)
    _assign_indices(root_body, [0])

    return root_body


def _assign_indices(body, counter):
    body.index = counter[0]
    counter[0] += 1
    for child in body.children:
        _assign_indices(child, counter)


def _iter_actuated_joints(root: Body):
    """Yield (body, joint_index) for each actuated joint in DFS order.
    Skips the root body (floating base)."""
    joint_idx = 0
    def _recurse(body):
        nonlocal joint_idx
        for child in body.children:
            if child.joint_name and child.joint_axis is not None:
                yield child, joint_idx
                joint_idx += 1
            yield from _recurse(child)
    yield from _recurse(root)


def _compute_world_poses(root: Body):
    """Compute world-frame position/orientation for all bodies via FK.

    Returns dict: body_index -> {"pos": np.array, "quat": np.array}
    """
    poses = {}

    def _fk(body, parent_world_pos, parent_world_quat):
        # Body frame world pose
        world_pos = parent_world_pos + _quat_rotate(parent_world_quat, body.pos)
        world_quat = _quat_multiply(parent_world_quat, body.quat)
        poses[body.index] = {"pos": world_pos, "quat": world_quat}
        for child in body.children:
            _fk(child, world_pos, world_quat)

    _fk(root, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    return poses


def _collect_all_bodies(root: Body):
    """Yield all bodies (including root) in DFS order."""
    yield root
    for child in root.children:
        yield from _collect_all_bodies(child)


# ── USD generation ────────────────────────────────────────────────────

def _gen_header(up_axis="Z") -> str:
    return f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "{up_axis}"
)
'''


def _gen_body_usda(body: Body, parent_path: str, root_path: str = "/World/G1", indent: int = 8) -> str:
    """Generate USD prim for a body (Capsule for all bodies, Xform wrapper for root)."""
    sp = " " * indent
    sp1 = " " * (indent + 4)
    is_root = (body.parent is None)
    name = body.name.replace('.', '_')
    api_schemas = ['PhysicsCollisionAPI', 'PhysicsRigidBodyAPI']
    schemas_str = '", "'.join(api_schemas)

    lines = []

    if is_root:
        # Root: Xform with articulation API, then separate pelvis_body Capsule
        # Both at same level inside /World/G1/
        lines.append(f'{sp}def Xform "{name}" (')
        lines.append(f'{sp1}prepend apiSchemas = ["PhysicsArticulationRootAPI"]')
        lines.append(f'{sp})')
        lines.append(f'{sp}{{')
        lines.append(f'{sp1}def PhysicsFixedJoint "rootJoint"')
        lines.append(f'{sp1}{{')
        lines.append(f'{sp1}    rel physics:body1 = <{root_path}/{name}_body>')
        lines.append(f'{sp1}    point3f physics:localPos0 = (0, 0, 0)')
        lines.append(f'{sp1}    point3f physics:localPos1 = (0, 0, 0)')
        lines.append(f'{sp1}    quatf physics:localRot0 = (1, 0, 0, 0)')
        lines.append(f'{sp1}    quatf physics:localRot1 = (1, 0, 0, 0)')
        lines.append(f'{sp1}}}')
        lines.append(f'{sp}}}')
        # Pelvis body as Capsule at G1 level (same level as the Xform)
        lines.append(f'{sp}def Capsule "{name}_body" (')
        lines.append(f'{sp1}prepend apiSchemas = ["{schemas_str}"]')
        lines.append(f'{sp})')
        lines.append(f'{sp}{{')
        lines.append(f'{sp1}uniform token axis = "X"')
        lines.append(f'{sp1}double height = 0.01')
        lines.append(f'{sp1}double radius = 0.01')
        lines.append(f'{sp1}point3f physics:centerOfMass = {_fmt_vec3(body.com_pos)}')
        if not np.allclose(body.com_quat, [1, 0, 0, 0]):
            lines.append(f'{sp1}quatf physics:principalAxes = {_fmt_quat(body.com_quat)}')
        lines.append(f'{sp1}float physics:mass = {body.mass:.8g}')
        lines.append(f'{sp1}float3 physics:diagonalInertia = {_fmt_vec3(body.diag_inertia)}')
        lines.append(f'{sp}}}')
    else:
        lines.append(f'{sp}def Capsule "{name}_body" (')
        lines.append(f'{sp1}prepend apiSchemas = ["{schemas_str}"]')
        lines.append(f'{sp})')
        lines.append(f'{sp}{{')
        lines.append(f'{sp1}uniform token axis = "X"')
        lines.append(f'{sp1}double height = 0.01')
        lines.append(f'{sp1}double radius = 0.01')
        lines.append(f'{sp1}point3f physics:centerOfMass = {_fmt_vec3(body.com_pos)}')
        if not np.allclose(body.com_quat, [1, 0, 0, 0]):
            lines.append(f'{sp1}quatf physics:principalAxes = {_fmt_quat(body.com_quat)}')
        lines.append(f'{sp1}float physics:mass = {body.mass:.8g}')
        lines.append(f'{sp1}float3 physics:diagonalInertia = {_fmt_vec3(body.diag_inertia)}')
        lines.append(f'{sp}}}')

    return '\n'.join(lines)


def _gen_joint_usda(body: Body, parent_path: str, root_path: str, indent: int = 8) -> str:
    """Generate a PhysicsRevoluteJoint prim for a joint connecting parent→child."""
    sp = " " * indent
    sp1 = " " * (indent + 4)

    parent_name = body.parent.name.replace('.', '_')
    child_name = body.name.replace('.', '_')
    joint_name = body.joint_name.replace('.', '_')
    is_parent_root = (body.parent.parent is None)

    force_limit = 50.0
    if body.actuatorfrcrange:
        parts = body.actuatorfrcrange.strip().split()
        if len(parts) >= 2:
            force_limit = max(abs(float(parts[0])), abs(float(parts[1])))

    kp, kd = _isaac_pd(body.joint_name)

    axis_parent = body.joint_axis
    axis_joint = _quat_rotate(_quat_inverse(body.quat), axis_parent)
    axis_token, _ = _axis_to_enum(axis_joint)

    local_pos0 = body.pos.copy()
    local_rot0 = body.quat.copy()
    local_pos1 = np.zeros(3)
    local_rot1 = np.array([1.0, 0.0, 0.0, 0.0])

    # Body refs: root's children reference pelvis_body, others reference <name>_body
    body0_ref = f"<{root_path}/pelvis_body>" if is_parent_root else f"<{root_path}/{parent_name}_body>"
    body1_ref = f"<{root_path}/{child_name}_body>"

    lines = []
    lines.append(f'{sp}def PhysicsRevoluteJoint "{joint_name}" (')
    lines.append(f'{sp1}prepend apiSchemas = ["PhysicsDriveAPI:angular"]')
    lines.append(f'{sp})')
    lines.append(f'{sp}{{')

    # Body references
    lines.append(f'{sp1}rel physics:body0 = {body0_ref}')
    lines.append(f'{sp1}rel physics:body1 = {body1_ref}')

    # Joint frame
    lines.append(f'{sp1}point3f physics:localPos0 = {_fmt_vec3(local_pos0)}')
    if np.allclose(local_rot0, [1, 0, 0, 0]):
        lines.append(f'{sp1}quatf physics:localRot0 = {_fmt_quat_identity()}')
    else:
        lines.append(f'{sp1}quatf physics:localRot0 = {_fmt_quat(local_rot0)}')
    lines.append(f'{sp1}point3f physics:localPos1 = {_fmt_vec3(local_pos1)}')
    lines.append(f'{sp1}quatf physics:localRot1 = {_fmt_quat_identity()}')

    # Joint properties (axis, limits)
    lines.append(f'{sp1}uniform token physics:axis = "{axis_token}"')
    lines.append(f'{sp1}float physics:lowerLimit = {body.joint_range[0]:.8g}')
    lines.append(f'{sp1}float physics:upperLimit = {body.joint_range[1]:.8g}')

    # Drive parameters (PD control) — ovphysx uses "drive:angular:physics:*" naming
    lines.append(f'{sp1}float drive:angular:physics:stiffness = {kp:.8g}')
    lines.append(f'{sp1}float drive:angular:physics:damping = {kd:.8g}')
    lines.append(f'{sp1}float drive:angular:physics:maxForce = {force_limit:.8g}')

    lines.append(f'{sp}}}')
    return '\n'.join(lines)


def generate_usd(root_body: Body, out_path: str, robot_name: str = "G1"):
    """Generate the complete .usda file.

    Interleaves body-joint-body-joint in tree order, matching the
    links_chain_sample convention that ovphysx expects.
    """
    num_bodies = len(list(_collect_all_bodies(root_body)))
    joints = list(_iter_actuated_joints(root_body))
    num_joints = len(joints)

    root_path = f"/World/{robot_name}"
    articulation_root = root_body.name.replace('.', '_')

    world_poses = _compute_world_poses(root_body)

    # Adjust body positions for collision geom local offsets
    _apply_geom_offsets(root_body, world_poses)

    # Root world position for relative body placement in articulation
    root_world_pos = world_poses[root_body.index]["pos"].copy()

    def _body_prim_path(body):
        name = body.name.replace('.', '_')
        return f"{root_path}/{articulation_root}/{name}_body"

    # Build body→joint lookup: for each child body, what joint connects it?
    body_to_joint = {}
    for body, j_idx in joints:
        body_to_joint[body.index] = (body, j_idx)

    lines = []
    lines.append(_gen_header())
    lines.append('')
    lines.append(f'def Xform "World"')
    lines.append('{')
    lines.append(f'    def Xform "{robot_name}"')
    lines.append(f'    {{')
    lines.append(f'        def Xform "{articulation_root}" (')
    lines.append(f'            prepend apiSchemas = ["PhysicsArticulationRootAPI"]')
    lines.append(f'        )')
    lines.append(f'        {{')

    # Root pelvis world pose
    pelvis_pose = world_poses[root_body.index]
    lines.append(f'            double3 xformOp:translate = {_fmt_vec3(pelvis_pose["pos"])}')
    lines.append(f'            quatf xformOp:orient = {_fmt_quat(pelvis_pose["quat"])}')
    lines.append(f'            float3 xformOp:scale = (1, 1, 1)')
    lines.append(f'            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]')

    # Root body (pelvis) — floating base: NO root joint, gravity affects COM
    lines.append(f'')
    lines.append(_gen_body_usda_v2(root_body, world_poses, articulation_root, root_path, root_world_pos))

    # Depth-first traversal: for each body, emit its children (body + joint pairs)
    def _emit_children(body, indent=""):
        result = []
        for child in body.children:
            # Emit child body
            result.append('')
            result.append(_gen_body_usda_v2(child, world_poses, articulation_root, root_path, root_world_pos))
            # Emit joint for this child (if actuated)
            if child.index in body_to_joint:
                j_body, _ = body_to_joint[child.index]
                result.append('')
                result.append(_gen_joint_usda_v2(j_body, world_poses, articulation_root, root_path))
            # Recurse
            result.extend(_emit_children(child))
        return result

    for line in _emit_children(root_body):
        lines.append(line)

    lines.append(f'        }}')
    lines.append(f'    }}')
    lines.append(f'}}')
    lines.append('')

    usda_text = '\n'.join(lines)

    with open(out_path, 'w') as f:
        f.write(usda_text)

    return num_bodies, num_joints


def _apply_geom_offsets(root: Body, world_poses: dict):
    """Adjust world positions for bodies that have explicit collision geoms.

    The collision prim is placed at the geom's world position (body frame + geom
    local offset). The joint localPos is adjusted by the inverse offset so the
    kinematic chain stays correct.

    Stores 'frame_pos' (original body frame position) for joint computation
    and updates 'pos' to the geom position for body prim placement.
    """
    for body in _collect_all_bodies(root):
        wp = world_poses[body.index]
        wp["frame_pos"] = wp["pos"].copy()  # save original for joint hinge

        # Only box geoms get position offsets (they're placed at ground-contact
        # positions). Cylinder/sphere offsets are cosmetic and introducing them
        # shifts body prims away from joint hinges, degrading arm chain accuracy.
        boxes = [g for g in body.geoms if g.geom_type == "box"]
        best = boxes[0] if boxes else None

        if best is not None and np.any(best.pos != 0):
            offset_world = _quat_rotate(wp["quat"], best.pos)
            wp["pos"] = wp["pos"] + offset_world
            wp["geom_offset_local"] = best.pos.copy()  # for joint adjustment


def _gen_body_usda_v2(body: Body, world_poses: dict, articulation_root: str, root_path: str, root_world_pos: np.ndarray) -> str:
    """Generate a collision body prim for ovphysx.

    Uses parsed MJCF geom data: Cube for box geoms, Capsule for cylinder/mesh geoms.
    Falls back to mesh_fallback table for mesh-only bodies.

    Body prims are placed at positions RELATIVE to the articulation root, which
    is itself positioned at root_world_pos in world space. This avoids double-
    counting the articulation root offset while keeping bodies at correct
    initial positions for the constraint solver.
    """
    sp = " " * 12
    sp1 = " " * 16
    name = body.name.replace('.', '_')
    wp = world_poses[body.index]

    # Body position relative to articulation root (not world position)
    # Root body stays at (0,0,0); child bodies use their FK position minus root pos
    body_pos_rel = wp["pos"] - root_world_pos

    # Geom local offset in body frame (set by _apply_geom_offsets for box geoms)
    geom_offset = wp.get("geom_offset_local", np.zeros(3))

    # ── Determine collision shape and dimensions ──────────────────────
    boxes = [g for g in body.geoms if g.geom_type == "box"]
    cylinders = [g for g in body.geoms if g.geom_type in ("cylinder", "capsule")]
    spheres = [g for g in body.geoms if g.geom_type == "sphere"]

    if boxes:
        prim_type = "Cube"
        best = boxes[0]
        sx = 2 * best.box_size[0]
        sy = 2 * best.box_size[1]
        sz = 2 * best.box_size[2]
        shape_lines = []
        scale = f"({sx:.6g}, {sy:.6g}, {sz:.6g})"
    elif cylinders:
        prim_type = "Capsule"
        best = cylinders[0]
        shape_lines = [
            f'{sp1}uniform token axis = "X"',
            f'{sp1}double height = {2 * best.half_height:.6g}',
            f'{sp1}double radius = {best.radius:.6g}',
        ]
        scale = "(1, 1, 1)"
    elif spheres:
        prim_type = "Sphere"
        best = spheres[0]
        shape_lines = [f'{sp1}double radius = {best.radius:.6g}']
        scale = "(1, 1, 1)"
    else:
        # Mesh collision geoms — use protective capsule to prevent ground
        # fall-through.  Full-size shapes from _mesh_fallback() would cause
        # self-collision between adjacent articulation links because ovphysx
        # does not expose per-link-pair collision filtering.  The feet
        # (ankle_roll, ankle_pitch) have box/cylinder geoms in the MJCF and
        # are NOT handled here — they get proper collision from the box/
        # cylinder branches above.
        prim_type = "Capsule"
        best = None
        radius, half_height = 0.02, 0.03  # small enough to avoid self-collision
        shape_lines = [
            f'{sp1}uniform token axis = "X"',
            f'{sp1}double height = {2 * half_height:.6g}',
            f'{sp1}double radius = {radius:.6g}',
        ]
        scale = "(1, 1, 1)"

    # ── Contact material parameters (MJCF → PhysX) ───────────────────
    # friction[0] = sliding friction → staticFriction + dynamicFriction
    # solref timeconst → contact/rest offset: smaller = stiffer
    if best is not None:
        friction_val = best.friction[0]
        timeconst = best.solref[0]
    else:
        friction_val = 0.5
        timeconst = 0.02

    # Map solref timeconst to contact offset range
    # MuJoCo default timeconst=0.02 (very stiff) → offset ~0.001
    # v17 ankle timeconst=0.04 (slightly softer) → offset ~0.002
    contact_offset = max(0.001, timeconst * 0.05)

    # ── Build prim ────────────────────────────────────────────────────
    lines = []
    lines.append(f'{sp}def {prim_type} "{name}_body" (')
    lines.append(f'{sp1}prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMaterialAPI", "PhysicsRigidBodyAPI"]')
    lines.append(f'{sp})')
    lines.append(f'{sp}{{')
    for sl in shape_lines:
        lines.append(sl)
    # xformOp:translate = body position relative to articulation root.
    # wp["pos"] already includes geom offset (from _apply_geom_offsets), so
    # body_pos_rel = (FK pos + geom offset) - root FK pos.
    lines.append(f'{sp1}double3 xformOp:translate = {_fmt_vec3(body_pos_rel)}')
    lines.append(f'{sp1}quatf xformOp:orient = (1, 0, 0, 0)')
    lines.append(f'{sp1}float3 xformOp:scale = {scale}')
    lines.append(f'{sp1}uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]')
    # Contact material
    lines.append(f'{sp1}float physics:dynamicFriction = {friction_val:.4g}')
    lines.append(f'{sp1}float physics:staticFriction = {friction_val:.4g}')
    lines.append(f'{sp1}float physics:restitution = 0')
    lines.append(f'{sp1}float physics:contactOffset = {contact_offset:.6g}')
    # Mass properties
    lines.append(f'{sp1}point3f physics:centerOfMass = {_fmt_vec3(body.com_pos)}')
    if not np.allclose(body.com_quat, [1, 0, 0, 0]):
        lines.append(f'{sp1}quatf physics:principalAxes = {_fmt_quat(body.com_quat)}')
    lines.append(f'{sp1}float physics:mass = {body.mass:.8g}')
    lines.append(f'{sp1}float3 physics:diagonalInertia = {_fmt_vec3(body.diag_inertia)}')
    lines.append(f'{sp}}}')
    return '\n'.join(lines)


# ── Mesh collision fallback table ──────────────────────────────────────
# Mirrors physx_loader._add_mesh_fallback().
# Isaac Sim does convex decomposition on STL mesh collision geoms.
# We approximate with Box/Capsule primitives of matching dimensions.
# Format: (prim_type, dims) where dims for Capsule=(radius, half_height)
# and dims for Box=(half_x, half_y, half_z) — same convention as physx_loader.

_MESH_FALLBACK = {
    "pelvis":           ("Cube",    (0.15, 0.12, 0.08)),
    "knee":             ("Capsule", (0.05, 0.12)),
    "hip_pitch":        ("Capsule", (0.06, 0.10)),
    "hip_roll":         ("Capsule", (0.06, 0.10)),
    "hip_yaw":          ("Capsule", (0.06, 0.10)),
    "ankle_pitch":      ("Capsule", (0.04, 0.06)),
    "torso":            ("Cube",    (0.12, 0.15, 0.20)),
    "shoulder_yaw":     ("Capsule", (0.04, 0.10)),
    "elbow":            ("Capsule", (0.04, 0.08)),
    "wrist_roll":       ("Capsule", (0.03, 0.05)),
    "wrist_pitch":      ("Capsule", (0.03, 0.05)),
    "wrist_yaw":        ("Capsule", (0.03, 0.05)),
    "rubber_hand":      ("Cube",    (0.04, 0.03, 0.06)),
    "logo":             ("Capsule", (0.03, 0.05)),
    "head":             ("Capsule", (0.03, 0.05)),
    "waist_support":    ("Capsule", (0.03, 0.05)),
}


def _mesh_fallback(name: str):
    """Look up the fallback (prim_type, dims) for a body name."""
    for pattern, shape_info in _MESH_FALLBACK.items():
        if pattern in name:
            return shape_info
    # Unknown body with mesh collision — default to small protective capsule
    return ("Capsule", (0.04, 0.06))


def _gen_joint_usda_v2(body: Body, world_poses: dict, articulation_root: str, root_path: str) -> str:
    """Generate a PhysicsRevoluteJoint prim nested inside the articulation root Xform.

    Computes localPos0/localPos1 from MuJoCo body kinematics:
    - localPos0: joint hinge offset from parent body frame (in parent local frame)
    - localPos1: joint hinge offset from child body frame (in child local frame)
    """
    sp = " " * 12
    sp1 = " " * 16

    parent = body.parent
    parent_name = parent.name.replace('.', '_')
    child_name = body.name.replace('.', '_')
    joint_name = body.joint_name.replace('.', '_')
    art_root = articulation_root

    parent_wp = world_poses[parent.index]
    child_wp = world_poses[body.index]

    force_limit = 50.0
    if body.actuatorfrcrange:
        parts = body.actuatorfrcrange.strip().split()
        if len(parts) >= 2:
            force_limit = max(abs(float(parts[0])), abs(float(parts[1])))

    kp, kd = _isaac_pd(body.joint_name)

    # Joint hinge is at parent body frame origin (MuJoCo convention: no <joint pos>)
    parent_frame_pos = parent_wp.get("frame_pos", parent_wp["pos"])
    child_frame_pos = child_wp.get("frame_pos", child_wp["pos"])
    parent_quat = parent_wp["quat"]
    child_quat = child_wp["quat"]

    # localPos0: hinge offset from parent body pose frame (hinge = frame origin)
    local_pos0 = np.zeros(3)
    # Adjust for parent collision geom offset (body prim moved to geom position)
    if "geom_offset_local" in parent_wp:
        local_pos0 = local_pos0 - parent_wp["geom_offset_local"]

    # localPos1: hinge offset from child body pose frame, in child local
    offset_world = parent_frame_pos - child_frame_pos
    local_pos1 = _quat_rotate(_quat_inverse(child_quat), offset_world)
    # Adjust for child collision geom offset (body prim moved to geom position)
    if "geom_offset_local" in child_wp:
        local_pos1 = local_pos1 - child_wp["geom_offset_local"]

    # localRot0/localRot1: joint frame orientation relative to each body's pose
    # localRot0 = identity: joint frame = body0's pose frame (world orientation)
    # localRot1 = inverse(body0_orient) * body1_orient: relative orientation
    local_rot0 = np.array([1.0, 0.0, 0.0, 0.0])
    local_rot1 = _quat_multiply(_quat_inverse(parent_quat), child_quat)

    # Joint axis is expressed in the JOINT frame (= body0's pose frame)
    # MJCF axis is in parent body frame = body0's pose frame for identity principalAxes
    axis_token, _ = _axis_to_enum(body.joint_axis)

    # Body references (all bodies are children of articulation root)
    body0_ref = f"<{root_path}/{art_root}/{parent_name}_body>"
    body1_ref = f"<{root_path}/{art_root}/{child_name}_body>"

    lines = []
    lines.append(f'{sp}def PhysicsRevoluteJoint "{joint_name}" (')
    lines.append(f'{sp1}prepend apiSchemas = ["PhysicsDriveAPI:angular"]')
    lines.append(f'{sp})')
    lines.append(f'{sp}{{')

    # Body references
    lines.append(f'{sp1}rel physics:body0 = {body0_ref}')
    lines.append(f'{sp1}rel physics:body1 = {body1_ref}')

    # Joint frame
    lines.append(f'{sp1}point3f physics:localPos0 = {_fmt_vec3(local_pos0)}')
    lines.append(f'{sp1}quatf physics:localRot0 = {_fmt_quat(local_rot0)}')
    lines.append(f'{sp1}point3f physics:localPos1 = {_fmt_vec3(local_pos1)}')
    lines.append(f'{sp1}quatf physics:localRot1 = {_fmt_quat(local_rot1)}')

    # Joint properties (ovphysx USD limits are in DEGREES, not radians!)
    lines.append(f'{sp1}uniform token physics:axis = "{axis_token}"')
    lines.append(f'{sp1}float physics:lowerLimit = {np.degrees(body.joint_range[0]):.8g}')
    lines.append(f'{sp1}float physics:upperLimit = {np.degrees(body.joint_range[1]):.8g}')

    # Drive parameters
    lines.append(f'{sp1}float drive:angular:physics:stiffness = {kp:.8g}')
    lines.append(f'{sp1}float drive:angular:physics:damping = {kd:.8g}')
    lines.append(f'{sp1}float drive:angular:physics:maxForce = {force_limit:.8g}')

    lines.append(f'{sp}}}')
    return '\n'.join(lines)


def _collect_all_joints(root: Body):
    return list(_iter_actuated_joints(root))


# ── Validation (runs after conversion) ────────────────────────────────

def validate_mjcf(xml_path: str) -> dict:
    """Quick validation of MJCF structure before USD conversion."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    issues = []

    # Check actuator count
    actuator = root.find("actuator")
    if actuator is not None:
        motors = actuator.findall("motor")
        num_motors = len(motors)
        if num_motors != 29:
            issues.append(f"Expected 29 motors, found {num_motors}")

    # Check for floating base
    worldbody = root.find("worldbody")
    if worldbody is not None:
        first_body = worldbody.find("body")
        if first_body is not None:
            first_joint = first_body.find("joint")
            if first_joint is not None and first_joint.get("type") != "free":
                issues.append("Root body should have free joint (floating base)")

    # Check body count
    body_count = len(root.findall(".//body"))

    return {
        "num_bodies": body_count,
        "num_motors": num_motors if actuator is not None else 0,
        "issues": issues,
    }


def print_tree(root_body: Body, indent: int = 0):
    """Debug: print the body/joint tree."""
    prefix = "  " * indent
    joint_info = ""
    if root_body.joint_name and root_body.joint_axis is not None:
        axis_str = root_body.joint_axis_str
        kp, kd = _isaac_pd(root_body.joint_name)
        joint_info = f"  [{root_body.joint_name}] axis={axis_str} kp={kp} kd={kd} range={root_body.joint_range}"
    print(f"{prefix}[{root_body.index}] {root_body.name} m={root_body.mass:.3f} I={root_body.diag_inertia}{joint_info}")
    for child in root_body.children:
        print_tree(child, indent + 1)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert G1 MJCF XML to ovphysx-compatible USD"
    )
    parser.add_argument("input", help="Path to MJCF XML file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .usda path (default: <input>.usda)")
    parser.add_argument("--name", default="G1",
                        help="Robot name in USD (/World/<name>/...)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate MJCF, don't generate USD")
    parser.add_argument("--print-tree", action="store_true",
                        help="Print the parsed body/joint tree")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    # Validate
    info = validate_mjcf(args.input)
    print(f"MJCF validation: {info['num_bodies']} bodies, {info['num_motors']} motors")
    if info["issues"]:
        for issue in info["issues"]:
            print(f"  WARNING: {issue}")

    if args.validate_only:
        return

    # Parse tree
    root_body = _parse_mjcf_tree(args.input)

    if args.print_tree:
        print_tree(root_body)

    # Generate USD
    out_path = args.output or os.path.splitext(args.input)[0] + ".usda"
    n_bodies, n_joints = generate_usd(root_body, out_path, args.name)

    # Print summary stats
    joint_list = _collect_all_joints(root_body)
    print(f"\nGenerated: {out_path}")
    print(f"  Bodies: {n_bodies}")
    print(f"  Actuated joints: {n_joints}")
    print(f"  Root body: {root_body.name}")
    print(f"  Articulation root API: /World/{args.name}/{root_body.name}")
    print(f"\nJoint summary:")
    for body, j_idx in joint_list:
        axis_joint = _quat_rotate(_quat_inverse(body.quat), body.joint_axis)
        axis_token, _ = _axis_to_enum(axis_joint)
        kp, kd = _isaac_pd(body.joint_name)
        frc = 50.0
        if body.actuatorfrcrange:
            parts = body.actuatorfrcrange.strip().split()
            if len(parts) >= 2:
                frc = max(abs(float(parts[0])), abs(float(parts[1])))
        print(f"  [{j_idx:2d}] {body.joint_name:35s} {axis_token} "
              f"kp={kp:6.1f} kd={kd:4.1f} flimit={frc:5.0f}")

    print(f"\nTo use with ovphysx on NPU:")
    print(f"  from ovphysx import PhysX, TensorType")
    print(f'  physx = PhysX(device="cpu")')
    print(f'  physx.add_usd("{out_path}")')
    print(f'  binding = physx.create_tensor_binding("/World/{args.name}/**",')
    print(f'      TensorType.ARTICULATION_DOF_POSITION)')


if __name__ == "__main__":
    main()
