#!/usr/bin/env python3
"""MJCF → PhysX 5 articulation converter.

Parses G1 MJCF XML (v17) and creates a PxArticulationReducedCoordinate
with full kinematics, mass/inertia, collision shapes, and PD drive config.

Usage:
    import physx_loader
    art = physx_loader.load_g1(px, "g1_29dof_v17.xml")
"""

import xml.etree.ElementTree as ET
import numpy as np
import os


_ISAAC_PD = {
    "hip_pitch": (99.1, 6.3), "hip_roll": (99.1, 6.3), "knee": (99.1, 6.3),
    "hip_yaw": (40.2, 2.6),
    "ankle_pitch": (28.5, 1.8), "ankle_roll": (28.5, 1.8),
    "waist_roll": (28.5, 1.8), "waist_pitch": (28.5, 1.8), "waist_yaw": (40.2, 2.6),
    "shoulder_pitch": (14.3, 0.9), "shoulder_roll": (14.3, 0.9),
    "shoulder_yaw": (14.3, 0.9), "elbow": (14.3, 0.9),
    "wrist_roll": (14.3, 0.9), "wrist_pitch": (16.8, 1.1), "wrist_yaw": (16.8, 1.1),
}


# eACCELERATION kp scaling by joint group.
# Isaac base kp (14-99) is too small for raw PhysX 5 eACCELERATION because
# raw PhysX uses kp directly as an acceleration gain (1/s²), while Isaac's
# omni.physx applies internal inertia normalization (τ = I × kp × ε).
# Scale factors determined by sweep (scripts/sweep_accel_kp.py):
#   legs ×10k, waist ×10k, arms ×200k
_ACCEL_KP_SCALE = {
    "hip": 15000, "knee": 15000, "ankle": 15000,
    "waist": 10000,
    "shoulder": 100000, "elbow": 100000, "wrist": 100000,
}
# Per-group damping ratio ζ = kd/(2√kp) in acceleration domain.
# Determined by sweep: legs prefer overdamped (0.8), arms prefer critical (1.0),
# waist stays at 0.4 (higher values degrade due to coupling).
_DAMPING_ZETA = {
    "hip": 0.8, "knee": 0.8, "ankle": 0.8,
    "waist": 0.4,
    "shoulder": 1.0, "elbow": 1.0, "wrist": 1.0,
}


def _isaac_pd_gains(jname):
    """Return Isaac Sim base PD gains (kp in Nm/rad, kd in Nm/(rad/s)) —
    BEFORE eACCELERATION scaling."""
    for pattern, gains in _ISAAC_PD.items():
        if pattern in jname:
            return gains
    return (100.0, 5.0)


# Per-joint effective inertia M_eff (kg·m²) about the joint axis — subtree
# inertia in the default pose, computed from the XML inertials (Steiner).
# Used by the ANALYTICAL drive-gain mode: kp_drive = k_isaac / M_eff gives
# torque-domain equivalence τ = k_isaac × err through the acceleration drive
# (τ = M × kp_drive × err).  The legacy ×15000 group scaling was swept on the
# BROKEN physics (pre foot-contact fix) and is ~4 orders of magnitude stiffer
# than the analytical values (hip: 1.49e6 vs 110).
_M_EFF = {
    "hip_pitch": 0.902, "hip_roll": 0.773, "hip_yaw": 0.033,
    "knee": 0.112,
    "ankle_pitch": 0.002, "ankle_roll": 0.001,
    "waist_yaw": 0.319, "waist_roll": 0.761, "waist_pitch": 0.601,
    "shoulder_pitch": 0.134, "shoulder_roll": 0.081, "shoulder_yaw": 0.041,
    "elbow": 0.034,
    "wrist_roll": 0.001, "wrist_pitch": 0.005, "wrist_yaw": 0.002,
}

# Isaac per-joint armature (kg·m²) — reflected rotor inertia added to the
# joint-space inertia.  Source: gear_sonic/envs/manager_env/robots/g1.py
# ARMATURE_7520_22/7520_14/5020/4010 constants (identical to
# gear_sonic_deploy policy_parameters.hpp).  Isaac's PD gains are DERIVED
# from these (kp = armature·ω², ω = 10Hz), so they are the source of truth.
_ISAAC_ARMATURE = {
    "hip_pitch": 0.025101925, "hip_roll": 0.025101925, "knee": 0.025101925,
    "hip_yaw": 0.010177520, "waist_yaw": 0.010177520,
    "ankle_pitch": 0.00721945, "ankle_roll": 0.00721945,   # 2 × ARMATURE_5020
    "waist_roll": 0.00721945, "waist_pitch": 0.00721945,   # 2 × ARMATURE_5020
    "shoulder_pitch": 0.003609725, "shoulder_roll": 0.003609725,
    "shoulder_yaw": 0.003609725, "elbow": 0.003609725,
    "wrist_roll": 0.003609725,
    "wrist_pitch": 0.00425, "wrist_yaw": 0.00425,
}

# Isaac velocity_limit_sim per group (rad/s), enforced by the solver via
# per-axis joint-velocity impulses (PhysX setMaxJointVelocity, default 100).
_ISAAC_VEL_LIMIT = {
    "hip_yaw": 32.0,
    "hip_pitch": 20.0, "hip_roll": 20.0, "knee": 20.0,
    "ankle_pitch": 37.0, "ankle_roll": 37.0,
    "waist_roll": 37.0, "waist_pitch": 37.0, "waist_yaw": 32.0,
    "shoulder_pitch": 37.0, "shoulder_roll": 37.0, "shoulder_yaw": 37.0,
    "elbow": 37.0, "wrist_roll": 37.0,
    "wrist_pitch": 22.0, "wrist_yaw": 22.0,
}


def _lookup_joint_table(table, jname):
    """Return the first value whose key is a substring of jname, else None."""
    for pattern, val in table.items():
        if pattern in jname:
            return val
    return None


def _analytical_pd_gains(jname, mult=1.0, kd_mult=1.0):
    """Drive-domain gains: kp_drive = mult × k_isaac / M_eff (1/s²),
    kd_drive = kd_mult × mult × kd_isaac / M_eff (1/s) — torque-domain Isaac
    semantics through the eACCELERATION drive.  Enabled by
    SONIC_PHYSX_DRIVE_ANALYTICAL=1; SONIC_PHYSX_DRIVE_MULT sweeps the kp
    scale, SONIC_PHYSX_DRIVE_KD_MULT sweeps damping independently."""
    kp, kd = _isaac_pd_gains(jname)
    for pattern, meff in _M_EFF.items():
        if pattern in jname:
            return mult * kp / meff, kd_mult * mult * kd / meff
    return mult * kp, kd_mult * mult * kd


def _scaled_pd_gains(jname, drive_type="ACCELERATION"):
    """Return PD gains for the given drive type.

    FORCE: raw Isaac torque-domain gains (kp N·m/rad, kd N·m/(rad/s)) —
    exactly Isaac Sim's articulation drive semantics (torque PD, clamped
    by the actuator forceLimit).  This is the faithful release-weight
    config; the eACCELERATION modes below approximate it via q̈ targets.

    ACCELERATION: kp → kp × group_scale, kd → 2ζ√(scaled_kp) per-group ζ,
    or the ANALYTICAL (kp/M_eff) mode when SONIC_PHYSX_DRIVE_ANALYTICAL=1,
    or RAW (Isaac inertia-normalization hypothesis q̈ = kp×ε + kd×ε̇) when
    SONIC_PHYSX_DRIVE_RAW=1."""
    if drive_type == "FORCE":
        return _isaac_pd_gains(jname)
    if os.environ.get("SONIC_PHYSX_DRIVE_RAW"):
        return _isaac_pd_gains(jname)
    if os.environ.get("SONIC_PHYSX_DRIVE_ANALYTICAL"):
        mult = float(os.environ.get("SONIC_PHYSX_DRIVE_MULT", "1.0"))
        kd_mult = float(os.environ.get("SONIC_PHYSX_DRIVE_KD_MULT", "1.0"))
        return _analytical_pd_gains(jname, mult, kd_mult)
    kp, _kd = _isaac_pd_gains(jname)
    for pattern, scale in _ACCEL_KP_SCALE.items():
        if pattern in jname:
            kp *= scale
            break
    zeta = 0.4  # fallback
    for pattern, z in _DAMPING_ZETA.items():
        if pattern in jname:
            zeta = z
            break
    kd = 2 * zeta * np.sqrt(kp)
    return kp, kd


def _fix_joint_frames(art, parser):
    """[DEPRECATED] Fixes FK but breaks physics stability — do NOT call.

    Setting parentPose via setParentPose after createLink creates a
    mismatch between the body positions (set by createLink) and the joint
    constraint frame (now corrected).  The solver sees this as a constraint
    violation and tears the robot apart.

    createLink(world_poses) already computes parentPose correctly
    (parentPose = child_local in parent body frame).  The remaining
    getGlobalPose FK error (~0.82m) is irrelevant because all observations
    use Python FK (physx_fk.py).
    """
    pass


def _parse_float(s, default=0.0):
    try: return float(s)
    except (ValueError, TypeError): return default


def _parse_vec3(s):
    """Parse "x y z" string → np.array([x,y,z], float64)."""
    parts = s.strip().split()
    if len(parts) >= 3:
        return np.array([float(x) for x in parts[:3]], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def _parse_quat(s):
    """Parse quaternion string → np.array([w,x,y,z], float32).

    MuJoCo quat convention is [w, x, y, z] — same as our numpy convention.
    """
    parts = s.strip().split()
    if len(parts) >= 4:
        return np.array([float(parts[0]), float(parts[1]),
                         float(parts[2]), float(parts[3])], dtype=np.float32)
    return np.array([1, 0, 0, 0], dtype=np.float32)


def load_g1(px, xml_path, pos_iters=8, vel_iters=1, drive_type="ACCELERATION"):
    """Load G1 robot from MJCF XML into a PhysX articulation.

    Args:
        px: physx_core module
        xml_path: Path to g1_29dof_v17.xml
        pos_iters: Solver position iterations (default 8, try 16/32 for precision)
        vel_iters: Solver velocity iterations (default 1, try 2/4 for precision)
        drive_type: "ACCELERATION" (default) or "FORCE"

    Returns:
        px.Articulation (finalized, ready to add to scene)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the first <worldbody> and its first <body> (the root body = pelvis)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> in MJCF XML")

    # Find the actuator element to get motor-joint mapping
    actuator = root.find("actuator")
    motor_joints = {}
    motor_order = []
    if actuator is not None:
        for motor in actuator.findall("motor"):
            jname = motor.get("joint", "")
            if jname:
                motor_order.append(jname)
                motor_joints[jname] = motor

    parser = _MJCFParser(px, motor_joints, motor_order, drive_type=drive_type)
    pelvis = worldbody.find("body")
    if pelvis is None:
        raise ValueError("No root <body> in <worldbody>")
    art = px.Articulation()
    parser.parse_body(art, pelvis, parent_idx=-1)

    # Build flattened arrays for finalize()
    masses     = np.array([l["mass"] for l in parser.links], dtype=np.float32)
    inertias   = np.array([l["inertia"] for l in parser.links], dtype=np.float32).flatten()
    positions  = np.array([l["pos"] for l in parser.links], dtype=np.float32).flatten()
    quats      = np.array([l["quat"] for l in parser.links], dtype=np.float32).flatten()
    parents    = np.array([l["parent"] for l in parser.links], dtype=np.int32)
    # Local poses + real CoM offsets: createLink computes parentPose =
    # parent_CoM⁻¹ * local. With local poses (correct joint frame) and real CoM
    # offsets (correct mass distribution), both FK and physics should be correct.
    com_pos    = np.array([l["com_pos"] for l in parser.links], dtype=np.float32).flatten()
    com_quat   = np.array([l["com_quat"] for l in parser.links], dtype=np.float32).flatten()
    jt_axis    = np.array([j["axis"] for j in parser.joints], dtype=np.int32)
    jt_lower   = np.array([j["lower"] for j in parser.joints], dtype=np.float32)
    jt_upper   = np.array([j["upper"] for j in parser.joints], dtype=np.float32)
    jt_fric    = np.array([j["frictionloss"] for j in parser.joints], dtype=np.float32)

    art.finalize(
        link_masses=masses,
        link_diag_inertia=inertias,
        link_local_pos=positions,
        link_local_quat=quats,
        link_parents=parents,
        link_com_pos=com_pos,
        link_com_quat=com_quat,
        joint_axis=jt_axis,
        joint_lower=jt_lower,
        joint_upper=jt_upper,
        joint_friction=jt_fric,
        position_iters=pos_iters,
        velocity_iters=vel_iters,
        drive_type=drive_type,
        local_poses=True,
    )

    # Isaac alignment (batch 3): armature + velocity limits + depenetration.
    # Default ON (these are Isaac values); env-var kill switches for ablation.
    jnames = [j["name"] for j in parser.joints]
    if os.environ.get("SONIC_PHYSX_ARMATURE", "1") != "0":
        armatures = []
        for n in jnames:
            v = _lookup_joint_table(_ISAAC_ARMATURE, n)
            if v is None:
                raise ValueError(f"No Isaac armature entry for joint {n}")
            armatures.append(v)
        art.set_joint_armatures(np.array(armatures, dtype=np.float32))
    if os.environ.get("SONIC_PHYSX_VEL_LIMIT", "1") != "0":
        vlims = []
        for n in jnames:
            v = _lookup_joint_table(_ISAAC_VEL_LIMIT, n)
            if v is None:
                raise ValueError(f"No Isaac velocity limit entry for joint {n}")
            vlims.append(v)
        art.set_joint_velocity_limits(np.array(vlims, dtype=np.float32))
    if os.environ.get("SONIC_PHYSX_DEPEN", "1") != "0":
        art.set_max_depenetration_velocity(1.0)

    # setParentPose fix DISABLED: it improves PhysX FK (getGlobalPose) from
    # 0.82m → 0.06m but breaks physics stability (NaN at step 11 vs 30-step
    # stable without).  createLink(world_poses) naturally computes parentPose =
    # child_local, which is correct for the joint constraint solver.  The FK
    # error is irrelevant — Python FK (physx_fk.py) supplies all body-tracking
    # positions for observations and rewards (T2 fix).

    # Attach collision shapes
    for shape_info in parser.shapes:
        fn = getattr(art, shape_info["method"])
        fn(*shape_info["args"])

    return art


class _MJCFParser:
    """Incremental MJCF body/joint parser.  Builds up lists of links, joints,
    and shapes for passing to Articulation.finalize()."""

    def __init__(self, px, motor_joints, motor_order, drive_type="ACCELERATION"):
        self.px = px
        self.motor_joints = motor_joints
        self.motor_order = motor_order
        self.drive_type = drive_type
        self.links = []
        self.joints = []
        self.shapes = []

    # ── Joint axis → PhysX enum ──────────────────────────────────────
    @staticmethod
    def _axis_enum(axis_str):
        """MJCF axis string → PhysX int (0=eTWIST/X, 1=eSWING1/Y, 2=eSWING2/Z)."""
        parts = axis_str.strip().split()
        if len(parts) < 3:
            return 0
        v = np.array([abs(float(x)) for x in parts[:3]])
        if v[0] > 0.9: return 0   # X → eTWIST
        if v[1] > 0.9: return 1   # Y → eSWING1
        if v[2] > 0.9: return 2   # Z → eSWING2
        return 0  # default X

    def parse_body(self, art, elem, parent_idx):
        """Recursively parse a <body> element and its children."""
        name = elem.get("name", f"body_{len(self.links)}")
        link_idx = len(self.links)

        # Register link with PhysX articulation
        art.add_link(parent_idx, name)

        self.links.append({
            "name": name,
            "parent": parent_idx,
            "mass": 1.0,
            "inertia": np.array([0.01, 0.01, 0.01], dtype=np.float32),
            "pos": np.zeros(3, dtype=np.float32),
            "quat": np.array([1, 0, 0, 0], dtype=np.float32),
            "com_pos": np.zeros(3, dtype=np.float32),    # CoM offset in body frame
            "com_quat": np.array([1, 0, 0, 0], dtype=np.float32),  # CoM orientation
        })

        # Body local position
        bpos_str = elem.get("pos", "0 0 0")
        self.links[-1]["pos"] = _parse_vec3(bpos_str)
        bquat_str = elem.get("quat", "")
        if bquat_str:
            self.links[-1]["quat"] = _parse_quat(bquat_str)

        # Parse children: inertial, joint, geom, body
        for child in elem:
            tag = child.tag.lower()

            if tag == "inertial":
                self._parse_inertial(child, link_idx)

            elif tag == "joint":
                self._parse_joint(art, child, link_idx, parent_idx)

            elif tag == "geom":
                self._parse_geom(child, link_idx)

            elif tag == "body":
                # Child body → recurse
                # Its joint was already parsed as a sibling <joint> in the parent
                self.parse_body(art, child, link_idx)

    def _parse_inertial(self, elem, link_idx):
        mass = _parse_float(elem.get("mass", "1.0"), 1.0)
        diag = elem.get("diaginertia", "")
        if diag:
            parts = diag.strip().split()
            inertia = np.array([float(x) for x in parts[:3]], dtype=np.float32)
        else:
            inertia = np.array([0.01, 0.01, 0.01], dtype=np.float32)
        self.links[link_idx]["mass"] = mass
        self.links[link_idx]["inertia"] = inertia
        # Center of mass offset in body frame
        ipos = elem.get("pos", "")
        if ipos:
            self.links[link_idx]["com_pos"] = _parse_vec3(ipos)
        iquat = elem.get("quat", "")
        if iquat:
            self.links[link_idx]["com_quat"] = _parse_quat(iquat)

    def _parse_joint(self, art, elem, child_idx, parent_idx):
        """Parse <joint> element inside a <body>.  In MJCF, the joint element
        inside a body defines the connection FROM parent TO this body (child_idx)."""
        jtype = elem.get("type", "hinge")
        if jtype == "free":
            return  # floating base — handled by articulation root

        # Parse axis: "1 0 0" → 0 (X/eTWIST), "0 1 0" → 1 (Y/eSWING1), "0 0 1" → 2 (Z/eSWING2)
        axis_str = elem.get("axis", "0 0 1")
        axis_enum = self._axis_enum(axis_str)
        # range: default depends on joint type
        range_str = elem.get("range", "-1.57 1.57")
        range_parts = range_str.strip().split()
        lower = float(range_parts[0]) if len(range_parts) >= 1 else -1.57
        upper = float(range_parts[1]) if len(range_parts) >= 2 else 1.57

        damping = _parse_float(elem.get("damping", "0"))
        frictionloss = _parse_float(elem.get("frictionloss", "0"))

        # Get actuator force range for this joint
        force_limit = 50.0
        jname = elem.get("name", "")
        actuatorfrcrange = elem.get("actuatorfrcrange", "")
        if actuatorfrcrange:
            parts = actuatorfrcrange.strip().split()
            if len(parts) >= 2:
                force_limit = max(abs(float(parts[0])), abs(float(parts[1])))
        elif jname in self.motor_joints:
            # Try to find torque limit from XML joint->motor mapping
            # The motor element doesn't have gear/torque info directly;
            # actuatorfrcrange on the joint is the source of truth
            pass

        kp, kd = _scaled_pd_gains(jname, drive_type=self.drive_type)

        # Joint connects parent_idx → child_idx
        art.add_joint(parent_idx, child_idx, axis_enum,
                       kp=kp, kd=kd, force_limit=force_limit)

        self.joints.append({
            "name": jname,
            "axis": axis_enum,
            "lower": lower,
            "upper": upper,
            "frictionloss": frictionloss,
            "damping": damping,
            "kp": kp,
            "kd": kd,
            "force_limit": force_limit,
        })

    def _parse_geom(self, elem, link_idx):
        """Parse a <geom> — check if it's a collision geom and add shape."""
        contype = elem.get("contype", "")
        conaffinity = elem.get("conaffinity", "")

        # Skip visual-only geoms (contype=0 and conaffinity=0)
        if contype == "0" and conaffinity == "0":
            return

        gtype = elem.get("type", "sphere")
        pos_str = elem.get("pos", "0 0 0")
        pos = _parse_vec3(pos_str)
        quat_str = elem.get("quat", "")
        quat = _parse_quat(quat_str) if quat_str else np.array([1, 0, 0, 0], dtype=np.float32)

        if gtype == "box":
            size_str = elem.get("size", "0.01 0.01 0.01")
            parts = size_str.strip().split()
            if len(parts) >= 3:
                hx, hy, hz = float(parts[0]), float(parts[1]), float(parts[2])
            else:
                hx = hy = hz = float(parts[0])
            self.shapes.append({
                "method": "attach_box",
                "args": [link_idx, hx, hy, hz, pos, quat],
            })

        elif gtype == "sphere":
            size_str = elem.get("size", "0.01")
            r = float(size_str.strip().split()[0])
            self.shapes.append({
                "method": "attach_sphere",
                "args": [link_idx, r, pos, quat],
            })

        elif gtype == "cylinder":
            # MuJoCo cylinder → PhysX capsule (intentional approximation; PhysX has no
            # native cylinder shape.  Capsule has rounded endcaps vs. cylinder's flat
            # ones.  For shoulder collision, the difference is acceptable.  See §4.4.2.)
            size_str = elem.get("size", "0.01 0.01")
            parts = size_str.strip().split()
            r = float(parts[0])
            hh = float(parts[1]) if len(parts) >= 2 else r
            self.shapes.append({
                "method": "attach_capsule",
                "args": [link_idx, r, hh, pos, quat],
            })

        elif gtype == "capsule":
            size_str = elem.get("size", "0.01 0.01")
            parts = size_str.strip().split()
            r = float(parts[0])
            hh = float(parts[1]) if len(parts) >= 2 else r
            self.shapes.append({
                "method": "attach_capsule",
                "args": [link_idx, r, hh, pos, quat],
            })

        elif gtype == "plane":
            # Ground plane — handled separately via scene.add_ground_plane()
            pass

        elif gtype == "mesh":
            # Mesh collision → approximate with primitive (capsule/box).
            # TODO: PxConvexMeshCooking from STL for exact collision.
            self._add_mesh_fallback(link_idx, pos, quat)

    def _add_mesh_fallback(self, link_idx, pos, quat):
        """Add capsule/box approximation for mesh collision geoms."""
        name = self.links[link_idx]["name"]
        if "pelvis" in name:
            self.shapes.append({"method": "attach_box", "args": [link_idx, 0.15, 0.12, 0.08, pos, quat]})
        elif "knee" in name:
            self.shapes.append({"method": "attach_capsule", "args": [link_idx, 0.05, 0.12, pos, quat]})
        elif "hip_pitch" in name or "hip_roll" in name or "hip_yaw" in name:
            self.shapes.append({"method": "attach_capsule", "args": [link_idx, 0.06, 0.10, pos, quat]})
        elif "ankle_pitch" in name:
            # Skip: the ankle capsule (r=0.04, hh=0.06) extends 10cm below the
            # ankle joint and penetrates the ground ~5cm at the mocap reset
            # height, launching the robot.  The ankle_roll foot box is the
            # actual sole contact surface.
            pass
        elif "torso" in name:
            self.shapes.append({"method": "attach_box", "args": [link_idx, 0.12, 0.15, 0.20, pos, quat]})
        elif "shoulder_yaw" in name:
            self.shapes.append({"method": "attach_capsule", "args": [link_idx, 0.04, 0.10, pos, quat]})
        elif "elbow" in name:
            self.shapes.append({"method": "attach_capsule", "args": [link_idx, 0.04, 0.08, pos, quat]})
        elif "wrist_roll" in name or "wrist_pitch" in name or "wrist_yaw" in name:
            self.shapes.append({"method": "attach_capsule", "args": [link_idx, 0.03, 0.05, pos, quat]})
        elif "rubber_hand" in name:
            self.shapes.append({"method": "attach_box", "args": [link_idx, 0.04, 0.03, 0.06, pos, quat]})
        elif "logo" in name or "head" in name or "waist_support" in name:
            self.shapes.append({"method": "attach_capsule", "args": [link_idx, 0.03, 0.05, pos, quat]})
        else:
            # Unknown body with mesh collision — default to small capsule
            import warnings
            warnings.warn(f"mesh fallback: no approximation for '{name}', using default capsule")
            self.shapes.append({"method": "attach_capsule", "args": [link_idx, 0.04, 0.06, pos, quat]})
        # The key collision shapes (foot boxes, shoulder cylinders) are covered above.
