#!/usr/bin/env python3
"""Compute per-joint effective subtree inertia M_eff about the joint axis at the
default pose (q = deploy default_angles), from the XML inertials (Steiner).

M_eff_j = Σ_{links in subtree(j)} [ aᵀ R I_diag Rᵀ a + m·|(c − p) × a|² ]
where a = world joint axis, p = world joint origin, c = world link CoM,
R = world link rotation (body quat ⊗ inertial quat).

Used for the ANALYTICAL eACCELERATION drive gains (kp_drive = k_isaac / M_eff).
Usage: python3 compute_meff.py [xml_path]
"""
import sys
import numpy as np
import xml.etree.ElementTree as ET

XML = "/Users/kevin/code/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof_v17.xml"

# Deploy-stack default angles (the reference default pose)
ACT_OFFSET = {
    "left_hip_pitch_joint": -0.312, "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669, "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363, "right_ankle_pitch_joint": -0.363,
    "left_shoulder_pitch_joint": 0.2, "right_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
    "left_elbow_joint": 0.6, "right_elbow_joint": 0.6,
}


def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def q_rot(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]])
    qc = np.array([q[0], -q[1], -q[2], -q[3]])
    return q_mult(q_mult(q, qv), qc)[1:]


def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def axis_quat(axis, angle):
    a = axis / (np.linalg.norm(axis) + 1e-12)
    return np.array([np.cos(angle/2), a[0]*np.sin(angle/2),
                     a[1]*np.sin(angle/2), a[2]*np.sin(angle/2)])


def parse_vec(s):
    return np.array([float(x) for x in s.split()])


def meff(xml_path):
    tree = ET.parse(xml_path)
    pelvis = tree.getroot().find("worldbody").find("body")

    # FK at default pose: world_T[body], world_R[body] (quat), joints get
    # world_pos (child-frame joint pos in world) and world_axis.
    world_T = {"pelvis": np.zeros(3)}
    world_R = {"pelvis": np.array([1.0, 0.0, 0.0, 0.0])}
    links = {}    # name -> {"T", "R", "inertial", "parent"}
    joints = {}   # name -> {"world_pos", "world_axis"}
    joint_child = {}  # joint name -> child body name

    def walk(body_el, pname):
        bname = body_el.get("name")
        bpos = parse_vec(body_el.get("pos", "0 0 0"))
        bquat = np.array([float(x) for x in (body_el.get("quat") or "1 0 0 0").split()])
        pT = world_T[pname]; pR = world_R[pname]
        # Joints declared in this body connect parent -> this body.  MuJoCo
        # convention: the joint pos/axis are in the CHILD body's local frame,
        # so with pos="0 0 0" the joint center coincides with the child body
        # origin (2026-08-20: verified against mujoco xanchor; the old
        # parent-frame placement put the ankle axis at the knee and inflated
        # ankle M_eff ~38x vs the deployed table in physx_loader.py).
        R_joint = pR
        body_joints = []
        for child in body_el.findall("joint"):
            jname = child.get("name")
            if jname == "floating_base_joint":
                continue
            jpos = parse_vec(child.get("pos", "0 0 0"))
            axis = parse_vec(child.get("axis", "0 0 1"))
            qj = ACT_OFFSET.get(jname, 0.0)
            R_joint = q_mult(R_joint, axis_quat(axis, qj))
            body_joints.append((jname, jpos, axis))
        R = q_mult(R_joint, bquat)
        T = pT + q_rot(R_joint, bpos)
        for jname, jpos, axis in body_joints:
            joints[jname] = {
                "world_pos": T + q_rot(R, jpos),
                "world_axis": q_rot(R, axis),
            }
            joint_child[jname] = bname
        inertial = body_el.find("inertial")
        links[bname] = {"T": T, "R": R, "inertial": inertial, "parent": pname}
        world_T[bname] = T; world_R[bname] = R
        for child in body_el.findall("body"):
            walk(child, bname)

    walk(pelvis, "pelvis")

    # subtree of a joint = all descendants of its child body
    children = {l: [] for l in links}
    for l, info in links.items():
        if info["parent"]:
            children[info["parent"]].append(l)

    def descendants(l):
        out = [l]
        for c in children.get(l, []):
            out.extend(descendants(c))
        return out

    results = {}
    for jname, j in joints.items():
        cb = joint_child.get(jname)
        if not cb:
            continue
        sub = descendants(cb)
        axis = j["world_axis"] / np.linalg.norm(j["world_axis"])
        jT = j["world_pos"]
        tot = 0.0
        for ln in sub:
            info = links[ln]
            inr = info["inertial"]
            if inr is None:
                continue
            m = float(inr.get("mass", "0"))
            diag = np.array([float(x) for x in inr.get("diaginertia", "0 0 0").split()])
            iq = np.array([float(x) for x in (inr.get("quat") or "1 0 0 0").split()])
            ipos = parse_vec(inr.get("pos", "0 0 0"))
            com_w = info["T"] + q_rot(info["R"], ipos)
            R_i = q_mult(info["R"], iq)
            a_local = q_rot(q_conj(R_i), axis)
            i_ax = float(np.sum(diag * a_local**2))
            d = com_w - jT
            d_perp2 = float(np.dot(d, d) - np.dot(d, axis)**2)
            tot += i_ax + m * d_perp2
        results[jname] = tot

    groups = {}
    for jname, v in results.items():
        key = jname.replace("left_", "").replace("right_", "").replace("_joint", "")
        groups[key] = v
    return results, groups


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else XML
    results, groups = meff(path)
    print("per-joint M_eff (kg·m²) about the joint axis at default pose:")
    for j in sorted(results):
        print(f"  {j:28s} {results[j]:.4f}")
    print("\nM_EFF table entries (L/R identical):")
    for k in sorted(groups):
        print(f'    "{k}": {groups[k]:.3f},')
    return 0


if __name__ == "__main__":
    sys.exit(main())
