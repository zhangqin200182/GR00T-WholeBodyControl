#!/usr/bin/env python3
"""Analytic whole-body composite inertia about ground pivots, from XML inertials.

Reference values for the kick-protocol I_eff probe (frozen-target tip-over
diagnosis): omega(0+) = J*h/I_pivot.  If the impulse-derived I_pivot falls
significantly below these numbers, the RC solver's composite-rigid-body
inertia is defective (D2 static audit and P1 root-fixed tests cannot see it).

Pivots (ground z=0):
  - ankle line midpoint: double-stance small-amplitude CoP pivot
  - toe edge (+10cm forward): CoP-saturated tip-over pivot
Also: I_CoM full tensor (free-air impulse reference), m_total/COM/mgh.

Usage: python3 compute_ipivot.py
"""
import sys
import numpy as np
import xml.etree.ElementTree as ET

from compute_meff import (q_mult, q_rot, q_conj, axis_quat, parse_vec,
                          ACT_OFFSET, XML)

# v3-verified release reset root quat (wxyz); yaw-heavy, small pitch/roll.
RESET_QUAT = np.array([0.7397, 0.0289, 0.0208, -0.672])


def fk_links(xml_path, root_quat=RESET_QUAT):
    tree = ET.parse(xml_path)
    pelvis = tree.getroot().find("worldbody").find("body")
    root_quat = root_quat / np.linalg.norm(root_quat)

    world_T = {"pelvis": np.zeros(3)}
    world_R = {"pelvis": root_quat}
    links = {}
    joints = {}

    def walk(body_el, pname):
        bname = body_el.get("name")
        bpos = parse_vec(body_el.get("pos", "0 0 0"))
        bquat = np.array([float(x) for x in (body_el.get("quat") or "1 0 0 0").split()])
        pT = world_T[pname]
        pR = world_R[pname]
        # MuJoCo convention: joint pos/axis in the CHILD body's local frame
        # (pos="0 0 0" -> joint center at the child body origin).
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
        inertial = body_el.find("inertial")
        links[bname] = {"T": T, "R": R, "inertial": inertial}
        world_T[bname] = T
        world_R[bname] = R
        for child in body_el.findall("body"):
            walk(child, bname)

    walk(pelvis, "pelvis")
    return links, joints, root_quat


def link_com_data(links):
    """(mass, world COM, world rotation, local diag inertia) per link."""
    out = []
    for name, info in links.items():
        inr = info["inertial"]
        if inr is None:
            continue
        m = float(inr.get("mass", "0"))
        if m <= 0:
            continue
        diag = np.array([float(x) for x in inr.get("diaginertia", "0 0 0").split()])
        iq = np.array([float(x) for x in (inr.get("quat") or "1 0 0 0").split()])
        ipos = parse_vec(inr.get("pos", "0 0 0"))
        com_w = info["T"] + q_rot(info["R"], ipos)
        R_i = q_mult(info["R"], iq)
        out.append((name, m, com_w, R_i, diag))
    return out


def rot_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def inertia_tensor(coms, pivot):
    """3x3 composite inertia about pivot (parallel axis over all links)."""
    # Steiner form: I_total = sum(R*diag*R^T + m*(|d|^2 E - d d^T))
    I = np.zeros((3, 3))
    for name, m, c, R, diag in coms:
        Rm = rot_matrix(R)
        I += Rm @ np.diag(diag) @ Rm.T
        d = c - pivot
        I += m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return I


def main():
    links, joints, root_quat = fk_links(XML)
    coms = link_com_data(links)

    m_total = sum(m for _, m, _, _, _ in coms)
    com_w = sum(m * c for _, m, c, _, _ in coms) / m_total
    print(f"m_total = {m_total:.3f} kg")
    print(f"COM    = {np.round(com_w, 4)} m (height {com_w[2]:.4f})")

    # body lateral axis (pitch axis) in world frame
    lat = q_rot(root_quat, np.array([0.0, 1.0, 0.0]))
    fwd = q_rot(root_quat, np.array([1.0, 0.0, 0.0]))

    # ankle line midpoint at ground
    ank = [joints[f"{s}_ankle_pitch_joint"]["world_pos"] for s in ("left", "right")]
    for s, p in zip(("left", "right"), ank):
        print(f"  {s} ankle joint @ {np.round(p, 4)}")
    ank_mid = np.mean(ank, axis=0)
    piv_ankle = np.array([ank_mid[0], ank_mid[1], 0.0])
    piv_toe = piv_ankle + 0.10 * fwd

    for label, piv in (("ankle-line", piv_ankle), ("toe-edge +10cm", piv_toe)):
        I = inertia_tensor(coms, piv)
        I_pitch = lat @ I @ lat
        print(f"\nI about {label} pivot {np.round(piv, 4)}:")
        print(f"  I_pitch (body lateral axis) = {I_pitch:.3f} kg*m^2")
        print(f"  full tensor:\n{np.round(I, 3)}")

    # COM-frame tensor (free-air impulse reference)
    I_com = inertia_tensor(coms, com_w)
    print(f"\nI about COM (free-air reference):")
    print(f"  I_pitch (body lateral axis) = {lat @ I_com @ lat:.3f} kg*m^2")
    print(f"  full tensor:\n{np.round(I_com, 3)}")

    # mgh cross-check (tip-over denominator claim mgh ~ 250 Nm/rad)
    for label, piv in (("ankle-line", piv_ankle), ("toe-edge +10cm", piv_toe)):
        d = com_w - piv
        h = np.linalg.norm(d)
        print(f"\nmgh about {label} = {m_total * 9.81 * h:.1f} Nm/rad "
              f"(COM-pivot dist {h:.3f} m)")

    # root-quat sensitivity: identity root (standing straight)
    links0, joints0, _ = fk_links(XML, np.array([1.0, 0.0, 0.0, 0.0]))
    coms0 = link_com_data(links0)
    ank0 = [joints0[f"{s}_ankle_pitch_joint"]["world_pos"] for s in ("left", "right")]
    piv0 = np.array([*np.mean(ank0, axis=0)[:2], 0.0])
    I0 = inertia_tensor(coms0, piv0)
    print(f"\nidentity-root ankle-line I_pitch = {I0[1, 1]:.3f} kg*m^2 "
          f"(sensitivity check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
