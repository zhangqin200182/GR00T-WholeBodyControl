#!/usr/bin/env python3
"""Generate g1_29dof_capsule_feet.xml: replace foot mesh/sphere collision with the
URDF capsule row that Isaac Sim actually uses (cylinders -> capsules on import).

Isaac (g1.py): UrdfFileCfg(replace_cylinders_with_capsules=True), so the policy
learned foot contact against a row of capsules, not the STL sole mesh.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

URDF = "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
XML_IN = "gear_sonic_deploy/g1/g1_29dof.xml"
XML_OUT = "gear_sonic_deploy/g1/g1_29dof_capsule_feet.xml"


def rpy_to_R(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def foot_capsules(link_name):
    root = ET.parse(URDF).getroot()
    out = []
    for link in root.findall("link"):
        if link.get("name") != link_name:
            continue
        for c in link.findall("collision"):
            cyl = c.find("geometry/cylinder")
            if cyl is None:
                continue
            r = float(cyl.get("radius"))
            L = float(cyl.get("length"))
            o = c.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
            z = rpy_to_R(*rpy) @ np.array([0, 0, 1.0])
            seg_half = max(L / 2 - r, 0)  # capsule total length == cylinder length
            a = xyz - z * seg_half
            b = xyz + z * seg_half
            out.append((r, a, b))
    return out


def disable_collision(geom):
    geom.set("contype", "0")
    geom.set("conaffinity", "0")


def main():
    tree = ET.parse(XML_IN)
    root = tree.getroot()
    for side in ("left", "right"):
        link = f"{side}_ankle_roll_link"
        caps = foot_capsules(link)
        print(f"{link}: {len(caps)} capsules from URDF")
        body = None
        for b in root.iter("body"):
            if b.get("name") == link:
                body = b
                break
        assert body is not None, link
        for g in list(body.findall("geom")):
            # collision sole mesh and the small sole spheres both become visual-only
            if g.get("mesh") == link or "size" in g.attrib:
                disable_collision(g)
        for r, a, b in caps:
            ET.SubElement(
                body, "geom",
                {"type": "capsule", "size": f"{r:.4f}",
                 "fromto": " ".join(f"{v:.5f}" for v in np.concatenate([a, b])),
                 "rgba": "0.9 0.3 0.3 0.4"},
            )
    tree.write(XML_OUT)
    print("wrote", XML_OUT)

    import mujoco
    m = mujoco.MjModel.from_xml_path(XML_OUT)
    print("model loads OK:", m.ngeom, "geoms")


if __name__ == "__main__":
    main()
