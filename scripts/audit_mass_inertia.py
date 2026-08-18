#!/usr/bin/env python3
"""Static parameter audit: compare XML (our PhysX env) vs URDF (Isaac-side asset).

Extracts per-link mass + inertia (diagonal) from both sources and diffs them.
Usage: python3 audit_mass_inertia.py
"""
import re, sys, xml.etree.ElementTree as ET

XML = "/Users/kevin/code/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof_v17.xml"
URDF = "/Users/kevin/code/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf"


def parse_xml(path):
    tree = ET.parse(path)
    out = {}
    for body in tree.iter("body"):
        name = body.get("name", "?")
        inertial = body.find("inertial")
        if inertial is None:
            continue
        m = float(inertial.get("mass", "0"))
        diag = [float(x) for x in inertial.get("diaginertia", "0 0 0").split()]
        out[name] = {"mass": m, "inertia": diag}
    return out


def parse_urdf(path):
    tree = ET.parse(path)
    out = {}
    for link in tree.iter("link"):
        name = link.get("name", "?")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_el = inertial.find("mass")
        m = float(mass_el.get("value", "0")) if mass_el is not None else 0.0
        inr = inertial.find("inertia")
        if inr is not None:
            diag = [float(inr.get(k, "0")) for k in ("ixx", "iyy", "izz")]
        else:
            diag = [0, 0, 0]
        out[name] = {"mass": m, "inertia": diag}
    return out


def main():
    xml = parse_xml(XML)
    urdf = parse_urdf(URDF)
    print(f"XML links with inertial: {len(xml)}; URDF links with inertial: {len(urdf)}")

    # Try to match by name directly, then by normalized name
    def norm(n):
        return n.lower().replace("_", "").replace("-", "")

    xml_n = {norm(k): k for k in xml}
    urdf_n = {norm(k): k for k in urdf}

    matched = matched_norm = 0
    print(f"\n{'link':28s} {'xml mass':>9s} {'urdf mass':>10s} {'Δmass%':>8s}  "
          f"{'xml ixx/iyy/izz':>26s} {'urdf ixx/iyy/izz':>26s}")
    for un, uk in sorted(urdf_n.items()):
        if un in xml_n:
            xk = xml_n[un]
            x, u = xml[xk], urdf[uk]
            dm = 100 * (u["mass"] - x["mass"]) / max(abs(x["mass"]), 1e-9)
            flag = "  <-- DIFF" if abs(dm) > 5 else ""
            xi = "/".join(f"{v:.4f}" for v in x["inertia"])
            ui = "/".join(f"{v:.4f}" for v in u["inertia"])
            print(f"{xk:28s} {x['mass']:9.4f} {u['mass']:10.4f} {dm:8.1f}  "
                  f"{xi:>26s} {ui:>26s}{flag}")
            matched += 1
        else:
            uu = urdf[uk]
            print(f"{uk:28s} {'-':>9s} {uu['mass']:10.4f} {'-':>8s}  (no XML counterpart)")
    xml_only = [xk for xk in xml if norm(xk) not in urdf_n]
    if xml_only:
        print(f"\nXML-only links: {xml_only}")

    # totals
    xml_total = sum(v["mass"] for v in xml.values())
    urdf_total = sum(v["mass"] for v in urdf.values())
    print(f"\nTOTAL mass: XML={xml_total:.3f} kg, URDF={urdf_total:.3f} kg, "
          f"Δ={urdf_total - xml_total:+.3f} kg ({100*(urdf_total-xml_total)/max(xml_total,1e-9):+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
