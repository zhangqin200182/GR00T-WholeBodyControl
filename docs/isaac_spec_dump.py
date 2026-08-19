#!/usr/bin/env python3
"""Isaac-side spec dump — unified request round 2, item A (A1-A3).

Run in the Isaac env (with Isaac Lab imported) and redirect output:
    python3 isaac_spec_dump.py > specs.txt

Each section has a "fallback" note — if the API below doesn't match your
setup, extract the same VALUES by hand from your config/USD and write
them into specs.txt directly.  Only the final numbers matter to us.
"""
import json

specs = {}


def report(section, data):
    specs[section] = data
    print(f"=== {section} ===")
    print(json.dumps(data, indent=2, default=str))
    print()


# ── A1: left foot collision geometry ────────────────────────────────
def dump_foot_colliders():
    out = {}
    try:
        from isaaclab.assets import ArticulationCfg
        # Adapt: locate your G1 articulation instance in the env and
        # enumerate its collision shapes (or read the USD foot prims).
        # For each foot-related shape, report: type, dims, local pose.
        print("A1: adapt to your env API — enumerate foot collision shapes")
        print("    and report type/dims/local pose per shape.")
    except Exception as e:
        print(f"A1 fallback (manual): {e}")
    return out


# ── A2: friction parameters ─────────────────────────────────────────
def dump_friction():
    out = {}
    # Adapt: ground plane material + robot foot material static/dynamic
    # friction from your env config (g1.py / ground plane cfg).
    print("A2: adapt — report ground plane and foot material friction values.")
    return out


# ── A3: scene contact parameters ────────────────────────────────────
def dump_scene_contact():
    out = {}
    # Adapt: report (or confirm defaults for) the omni.physx scene:
    #   bounceThresholdVelocity, frictionType, contactOffset, restOffset
    print("A3: adapt — report bounceThresholdVelocity / frictionType /")
    print("    contactOffset / restOffset (or state 'all PhysX defaults').")
    return out


if __name__ == "__main__":
    report("A1_foot_colliders", dump_foot_colliders())
    report("A2_friction", dump_friction())
    report("A3_scene_contact", dump_scene_contact())
