#!/usr/bin/env python3
"""Isaac-aligned settle probe (MuJoCo side): pinned root tiers, joint-locked.

Mirrors the Isaac R6-B/C protocol: root pinned each control step to
default-height + offset, joints locked at q0, record foot forces + sole
separation. Compare against Isaac official-env tiers:
  force-bearing band sole_sep [+4.0,+6.3]mm, ~118-156N/foot at -20/-25mm tiers.
"""
import sys
import numpy as np
import mujoco

XML = sys.argv[1] if len(sys.argv) > 1 else \
    "gear_sonic_deploy/g1/g1_29dof_v18_isaac_aligned.xml"

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
DT = m.opt.timestep
print(f"model={XML.split('/')[-1]} timestep={DT} nq={m.nq}")

# standing q: all joints 0 (MuJoCo model zero pose = standing), root upright
q0 = np.zeros(m.nq)
q0[3] = 1.0  # quat w
# find root height that gives sole ~0 clearance: binary via forward kinematics
def root_z_for_sole(target_sole):
    lo, hi = 0.5, 1.2
    for _ in range(50):
        mid = (lo + hi) / 2
        d.qpos[:] = q0; d.qpos[2] = mid
        mujoco.mj_forward(m, d)
        coll = [g for g in range(m.ngeom)
                if m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX)
                and m.geom_contype[g] != 0]
        sole = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in coll)
        if sole > target_sole: hi = mid
        else: lo = mid
    return (lo + hi) / 2

ankle_roll_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
z_touch = root_z_for_sole(0.0)
# sole offset below ankle_roll body origin at standing pose (per model)
d.qpos[:] = q0; d.qpos[2] = z_touch; mujoco.mj_forward(m, d)
sole_below_ankle = d.xpos[ankle_roll_bid][2] - 0.0  # sole at world 0
print(f"touchdown root_z (sole clearance 0) = {z_touch:.4f}  "
      f"sole_below_ankle={-sole_below_ankle:.4f}")

OFFSETS_MM = [8, 6, 5, 4, 2, 0, -6, -25]

for off in OFFSETS_MM:
    z_pin = z_touch + off / 1000.0
    forces, seps = [], []
    for step in range(200):  # 200 control steps, joints locked each step
        d.qpos[:] = q0
        d.qpos[2] = z_pin
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d)
        if step >= 190:
            # total contact force on both feet (all capsule geoms)
            fz_total = 0.0
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = c.geom1, c.geom2
                if m.geom_type[g1] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX) or \
                   m.geom_type[g2] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX):
                    f6 = np.zeros(6)
                    mujoco.mj_contactForce(m, d, i, f6)
                    frame = c.frame.reshape(3, 3)
                    normal_world = frame[0]           # row 0 = contact normal (world)
                    f_world = normal_world * f6[0]    # normal force in world frame
                    fz_total += abs(f_world[2])       # support force (world +z)
            forces.append(fz_total)
            if step == 199: ncon_log = d.ncon
            az = d.xpos[ankle_roll_bid][2]
            seps.append(az - sole_below_ankle)
    print(f"off {off:+4d}mm  pin_z={z_pin:.4f}  ankle_z={np.mean(seps):.4f}  "
          f"sole_sep={np.mean(seps):+.4f}m ncon={ncon_log} "
          f"foot_force={np.mean(forces):8.1f}N", flush=True)
