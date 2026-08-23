#!/usr/bin/env python3
"""R6-A scene-1 (settle/MySceneCfg): runtime contact/rest offset readout.

Tries every available physics-API route (warp view, isaacsim core view,
USD PhysxCollisionAPI authored values) after 1 sim step.
"""
import argparse

from isaaclab.app import AppLauncher

_plauncher_parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_plauncher_parser)
_args_cli = _plauncher_parser.parse_known_args()[0]
_args_cli.headless = True
app_launcher = AppLauncher(_args_cli)

import omni.kit.app as _oka
_ext_mgr = _oka.get_app_interface().get_extension_manager()
_ext_mgr.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext

from gear_sonic.envs.manager_env.modular_tracking_env_cfg import MySceneCfg
from gear_sonic.envs.manager_env.robots.g1 import G1_CYLINDER_MODEL_12_DEX_CFG

OUT = "/tmp/isaac_r6/offsets_scene1.txt"
lines = []


def emit(s=""):
    print(s, flush=True)
    lines.append(s)


def build_scene(device):
    scene_cfg = MySceneCfg(config=dict(num_envs=1, terrain_type="plane"))
    scene_cfg.robot = G1_CYLINDER_MODEL_12_DEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    sim_cfg = SimulationCfg(
        dt=0.005,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    sim_cfg.physx.gpu_max_rigid_patch_count = 10 * 2**15
    sim_cfg.physx.gpu_collision_stack_size = 2**26
    sim_cfg.device = device
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    return sim, scene, robot


def try_view_methods(robot):
    emit("=== route 1: isaaclab root_physx_view shape-ish methods ===")
    v = robot.root_physx_view
    cands = [m for m in dir(v) if any(k in m.lower() for k in
            ("offset", "contact", "shape", "material", "solver"))]
    emit(f"view type: {type(v).__name__}; candidate methods: {cands}")
    for m in cands:
        try:
            attr = getattr(v, m)
            if callable(attr):
                res = attr()
                if torch.is_tensor(res):
                    emit(f"  {m}() -> tensor shape={tuple(res.shape)} "
                         f"head={res.flatten()[:8].cpu().tolist()}")
                else:
                    emit(f"  {m}() -> {res}")
        except Exception as e:
            emit(f"  {m}() ERR {type(e).__name__}: {e}")


def try_isaacsim_core_view():
    emit("=== route 2: isaacsim.core RigidPrimView ===")
    try:
        from isaacsim.core.prims.rigid_prim_view import RigidPrimView
        rv = RigidPrimView(prim_paths_expr="/World/envs/env_0/Robot/.*", name="probe_rv")
        rv.initialize()
        for prop in ("contact_offset", "rest_offset"):
            try:
                getter = getattr(rv, f"get_{prop}", None)
                if getter is None:
                    emit(f"  rv.get_{prop}: MISSING")
                    continue
                val = getter()
                arr = np.asarray(val).flatten()
                emit(f"  rv.get_{prop}() -> len={len(arr)} head={arr[:10].tolist()}")
            except Exception as e:
                emit(f"  rv.get_{prop} ERR {type(e).__name__}: {e}")
    except Exception as e:
        emit(f"  RigidPrimView import/init ERR: {e}")


def try_usd_authored():
    emit("=== route 3: USD PhysxCollisionAPI authored values (foot links) ===")
    from pxr import Usd, PhysxSchema
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    n_caps, shown = 0, []
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if "/Robot/" not in p:
            continue
        if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            continue
        api = PhysxSchema.PhysxCollisionAPI(prim)
        co = api.GetContactOffsetAttr().Get()
        ro = api.GetRestOffsetAttr().Get()
        if prim.GetTypeName() in ("Capsule", "Sphere", "Cylinder") or "ankle" in p:
            n_caps += 1
            if len(shown) < 16:
                shown.append((p.split("Robot/")[-1], prim.GetTypeName(), co, ro))
    for name, t, co, ro in shown:
        emit(f"  {name:44s} {t:8s} contactOffset={co} restOffset={ro}")
    emit(f"  (collision-API prims shown {len(shown)}, total traversed {n_caps})")
    # ground
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p.startswith("/World/ground") and prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            api = PhysxSchema.PhysxCollisionAPI(prim)
            emit(f"  GROUND {p} {prim.GetTypeName()} "
                 f"contactOffset={api.GetContactOffsetAttr().Get()} "
                 f"restOffset={api.GetRestOffsetAttr().Get()}")


def try_scene_query():
    emit("=== route 4: scene solver params ===")
    try:
        import omni.physx
        from pxr import UsdPhysics, PhysxSchema
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                ps = PhysxSchema.PhysxSceneAPI(prim)
                emit(f"  PhysicsScene {prim.GetPath()}: "
                     f"solverType={ps.GetSolverTypeAttr().Get()} "
                     f"bounceThreshold={ps.GetBounceThresholdVelocityAttr().Get()}")
    except Exception as e:
        emit(f"  ERR {e}")


def main():
    import os
    os.makedirs("/tmp/isaac_r6", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim, scene, robot = build_scene(device)
    # run exactly 1 step as protocol requires
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    robot.write_data_to_sim()
    for _ in range(4):
        sim.step()
    robot.update(0.005)

    emit("#### SCENE 1: settle/MySceneCfg (G1_CYLINDER_MODEL_12_DEX_CFG) ####")
    try_view_methods(robot)
    try_isaacsim_core_view()
    try_usd_authored()
    try_scene_query()

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"saved {OUT}")
    try:
        app_launcher.app.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
