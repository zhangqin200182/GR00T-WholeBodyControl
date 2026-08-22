#!/usr/bin/env python3
"""Isaac-side probe: print what the official scene ACTUALLY contains.

Answers without any simulation beyond spawn:
  - robot collision shapes: prim type (Capsule/Cylinder/Mesh/ConvexMesh),
    radius/height for capsules, mesh approximation mode
  - contact/rest offset authored on robot collision shapes + ground plane
  - physics materials bound to robot collision shapes + ground plane
    (friction/restitution; "unbound" = inherits the scene default material)

Usage:
    python gear_sonic/scripts/isaac_replay_kit/probe_contact_cfg.py \
        --out /tmp/isaac_replay_results/probe.txt
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="/tmp/isaac_replay_results/probe.txt")
parser.add_argument("--device", default="cuda")
args_cli = parser.parse_args()

try:
    app_launcher = AppLauncher({"headless": True})
except TypeError:
    app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema  # noqa: E402

from gear_sonic.envs.manager_env.modular_tracking_env_cfg import MySceneCfg  # noqa: E402
from gear_sonic.envs.manager_env.robots.g1 import G1_CYLINDER_MODEL_12_DEX_CFG  # noqa: E402

SIM_DT = 0.005


def build_scene(device):
    scene_cfg = MySceneCfg(config=dict(num_envs=1, terrain_type="plane"))
    scene_cfg.robot = G1_CYLINDER_MODEL_12_DEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    sim_cfg = SimulationCfg(
        dt=SIM_DT,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    # Official env override_settings (modular_tracking_env_cfg.py)
    sim_cfg.physx.gpu_max_rigid_patch_count = 10 * 2**15
    sim_cfg.physx.gpu_collision_stack_size = 2**26
    sim = SimulationContext(sim_cfg, device=device)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    return sim


def get_physx_collision(prim):
    if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        return None
    api = PhysxSchema.PhysxCollisionAPI(prim)
    co = api.GetContactOffsetAttr().Get()
    ro = api.GetRestOffsetAttr().Get()
    approx = api.GetApproximationAttr().Get()
    return co, ro, approx


def get_material(prim):
    bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not bound:
        return "unbound (scene default)"
    api = PhysxSchema.PhysxMaterialAPI(bound)
    vals = {}
    for name in ("StaticFriction", "DynamicFriction", "Restitution"):
        attr = getattr(api, f"Get{name}Attr")()
        vals[name] = attr.Get()
    return f"{bound.GetPath()} {vals}"


def main():
    sim = build_scene(args_cli.device)
    stage = sim.stage
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("=" * 72)
    emit("ROBOT collision shapes (/World/envs/env_0/Robot):")
    n_caps = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/Robot/" not in path:
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        tname = prim.GetTypeName()
        co, ro, approx = get_physx_collision(prim) or (None, None, None)
        if tname == "Capsule":
            cap = UsdGeom.Capsule(prim)
            r = cap.GetRadiusAttr().Get()
            h = cap.GetHeightAttr().Get()
            n_caps += 1
            emit(f"  {path}  Capsule r={r} height={h} "
                 f"(PhysX halfHeight={h / 2 - r:.5f})  contactOffset={co} restOffset={ro}")
        else:
            emit(f"  {path}  {tname}  approx={approx}  contactOffset={co} restOffset={ro}")
    emit(f"  -> total capsules: {n_caps}")
    emit("=" * 72)
    emit("ROBOT collision material bindings (foot shapes):")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/Robot/" not in path:
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if "ankle" in path.lower() or "foot" in path.lower():
            emit(f"  {path}  material: {get_material(prim)}")
    emit("=" * 72)
    emit("GROUND (/World/ground):")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/ground"):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        tname = prim.GetTypeName()
        co, ro, approx = get_physx_collision(prim) or (None, None, None)
        size = None
        if tname in ("Cube", "Mesh"):
            try:
                size = prim.GetAttribute("size").Get()
            except Exception:
                size = None
        emit(f"  {path}  {tname}  size={size}  approx={approx} "
             f"contactOffset={co} restOffset={ro}  material: {get_material(prim)}")
    emit("=" * 72)

    with open(args_cli.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nprobe report written to {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
