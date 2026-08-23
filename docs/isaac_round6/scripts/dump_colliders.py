import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_known_args()[0]; a.headless = True
app = AppLauncher(a)
import omni.kit.app as _oka
_ext = _oka.get_app_interface().get_extension_manager()
_ext.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import MySceneCfg
from gear_sonic.envs.manager_env.robots.g1 import G1_CYLINDER_MODEL_12_DEX_CFG

scene_cfg = MySceneCfg(config=dict(num_envs=1, terrain_type="plane"))
scene_cfg.robot = G1_CYLINDER_MODEL_12_DEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
sc = SimulationCfg(dt=0.005); sc.device = "cuda"
sim = SimulationContext(sc); scene = InteractiveScene(scene_cfg); sim.reset()

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
stage = omni.usd.get_context().get_stage()
lines = []
def emit(s):
    print(s, flush=True); lines.append(s)

TARGETS = ("waist_roll_link", "left_shoulder_pitch_link",
           "torso_link", "left_shoulder_yaw_link")
emit("#### R6-D collider dump (settle scene USD stage) ####")
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if "/Robot/" not in p:
        continue
    parts = p.split("/")
    link = next((x for x in parts if x in TARGETS), None)
    if link is None:
        continue
    t = prim.GetTypeName()
    if t in ("Xform", "Scope"):
        continue
    info = f"{p}  type={t}"
    if t == "Mesh":
        m = UsdGeom.Mesh(prim)
        Ext = m.GetExtentAttr().Get()
        info += f" extent={Ext}"
    elif t == "Capsule":
        c = UsdGeom.Capsule(prim)
        info += f" r={c.GetRadiusAttr().Get()} h={c.GetHeightAttr().Get()}"
    elif t == "Cube":
        c = UsdGeom.Cube(prim)
        info += f" size={c.GetSizeAttr().Get()}"
    elif t == "Cylinder":
        c = UsdGeom.Cylinder(prim)
        info += f" r={c.GetRadiusAttr().Get()} h={c.GetHeightAttr().Get()}"
    xf = UsdGeom.Xformable(prim)
    tr = xf.ComputeLocalTransform(Usd.TimeCode.Default())
    vals = [round(v, 5) for row in [tr.ExtractTranslation()] for v in row] if hasattr(tr, "ExtractTranslation") else None
    info += f" localT={vals}"
    apis = []
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        apis.append("CollisionAPI")
    if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        pca = PhysxSchema.PhysxCollisionAPI(prim)
        apis.append(f"PhysxCollision(approx={pca.GetApproximationAttr().Get()})")
    if prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
        apis.append("ConvexDecomp")
    info += f"  apis={apis}"
    emit(info)

emit("")
emit("#### filtered pairs (articulation-level) ####")
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if p.endswith("/Robot") or "/Robot" in p:
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            try:
                from pxr import PhysxSchema as PS
                fp = PS.PhysxFilteredPairsAPI.Apply  # existence check only
            except Exception:
                pass
try:
    import omni.physx
    physx_interface = omni.physx.get_physx_interface()
    emit("(pair filter via raw PhysX API not exposed; URDF importer default = parent-child filtering)")
except Exception as e:
    emit(f"physx interface ERR {e}")

with open("/tmp/isaac_r6/collider_dump.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
emit("saved /tmp/isaac_r6/collider_dump.txt")
import os; os._exit(0)
