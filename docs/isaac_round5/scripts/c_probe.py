#!/usr/bin/env python3
"""Round-5 C 收口探针：C1 per-body 力（站姿 + 深屈）+ C2 每脚 manifold 点数。

复用 settle_contact_probe 已验证的场景构建（ContactSensor 直连 + 手动 init +
每步写 target）。零状态假设，只打印实测。
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
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from gear_sonic.envs.manager_env.modular_tracking_env_cfg import MySceneCfg
from gear_sonic.envs.manager_env.robots.g1 import G1_CYLINDER_MODEL_12_DEX_CFG

SIM_DT = 0.005
DECIMATION = 4


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
    sim_cfg.physx.gpu_max_rigid_patch_count = 10 * 2**15
    sim_cfg.physx.gpu_collision_stack_size = 2**26
    sim_cfg.device = device
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    robot = scene["robot"]
    sensor_cfg = ContactSensorCfg(prim_path="/World/envs/env_0/Robot/.*",
                                  update_period=0.0, track_air_time=False,
                                  history_length=0)
    sensor = ContactSensor(sensor_cfg)
    sim.reset()
    sensor._initialize_impl()
    return sim, robot, sensor


def run_pose(sim, robot, sensor, q_target, root_z_offset, steps, label, out_lines):
    default_root = robot.data.default_root_state.clone()
    robot.write_joint_state_to_sim(q_target, torch.zeros_like(q_target))
    root_state = default_root.clone()
    root_state[0, 2] += root_z_offset
    robot.write_root_state_to_sim(root_state)
    for _ in range(steps):
        robot.set_joint_position_target(q_target)
        robot.write_data_to_sim()
        for _ in range(DECIMATION):
            sim.step()
        robot.update(SIM_DT)
        sensor.update(SIM_DT)

    # --- C1: per-body net forces (last step) ---
    nf = sensor.data.net_forces_w[0].cpu().numpy()  # (B, 3)
    mags = np.linalg.norm(nf, axis=1)
    out_lines.append(f"\n=== [{label}] per-body |net force| (N), {len(mags)} bodies ===")
    for name, m in zip(robot.body_names, mags):
        if m > 1e-6:
            out_lines.append(f"  {name:32s} {m:10.3f} N")
    n_zero = int((mags <= 1e-6).sum())
    out_lines.append(f"  -> nonzero: {int((mags > 1e-6).sum())}, zero: {n_zero}")

    # --- C2: contact point counts ---
    view = sensor.contact_physx_view
    try:
        counts = view.get_contact_count(dt=SIM_DT)
        out_lines.append(f"[{label}] contact_physx_view.get_contact_count() -> "
                         f"shape={tuple(counts.shape)} values={counts.cpu().tolist()}")
    except Exception as e:
        out_lines.append(f"[{label}] get_contact_count ERR: {type(e).__name__} {e}")
    try:
        forces = view.get_contact_forces(dt=SIM_DT)
        out_lines.append(f"[{label}] get_contact_forces shape={tuple(forces.shape)}")
    except Exception as e:
        out_lines.append(f"[{label}] get_contact_forces ERR: {type(e).__name__} {e}")

    rp = robot.data.root_pos_w[0].cpu().numpy()
    out_lines.append(f"[{label}] final root_z={rp[2]:.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim, robot, sensor = build_scene(device)
    out_lines = []
    default_q = robot.data.default_joint_pos.clone()

    # 1) standing pose
    run_pose(sim, robot, sensor, default_q, -0.005, 100, "STAND", out_lines)

    # 2) squat / deep-flexion pose (knee 1.0, hip -0.75, ankle ~-0.5 both legs)
    squat_q = default_q.clone()
    names = list(robot.joint_names)
    for side in ("left", "right"):
        squat_q[0, names.index(f"{side}_hip_pitch_joint")] = -0.75
        squat_q[0, names.index(f"{side}_knee_joint")] = 1.0
        squat_q[0, names.index(f"{side}_ankle_pitch_joint")] = -0.5
    run_pose(sim, robot, sensor, squat_q, -0.15, 100, "SQUAT", out_lines)

    text = "\n".join(out_lines)
    print(text, flush=True)
    with open("/tmp/isaac_settle_probe/c_probe.txt", "w") as f:
        f.write(text + "\n")
    try:
        app_launcher.app.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
