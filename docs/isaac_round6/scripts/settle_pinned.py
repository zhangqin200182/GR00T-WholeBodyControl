#!/usr/bin/env python3
"""Round-5 B(修正版) + C: PINNED-root contact compliance + per-body probe.

原协议（自由 root settle）在官方软踝增益下 2s 内必倒（7 档实测 root_z→0.087），
无法测穿透-力关系。修正：root 每控制步重写到 默认高度+offset（速度清零），
穿透由 root 高度唯一决定 → 纯接触合规曲线。C 探针同场景读 per-body 力。
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

from pathlib import Path

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
FOOT_BODY_SUBSTRINGS = ("ankle_roll_link", "ankle_pitch_link")


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


def run_pinned_tier(sim, robot, sensor, q_target, root_state_tpl, steps, label, out_lines):
    foot_idx = {n: i for i, n in enumerate(robot.body_names)
                if any(s in n for s in FOOT_BODY_SUBSTRINGS)}
    n = steps
    rec_root = np.zeros((n, 3))
    rec_q = np.zeros((n, len(robot.joint_names)))
    rec_tau = np.zeros((n, len(robot.joint_names)))
    rec_pose = {name: np.zeros((n, 7)) for name in foot_idx}
    rec_force = {name: np.zeros((n, 3)) for name in foot_idx}

    for s in range(n):
        robot.set_joint_position_target(q_target)
        robot.write_root_state_to_sim(root_state_tpl)  # PIN every ctrl step
        # LOCK joints: soft PD alone lets legs sag under gravity (feet never
        # reach ground, empirically proven); rewriting state pins feet too
        robot.write_joint_state_to_sim(q_target, torch.zeros_like(q_target))
        robot.write_data_to_sim()
        for _ in range(DECIMATION):
            sim.step()
        robot.update(SIM_DT)
        sensor.update(SIM_DT)
        rec_root[s] = robot.data.root_pos_w[0].cpu().numpy()
        rec_q[s] = robot.data.joint_pos[0].cpu().numpy()
        rec_tau[s] = robot.data.applied_torque[0].cpu().numpy()
        for name, idx in foot_idx.items():
            rec_pose[name][s, :3] = robot.data.body_pos_w[0, idx].cpu().numpy()
            rec_pose[name][s, 3:] = robot.data.body_quat_w[0, idx].cpu().numpy()
            rec_force[name][s] = sensor.data.net_forces_w[0, idx].cpu().numpy()

    npz = {"root_pos": rec_root, "joint_q": rec_q, "applied_torque": rec_tau}
    for name in foot_idx:
        npz[f"pose_{name}"] = rec_pose[name]
        npz[f"force_{name}"] = rec_force[name]
    np.savez(OUT_DIR / f"pinned_{label}.npz", **npz)

    tail = slice(n - 10, n)
    lines = [f"pinned {label}  ({n} steps, last-10-step means)"]
    for name in foot_idx:
        f = np.linalg.norm(rec_force[name][tail], axis=1).mean()
        p = rec_pose[name][tail, :3].mean(axis=0)
        lines.append(f"  {name}: pos=({p[0]:.4f},{p[1]:.4f},{p[2]:.4f}) |F|={f:8.2f}N")
    lines.append(f"  root_z tail={rec_root[tail, 2].mean():.4f} "
                 f"|tau| tail={np.abs(rec_tau[tail]).max():.2f} Nm")
    txt = "\n".join(lines)
    print(txt, flush=True)
    out_lines.append(txt)
    return rec_root[tail, 2].mean()


def per_body_dump(robot, sensor, label, out_lines):
    nf = sensor.data.net_forces_w[0].cpu().numpy()
    mags = np.linalg.norm(nf, axis=1)
    out_lines.append(f"\n=== [{label}] per-body |net contact force| (N) ===")
    for name, m in zip(robot.body_names, mags):
        if m > 1e-6:
            out_lines.append(f"  {name:32s} {m:10.3f} N")
    out_lines.append(f"  nonzero: {int((mags > 1e-6).sum())} / {len(mags)}")


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/isaac_settle_probe_pinned")
    parser.add_argument("--offsets-mm", default="-5,0,2,5,10,15,20")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim, robot, sensor = build_scene(device)

    # actuator table (deliverable)
    out_lines = ["=== actuator runtime info (per group, joint-mean) ==="]
    for name, act in robot.actuators.items():
        info = {"joints": act.num_joints}
        for attr in ("stiffness", "damping", "armature", "effort_limit"):
            try:
                v = getattr(act, attr)
                info[attr] = round(float(v.float().mean().cpu().item()), 6) if isinstance(v, torch.Tensor) else v
            except Exception:
                pass
        out_lines.append(f"  {name}: {info}")
    out_lines.append("=== end actuator info ===")

    default_q = robot.data.default_joint_pos.clone()
    default_root = robot.data.default_root_state.clone()

    # --- B: pinned compliance tiers ---
    for off in [float(x) for x in args.offsets_mm.split(",")]:
        rs = default_root.clone()
        rs[0, 2] += off / 1000.0
        rs[0, 7:] = 0.0  # zero velocities
        run_pinned_tier(sim, robot, sensor, default_q, rs, args.steps,
                        f"{off:+g}mm", out_lines)

    # --- C1: pinned squat, per-body forces ---
    squat_q = default_q.clone()
    names = list(robot.joint_names)
    for side in ("left", "right"):
        squat_q[0, names.index(f"{side}_hip_pitch_joint")] = -0.75
        squat_q[0, names.index(f"{side}_knee_joint")] = 1.0
        squat_q[0, names.index(f"{side}_ankle_pitch_joint")] = -0.5
    rs = default_root.clone()
    rs[0, 2] -= 0.10  # squat depth: lower root so bent legs keep ground contact
    rs[0, 7:] = 0.0
    run_pinned_tier(sim, robot, sensor, squat_q, rs, args.steps, "squat-100mm", out_lines)
    per_body_dump(robot, sensor, "PINNED-SQUAT", out_lines)

    # standing per-body for reference
    rs = default_root.clone()
    rs[0, 7:] = 0.0
    run_pinned_tier(sim, robot, sensor, default_q, rs, args.steps, "stand0mm", out_lines)
    per_body_dump(robot, sensor, "PINNED-STAND", out_lines)

    text = "\n".join(out_lines)
    with open(OUT_DIR / "pinned_summary.txt", "w") as f:
        f.write(text + "\n")
    print(text, flush=True)
    try:
        app_launcher.app.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
