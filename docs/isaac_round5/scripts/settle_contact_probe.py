#!/usr/bin/env python3
"""Isaac-side contact compliance settle probe (Round-5 B).

Calibrates the penetration-vs-force relationship of the official URDF
(capsule-feet) robot: spawn at the default standing posture, hold it with the
official PD targets, settle at a series of root-height offsets, and record
raw foot body poses + per-body contact forces at every step.

We (the PhysX side) compute the true capsule-surface lowest point locally
from the recorded body poses with our verified formula (min over
center +/- axis*half - r), so this script intentionally does NOT embed any
geometry assumptions -- raw poses and forces only.

Scene build, launcher, and sim-loop patches mirror the validated
isaac_one_step_replay.py (write_data_to_sim + robot.update are REQUIRED).

Usage:
    python gear_sonic/scripts/isaac_round2_kit/settle_contact_probe.py \
        --out /tmp/isaac_settle_probe
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
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext

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
    sim.reset()
    robot = scene["robot"]
    # per-body contact forces: construct the ContactSensor directly (the scene
    # cfg's sensors field is not a recognized entity type in MySceneCfg);
    # G1 cfg sets activate_contact_sensors=True on the articulation prims
    sensor_cfg = ContactSensorCfg(prim_path="/World/envs/env_0/Robot/.*",
                                  update_period=0.0, track_air_time=False,
                                  history_length=1)
    sensor = ContactSensor(sensor_cfg)
    sensor._initialize_impl()  # manual init after play; we never stop/play again
    return sim, robot, sensor


def dump_actuator_info(robot):
    print("=== actuator runtime info (per group) ===")
    for name, act in robot.actuators.items():
        info = {"joints": act.num_joints}
        for attr in ("stiffness", "damping", "armature", "effort_limit", "velocity_limit"):
            try:
                val = getattr(act, attr)
                if isinstance(val, torch.Tensor):
                    info[attr] = float(val.float().mean().cpu().item())
                else:
                    info[attr] = val
            except Exception:
                pass
        print(f"  {name}: {info}")
    print("=== end actuator info ===", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/isaac_settle_probe", help="results dir")
    parser.add_argument(
        "--offsets-mm",
        default="-5,0,2,5,10,15,20",
        help="comma-separated root_z offsets in mm, relative to default spawn height",
    )
    parser.add_argument("--settle-steps", type=int, default=100,
                        help="control steps per offset (1 step = 4 x 0.005s = 0.02s)")
    args = parser.parse_args()

    offsets_mm = [float(x) for x in args.offsets_mm.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim, robot, sensor = build_scene(device)

    dump_actuator_info(robot)

    names = list(robot.body_names)
    foot_idx = {}
    for i, name in enumerate(names):
        for sub in FOOT_BODY_SUBSTRINGS:
            if sub in name:
                foot_idx[name] = i
    print(f"foot bodies found: {foot_idx}", flush=True)
    if len(foot_idx) != 4:
        print(f"[warn] expected 4 foot bodies (2x ankle_roll + 2x ankle_pitch), "
              f"found {len(foot_idx)}: {list(foot_idx)}", flush=True)

    default_q = robot.data.default_joint_pos.clone()
    default_root = robot.data.default_root_state.clone()
    print(f"default root state: pos={default_root[0, :3].tolist()} "
          f"quat={default_root[0, 3:7].tolist()} (wxyz)")
    print(f"default joint pos: {default_q[0].tolist()}")
    print(f"body names: {names}", flush=True)

    for off_mm in offsets_mm:
        # --- reset to default posture at controlled height ---
        # NOTE: no sim.reset() here -- stop/play drops the articulation drive
        # state (actuator gains/targets read back as zero torque)
        robot.write_joint_state_to_sim(
            default_q, torch.zeros_like(default_q, device=device))
        root_state = default_root.clone()
        root_state[0, 2] += off_mm / 1000.0
        robot.write_root_state_to_sim(root_state)
        n = args.settle_steps
        rec_root = np.zeros((n, 3))
        rec_q = np.zeros((n, len(robot.joint_names)))
        rec_tau = np.zeros((n, len(robot.joint_names)))
        rec_pose = {name: np.zeros((n, 7)) for name in foot_idx}
        rec_force = {name: np.zeros((n, 3)) for name in foot_idx}

        print(f"\n--- offset {off_mm:+.0f} mm: settling {n} control steps ---", flush=True)
        for s in range(n):
            robot.set_joint_position_target(default_q)
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

        npz_path = out_dir / f"offset_{off_mm:+.0f}mm.npz"
        np.savez(
            npz_path,
            root_pos=rec_root,
            joint_q=rec_q,
            applied_torque=rec_tau,
            **{f"pos_{name}": rec_pose[name] for name in foot_idx},
            **{f"force_{name}": rec_force[name] for name in foot_idx},
        )

        # --- summary: last-10-step means ---
        def tail_mean(arr):
            return arr[-10:].mean(axis=0)

        lines = [f"offset {off_mm:+.0f} mm  ({n} steps, last-10-step means)"]
        for name in foot_idx:
            fz = tail_mean(rec_force[name])
            p = tail_mean(rec_pose[name])
            lines.append(
                f"  {name}: pos=({p[0]:.4f},{p[1]:.4f},{p[2]:.4f}) "
                f"quat_wxyz=({p[3]:.4f},{p[4]:.4f},{p[5]:.4f},{p[6]:.4f}) "
                f"|F|={np.linalg.norm(fz):.1f}N Fz={fz[2]:.1f}N")
        lines.append(f"  root_z tail={tail_mean(rec_root)[2]:.4f} "
                     f"|tau| tail={np.abs(tail_mean(rec_tau)).mean():.2f} Nm "
                     f"|q-q0| tail={np.abs(tail_mean(rec_q) - default_q[0].cpu().numpy()).mean():.4f} rad")
        summary = "\n".join(lines)
        print(summary)
        (out_dir / f"offset_{off_mm:+.0f}mm_summary.txt").write_text(summary + "\n")

    print(f"\ndone. results in {out_dir}")
    try:
        simulation_app.close()
    except NameError:
        pass


if __name__ == "__main__":
    main()
