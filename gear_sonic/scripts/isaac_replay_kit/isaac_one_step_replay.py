#!/usr/bin/env python3
"""Isaac-side one-step replay: measure Isaac's OWN one-step residual under the
same protocol as the PhysX-side A/B plant tests.

For each npz clip and each control step k in [start, end):
  - set full state to npz frame k (qpos[k], qvel[k], root[k]; root velocity by
    differencing frames k-1 -> k, since the npz has no root velocities)
  - feed one command, depending on mode:
      target      joint position target = joint_target[k+1]
                  (bit-exact: the recorded target channel lags one frame,
                   recorded[t] ~= true[t-1], so the target actually active
                   during step k is recorded[k+1]; verified corr 0.99993)
      affine      joint position target = deploy_affine(action_raw[k])
                  (offset + scale * action, isaaclab order; cross-check of the
                   deploy mapping, verified vs joint_target[k+1] err 0.005 rad)
      torque      feedforward torque = applied_torque[k] with PD zeroed
                  (matches the PhysX-side B test; applied_torque is NOT lagged:
                   it pairs with state[k], verified corr 0.97)
      zero        no feed at all, PD zeroed (zero-torque control; our side:
                   GROUND 0.0106 / AIR 0.0021)
      target_lag  joint position target = joint_target[k]  (control: the lagged
                   channel; residual should be much larger than "target")
  - step exactly ONE control step = 4 physics steps @ dt=0.005 (decimation=4)
  - compare against npz qpos[k+1] / qvel[k+1] / applied_torque[k+1]

Output per clip: per-joint signed dq, |dq|, dvel, dtorque arrays plus a
summary table (global + per-joint median |dq|, mean signed dq, max |dq|,
mean |dtorque|, root pos/rot error).

Usage:
    python gear_sonic/scripts/isaac_replay_kit/isaac_one_step_replay.py \
        --npz-dir gear_sonic/scripts/isaac_replay_kit/data \
        --out /tmp/isaac_replay_results
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--npz-dir", required=True, help="directory with release_*.npz")
parser.add_argument("--out", default="/tmp/isaac_replay_results", help="results dir")
parser.add_argument("--start", type=int, default=100, help="first control step k")
parser.add_argument("--end", type=int, default=200, help="one past last control step")
parser.add_argument(
    "--modes",
    default="target,affine,torque,target_lag,zero",
    help="comma-separated subset of {target,affine,torque,target_lag,zero}",
)
parser.add_argument("--clips", default="", help="substring filter on npz filenames")
parser.add_argument("--device", default="cuda", help="cuda or cpu")
args_cli = parser.parse_args()

try:
    app_launcher = AppLauncher({"headless": True})
except TypeError:
    app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import glob  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from gear_sonic.envs.manager_env.modular_tracking_env_cfg import MySceneCfg  # noqa: E402
from gear_sonic.envs.manager_env.robots.g1 import G1_CYLINDER_MODEL_12_DEX_CFG  # noqa: E402

CONTROL_DT = 0.02      # one control step (sim_dt 0.005 x decimation 4)
SIM_DT = 0.005
DECIMATION = 4

# Deploy action affine (policy_parameters.hpp / physx_env.py _ISAAC_ACT_*).
# Tables below are in XML (MuJoCo) joint order; isaaclab[j] = xml[ISAAC_REORDER[j]].
ISAAC_REORDER = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)
ACT_SCALE_XML = np.array(
    [0.3506, 0.3506, 0.5475, 0.3506, 0.4386, 0.4386,   # left leg
     0.3506, 0.3506, 0.5475, 0.3506, 0.4386, 0.4386,   # right leg
     0.5475, 0.4386, 0.4386,                            # waist
     0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,  # left arm
     0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745]  # right arm
)
ACT_OFFSET_XML = np.array(
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     0.0, 0.0, 0.0,
     0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
     0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0]
)
ACT_SCALE = ACT_SCALE_XML[ISAAC_REORDER]    # isaaclab order
ACT_OFFSET = ACT_OFFSET_XML[ISAAC_REORDER]  # isaaclab order


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def root_velocity_at(root_pos, root_quat, k):
    """Root linear/angular velocity at frame k by differencing frames k-1 -> k."""
    lin = (root_pos[k] - root_pos[k - 1]) / CONTROL_DT
    q0 = root_quat[k - 1]
    inv_q0 = q0 * np.array([1.0, -1.0, -1.0, -1.0])
    inv_q0 = inv_q0 / np.dot(inv_q0, inv_q0)
    dq = quat_mul(root_quat[k], inv_q0)
    dq = dq / np.linalg.norm(dq)
    ang = 2.0 * dq[1:] / CONTROL_DT
    return lin, ang


def zero_actuator_gains(robot):
    saved = {}
    for name, act in robot.actuators.items():
        try:
            saved[name] = (act.stiffness.clone(), act.damping.clone())
            act.set_gains(
                kp=torch.zeros_like(act.stiffness),
                kd=torch.zeros_like(act.damping),
            )
        except AttributeError:
            print(f"[warn] actuator '{name}': set_gains unavailable; "
                  "torque mode relies on target==state fallback only")
            saved[name] = None
    return saved


def restore_actuator_gains(robot, saved):
    for name, gains in saved.items():
        if gains is None:
            continue
        kp, kd = gains
        robot.actuators[name].set_gains(kp=kp, kd=kd)


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
    robot = scene["robot"]
    return sim, robot


def main():
    device = args_cli.device
    npz_dir = Path(args_cli.npz_dir)
    out_dir = Path(args_cli.out)
    modes = [m for m in args_cli.modes.split(",") if m]

    files = sorted(npz_dir.glob("release_*.npz"))
    if args_cli.clips:
        files = [f for f in files if args_cli.clips in f.name]
    print(f"clips: {len(files)}  modes: {modes}  window: [{args_cli.start}, {args_cli.end})")

    sim, robot = build_scene(device)
    dev = torch.device(device)

    for f in files:
        print(f"\n=== {f.name} ===")
        d = np.load(f)
        qpos = d["qpos"]
        qvel = d["qvel"]
        root_pos = d["root_pos"]
        root_quat = d["root_quat"]
        action = d["action_raw"]
        joint_target = d["joint_target"]
        applied = d["applied_torque"]
        n_steps = args_cli.end - args_cli.start

        clip_out = out_dir / f.stem
        clip_out.mkdir(parents=True, exist_ok=True)
        summary_lines = []

        for mode in modes:
            if mode in ("torque", "zero"):
                saved_gains = zero_actuator_gains(robot)

            dq_all = np.zeros((n_steps, 29))
            dv_all = np.zeros((n_steps, 29))
            dtau_all = np.zeros((n_steps, 29))
            droot_pos = np.zeros(n_steps)
            droot_rot = np.zeros(n_steps)
            dtgt_affine = np.zeros((n_steps, 29))

            for i, k in enumerate(range(args_cli.start, args_cli.end)):
                # --- set state to npz frame k ---
                lin, ang = root_velocity_at(root_pos, root_quat, k)
                root_state = torch.zeros((1, 13), dtype=torch.float32, device=dev)
                root_state[0, 0:3] = torch.tensor(root_pos[k], dtype=torch.float32)
                root_state[0, 3:7] = torch.tensor(root_quat[k], dtype=torch.float32)
                root_state[0, 7:10] = torch.tensor(lin, dtype=torch.float32)
                root_state[0, 10:13] = torch.tensor(ang, dtype=torch.float32)
                robot.write_root_state_to_sim(root_state)
                robot.write_joint_state_to_sim(
                    torch.tensor(qpos[k][None], dtype=torch.float32, device=dev),
                    torch.tensor(qvel[k][None], dtype=torch.float32, device=dev),
                )

                # --- feed command ---
                if mode == "target":
                    tgt = joint_target[k + 1]
                    robot.set_joint_position_target(torch.tensor(tgt[None], dtype=torch.float32, device=dev))
                elif mode == "target_lag":
                    tgt = joint_target[k]
                    robot.set_joint_position_target(torch.tensor(tgt[None], dtype=torch.float32, device=dev))
                elif mode == "affine":
                    tgt = ACT_OFFSET + ACT_SCALE * action[k]
                    robot.set_joint_position_target(torch.tensor(tgt[None], dtype=torch.float32, device=dev))
                    dtgt_affine[i] = tgt - joint_target[k + 1]
                elif mode == "torque":
                    # PD terms vanish (targets = current state, gains zeroed);
                    # force = feedforward effort target only.
                    robot.set_joint_position_target(
                        torch.tensor(qpos[k][None], dtype=torch.float32, device=dev))
                    robot.set_joint_velocity_target(
                        torch.tensor(qvel[k][None], dtype=torch.float32, device=dev))
                    robot.set_joint_effort_target(
                        torch.tensor(applied[k][None], dtype=torch.float32, device=dev))
                elif mode == "zero":
                    # true zero-torque control: gains zeroed, no effort feed
                    robot.set_joint_position_target(
                        torch.tensor(qpos[k][None], dtype=torch.float32, device=dev))
                    robot.set_joint_velocity_target(
                        torch.tensor(qvel[k][None], dtype=torch.float32, device=dev))
                    robot.set_joint_effort_target(
                        torch.zeros((1, 29), dtype=torch.float32, device=dev))

                # --- one control step = 4 physics substeps, target held ---
                for _ in range(DECIMATION):
                    sim.step()

                q_next = robot.data.joint_pos[0].cpu().numpy()
                v_next = robot.data.joint_vel[0].cpu().numpy()
                tau_applied = robot.data.applied_torque[0].cpu().numpy()
                rp = robot.data.root_pos_w[0].cpu().numpy()
                rq = robot.data.root_quat_w[0].cpu().numpy()

                dq_all[i] = q_next - qpos[k + 1]
                dv_all[i] = v_next - qvel[k + 1]
                dtau_all[i] = tau_applied - applied[k + 1]
                droot_pos[i] = np.linalg.norm(rp - root_pos[k + 1])
                droot_rot[i] = np.linalg.norm(rq - root_quat[k + 1])

            if mode in ("torque", "zero"):
                restore_actuator_gains(robot, saved_gains)

            np.save(clip_out / f"dq_{mode}.npy", dq_all)
            np.save(clip_out / f"dv_{mode}.npy", dv_all)
            np.save(clip_out / f"dtau_{mode}.npy", dtau_all)
            np.save(clip_out / f"droot_{mode}.npy", np.stack([droot_pos, droot_rot], axis=1))
            if mode == "affine":
                np.save(clip_out / "dtgt_affine.npy", dtgt_affine)

            med = np.median(np.abs(dq_all))
            mean = np.abs(dq_all).mean()
            mx = np.abs(dq_all).max()
            per_joint_med = np.median(np.abs(dq_all), axis=0)
            per_joint_bias = dq_all.mean(axis=0)
            dtau_med = np.median(np.abs(dtau_all))
            droot_med = np.median(droot_pos)
            print(f"[{mode}] global med|dq|={med:.5f} mean|dq|={mean:.5f} max|dq|={mx:.5f} "
                  f"med|dtau|={dtau_med:.4f} Nm med root_pos_err={droot_med:.5f} m")
            summary_lines.append(
                f"[{mode}] med|dq|={med:.5f} mean|dq|={mean:.5f} max|dq|={mx:.5f} "
                f"med|dtau|={dtau_med:.4f} med root_err={droot_med:.5f} m"
            )
            summary_lines.append("per-joint median |dq|:")
            summary_lines.append("  " + " ".join(f"{v:.4f}" for v in per_joint_med))
            summary_lines.append("per-joint mean signed dq (bias):")
            summary_lines.append("  " + " ".join(f"{v:+.4f}" for v in per_joint_bias))

        with open(clip_out / "summary.txt", "w") as fh:
            fh.write("\n".join(summary_lines) + "\n")

    print(f"\ndone. results in {out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
